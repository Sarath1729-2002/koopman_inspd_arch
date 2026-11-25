"""VBench-style per-frame metrics over time for generated long videos (own implementation, standard models):
  aesthetic   LAION aesthetic predictor (improved-aesthetic-predictor MLP on CLIP ViT-L/14 image embeddings), 1-10 scale
  clip        text-image alignment: 100 x cosine(CLIP ViT-L/14 image, prompt)
  dino_first  subject/background consistency: cosine(DINOv2 ViT-B/14 CLS of frame, of the first frame)
  dino_adj    temporal consistency: cosine with the previous sampled frame
  flow        motion: mean RAFT-small optical-flow magnitude between frames 2 apart (16 fps), pixels at 240x416
  bright, sat, sharp   mean luminance, saturation (max-min over RGB), Laplacian variance
Frames are sampled at 2 fps.  Writes <video>.metrics.json next to each <video>.mp4 (skips existing).
Usage: python -m sf_cc.metrics --root results_sf"""
import argparse, glob, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

AES_URL = 'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth'


class AesMLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.layers = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(s, x):
        return s.layers(x)


def load_models(dev):
    import open_clip, torchvision
    clip, _, prep = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=dev)
    tok = open_clip.get_tokenizer('ViT-L-14')
    aes = AesMLP().to(dev)
    sd = torch.hub.load_state_dict_from_url(AES_URL, map_location=dev, progress=False)
    aes.load_state_dict(sd); aes.eval()
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(dev).eval()
    raft = torchvision.models.optical_flow.raft_small(weights=torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT).to(dev).eval()
    return dict(clip=clip.eval(), tok=tok, aes=aes, dino=dino, raft=raft)


def read_frames(path):
    import av
    fr = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            fr.append(f.to_ndarray(format='rgb24'))
    return np.stack(fr)  # [T,H,W,3] uint8


IM_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def video_metrics(M, frames, prompt, dev, fps=16, every=8):
    idx = np.arange(0, len(frames), every)
    x = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2).float().div(255)           # [N,3,H,W]
    out = {k: [] for k in ('aesthetic', 'clip', 'dino_first', 'dino_adj', 'flow', 'bright', 'sat', 'sharp')}
    txt = M['clip'].encode_text(M['tok']([prompt[:300]]).to(dev)); txt = txt / txt.norm(dim=-1, keepdim=True)
    feats_d = []
    for b in range(0, len(x), 32):
        xb = x[b:b + 32].to(dev)
        xc = F.interpolate(xb, size=(224, 224), mode='bicubic', align_corners=False)
        e = M['clip'].encode_image(((xc - IM_MEAN.to(dev)) / IM_STD.to(dev)))
        e = e / e.norm(dim=-1, keepdim=True)
        out['aesthetic'] += M['aes'](e.float()).squeeze(-1).tolist()
        out['clip'] += (100 * (e @ txt.T)).squeeze(-1).tolist()
        xd = F.interpolate(xb, size=(224, 392), mode='bilinear', align_corners=False)
        feats_d.append(F.normalize(M['dino']((xd - IN_MEAN.to(dev)) / IN_STD.to(dev)), dim=-1))
        g = xb.mean(1, keepdim=True)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=g.dtype, device=dev).view(1, 1, 3, 3)
        out['sharp'] += F.conv2d(g, lap).flatten(1).var(1).tolist()
        out['bright'] += xb.flatten(1).mean(1).tolist()
        out['sat'] += (xb.max(1).values - xb.min(1).values).flatten(1).mean(1).tolist()
    D = torch.cat(feats_d)
    out['dino_first'] = (D @ D[0]).tolist()
    out['dino_adj'] = [1.0] + (D[1:] * D[:-1]).sum(-1).tolist()
    # motion: RAFT between frame t and t+2 at each sampled t
    pairs = [(i, min(i + 2, len(frames) - 1)) for i in idx]
    for b in range(0, len(pairs), 16):
        pb = pairs[b:b + 16]
        a1 = torch.from_numpy(frames[[p[0] for p in pb]]).permute(0, 3, 1, 2).float().div(255).to(dev)
        a2 = torch.from_numpy(frames[[p[1] for p in pb]]).permute(0, 3, 1, 2).float().div(255).to(dev)
        a1 = F.interpolate(a1, size=(240, 416), mode='bilinear', align_corners=False) * 2 - 1
        a2 = F.interpolate(a2, size=(240, 416), mode='bilinear', align_corners=False) * 2 - 1
        fl = M['raft'](a1, a2)[-1]
        out['flow'] += fl.norm(dim=1).flatten(1).mean(1).tolist()
    out['t_sec'] = (idx / fps).tolist()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_sf'); ap.add_argument('--arms', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = load_models(dev)
    arms = a.arms.split(',') if a.arms else sorted(os.listdir(a.root))
    for arm in arms:
        for mp4 in sorted(glob.glob(f'{a.root}/{arm}/*.mp4')):
            fj = mp4[:-4] + '.metrics.json'
            if os.path.exists(fj):
                continue
            if not os.path.exists(mp4[:-4] + '.json'):   # video from an interrupted run (no meta): skip it
                print('skip (no meta):', mp4, flush=True); continue
            meta = json.load(open(mp4[:-4] + '.json'))
            m = video_metrics(M, read_frames(mp4), meta['prompt'], dev)
            json.dump(m, open(fj + '.tmp', 'w')); os.replace(fj + '.tmp', fj)
            print(f'[metrics] {arm} {os.path.basename(mp4)}  aesthetic first/last {np.mean(m["aesthetic"][:20]):.2f}/'
                  f'{np.mean(m["aesthetic"][-20:]):.2f}  flow {np.mean(m["flow"]):.2f}', flush=True)
    print('done metrics')
