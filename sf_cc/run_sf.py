"""Course correction in Self-Forcing (Wan2.1-T2V-1.3B, chunk-wise autoregressive, DMD 4-step): long videos with a
rolling attention window, with/without decode -> re-encode of each finished chunk BEFORE it is written into the KV cache.

Arms (name tokens joined by '_'):
  base          rolling window of 21 latent frames (Self-Forcing's training length), no sink, no correction
  reencP        every P chunks: chunk latents z <- E(D([previous W latents, z]))[-3:]   (the projection)
  noiseP        control: z <- z + eps, |eps| = |E(D(.)) - z| per sample (same size, random direction)
  sinkK         attention sink: keep the first K latent frames in the cache (StreamingLLM / Deep-Forcing-style baseline)
Per video we save the mp4 (16 fps) and a json with per-chunk drift signal ||E(D(z)) - z||^2 / ||z||^2 and timings.
Run from the repo root:  python -m sf_cc.run_sf --arm reenc1 --prompts 0-31 --latents 240"""
import argparse, json, os, sys, time, numpy as np, torch

SF = os.path.abspath(os.environ.get('SF_DIR', 'ext/Self-Forcing'))


def arm_cfg(arm):
    c = dict(P=0, noise=False, sink=0)
    for q in arm.split('_'):
        if q.startswith('reenc'): c['P'] = int(q[5:])
        elif q.startswith('noise'): c['P'], c['noise'] = int(q[5:]), True
        elif q.startswith('sink'): c['sink'] = int(q[4:])
    return c


def write_mp4(path, frames_u8, fps=16):
    import av
    T, H, W, _ = frames_u8.shape
    with av.open(path + '.tmp.mp4', 'w') as c:
        st = c.add_stream('libx264', rate=fps); st.width, st.height, st.pix_fmt = W, H, 'yuv420p'
        st.options = {'crf': '18'}
        for f in frames_u8:
            for pk in st.encode(av.VideoFrame.from_ndarray(f, format='rgb24')):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
    os.replace(path + '.tmp.mp4', path)


def build(sink, local=21):
    sys.path.insert(0, SF); os.chdir(SF)
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    cfg = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), OmegaConf.load('configs/self_forcing_dmd.yaml'))
    cfg.model_kwargs.local_attn_size = local
    cfg.model_kwargs.sink_size = sink
    pipe = CausalInferencePipeline(cfg, device='cuda')
    sd = torch.load('checkpoints/self_forcing_dmd.pt', map_location='cpu')
    missing = pipe.generator.load_state_dict(sd['generator_ema'])
    print('generator load:', missing, flush=True)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to('cuda'); pipe.generator.to('cuda'); pipe.vae.to('cuda')
    return pipe, cfg


@torch.no_grad()
def generate(pipe, cfg, prompt, n_lat, c, seed, W=3, diag_every=4):
    """Re-implementation of CausalInferencePipeline.inference (T2V, no initial latent) with the correction hook."""
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn([1, n_lat, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
    torch.manual_seed(seed)
    B, F_ = 1, pipe.num_frame_per_block
    cond = pipe.text_encoder(text_prompts=[prompt])
    if pipe.kv_cache1 is None:
        pipe._initialize_kv_cache(batch_size=B, dtype=noise.dtype, device=dev)
        pipe._initialize_crossattn_cache(batch_size=B, dtype=noise.dtype, device=dev)
    else:
        for b in range(pipe.num_transformer_blocks):
            pipe.crossattn_cache[b]['is_init'] = False
        for b in range(len(pipe.kv_cache1)):
            pipe.kv_cache1[b]['global_end_index'] = torch.tensor([0], dtype=torch.long, device=dev)
            pipe.kv_cache1[b]['local_end_index'] = torch.tensor([0], dtype=torch.long, device=dev)
    out = torch.zeros_like(noise)
    shifts, diag_time = [], 0.0
    torch.cuda.synchronize(); t0 = time.time()
    cur = 0
    for blk in range(n_lat // F_):
        x = noise[:, cur:cur + F_]
        for i, ts in enumerate(pipe.denoising_step_list):
            t = torch.ones([B, F_], device=dev, dtype=torch.int64) * ts
            _, z = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=t, kv_cache=pipe.kv_cache1,
                                  crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                x = pipe.scheduler.add_noise(z.flatten(0, 1), torch.randn_like(z.flatten(0, 1)),
                                             nt * torch.ones([B * F_], device=dev, dtype=torch.long)).unflatten(0, z.shape[:2])
        act = c['P'] > 0 and (blk + 1) % c['P'] == 0 and blk > 0
        diag = (not act) and blk % diag_every == 0
        if act or diag:
            torch.cuda.synchronize(); td = time.time()
            win = torch.cat([out[:, max(0, cur - W):cur], z], 1)
            pix = pipe.vae.decode_to_pixel(win)                                   # [B,T,C,H,W] in [-1,1]
            zr = pipe.vae.encode_to_latent(pix.permute(0, 2, 1, 3, 4).to(torch.bfloat16))[:, -F_:].to(z.dtype)
            s = ((zr.float() - z.float()) ** 2).flatten(1).sum(1) / (z.float() ** 2).flatten(1).sum(1)
            shifts.append([blk, float(s.mean())])
            if act:
                if c['noise']:
                    d = (zr.float() - z.float()).flatten(1).norm(dim=1)
                    e = torch.randn_like(z, dtype=torch.float32)
                    e = e / e.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1) * d.view(-1, 1, 1, 1, 1)
                    z = (z.float() + e).to(z.dtype)
                else:
                    z = zr
            torch.cuda.synchronize()
            if diag:
                diag_time += time.time() - td
        out[:, cur:cur + F_] = z
        pipe.generator(noisy_image_or_video=z, conditional_dict=cond,
                       timestep=torch.ones([B, F_], device=dev, dtype=torch.int64) * cfg.context_noise,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=cur * pipe.frame_seq_length)
        cur += F_
    torch.cuda.synchronize(); gen_time = time.time() - t0 - diag_time
    video = pipe.vae.decode_to_pixel(out)
    pipe.vae.model.clear_cache()
    video = ((video * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)[0].permute(0, 2, 3, 1).cpu().numpy()
    return video, shifts, gen_time


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='base'); ap.add_argument('--prompts', default='0-31')
    ap.add_argument('--latents', type=int, default=240, help='latent frames (x4 = video frames at 16 fps); 240 = 60 s')
    ap.add_argument('--out', default='results_sf'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    root = os.getcwd(); out = os.path.join(root, a.out, a.arm); os.makedirs(out, exist_ok=True)
    lo, hi = (int(x) for x in a.prompts.split('-'))
    c = arm_cfg(a.arm)
    pipe, cfg = build(c['sink'])
    ext = [l.strip() for l in open('prompts/MovieGenVideoBench_extended.txt', encoding='utf-8')]
    short = [l.strip() for l in open('prompts/MovieGenVideoBench.txt', encoding='utf-8')]
    n_lat = 9 if a.smoke else a.latents
    for idx in range(lo, hi + 1):
        f = os.path.join(out, f'{idx:04d}.json')
        if os.path.exists(f) and not a.smoke:
            continue
        video, shifts, gt = generate(pipe, cfg, ext[idx], n_lat, c, seed=1000 + idx)
        write_mp4(os.path.join(out, f'{idx:04d}.mp4'), video)
        json.dump({'arm': a.arm, 'cfg': c, 'idx': idx, 'prompt': short[idx], 'latents': n_lat, 'frames': int(video.shape[0]),
                   'shift': shifts, 'gen_seconds': gt}, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
        sh = [s for _, s in shifts]
        print(f'[{a.arm}] prompt {idx}: {video.shape[0]} frames in {gt:.0f}s  drift signal first/last '
              f'{(sh[0] if sh else float("nan")):.4f}/{(sh[-1] if sh else float("nan")):.4f}', flush=True)
        if a.smoke:
            print('SMOKE OK'); break
