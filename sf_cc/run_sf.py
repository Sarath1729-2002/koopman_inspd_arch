"""Course correction in Self-Forcing (Wan2.1-T2V-1.3B, chunk-wise autoregressive, DMD 4-step): long videos with a
rolling attention window, with/without decode -> re-encode of each finished chunk BEFORE it is written into the KV cache.

Arms (name tokens joined by '_'):
  base          rolling window of 21 latent frames (Self-Forcing's training length), no sink, no correction
  reencP        every P chunks: chunk latents z <- E(D([previous W latents, z]))[-3:]   (the projection)
  noiseP        control: z <- z + eps, |eps| = |E(D(.)) - z| per sample (same size, random direction)
  sinkK         attention sink: keep the first K latent frames in the cache (StreamingLLM / Deep-Forcing-style baseline)
  cache         apply the correction only to the context written into the KV cache; the output video keeps the
                generated latents (separates "clean context" from "lossy VAE round trip on what the viewer sees")
  aNN           partial projection: z <- z + NN/100 * (E(D(z)) - z)   (e.g. reenc2_a50)
  evtK          drift-triggered: re-encode a chunk only when its drift signal > K x the video's own floor
                (mean drift of its first 2 chunks); e.g. evt2
  dproj[N]      GENERATOR-manifold projection (no VAE): re-noise the finished chunk to the N-th smallest non-zero
                denoising timestep (default 1 = smallest) and denoise it once with the model itself. Rationale: a
                self-forced model's context was trained on its OWN outputs, so the manifold to project onto is the
                generator's, not the VAE's. One extra generator call per chunk (cheaper than a VAE round trip)
  stat          statistics anchoring: per-channel latent mean/std pulled halfway back to those of the first 2 chunks
                (targets slow colour / saturation drift; no model call)
  blo / bhi     band-limited re-encode (Nov 25, after FreqForcing's low/high-frequency split): blo takes only the
                LOW spatial frequencies of E(D(z)) (colour, layout) and keeps the model's own high frequencies (detail,
                motion); bhi the reverse. Low = 5x5 box blur over each latent frame
  ANCHOR arms (Nov 26, for content drift, which the VAE round trip cannot see). A second KV cache, the anchor cache,
  holds the opening chunk at the sink AND copies of the opening chunk at every recent position, so attending to it
  means "the recent past looked like the opening". One extra context write per chunk.
  hgNN          anchor guidance: at every denoising step z <- z_full + NN/100 * (z_anchor - z_full), where z_anchor is the
                same denoiser call made with the anchor cache (2x denoiser calls per chunk)
  adpN          anchor re-denoise ("content re-encoding"): every 4 chunks, re-noise the finished chunk to the N-th
                smallest non-zero timestep and denoise it once WITH THE ANCHOR CACHE, so deviations from the opening
                are treated as error (dproj does the same with the model's own context and changed nothing)
  adtK          drift-triggered anchor re-denoise: only when the chunk's latent statistics (per-channel mean/std) are
                more than K x the early variation (chunks 1-2) away from the opening chunk's
  ode           use checkpoints/ode_init.pt: the SAME causal 4-step student BEFORE Self-Forcing's self-rollout (DMD)
                training, i.e. a model never trained on its own outputs (tests: does correction substitute for that?)
Per video we save the mp4 (16 fps) and a json with per-chunk drift signal ||E(D(z)) - z||^2 / ||z||^2 and timings.
Run from the repo root:  python -m sf_cc.run_sf --arm reenc1 --prompts 0-31 --latents 240"""
import argparse, json, os, sys, time, numpy as np, torch

SF = os.path.abspath(os.environ.get('SF_DIR', 'ext/Self-Forcing'))


def arm_cfg(arm):
    c = dict(P=0, noise=False, sink=0, ckpt='dmd', cache=False, alpha=1.0, evt=0.0, dproj=0, stat=False, dpn=False, band=None, hg=0.0, adp=0, adt=0.0)
    for q in arm.split('_'):
        if q.startswith('reenc'): c['P'] = int(q[5:])
        elif q.startswith('noise'): c['P'], c['noise'] = int(q[5:]), True
        elif q.startswith('sink'): c['sink'] = int(q[4:])
        elif q == 'ode': c['ckpt'] = 'ode'
        elif q == 'causvid': c['ckpt'] = 'causvid'   # CausVid (Yin et al. 2025): same Wan-1.3B causal student, DMD-distilled
                                                    # with teacher-forced (clean) context, i.e. NOT trained on its own rollouts
        elif q == 'cache': c['cache'] = True
        elif q.startswith('evt'): c['evt'] = float(q[3:])
        elif q.startswith('dproj'): c['dproj'] = int(q[5:] or 1)
        elif q == 'stat': c['stat'] = True
        elif q in ('blo', 'bhi'): c['band'] = q[1:]
        elif q.startswith('hg'): c['hg'] = int(q[2:]) / 100
        elif q.startswith('adp'): c['adp'] = int(q[3:] or 2)
        elif q.startswith('adt'): c['adt'], c['adp'] = float(q[3:]), c['adp'] or 2
        elif q == 'dpn': c['dpn'] = True   # control for dproj: random nudge of the same size as the projection's change
        elif q[0] == 'a' and q[1:].isdigit(): c['alpha'] = int(q[1:]) / 100   # correction goes ONLY into the KV-cache context; the output keeps the original latents
    return c


def band_mix(z, zr, band):
    """z, zr [B, F, C, h, w]: low spatial frequencies from one, high from the other"""
    if band is None:
        return zr
    import torch.nn.functional as Fn
    sh = z.shape
    lo = lambda x: Fn.avg_pool2d(x.float().flatten(0, 1), 5, 1, 2, count_include_pad=False).view(sh)
    zl, rl = lo(z), lo(zr)
    return (rl + (z.float() - zl)) if band == 'lo' else (zl + (zr.float() - rl))


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


def sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                         causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """Drop-in for wan.modules.attention.flash_attention using PyTorch SDPA (q, k, v: [B, L, heads, dim]).
    Honours k_lens (key padding) and q_scale; returns [B, Lq, heads, dim] in q's dtype."""
    out_dtype = q.dtype
    cast = (lambda t: t if t.dtype in (torch.float16, torch.bfloat16) else t.to(dtype))
    q, k, v = cast(q), cast(k), cast(v)
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        Lk = k.shape[1]
        mask = (torch.arange(Lk, device=k.device)[None, :] < k_lens.to(k.device)[:, None])[:, None, None, :]
    o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                         attn_mask=mask, dropout_p=dropout_p, is_causal=causal,
                                                         scale=softmax_scale)
    return o.transpose(1, 2).contiguous().to(out_dtype)


def patch_flash_attention():
    """flash-attn wheels on the cluster do not match this torch; Wan's cross-attention calls flash_attention directly."""
    import wan.modules.attention as A
    if A.FLASH_ATTN_2_AVAILABLE or A.FLASH_ATTN_3_AVAILABLE:
        return
    import wan.modules.model  # noqa: F401  (load the modules that imported the name)
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.startswith('wan') and getattr(mod, 'flash_attention', None) is not None:
            mod.flash_attention = sdpa_flash_attention; n += 1
    print(f'flash-attn unavailable: patched flash_attention -> PyTorch SDPA in {n} modules', flush=True)


class StreamVAE:
    """Temporally-consistent decode -> re-encode for Wan's causal 3D VAE (4x temporal compression).
    Keeps the decoder's and encoder's causal feature caches across chunks, so re-encoding a chunk gives exactly the
    latents that encoding the whole decoded video would give (first latent <-> 1 frame, later latents <-> 4 frames).
    The windowed version used in the first run decoded each window as if it started a new clip."""
    def __init__(s, wrapper):
        s.m, s.mean, s.std = wrapper.model, wrapper.mean, wrapper.std
        s.reset()

    def reset(s):
        cc = sys.modules[type(s.m).__module__].count_conv3d
        s.dec, s.enc, s.first = [None] * cc(s.m.decoder), [None] * cc(s.m.encoder), True

    @torch.no_grad()
    def roundtrip(s, z):
        """z: [B, T, C, h, w] = the next latents in video order. Returns E(D(z)) in the same layout."""
        mean = s.mean.to(z.device, z.dtype).view(1, -1, 1, 1, 1); std = s.std.to(z.device, z.dtype).view(1, -1, 1, 1, 1)
        x = s.m.conv2(z.permute(0, 2, 1, 3, 4) * std + mean)
        pix = torch.cat([s.m.decoder(x[:, :, i:i + 1], feat_cache=s.dec, feat_idx=[0]) for i in range(x.shape[2])], 2)
        pix = pix.clamp(-1, 1)
        outs, p = [], 0
        if s.first:
            outs.append(s.m.encoder(pix[:, :, :1], feat_cache=s.enc, feat_idx=[0])); p, s.first = 1, False
        while p < pix.shape[2]:
            outs.append(s.m.encoder(pix[:, :, p:p + 4], feat_cache=s.enc, feat_idx=[0])); p += 4
        mu, _ = s.m.conv1(torch.cat(outs, 2)).chunk(2, dim=1)
        return ((mu - mean) / std).permute(0, 2, 1, 3, 4)


@torch.no_grad()
def stream_decode(wrapper, z):
    """Same computation as Wan's decode_to_pixel (one latent at a time through the causal decoder with its feature
    cache), but each decoded piece goes to uint8 on the CPU at once, so a 4-minute video never sits on the GPU whole."""
    m = wrapper.model
    mean = wrapper.mean.to(z.device, z.dtype).view(1, -1, 1, 1, 1); std = wrapper.std.to(z.device, z.dtype).view(1, -1, 1, 1, 1)
    x = m.conv2(z.permute(0, 2, 1, 3, 4) * std + mean)
    cache = [None] * sys.modules[type(m).__module__].count_conv3d(m.decoder)
    frames = []
    for i in range(x.shape[2]):
        px = m.decoder(x[:, :, i:i + 1], feat_cache=cache, feat_idx=[0]).clamp(-1, 1)   # [1, 3, t, H, W]
        frames.append(((px[0] * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu())   # same rounding as before
    return torch.cat(frames, 0).numpy()


def build(sink, local=21, ckpt='dmd'):
    sys.path.insert(0, SF); os.chdir(SF)
    patch_flash_attention()
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    cfg = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), OmegaConf.load('configs/self_forcing_dmd.yaml'))
    if ckpt == 'causvid':   # CausVid's own sampler settings (denoising steps, chunk size) from its config
        cv = os.environ.get('CAUSVID_CFG', os.path.join(os.path.dirname(SF), 'CausVid', 'configs', 'wan_causal_dmd.yaml'))
        cvc = OmegaConf.load(cv)
        for k in ('denoising_step_list', 'num_frame_per_block', 'warp_denoising_step', 'context_noise'):
            if k in cvc:
                cfg[k] = cvc[k]
        print('CausVid sampler:', {k: cfg.get(k) for k in ('denoising_step_list', 'num_frame_per_block', 'warp_denoising_step', 'context_noise')}, flush=True)
    cfg.model_kwargs.local_attn_size = local
    cfg.model_kwargs.sink_size = sink
    pipe = CausalInferencePipeline(cfg, device='cuda')
    f = {'dmd': 'checkpoints/self_forcing_dmd.pt', 'ode': 'checkpoints/ode_init.pt',
         'causvid': 'checkpoints/causvid/model.pt'}[ckpt]
    sd = torch.load(f, map_location='cpu')
    key = next((k for k in ('generator_ema', 'generator') if k in sd), None)
    sd = sd[key] if key else sd
    print(f'checkpoint {f} key {key}', flush=True)
    tgt = pipe.generator.state_dict().keys()
    if not set(sd) & set(tgt):   # prefix mismatch between codebases: try adding / stripping 'model.'
        for cand in ({('model.' + k): v for k, v in sd.items()}, {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}):
            if set(cand) & set(tgt):
                sd = cand; break
    missing = pipe.generator.load_state_dict(sd, strict=True)
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
    anchor = c['hg'] > 0 or c['adp'] > 0
    if anchor:
        if getattr(pipe, '_kv_anchor', None) is None:   # a second cache with the same layout as the main one
            main = pipe.kv_cache1
            pipe._initialize_kv_cache(batch_size=B, dtype=noise.dtype, device=dev)
            pipe._kv_anchor, pipe.kv_cache1 = pipe.kv_cache1, main
        for b in range(len(pipe._kv_anchor)):
            pipe._kv_anchor[b]['global_end_index'] = torch.tensor([0], dtype=torch.long, device=dev)
            pipe._kv_anchor[b]['local_end_index'] = torch.tensor([0], dtype=torch.long, device=dev)
    kva = getattr(pipe, '_kv_anchor', None)
    z0, st0, st_early = None, None, []
    stats = lambda zz: torch.cat([zz.float().mean(dim=(1, 3, 4)), zz.float().std(dim=(1, 3, 4))], -1)
    out = torch.zeros_like(noise)
    shifts, diag_time, floor, ref, n_act = [], 0.0, [], [], 0
    sv = StreamVAE(pipe.vae)   # one causal decode/encode stream per video, fed every chunk in order
    torch.cuda.synchronize(); t0 = time.time()
    cur = 0
    for blk in range(n_lat // F_):
        x = noise[:, cur:cur + F_]
        for i, ts in enumerate(pipe.denoising_step_list):
            t = torch.ones([B, F_], device=dev, dtype=torch.int64) * ts
            _, z = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=t, kv_cache=pipe.kv_cache1,
                                  crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
            if c['hg'] > 0 and blk > 0:   # anchor guidance
                _, za = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=t, kv_cache=kva,
                                       crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
                z = (z.float() + c['hg'] * (za.float() - z.float())).to(z.dtype)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                x = pipe.scheduler.add_noise(z.flatten(0, 1), torch.randn_like(z.flatten(0, 1)),
                                             nt * torch.ones([B * F_], device=dev, dtype=torch.long)).unflatten(0, z.shape[:2])
        act = c['P'] > 0 and (blk + 1) % c['P'] == 0 and blk > 0
        if c['stat']:   # statistics anchoring toward the first 2 chunks
            zf = z.float(); mu = zf.mean(dim=(1, 3, 4), keepdim=True); sd = zf.std(dim=(1, 3, 4), keepdim=True) + 1e-6
            if blk < 2:
                ref.append((mu, sd))
            else:
                rmu = torch.stack([m for m, _ in ref]).mean(0); rsd = torch.stack([v for _, v in ref]).mean(0)
                z = (zf + 0.5 * ((zf - mu) / sd * rsd + rmu - zf)).to(z.dtype)
        # the stream must see every chunk to stay causal-consistent, so the round trip runs every chunk;
        # its cost is counted only for arms that use it (act), and excluded otherwise
        z_out = z   # what the viewer sees (unless the correction is also applied to the output)
        torch.cuda.synchronize(); td = time.time()
        zr = sv.roundtrip(z).to(z.dtype)
        s = ((zr.float() - z.float()) ** 2).flatten(1).sum(1) / (z.float() ** 2).flatten(1).sum(1)
        if blk < 2:
            floor.append(float(s.mean()))
        if c['evt'] > 0:
            act = blk >= 2 and float(s.mean()) > c['evt'] * np.mean(floor)
        n_act += int(act)
        if blk % diag_every == 0 or act:
            shifts.append([blk, float(s.mean())])
        if True:
            if act:
                if c['noise']:
                    d = (zr.float() - z.float()).flatten(1).norm(dim=1)
                    e = torch.randn_like(z, dtype=torch.float32)
                    e = e / e.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1) * d.view(-1, 1, 1, 1, 1)
                    z = (z.float() + e).to(z.dtype)
                else:
                    zb = band_mix(z, zr, c['band']).float()
                    z = (z.float() + c['alpha'] * (zb - z.float())).to(z.dtype)
            torch.cuda.synchronize()
            if not act and c['P'] == 0 and c['evt'] == 0:
                diag_time += time.time() - td
        if c['dproj'] and blk > 0:   # generator-manifold projection: re-noise to a small t, denoise once
            nz = [int(v) for v in pipe.denoising_step_list if int(v) > 0]
            tp = nz[-c['dproj']]
            zn = pipe.scheduler.add_noise(z.flatten(0, 1), torch.randn_like(z.flatten(0, 1)),
                                          tp * torch.ones([B * F_], device=dev, dtype=torch.long)).unflatten(0, z.shape[:2])
            _, zp = pipe.generator(noisy_image_or_video=zn, conditional_dict=cond,
                                   timestep=torch.ones([B, F_], device=dev, dtype=torch.int64) * tp, kv_cache=pipe.kv_cache1,
                                   crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
            if c['dpn']:
                d = (zp.float() - z.float()).flatten(1).norm(dim=1)
                e = torch.randn_like(z, dtype=torch.float32)
                zp = (z.float() + e / e.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1) * d.view(-1, 1, 1, 1, 1)).to(z.dtype)
            z = zp
            n_act += 1
        if anchor and blk == 0:
            z0, st0 = z.clone(), stats(z)
        if c['adp'] and blk > 0:   # anchor re-denoise: periodic (every 4 chunks) or drift-triggered (adt)
            if c['adt'] > 0:
                dd = float((stats(z) - st0).norm())
                if blk <= 2:
                    st_early.append(dd)
                go = blk > 2 and dd > c['adt'] * max(np.mean(st_early), 1e-6)
            else:
                go = blk % 4 == 0
            if go:
                nz = [int(v) for v in pipe.denoising_step_list if int(v) > 0]
                tp = nz[-c['adp']]
                zn = pipe.scheduler.add_noise(z.flatten(0, 1), torch.randn_like(z.flatten(0, 1)),
                                              tp * torch.ones([B * F_], device=dev, dtype=torch.long)).unflatten(0, z.shape[:2])
                _, z = pipe.generator(noisy_image_or_video=zn, conditional_dict=cond,
                                      timestep=torch.ones([B, F_], device=dev, dtype=torch.int64) * tp, kv_cache=kva,
                                      crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
                n_act += 1
        out[:, cur:cur + F_] = z_out if c.get('cache') else z
        pipe.generator(noisy_image_or_video=z, conditional_dict=cond,
                       timestep=torch.ones([B, F_], device=dev, dtype=torch.int64) * cfg.context_noise,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=cur * pipe.frame_seq_length)
        if anchor:   # the anchor cache sees the opening chunk at every position
            pipe.generator(noisy_image_or_video=z0, conditional_dict=cond,
                           timestep=torch.ones([B, F_], device=dev, dtype=torch.int64) * cfg.context_noise,
                           kv_cache=kva, crossattn_cache=pipe.crossattn_cache, current_start=cur * pipe.frame_seq_length)
        cur += F_
    torch.cuda.synchronize(); gen_time = time.time() - t0 - diag_time
    video = stream_decode(pipe.vae, out)
    pipe.vae.model.clear_cache()
    return video, shifts, gen_time, n_act


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='base'); ap.add_argument('--prompts', default='0-31')
    ap.add_argument('--latents', type=int, default=240, help='latent frames (x4 = video frames at 16 fps); 240 = 60 s')
    ap.add_argument('--out', default='results_sf'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    root = os.getcwd(); out = os.path.join(root, a.out, a.arm); os.makedirs(out, exist_ok=True)
    lo, hi = (int(x) for x in a.prompts.split('-'))
    c = arm_cfg(a.arm)
    pipe, cfg = build(c['sink'], ckpt=c['ckpt'])
    ext = [l.strip() for l in open('prompts/MovieGenVideoBench_extended.txt', encoding='utf-8')]
    short = [l.strip() for l in open('prompts/MovieGenVideoBench.txt', encoding='utf-8')]
    n_lat = 9 if a.smoke else a.latents
    for idx in range(lo, hi + 1):
        f = os.path.join(out, f'{idx:04d}.json')
        if os.path.exists(f) and not a.smoke:
            continue
        video, shifts, gt, n_act = generate(pipe, cfg, ext[idx], n_lat, c, seed=1000 + idx)
        write_mp4(os.path.join(out, f'{idx:04d}.mp4'), video)
        json.dump({'arm': a.arm, 'cfg': c, 'idx': idx, 'prompt': short[idx], 'latents': n_lat, 'frames': int(video.shape[0]),
                   'shift': shifts, 'gen_seconds': gt, 'n_corrections': n_act}, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
        sh = [s for _, s in shifts]
        print(f'[{a.arm}] prompt {idx}: {video.shape[0]} frames in {gt:.0f}s  drift signal first/last '
              f'{(sh[0] if sh else float("nan")):.4f}/{(sh[-1] if sh else float("nan")):.4f}', flush=True)
        if a.smoke:
            print('SMOKE OK'); break
