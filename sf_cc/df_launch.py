"""Runs Deep Forcing's inference.py (cwd = ext/DeepForcing) with two compatibility shims for our cluster env, the
same ones run_sf.py uses for Self-Forcing, plus (0) a stub for lmdb, which only the training datasets use: (1) torchvision >= 0.24 has no torchvision.io.write_video -> an av/libx264
writer (crf 18, like run_sf.write_mp4); (2) flash-attn wheels do not match this torch -> PyTorch SDPA.
Deep Forcing's own code is not modified. Usage: python /path/sf_cc/df_launch.py <inference.py args>"""
import os, runpy, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd()); sys.path.insert(1, REPO)
import numpy as np, torch, torchvision.io as tio


def write_video(path, video, fps=16, **kw):
    import av
    x = video.detach().cpu() if torch.is_tensor(video) else torch.as_tensor(video)
    x = x.clamp(0, 255).to(torch.uint8).numpy()   # [T, H, W, C]
    with av.open(path, 'w') as c:
        st = c.add_stream('libx264', rate=int(fps)); st.width, st.height, st.pix_fmt = x.shape[2], x.shape[1], 'yuv420p'
        st.options = {'crf': '18'}
        for f in x:
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format='rgb24')):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


import importlib, types
for opt in ('lmdb',):   # only used by Deep Forcing's training datasets; inference never calls it
    try:
        importlib.import_module(opt)
    except ImportError:
        sys.modules[opt] = types.ModuleType(opt)
if not hasattr(tio, 'write_video'):
    tio.write_video = write_video
import pipeline  # noqa: F401  Deep Forcing's pipeline -> loads its wan modules
from sf_cc.run_sf import patch_flash_attention
patch_flash_attention()
sys.argv = ['inference.py'] + sys.argv[1:]
runpy.run_path('inference.py', run_name='__main__')
