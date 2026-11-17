"""Course-correction policies, applied to the state after every loop past T_train.
Controls (no dynamics model):
  norm_match      rescale each token's state to its norm at depth T_train   (gain control)
  pca_project@a   pull the state toward the PCA subspace of trained-depth states  (manifold projection)
Koopman (use dynamics learned inside the trained depth):
  dmd_pull@a      pull the PCA part toward the DMD one-step prediction from the previous state
  kae_pull@a/k    every k loops, pull the PCA part toward the Koopman-AE k-step latent rollout from
                  the last corrected anchor (k=1: one-step). This is periodic course correction:
                  the real model runs freely between corrections.
"""
import torch

from .fit import KAE


class Policies:
    def __init__(self, fit, device):
        self.mu = fit["mu"].to(device).float()
        self.W = fit["W"].to(device).float()
        self.A = fit["A"].to(device).float()
        self.c = fit["c"].to(device).float()
        self.sd = fit["sd"].to(device).float()
        self.t_train = fit["t_train"]
        self.kae = KAE(fit["rank"]).to(device)
        self.kae.load_state_dict(fit["kae"])
        self.kae.eval()

    def z(self, h):
        return (h.float() - self.mu) @ self.W

    def make(self, name):
        if name == "none":
            return None
        kind, _, arg = name.partition("@")
        T = self.t_train

        def finish(h, new):
            return new.to(h.dtype)

        if kind == "norm_match":
            def f(d, h, ctl):
                ref = ctl.states[T - 1].float()
                return finish(h, h.float() * (ref.norm(dim=-1, keepdim=True) / h.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)))
            return f
        if kind == "pca_project":
            a = float(arg)
            def f(d, h, ctl):
                hf = h.float()
                proj = self.mu + self.z(hf) @ self.W.T
                return finish(h, hf + a * (proj - hf))
            return f
        if kind == "dmd_pull":
            a = float(arg)
            def f(d, h, ctl):
                zp = self.z(ctl.states[d - 2]) @ self.A.T + self.c
                hf = h.float()
                return finish(h, hf + a * (zp - self.z(hf)) @ self.W.T)
            return f
        if kind == "kae_pull":
            a, k = arg.split("/")
            a, k = float(a), int(k)
            def f(d, h, ctl):
                if (d - T) % k != 0:
                    return h
                anchor = ctl.states[d - k - 1]  # state k loops ago (corrected anchor, or depth T)
                with torch.no_grad():
                    y = self.kae.lift(self.z(anchor) / self.sd)
                    for _ in range(k):
                        y = self.kae.step(y)
                    zp = y[..., : self.W.shape[1]] * self.sd
                hf = h.float()
                return finish(h, hf + a * (zp - self.z(hf)) @ self.W.T)
            return f
        raise ValueError(name)


CONTROLS = ("norm_match", "pca_project")
KOOPMAN = ("dmd_pull", "kae_pull")
DEFAULT = ["none", "norm_match", "pca_project@0.5", "pca_project@1.0",
           "dmd_pull@0.25", "dmd_pull@0.5", "dmd_pull@1.0",
           "kae_pull@0.5/1", "kae_pull@1.0/1", "kae_pull@0.5/2", "kae_pull@1.0/2"]
