"""AdaGC: per-tensor adaptive gradient clipping (arXiv 2502.11034), CONTINUITY 2026-09-26.

Each trainable tensor's gradient g is scaled by

    min(1, lam * gamma / ||g||),    gamma <- beta * gamma + (1 - beta) * ||g_clipped||

where gamma is an EMA of that tensor's CLIPPED norms, so a spike cannot raise its own threshold.
The paper's defaults are lam = 1.04 (continuous trimming) and beta = 0.99; lam around 3 makes it a
rare-event clip that leaves ordinary steps alone.

WHY HERE. It runs in Lightning's configure_gradient_clipping, i.e. after backward and after the
flight recorder has measured the RAW gradients, and before optimizer.step(). So a clipped gradient
is what reaches BOTH the momentum buffer and the Shampoo accumulators (which average G Gᵀ, so an
unclipped 42x spike would enter them at ~1764x). With a normalised (spectral) step, clipping never
changes the step's size, only how much one step can pull on momentum and the preconditioner.
Background: papers/deep-research/2026-09-26-gradient-noise-spike-handling-triage.md.

Details that matter:
  * warm-up (default 100 steps): only builds gamma as the running mean of each tensor's finite,
    nonzero norms; nothing is clipped;
  * a tensor with no history yet (gamma == 0: LoRA A gets zero gradient while B is still 0) is
    never clipped; its first real norm initialises gamma;
  * non-finite gradients are left alone and not averaged in (the loss guard deals with them);
  * one fused norm reduction and one fused multiply per step, no host sync except in stats().
"""
from __future__ import annotations

import torch


class AdaGC:
    def __init__(self, named_params, lam: float = 1.04, beta: float = 0.99, warmup: int = 100,
                 eps: float = 1e-12):
        self.names: list[str] = []
        self.params: list[torch.nn.Parameter] = []
        for n, p in named_params:
            if p.requires_grad:
                self.names.append(n)
                self.params.append(p)
        dev = self.params[0].device if self.params else torch.device("cpu")
        self.lam = float(lam)
        self.beta = float(beta)
        self.warmup = int(warmup)
        self.eps = float(eps)
        self.gamma = torch.zeros(len(self.params), dtype=torch.float32, device=dev)
        self.count = torch.zeros(len(self.params), dtype=torch.float32, device=dev)
        self.n_seen = 0
        self._last_factor: torch.Tensor | None = None

    @torch.no_grad()
    def clip_(self) -> None:
        live = [i for i, p in enumerate(self.params) if p.grad is not None]
        if not live:
            return
        grads = [self.params[i].grad for i in live]
        idx = torch.tensor(live, device=self.gamma.device)
        norms = torch.stack(torch._foreach_norm(grads)).float().to(self.gamma.device)
        g_old = self.gamma[idx]
        finite = torch.isfinite(norms)
        use = finite & (norms > 0)

        if self.n_seen < self.warmup:
            cnt = self.count[idx] + use.float()
            self.gamma[idx] = torch.where(use, g_old + (norms - g_old) / cnt.clamp_min(1.0), g_old)
            self.count[idx] = cnt
            factor = torch.ones_like(norms)
        else:
            have = g_old > 0
            active = finite & have
            factor = torch.where(active, (self.lam * g_old / (norms + self.eps)).clamp(max=1.0),
                                 torch.ones_like(norms))
            clipped = norms * factor
            new = torch.where(active, self.beta * g_old + (1.0 - self.beta) * clipped, g_old)
            new = torch.where(use & ~have, norms, new)  # first real norm of a tensor with no history
            self.gamma[idx] = new
            torch._foreach_mul_(grads, list(factor.to(grads[0].device).unbind()))

        self.n_seen += 1
        full = torch.ones(len(self.params), dtype=torch.float32, device=self.gamma.device)
        full[idx] = factor
        self._last_factor = full

    def stats(self) -> dict[str, float]:
        """Clip statistics of the last step (one host sync)."""
        f = self._last_factor
        if f is None:
            return {}
        return {
            "adagc/clipped_frac": float((f < 1.0).float().mean()),
            "adagc/mean_factor": float(f.mean()),
            "adagc/min_factor": float(f.min()),
            "adagc/warmup": float(self.n_seen <= self.warmup),
        }

    def state_dict(self) -> dict:
        return {"names": list(self.names), "gamma": self.gamma.detach().cpu(),
                "count": self.count.detach().cpu(), "n_seen": self.n_seen,
                "lam": self.lam, "beta": self.beta, "warmup": self.warmup}

    def load_state_dict(self, sd: dict) -> None:
        """Restore averages by tensor NAME, so a changed parameter order cannot misassign them."""
        pos = {n: i for i, n in enumerate(sd["names"])}
        g = sd["gamma"].to(self.gamma.device)
        c = sd["count"].to(self.count.device)
        for i, n in enumerate(self.names):
            j = pos.get(n)
            if j is not None:
                self.gamma[i] = g[j]
                self.count[i] = c[j]
        self.n_seen = int(sd["n_seen"])
