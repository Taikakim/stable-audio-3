"""Pure weight-mutation core for scripts/mutate_weights.py.

Every op is a pure function (tensor in, tensor out), deterministic given a
torch.Generator, and device-agnostic: random draws happen on CPU with the
supplied generator, then move to the weight's device, so a mutation seed
reproduces the exact same model state on CPU or GPU.

Tested in tests/test_weight_mutations.py (CPU, no checkpoints).
"""
from __future__ import annotations

import math
import re
from typing import NamedTuple

import torch
import torch.nn.functional as F

# SA3's ContinuousTransformer stores blocks as `layers.N` (NOT `blocks.N` —
# matching only "blocks." silently mutates nothing; that bug shipped once).
_BLOCK_RE = re.compile(r"(?:^|\.)(?:blocks|layers)\.(\d+)\.")


# ------------------------------------------------------------------ decay

def decay_profile(n_blocks, rate, direction="late", focus=None, width=3.0):
    """Per-block mutation multiplier in [0, 1].

    direction: 'late'  — peaks at the last block (surface/texture damage)
               'early' — peaks at block 0 (structural damage)
               'flat'  — every block equally
               'focus' — gaussian bump centred on `focus` with std `width`
    """
    if direction == "flat":
        return [1.0] * n_blocks
    if direction == "late":
        return [math.exp(rate * (i - (n_blocks - 1))) for i in range(n_blocks)]
    if direction == "early":
        return [math.exp(-rate * i) for i in range(n_blocks)]
    if direction == "focus":
        if focus is None:
            raise ValueError("direction='focus' needs focus=<block index>")
        return [math.exp(-((i - focus) ** 2) / (2.0 * width ** 2)) for i in range(n_blocks)]
    raise ValueError(f"unknown decay direction {direction!r}")


# ------------------------------------------------------------------ targets

def classify_param(name):
    """'attn' | 'mlp' | 'norm' | 'other' from a parameter name."""
    if "norm" in name:
        return "norm"
    if "attn" in name or any(t in name for t in ("to_q", "to_kv", "to_qkv", "to_out")):
        return "attn"
    if ".ff" in name:
        return "mlp"
    return "other"


class Target(NamedTuple):
    name: str
    block: int
    kind: str
    param: torch.Tensor


def collect_targets(model, target="all"):
    """Mutable parameters inside transformer blocks: 2-D Linear weights
    (attn / mlp) and 1-D norm gains. `target` filters by kind."""
    out = []
    for name, param in model.named_parameters():
        m = _BLOCK_RE.search(name)
        if not m:
            continue
        kind = classify_param(name)
        if kind in ("attn", "mlp") and not (param.ndim == 2 and name.endswith(".weight")):
            continue
        if kind == "norm" and param.ndim != 1:
            continue
        if kind == "other":
            continue
        if target != "all" and kind != target:
            continue
        out.append(Target(name, int(m.group(1)), kind, param))
    return out


def snapshot_targets(targets):
    """CPU copies of the pristine weights, keyed by name."""
    return {t.name: t.param.detach().clone().cpu() for t in targets}


def restore_targets(targets, snap):
    with torch.no_grad():
        for t in targets:
            t.param.copy_(snap[t.name])


# ------------------------------------------------------------------ ops

def _cpu_randn_like(w, gen):
    return torch.randn(w.shape, generator=gen, dtype=torch.float32).to(
        device=w.device, dtype=w.dtype)


def drift(w, amount, gen):
    """Gaussian noise, sigma = amount * std(w)."""
    if amount == 0:
        return w
    return w + _cpu_randn_like(w, gen) * (amount * w.float().std().item())


def shuffle(w, frac, gen):
    """Randomly permute `frac` of the entries among themselves.
    Value-preserving: the multiset of weights is unchanged."""
    if frac <= 0:
        return w
    flat = w.flatten().clone()
    n = flat.numel()
    k = max(2, int(n * frac))
    idx = torch.randperm(n, generator=gen)[:k]
    perm = torch.randperm(k, generator=gen)
    flat[idx.to(w.device)] = flat[idx[perm].to(w.device)]
    return flat.view_as(w)


def blur(w, amount, axis=1):
    """Lerp toward a 3-tap moving average along `axis` (replicate edges)."""
    if amount == 0:
        return w
    x = w if axis == 1 else w.T
    padded = torch.cat([x[:, :1], x, x[:, -1:]], dim=1)
    avg = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
    out = x * (1.0 - amount) + avg * amount
    return out if axis == 1 else out.T


def contrast(w, amount):
    """Stretch (+) or flatten (−) around the mean. amount=-1 erases to mean."""
    mean = w.mean()
    return mean + (w - mean) * (1.0 + amount)


def life_step(w, alive_thresh):
    """One Game-of-Life generation on the weight matrix.

    A cell is alive if |w| > alive_thresh. Live cell with 2-3 live neighbours
    keeps its value; otherwise it dies to 0. Dead cell with exactly 3 live
    neighbours is born as the mean of its live neighbours' values; other dead
    cells keep their (sub-threshold) value untouched.
    """
    alive = (w.abs() > alive_thresh).to(w.dtype)
    kernel = torch.ones(1, 1, 3, 3, dtype=w.dtype, device=w.device)
    kernel[0, 0, 1, 1] = 0.0
    def conv(x):
        return F.conv2d(x[None, None], kernel, padding=1)[0, 0]
    ncount = conv(alive)
    nsum = conv(w * alive)
    born_val = nsum / ncount.clamp(min=1.0)
    survives = (alive > 0) & (ncount >= 2) & (ncount <= 3)
    born = (alive == 0) & (ncount == 3)
    out = torch.where(survives, w, torch.zeros_like(w))
    out = torch.where(born, born_val, out)
    keep_dead = (alive == 0) & ~born
    return torch.where(keep_dead, w, out)


class Condition(NamedTuple):
    """One reproducible mutation recipe. `ops` is an ordered list of dicts:
    {"op": "drift"|"shuffle"|"blur"|"contrast"|"tilt"|"life", "amount": float,
     + op-specific keys ("axis" for blur, "quantile" for life)}."""
    name: str
    ops: list
    target: str = "all"
    decay_rate: float = 0.5
    decay_direction: str = "late"
    decay_focus: int | None = None
    mutation_seed: int = 1234


def apply_condition(model, cond, decay_eps=1e-3):
    """Apply a Condition's ops to the model's block weights, in place.

    Deterministic: a fresh generator seeded with cond.mutation_seed drives all
    randomness, and targets are visited in name order. Per-block op amounts
    are scaled by the decay profile ('life' is gated at decay >= 0.5 instead
    of scaled — a half-dead automaton isn't meaningful).
    Returns a summary dict (params_touched, blocks_touched).
    """
    targets = sorted(collect_targets(model, target=cond.target))
    if not targets:
        raise RuntimeError(
            f"no mutable parameters matched target={cond.target!r} — "
            "wrong module tree? (this fails loud so a no-op run can't ship)")
    n_blocks = max(t.block for t in targets) + 1
    decay = decay_profile(n_blocks, cond.decay_rate,
                          direction=cond.decay_direction, focus=cond.decay_focus)
    gen = torch.Generator()
    gen.manual_seed(cond.mutation_seed)
    touched, blocks = 0, set()
    with torch.no_grad():
        for spec in cond.ops:
            op, amount = spec["op"], spec.get("amount", 0.0)
            for t in targets:
                d = decay[t.block]
                if d < decay_eps:
                    continue
                w = t.param
                if op == "drift":
                    out = drift(w, amount * d, gen)
                elif op == "shuffle":
                    out = shuffle(w, amount * d, gen)
                elif op == "blur":
                    if w.ndim < 2:
                        continue
                    out = blur(w, amount * d, axis=spec.get("axis", 1))
                elif op == "contrast":
                    out = contrast(w, amount * d)
                elif op == "tilt":
                    if w.ndim < 2:
                        continue
                    out = spectral_tilt(w, amount * d)
                elif op == "life":
                    if w.ndim < 2 or d < 0.5:
                        continue
                    q = spec.get("quantile", 0.75)
                    thresh = w.float().abs().quantile(q).item()
                    out = life_step(w, alive_thresh=thresh)
                else:
                    raise ValueError(f"unknown op {op!r}")
                if out is not w:
                    w.copy_(out)
                    touched += 1
                    blocks.add(t.block)
    return {"params_touched": touched, "blocks_touched": sorted(blocks)}


def spectral_tilt(w, tilt):
    """Re-weight the singular value spectrum, Frobenius norm preserved.

    tilt > 0 boosts the tail (small singular values) — structured weirdness;
    tilt < 0 boosts the head — coherent simplification. tilt=0 is identity.
    """
    if tilt == 0:
        return w
    x = w.float()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    pos = torch.linspace(0.0, 1.0, s.numel(), device=s.device)  # 0 = largest
    s2 = s * (1.0 + tilt * pos).clamp(min=0.0)
    s2 = s2 * (s.norm() / s2.norm().clamp(min=1e-12))
    return (u @ torch.diag(s2) @ vh).to(w.dtype)
