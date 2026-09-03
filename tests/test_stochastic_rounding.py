"""Tests for the stochastic-rounding bf16 optimizer path.

Correctness gate = UNBIASEDNESS. A biased SR silently corrupts training, so it is
proven here BEFORE the primitive is wired into any optimizer. All CPU, fast.
"""

import torch
import pytest

from stable_audio_3.training.stochastic_rounding import (
    stochastic_round_to_bf16,
    AdamWSR,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bf16_grid_neighbors(a_bf16: torch.Tensor):
    """Given a bf16 value `a`, return (a, b) where b is the adjacent bf16 grid
    point one ULP further from zero (larger magnitude)."""
    assert a_bf16.dtype == torch.bfloat16
    bits = a_bf16.view(torch.int16).to(torch.int32) & 0xFFFF  # unsigned 16-bit pattern
    # increment magnitude: for the sign-magnitude bf16 layout, +1 to the unsigned
    # pattern moves away from zero (larger magnitude) for both signs.
    b_bits = (bits + 1) & 0xFFFF
    b_i16 = torch.where(b_bits >= 2**15, b_bits - 2**16, b_bits).to(torch.int16)
    b_bf16 = b_i16.view(torch.bfloat16)
    return a_bf16, b_bf16


# ---------------------------------------------------------------------------
# Step 1 — the SR primitive
# ---------------------------------------------------------------------------

# A spread of bf16 anchors across magnitudes / signs to exercise the bit math.
_ANCHORS = [0.5, 1.0, 1.5, 3.0, 0.01, -1.0, -0.25, 100.0]
_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.mark.parametrize("anchor", _ANCHORS)
@pytest.mark.parametrize("f", _FRACS)
def test_unbiasedness(anchor, f):
    """(a) THE GATE: mean of many SR draws == the true fp32 value (MC tolerance)."""
    torch.manual_seed(0)
    a = torch.tensor(anchor, dtype=torch.bfloat16)
    a, b = _bf16_grid_neighbors(a)
    a_f = a.float()
    b_f = b.float()
    w = a_f + f * (b_f - a_f)  # fp32 value strictly between two bf16 grid points

    N = 200_000
    wv = w.repeat(N)
    samples = stochastic_round_to_bf16(wv).float()
    mean = samples.mean().item()
    grid = (b_f - a_f).abs().item()
    err = abs(mean - w.item()) / grid
    assert err < 0.01, f"anchor={anchor} f={f}: |mean-w|/grid={err:.4f} (mean={mean}, w={w.item()})"


@pytest.mark.parametrize("anchor", _ANCHORS)
@pytest.mark.parametrize("f", _FRACS)
def test_boundedness(anchor, f):
    """(b) Every output is one of the two bracketing bf16 grid points, never outside."""
    torch.manual_seed(1)
    a = torch.tensor(anchor, dtype=torch.bfloat16)
    a, b = _bf16_grid_neighbors(a)
    a_f, b_f = a.float(), b.float()
    w = a_f + f * (b_f - a_f)
    samples = stochastic_round_to_bf16(w.repeat(50_000))
    uniq = torch.unique(samples)
    allowed = torch.tensor([a.item(), b.item()], dtype=torch.bfloat16)
    for u in uniq:
        assert u in allowed, f"output {u} not in bracket {allowed.tolist()}"


@pytest.mark.parametrize("anchor", _ANCHORS)
@pytest.mark.parametrize("f", _FRACS)
def test_probability(anchor, f):
    """(c) fraction rounding UP ~= f (fractional position), within MC tolerance."""
    torch.manual_seed(2)
    a = torch.tensor(anchor, dtype=torch.bfloat16)
    a, b = _bf16_grid_neighbors(a)
    a_f, b_f = a.float(), b.float()
    w = a_f + f * (b_f - a_f)
    N = 200_000
    samples = stochastic_round_to_bf16(w.repeat(N)).float()
    frac_up = (samples == b_f).float().mean().item()
    assert abs(frac_up - f) < 0.01, f"anchor={anchor} f={f}: P(up)={frac_up:.4f}"


@pytest.mark.parametrize("anchor", _ANCHORS)
def test_exact_values_roundtrip(anchor):
    """(d) values already representable in bf16 round to themselves with prob 1."""
    torch.manual_seed(3)
    a = torch.tensor(anchor, dtype=torch.bfloat16).float()  # exactly a bf16 value
    samples = stochastic_round_to_bf16(a.repeat(10_000))
    assert torch.all(samples.float() == a), f"exact bf16 {anchor} did not round to itself"


def test_nan_inf_passthrough():
    """NaN/Inf pass through (as bf16 NaN/Inf), do not crash or become finite garbage."""
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0, -2.5])
    out = stochastic_round_to_bf16(x)
    assert out.dtype == torch.bfloat16
    assert torch.isnan(out[0])
    assert torch.isinf(out[1]) and out[1] > 0
    assert torch.isinf(out[2]) and out[2] < 0
    assert torch.isfinite(out[3]) and torch.isfinite(out[4])


def test_accumulation_beats_deterministic():
    """(e) THE POINT: tiny sub-ULP updates ACCUMULATE under SR but STALL deterministically.

    Pick w0 (a bf16 value) and a delta so small that (w0 + delta) casts back to w0
    under deterministic bf16 rounding (the update vanishes). Show:
      - deterministic .add_ stays pinned at w0 forever,
      - SR drifts ~= M*delta after M steps.
    """
    torch.manual_seed(4)
    w0 = torch.tensor(1.0, dtype=torch.bfloat16)
    # bf16 ULP near 1.0 is 2^-7 ~= 0.0078. delta = ULP/16 is deeply sub-dominant.
    ulp = (_bf16_grid_neighbors(w0)[1].float() - w0.float()).abs()
    delta = (ulp / 16.0).item()
    M = 4000

    # deterministic bf16-master accumulation
    wd = w0.clone()
    for _ in range(M):
        wd = (wd.float() + delta).to(torch.bfloat16)  # deterministic cast back

    # SR bf16-master accumulation
    ws = w0.clone()
    for _ in range(M):
        ws = stochastic_round_to_bf16(ws.float() + delta)

    det_drift = (wd.float() - w0.float()).item()
    sr_drift = (ws.float() - w0.float()).item()
    expected = M * delta

    assert det_drift == 0.0, f"deterministic should stall at w0, drifted {det_drift}"
    # SR should track the true accumulated sum within ~15% (MC noise on a single trajectory)
    assert abs(sr_drift - expected) / expected < 0.15, (
        f"SR drift {sr_drift:.4f} vs expected {expected:.4f}"
    )


# ---------------------------------------------------------------------------
# Step 2 — AdamWSR on a tiny convex problem
# ---------------------------------------------------------------------------

def _fit_linear(optimizer_factory, master_dtype, steps=4000, seed=0):
    """Fit a small linear layer y = W x + b to a fixed random target. Returns final MSE.

    `master_dtype` controls the stored param dtype (bf16 = the memory-constrained case,
    fp32 = the reference). AdamWSR ignores it (always bf16 master internally)."""
    torch.manual_seed(seed)
    D_in, D_out, N = 8, 4, 256
    Wt = torch.randn(D_out, D_in)
    bt = torch.randn(D_out)
    X = torch.randn(N, D_in)
    Y = X @ Wt.t() + bt

    W = torch.zeros(D_out, D_in, dtype=master_dtype, requires_grad=True)
    b = torch.zeros(D_out, dtype=master_dtype, requires_grad=True)
    opt = optimizer_factory([W, b])

    for _ in range(steps):
        opt.zero_grad()
        pred = X @ W.float().t() + b.float()
        loss = torch.nn.functional.mse_loss(pred, Y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = X @ W.float().t() + b.float()
        return torch.nn.functional.mse_loss(pred, Y).item()


def test_adamwsr_beats_deterministic_bf16():
    """AdamWSR converges near fp32 and clearly beats a deterministic bf16-master AdamW,
    which stalls because sub-ULP updates vanish."""
    lr = 5e-3

    def fp32_opt(ps):
        return torch.optim.AdamW(ps, lr=lr)

    def det_bf16_opt(ps):
        # deterministic bf16 master: params are bf16, plain AdamW writes back via cast
        return torch.optim.AdamW(ps, lr=lr)

    def sr_opt(ps):
        return AdamWSR(ps, lr=lr)

    loss_fp32 = _fit_linear(fp32_opt, torch.float32)
    loss_det = _fit_linear(det_bf16_opt, torch.bfloat16)
    loss_sr = _fit_linear(sr_opt, torch.bfloat16)

    print(f"\nconvex fit final MSE  fp32={loss_fp32:.3e}  SR-bf16={loss_sr:.3e}  det-bf16={loss_det:.3e}")

    # SR must MASSIVELY beat deterministic bf16 — det stalls near init because sub-ULP
    # Adam updates vanish, while SR keeps converging. (Empirically det/SR ~ 1e3-1e4.)
    assert loss_sr < loss_det * 0.05, f"SR {loss_sr:.3e} did not beat det-bf16 {loss_det:.3e}"

    # SR must land near fp32 — but NOT below the bf16 storage noise floor. A bf16 master
    # stores each converged weight with ~2^-8 relative quantization noise; that noise
    # propagates to a residual prediction MSE floor of order (2^-8)^2 * D_in ~ 4e-5 here,
    # i.e. ~1 order of magnitude above fp32's floor. This is fundamental to *any*
    # bf16-master optimizer (SR keeps it UNBIASED, it cannot remove the per-weight
    # quantization noise). So "close to fp32" realistically means within the bf16 floor,
    # not machine-fp32 precision. 20x gives headroom for cross-seed MC variation while
    # still being far tighter than det-bf16 (which is ~7e4x fp32).
    assert loss_sr < loss_fp32 * 20 + 1e-6, f"SR {loss_sr:.3e} beyond bf16 floor vs fp32 {loss_fp32:.3e}"
