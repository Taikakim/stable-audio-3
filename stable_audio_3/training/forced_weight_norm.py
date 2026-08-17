"""forced_weight_norm.py — EDM2-style forced weight normalization, as a RETROFIT for a fine-tune.

EDM2 (Karras et al., arXiv 2312.02696) names our exact pathology: "uncontrolled magnitude changes
in activations AND weights over training". Our full-FT drone was measured as latent output-scale
runaway (global z0 std 0.7 → 1.3 @ep3 → 5.6 @ep7; latent channels with std>2.0 going 0 → ~4 →
166/256). The mechanism matters for choosing the fix: the runaway lives in the **output
projection**, which is NOT scale-invariant, so weight-norm growth IS output-magnitude growth — not
merely an effective-LR shift the way it would be for a normalized hidden layer (Rotational
Equilibrium, arXiv 2305.17212).

WHAT THIS DOES
    W_eff[i, :] = gain[i] * W[i, :] / ‖W[i, :]‖
  i.e. the weight supplies only a DIRECTION per output channel, and all magnitude is carried by a
  learnable 1-D `gain`. Magnitude becomes one well-conditioned parameter per output channel instead
  of something that can drift across millions of entries. `force_unit_rows()` is the "forced" half:
  it projects the stored weight back to unit rows so raw magnitude cannot accumulate at all.

WHY THIS IS *NOT* THE SAME ARM AS --hyperball (they are the #1 vs #2 contrast Kim asked for):
  * hyperball  — whole-tensor ‖W‖_F FROZEN at ‖W0‖_F, optimizer-side, applied to EVERY spectral 2-D
                 matrix. Output magnitude cannot change at all.
  * this       — PER-OUTPUT-CHANNEL direction normalised with a LEARNABLE magnitude, applied to a
                 TARGETED set (the output projection). Magnitude can still adapt, but only through
                 `gain`. Strictly weaker constraint, strictly more expressive.
  Running both (MODE=both) = hyperball on the bulk + per-channel gain control on the output.

RETROFIT-SAFETY IS THE WHOLE DESIGN CONSTRAINT
  `gain` is initialised to the pretrained per-output-channel norms, so step 0 is functionally
  IDENTICAL to the loaded checkpoint — no loss spike. (This is why EDM2's forced-weight-norm is
  retrofittable while MaP-DiT's rotation modulation is not: rotation changes conditioning.)
  EDM2's from-scratch pieces — MP-SiLU, MP-sum, the magnitude-preserving layer redesign — are
  deliberately NOT implemented here; they only make sense when training from scratch.

TARGET SELECTION (SA3 specifics)
  The output projection is `transformer.py:1171  self.project_out = nn.Linear(dim, dim_out)`.
  ⚠️ `dit.py:145-146  self.postprocess_conv` is **zero-initialised by design**, so W/‖W‖ is 0/0 →
  NaN on the first forward. Zero-norm rows are detected and such modules are SKIPPED, not wrapped.
  Five independent sources agree the output projection + AdaLN modulation are what need bounded
  treatment rather than unconstrained orthogonalization (SA3's own recipe, Rotational Equilibrium,
  MaP-DiT, EDM2, SFWN) — see papers/CONTINUITY-drone-optimizer-synthesis-2026-08-11.md.

OPTIMIZER INTERACTION
  `gain` is 1-D on purpose: FusionOpt's `build_fusion_param_groups` routes 2-D matrices to
  Muon/NS5 and 1-D params to ScheduleFree-AdamW. Orthogonalizing a scale parameter would be
  meaningless — the same class of error CMuon exists to fix.
"""
import re

import torch


class ForcedWeightNorm(torch.nn.Module):
    """Wraps a Linear/Conv1d so its magnitude lives in a learnable per-output-channel `gain`.

    The wrapped module's own weight tensor is adopted (same values, same name `weight`), so a
    state_dict gains exactly one key per wrapped site (`gain`) and `weight` keeps its meaning.
    """

    def __init__(self, module: torch.nn.Module, eps: float = 1e-8):
        super().__init__()
        if not hasattr(module, "weight") or module.weight is None:
            raise ValueError(f"{type(module).__name__} has no weight to normalise")
        if getattr(module, "bias", None) is not None:
            # SA3's project_out/postprocess_conv are bias=False. Supporting bias would mean deciding
            # whether it scales with the gain; refuse rather than guess wrong silently.
            raise ValueError("ForcedWeightNorm does not support a bias term — the wrapped layer "
                             "must be bias=False (SA3's output projections are)")
        self.eps = float(eps)
        self.inner = module
        # adopt the pretrained weight as our own parameter, then drop it from the inner module so
        # there is exactly ONE copy and one state_dict entry for it.
        w = module.weight.detach().clone()
        del module._parameters["weight"]
        self.weight = torch.nn.Parameter(w)
        # gain = the pretrained row norms => step 0 reproduces the pretrained function exactly
        self.gain = torch.nn.Parameter(self._row_norms(w).clone())
        self._op = self._pick_op(module)

    # ── helpers ─────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _row_norms(w: torch.Tensor) -> torch.Tensor:
        """Norm per OUTPUT channel (dim 0), flattening everything else. Works for Linear (O,I) and
        Conv1d (O,I,K) alike."""
        return w.reshape(w.shape[0], -1).norm(dim=1)

    @staticmethod
    def _pick_op(module: torch.nn.Module):
        import torch.nn.functional as F
        if isinstance(module, torch.nn.Linear):
            return lambda x, w: F.linear(x, w, None)
        if isinstance(module, torch.nn.Conv1d):
            return lambda x, w: F.conv1d(x, w, None, module.stride, module.padding,
                                         module.dilation, module.groups)
        raise ValueError(f"ForcedWeightNorm supports Linear and Conv1d, got {type(module).__name__}")

    @staticmethod
    def has_zero_rows(module: torch.nn.Module) -> bool:
        """True if any output channel has zero norm — wrapping would divide by zero.

        SA3's postprocess_conv is entirely zero-initialised (dit.py:146), which is the case this
        exists to catch.
        """
        w = getattr(module, "weight", None)
        if w is None:
            return True
        return bool((ForcedWeightNorm._row_norms(w.detach()) == 0).any())

    # ── the reparameterization ──────────────────────────────────────────────────────────────────
    def effective_weight(self) -> torch.Tensor:
        n = self._row_norms(self.weight).clamp_min(self.eps)
        shape = [-1] + [1] * (self.weight.dim() - 1)
        return self.weight * (self.gain / n).reshape(shape)

    def forward(self, x):
        return self._op(x, self.effective_weight())

    # ── maintenance ops ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def force_unit_rows(self) -> None:
        """The "forced" half of EDM2: project the stored weight to unit rows. Output-preserving —
        the forward divides by the row norm anyway — but it stops raw magnitude accumulating, which
        keeps the parameterization well-conditioned instead of merely harmless."""
        n = self._row_norms(self.weight).clamp_min(self.eps)
        shape = [-1] + [1] * (self.weight.dim() - 1)
        self.weight.div_(n.reshape(shape))

    @torch.no_grad()
    def reinit_gain_from_weight(self) -> None:
        """Re-derive `gain` from the current weight's row norms.

        REQUIRED when resuming a checkpoint that predates this feature: `gain` comes back missing
        from load_state_dict(strict=False) and would keep whatever this instance was constructed
        with, silently rescaling the model. Re-deriving reproduces the pre-FWN function exactly.
        """
        self.gain.copy_(self._row_norms(self.weight))

    def extra_repr(self) -> str:
        return f"channels={self.gain.numel()}, inner={type(self.inner).__name__}"


def apply_forced_weight_norm(model: torch.nn.Module,
                             patterns=(r"project_out$",),
                             verbose: bool = False):
    """Wrap every submodule whose qualified name matches any regex in `patterns`.

    Returns the list of wrapped names. Skips (and reports) any module with a zero-norm output
    channel, because W/‖W‖ would be 0/0 — SA3's zero-init postprocess_conv is exactly that case.
    """
    rx = [re.compile(p) for p in patterns]
    targets = []
    for name, mod in model.named_modules():
        if not name or isinstance(mod, ForcedWeightNorm):
            continue
        if not isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
            continue
        if any(r.search(name) for r in rx):
            targets.append((name, mod))

    wrapped, skipped = [], []
    for name, mod in targets:
        if getattr(mod, "bias", None) is not None:
            skipped.append((name, "has bias"))
            continue
        if ForcedWeightNorm.has_zero_rows(mod):
            skipped.append((name, "zero-norm output channel (zero-init layer)"))
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ForcedWeightNorm(mod))
        wrapped.append(name)

    if verbose or skipped:
        print(f"[fwn] forced weight norm applied to {len(wrapped)} module(s): {wrapped}")
        for name, why in skipped:
            print(f"[fwn] SKIPPED {name}: {why}")
        if not wrapped:
            print(f"[fwn] WARNING: patterns {list(patterns)} matched NOTHING wrappable — the "
                  f"arm would run as a plain full-FT with no magnitude bound at all.")
    return wrapped


def force_unit_rows_everywhere(model: torch.nn.Module) -> int:
    """Call force_unit_rows() on every wrapped site. Cheap; safe to call after each optimizer step."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, ForcedWeightNorm):
            mod.force_unit_rows()
            n += 1
    return n
