"""EDM2-style forced weight normalization (arXiv 2312.02696) as a retrofit for a fine-tune.

WHAT IT MUST DO, and why each property is load-bearing:

 1. STEP-0 IDENTITY. Wrapping a pretrained layer must not change its function at all. A fine-tune
    that starts with a loss spike has already lost the thing we are trying to preserve, and it is
    what makes this retrofit-safe where MaP-DiT's rotation modulation is not. The gain is therefore
    initialised to the pretrained per-output-channel norms, not to 1.0.

 2. MAGNITUDE OF THE RAW WEIGHT MUST NOT REACH THE OUTPUT. This is the whole point: the drone was
    output-scale runaway, and the output projection is NOT scale-invariant, so weight-norm growth IS
    output-magnitude growth. After wrapping, scaling the stored weight by 1000x must leave the output
    unchanged -- magnitude lives ONLY in the gain.

 3. THE GAIN MUST BE 1-D. FusionOpt routes 2-D matrices to Muon/NS5 and 1-D params to AdamW. A gain
    that arrived as 2-D would be orthogonalized by NS5, which is meaningless for a scale parameter
    and is the same class of mistake CMuon exists to fix (orthogonalizing functionally-distinct
    fused weights).

 4. ZERO-INITIALISED TENSORS MUST BE SKIPPED. The SA3 DiT's `postprocess_conv` is zeros by design
    (dit.py:146), so W/||W|| is 0/0. Wrapping it would produce NaN on the first forward.

 5. STATE-DICT ROUND-TRIP, INCLUDING FROM A CHECKPOINT THAT PREDATES THIS. Resuming a non-FWN run
    into an FWN model must be possible: `gain` will be missing, and the correct recovery is to
    re-derive it from the loaded weight's row norms rather than default it to 1.0 (which would
    silently rescale the model).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_audio_3.training.forced_weight_norm import (  # noqa: E402
    ForcedWeightNorm, apply_forced_weight_norm,
)


def _lin(out=8, inp=6, seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Linear(inp, out, bias=False)
    with torch.no_grad():
        m.weight.mul_(1.7)          # a non-unit pretrained magnitude, like a real checkpoint
    return m


def test_step0_is_functionally_identical_to_the_pretrained_layer():
    lin = _lin()
    x = torch.randn(4, 6)
    before = lin(x).clone()
    fwn = ForcedWeightNorm(lin)
    after = fwn(x)
    assert torch.allclose(before, after, atol=1e-6), "wrapping changed the function at step 0"


def test_raw_weight_magnitude_cannot_reach_the_output():
    fwn = ForcedWeightNorm(_lin())
    x = torch.randn(4, 6)
    base = fwn(x).clone()
    with torch.no_grad():
        fwn.weight.mul_(1000.0)     # simulate a runaway in the stored weight
    assert torch.allclose(base, fwn(x), atol=1e-5), \
        "output changed when raw weight magnitude grew -- normalization is not in the forward path"


def test_gain_carries_the_scale_and_is_one_dimensional():
    lin = _lin()
    fwn = ForcedWeightNorm(lin)
    assert fwn.gain.ndim == 1, "gain must be 1-D so FusionOpt routes it to AdamW, not NS5"
    assert fwn.gain.numel() == lin.out_features, "gain must be per-output-channel"
    x = torch.randn(4, 6)
    base = fwn(x).clone()
    with torch.no_grad():
        fwn.gain.mul_(2.0)
    assert torch.allclose(fwn(x), base * 2.0, atol=1e-5), "gain does not scale the output"


def test_gain_initialised_to_pretrained_row_norms():
    lin = _lin()
    expected = lin.weight.detach().norm(dim=1)
    fwn = ForcedWeightNorm(lin)
    assert torch.allclose(fwn.gain.detach(), expected, atol=1e-6)


def test_gradients_flow_to_both_weight_and_gain():
    fwn = ForcedWeightNorm(_lin())
    out = fwn(torch.randn(4, 6)).sum()
    out.backward()
    assert fwn.weight.grad is not None and fwn.weight.grad.abs().sum() > 0
    assert fwn.gain.grad is not None and fwn.gain.grad.abs().sum() > 0


def test_zero_initialised_layer_is_skipped_not_nan():
    model = torch.nn.Module()
    model.project_out = _lin()
    model.postprocess_conv = torch.nn.Conv1d(4, 4, 1, bias=False)
    torch.nn.init.zeros_(model.postprocess_conv.weight)   # SA3 does exactly this
    wrapped = apply_forced_weight_norm(model, patterns=(r"project_out", r"postprocess_conv"))
    assert "project_out" in wrapped
    assert "postprocess_conv" not in wrapped, "a zero-init tensor must be skipped (0/0 -> NaN)"
    y = model.postprocess_conv(torch.randn(2, 4, 5))
    assert torch.isfinite(y).all()


def test_apply_matches_by_pattern_and_reports_what_it_wrapped():
    model = torch.nn.Module()
    model.project_out = _lin()
    model.other = _lin()
    wrapped = apply_forced_weight_norm(model, patterns=(r"project_out",))
    assert wrapped == ["project_out"]
    assert isinstance(model.project_out, ForcedWeightNorm)
    assert not isinstance(model.other, ForcedWeightNorm)


def test_state_dict_round_trip_preserves_function():
    model = torch.nn.Module()
    model.project_out = _lin()
    apply_forced_weight_norm(model, patterns=(r"project_out",))
    x = torch.randn(4, 6)
    before = model.project_out(x).clone()

    model2 = torch.nn.Module()
    model2.project_out = _lin(seed=99)          # different weights
    apply_forced_weight_norm(model2, patterns=(r"project_out",))
    model2.load_state_dict(model.state_dict())
    assert torch.allclose(model2.project_out(x), before, atol=1e-6)


def test_resuming_a_pre_fwn_checkpoint_rederives_gain_instead_of_defaulting_to_one():
    # a checkpoint from before this feature has `weight` but no `gain`
    plain = _lin()
    old_sd = {"project_out.weight": plain.weight.detach().clone()}
    x = torch.randn(4, 6)
    expected = plain(x).clone()

    model = torch.nn.Module()
    model.project_out = _lin(seed=123)
    apply_forced_weight_norm(model, patterns=(r"project_out",))
    missing, unexpected = model.load_state_dict(old_sd, strict=False)
    assert any("gain" in k for k in missing), "expected gain to be reported missing"
    # the recovery step the trainer must perform
    model.project_out.reinit_gain_from_weight()
    assert torch.allclose(model.project_out(x), expected, atol=1e-6), \
        "resume did not reproduce the pre-FWN function -- gain was not re-derived"


def test_conv1d_is_supported_and_normalises_per_output_channel():
    conv = torch.nn.Conv1d(4, 6, 1, bias=False)
    with torch.no_grad():
        conv.weight.mul_(2.3)
    x = torch.randn(2, 4, 7)
    before = conv(x).clone()
    fwn = ForcedWeightNorm(conv)
    assert torch.allclose(fwn(x), before, atol=1e-5)
    assert fwn.gain.numel() == 6


def test_forced_projection_renormalises_stored_weight_without_changing_output():
    """The 'forced' half of EDM2: the stored weight is projected back to unit rows, so raw magnitude
    cannot accumulate at all. Output must be unchanged by the projection."""
    fwn = ForcedWeightNorm(_lin())
    x = torch.randn(4, 6)
    base = fwn(x).clone()
    with torch.no_grad():
        fwn.weight.mul_(37.0)
    fwn.force_unit_rows()
    assert torch.allclose(fwn.weight.detach().norm(dim=1),
                          torch.ones(fwn.gain.numel()), atol=1e-5)
    assert torch.allclose(fwn(x), base, atol=1e-5)


def test_gain_routes_to_the_scalar_group_not_the_spectral_one():
    """The load-bearing optimizer claim: FusionOpt must send `gain` to ScheduleFree-AdamW, NOT to
    Muon/NS5. Orthogonalizing a per-channel SCALE parameter is meaningless — it is the same class of
    error CMuon exists to fix (orthogonalizing functionally-distinct weights that share a tensor).
    Asserted against the real build_fusion_param_groups, not assumed from the 1-D convention.
    """
    import sys as _s
    _s.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools")
    from stable_audio_tools.training.fusion_groups import build_fusion_param_groups

    # REAL shape matters: build_fusion_param_groups classifies 2-D with min(shape) >= 128 as
    # spectral and everything else (incl. "small/odd projections") as scalar. An 8x6 toy fixture
    # lands in the SCALAR group and the test would pass for the wrong reason. SA3's project_out is
    # Linear(dim=1536 -> 256*patch), so both dims clear 128.
    model = torch.nn.Module()
    model.project_out = _lin(out=256, inp=1536)
    apply_forced_weight_norm(model, patterns=(r"project_out",))

    groups = build_fusion_param_groups(model)
    gain = model.project_out.gain
    weight = model.project_out.weight
    where = {}
    for gi, g in enumerate(groups):
        for p in g["params"]:
            if p is gain:
                where["gain"] = g
            if p is weight:
                where["weight"] = g
    assert "gain" in where, "gain never reached any param group — it would not be optimized at all"
    assert "weight" in where
    assert where["gain"] is not where["weight"], "gain and weight must not share a group"
    # the spectral group is the orthogonalized one; identify it however this build labels it
    g_gain, g_w = where["gain"], where["weight"]
    spectral_flag = [k for k in ("spectral", "use_spectral", "is_spectral") if k in g_w or k in g_gain]
    if spectral_flag:
        k = spectral_flag[0]
        assert g_w.get(k), "the 2-D weight should be in the spectral/orthogonalized group"
        assert not g_gain.get(k), "the 1-D gain must NOT be in the spectral/orthogonalized group"
