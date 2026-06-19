import pytest
import torch
from stable_audio_3.inference.longform import (
    ChunkGenerator, CrossfadeStitcher, DriftMonitor, FakeChunkGenerator,
    InpaintContinuationGenerator, LongFormRenderer, PromptSchedule, slerp,
)

def test_single_prompt_no_transitions():
    s = PromptSchedule("acid techno")
    assert s.total_entries() == 1
    p, is_tr, xf = s.resolve(0.0)
    assert p == "acid techno" and is_tr is False
    assert s.resolve(99.0)[0] == "acid techno"

def test_schedule_transitions_fire_once():
    s = PromptSchedule([(0.0, "A"), (10.0, "B")], crossfade_sec=4.0)
    assert s.resolve(0.0) == ("A", False, 4.0)
    assert s.resolve(5.0) == ("A", False, 4.0)
    assert s.resolve(10.0) == ("B", True, 4.0)   # boundary crossed -> transition
    assert s.resolve(12.0) == ("B", False, 4.0)  # already in B -> no repeat

def test_missing_t0_entry_raises():
    with pytest.raises(ValueError):
        PromptSchedule([(5.0, "A"), (10.0, "B")])      # no t=0 entry
    with pytest.raises(ValueError):
        PromptSchedule([(-1.0, "A"), (5.0, "B")])      # negative first time, still no t=0


def test_slerp_endpoints():
    a = torch.randn(1, 8, 4)
    b = torch.randn(1, 8, 4)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-6)


def test_continuation_join_length_and_seam():
    st = CrossfadeStitcher(blend_frames=3)
    cur_tail = torch.zeros(1, 8, 3)          # end of current output
    new_region = torch.ones(1, 8, 10)        # start of next chunk's new region
    out = st.continuation_join(cur_tail, new_region)
    assert out.shape == (1, 8, 10)           # same length as new_region
    # first frame blended toward cur_tail (0), last frames untouched (1)
    assert out[..., 0].abs().mean() < 0.1
    assert torch.allclose(out[..., -1], torch.ones(1, 8))


def test_transition_join_length():
    st = CrossfadeStitcher(blend_frames=3)
    out = st.transition_join(torch.zeros(1, 8, 5), torch.ones(1, 8, 5), n=5)
    assert out.shape == (1, 8, 5)
    assert torch.allclose(out[..., 0], torch.zeros(1, 8), atol=1e-5)
    assert torch.allclose(out[..., -1], torch.ones(1, 8), atol=1e-5)


def test_drift_monitor_flags_collapse():
    m = DriftMonitor(rms_drop_frac=0.6)
    for _ in range(5):
        st = m.observe(torch.randn(1, 8, 16))   # ~unit RMS
        assert m.should_reanchor(st) is False
    collapsed = m.observe(torch.randn(1, 8, 16) * 0.05)  # RMS collapse
    assert m.should_reanchor(collapsed) is True


def test_fake_generator_honors_prefix_and_shape():
    g = FakeChunkGenerator(channels=8)
    prefix = torch.full((1, 8, 4), 0.5)
    out = g.generate("p", prefix_latents=prefix, prefix_frames=4, n_frames=10, seed=0)
    assert out.shape == (1, 8, 10)
    assert torch.allclose(out[..., :4], prefix)        # clamp region preserved
    assert torch.allclose(out[..., 4:], out[..., 4:5].expand(1, 8, 6))  # constant tail


def test_chunkgenerator_is_abstract():
    with pytest.raises(TypeError):
        ChunkGenerator()


def test_fake_generator_no_prefix_returns_constant():
    g = FakeChunkGenerator(channels=4)
    out = g.generate("p", prefix_latents=None, prefix_frames=0, n_frames=6, seed=0)
    assert out.shape == (1, 4, 6)
    assert torch.allclose(out, torch.ones(1, 4, 6))  # first call -> fill value 1.0


def test_renderer_length_and_continuity():
    g = FakeChunkGenerator(channels=8)
    r = LongFormRenderer(g, channels=8, fps=10.0, window_frames=20,
                         overlap_frames=5, blend_frames=2)
    lat = r.render_latents(PromptSchedule("x"), total_frames=50, base_seed=0)
    assert lat.shape == (1, 8, 50)                    # exact requested length
    assert torch.isfinite(lat).all()
    assert len(r.drift_log) >= 3                      # one entry per chunk


def test_renderer_transition_branch_exact_length():
    g = FakeChunkGenerator(channels=8)
    r = LongFormRenderer(g, channels=8, fps=10.0, window_frames=20,
                         overlap_frames=5, blend_frames=2)
    sched = PromptSchedule([(0.0, "A"), (2.0, "B")], crossfade_sec=0.4)  # transition fires
    lat = r.render_latents(sched, total_frames=60)
    assert lat.shape == (1, 8, 60)
    assert torch.isfinite(lat).all()


def test_renderer_transition_zero_crossfade_floored():
    # crossfade_sec rounds to 0 frames -> n must floor to >=1, not crash
    g = FakeChunkGenerator(channels=8)
    r = LongFormRenderer(g, channels=8, fps=10.0, window_frames=20,
                         overlap_frames=5, blend_frames=2)
    sched = PromptSchedule([(0.0, "A"), (2.0, "B")], crossfade_sec=0.02)
    lat = r.render_latents(sched, total_frames=60)
    assert lat.shape == (1, 8, 60)
    assert torch.isfinite(lat).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + model")
def test_sdedit_reanchor_preserves_shape():
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import SDEditReanchor
    m = StableAudioModel.from_pretrained("small-music-base", model_half=False)
    r = SDEditReanchor(m)
    C = m.model.io_channels
    z = torch.randn(1, C, 128, device="cuda")
    out = r.reanchor(z, sigma_peak=0.5, prompt="acid techno", seed=0)
    assert out.shape == z.shape and torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + model")
def test_inpaint_continuation_shape_and_clamp():
    from stable_audio_3 import StableAudioModel
    m = StableAudioModel.from_pretrained("small-music-base", model_half=False)
    g = InpaintContinuationGenerator(m)
    C = m.model.io_channels
    prefix = torch.randn(1, C, 16, device="cuda")
    out = g.generate("acid techno", prefix_latents=prefix, prefix_frames=16,
                     n_frames=128, seed=0)
    assert out.shape == (1, C, 128) and torch.isfinite(out).all()
    # clamp-hardness probe (informational): how close is the clamp region to the prefix?
    err = (out[..., :16].cpu() - prefix.cpu()).abs().mean().item()
    print(f"[clamp] mean abs err in clamp region = {err:.4e}")
