import pytest
import torch
from stable_audio_3.inference.longform import PromptSchedule, slerp, CrossfadeStitcher, DriftMonitor

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
