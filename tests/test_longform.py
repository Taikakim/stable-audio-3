import pytest
from stable_audio_3.inference.longform import PromptSchedule

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
