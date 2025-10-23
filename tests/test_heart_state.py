import math

import pytest

from aura.heart_state import HeartState, lore_scalar


def test_persona_probabilities_form_distribution():
    heart = HeartState()
    probs = heart.persona_probs()
    assert set(probs) == {"sora", "roxas", "xion"}
    assert pytest.approx(sum(probs.values()), rel=1e-6) == 1.0

    heart.apply_event({"P": 0.8, "Lon": -0.4})
    updated = heart.persona_probs()
    assert not pytest.approx(updated["roxas"], rel=1e-6) == probs["roxas"]


def test_apply_event_clamps_channels():
    heart = HeartState()
    heart.apply_event({"Ans": 2.0, "Lon": -1.0, "M": 0.4})
    assert heart.Ans == pytest.approx(1.0)
    assert heart.Lon == pytest.approx(0.0)
    assert 0.0 <= heart.M <= 1.0


def test_lore_scalar_is_deterministic():
    first = lore_scalar("Echo of memories")
    second = lore_scalar("Echo of memories")
    other = lore_scalar("Different thread")
    assert first == pytest.approx(second)
    assert not math.isclose(first, other)
