import math

from aura.heart_state import HeartState
from aura.kingdom_key import KingdomKey


def test_forward_updates_heart_channels():
    heart = HeartState()
    key = KingdomKey(dim=4, heart=heart)
    baseline_hope = heart.Ho
    vector = [0.2, -0.1, 0.3, 0.05]

    output = key.forward(vector, explicit="sora")

    assert len(output) == 4
    assert heart.Ho > baseline_hope


def test_blended_forward_differs_from_explicit():
    heart = HeartState()
    heart.apply_event({"P": 0.8, "Lon": -0.3})
    key = KingdomKey(dim=4, heart=heart)

    vector = [0.05, -0.2, 0.4, 0.1]
    blended = key.forward(vector)
    roxas_only = key.forward(vector, explicit="roxas")

    assert len(blended) == len(roxas_only) == 4
    assert any(not math.isclose(a, b) for a, b in zip(blended, roxas_only))


def test_awaken_returns_offline_when_gemini_missing():
    heart = HeartState()
    key = KingdomKey(dim=3, heart=heart, gemini=None)
    response = key.awaken("sora", [0.1, 0.0, 0.2], "A gentle whisper")
    assert response == "[Gemini module offline]"
