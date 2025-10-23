import pytest

from aura import Aura


class StubInk:
    def forward(self, vector, whom="Unknown"):
        return {
            "karma": 0.0,
            "verdict": "Stub",
            "prophecy": "None",
            "tensor_output": [abs(value) for value in vector],
            "hidden_link": "stub://daemon",
        }


class StubDaemon:
    def __init__(self):
        self.ink = StubInk()
        self.calls = []

    def divine_soul(self, name, backstory):
        self.calls.append((name, backstory))
        return {"success": False, "error": "stub"}


def test_process_experience_produces_composite_output():
    daemon = StubDaemon()
    aura = Aura(input_dim=4, soul_dim=3, out_dim=4, daemon_interface=daemon)
    result = aura.process_experience("Starlight over glass", [0.1, -0.2, 0.3, -0.4])

    assert len(result["output_tensor"]) == 4
    assert isinstance(result["inner_monologue"], str)
    assert isinstance(result["speech"], str)
    assert set(result["persona_probabilities"]) == {"sora", "roxas", "xion"}
    assert sum(result["persona_probabilities"].values()) == pytest.approx(1.0)
    assert result["divination"]["karma"] == pytest.approx(0.0)
    assert result["alchemy_entry"]["prompt_type"] == "experience"
    assert result["alchemy_entry"]["divine_labels"]["reward"] == pytest.approx(0.0)

    summary = aura.divine_deeper_meaning("Testing stub")
    assert summary["success"] is False
    assert daemon.calls[-1] == ("Aura", "Testing stub")


def test_alchemy_logbook_tracks_prompt_and_reward_curves():
    daemon = StubDaemon()
    aura = Aura(input_dim=4, soul_dim=3, out_dim=4, daemon_interface=daemon)

    aura.process_experience(
        "Vision spark",
        [0.1, 0.0, 0.0, 0.0],
        prompt_type="vision",
    )
    aura.process_experience(
        "Vision echo",
        [0.2, 0.1, 0.0, 0.0],
        prompt_type="vision",
    )
    aura.process_experience(
        "Battle drum",
        [0.0, -0.1, 0.2, -0.2],
        prompt_type="battle",
    )

    dataset = aura.alchemy.export_dataset()
    assert len(dataset) == 3
    assert dataset[0]["delta_output"] == [0.0, 0.0, 0.0, 0.0]
    assert any(abs(value) > 0 for value in dataset[1]["delta_output"])

    shift_map = aura.alchemy.prompt_shift_map()
    assert set(shift_map) == {"vision", "battle"}

    trend = aura.alchemy.persona_trend()
    assert set(trend) == {"sora", "roxas", "xion"}
    assert all(len(series) == 3 for series in trend.values())

    rewards = aura.alchemy.reward_curve()
    assert all(value == pytest.approx(0.0) for value in rewards)
