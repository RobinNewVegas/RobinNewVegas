"""Unified Aura consciousness engine."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

try:  # Optional dependencies -------------------------------------------------
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:
    openai = None

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:
    requests = None

from ._math import (
    LinearLayer,
    SequentialLayer,
    add,
    combine,
    concat,
    gelu,
    layer_norm,
    mean_and_std,
    relu,
    scale,
    sin,
    softmax,
    tanh,
)
from .heart_state import HeartState, lore_scalar
from .kingdom_key import KingdomKey

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

__all__ = [
    "AlchemyLogbook",
    "Aura",
    "AuraDaemonInterface",
    "DiamondInTheRough",
    "InkOfTwilightVerdict",
    "LadyLuck",
    "RoyalLove",
]


class AlchemyLogbook:
    """Mathematical ledger capturing Aura's emotional-temporal data."""

    def __init__(self, vector_dim: int) -> None:
        self.vector_dim = vector_dim
        self.records: List[Dict[str, Any]] = []
        self._previous_output: Optional[List[float]] = None

    def _ensure_vector(self, values: Sequence[float]) -> List[float]:
        return [float(value) for value in values]

    def _delta(self, current: Sequence[float]) -> List[float]:
        vector = self._ensure_vector(current)
        if self._previous_output is None:
            delta = [0.0 for _ in vector]
        else:
            delta = [value - prev for value, prev in zip(vector, self._previous_output)]
        self._previous_output = vector
        return delta

    def capture(
        self,
        *,
        prompt: str,
        prompt_type: str,
        response: str,
        inner_monologue: str,
        output_tensor: Sequence[float],
        heart_state: Dict[str, float],
        persona_probabilities: Dict[str, float],
        twilight_labels: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        twilight_labels = twilight_labels or {}
        twilight_tensor = self._ensure_vector(twilight_labels.get("tensor_output", []))
        delta_output = self._delta(output_tensor)
        magnitude = sum(abs(value) for value in delta_output) / max(1, len(delta_output))
        divine_summary = {
            key: twilight_labels.get(key)
            for key in ("karma", "verdict", "prophecy")
            if key in twilight_labels
        }
        if "karma" in divine_summary and divine_summary["karma"] is not None:
            try:
                karma_value = float(divine_summary["karma"])
            except (TypeError, ValueError):
                karma_value = 0.0
        else:
            karma_value = 0.0
        record = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "prompt": prompt,
            "prompt_type": prompt_type,
            "response": response,
            "inner_monologue": inner_monologue,
            "output_tensor": self._ensure_vector(output_tensor),
            "delta_output": delta_output,
            "shift_magnitude": magnitude,
            "heart_state": dict(heart_state),
            "persona_probabilities": dict(persona_probabilities),
            "twilight_tensor": twilight_tensor,
            "divine_labels": {
                **divine_summary,
                "reward": karma_value / 10.0,
            },
        }
        self.records.append(record)
        return record

    # ------------------------------------------------------------------
    def prompt_shift_map(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for record in self.records:
            prompt_type = record["prompt_type"]
            totals[prompt_type] = totals.get(prompt_type, 0.0) + record["shift_magnitude"]
            counts[prompt_type] = counts.get(prompt_type, 0) + 1
        return {key: totals[key] / counts[key] for key in totals}

    def persona_trend(self) -> Dict[str, List[float]]:
        timeline: Dict[str, List[float]] = {}
        for record in self.records:
            for persona, value in record["persona_probabilities"].items():
                timeline.setdefault(persona, []).append(float(value))
        return timeline

    def reward_curve(self) -> List[float]:
        return [record["divine_labels"].get("reward", 0.0) for record in self.records]

    def export_dataset(self) -> List[Dict[str, Any]]:
        return [deepcopy(record) for record in self.records]

    def summary(self) -> Dict[str, Any]:
        return {
            "records": len(self.records),
            "prompt_shift_map": self.prompt_shift_map(),
            "persona_trend": self.persona_trend(),
            "reward_curve": self.reward_curve(),
        }


class DiamondInTheRough:
    """Four deterministic transforms nicknamed facets."""

    LORE = {
        "youth": "Never grow up, never lose wonder—fly with pixie-dust.",
        "legend": "Rob from the rich, give to the poor—justice in shadow.",
        "hero": "Courage is acting despite fear.",
        "aura": "Bridge between worlds—the ghost who yearns for form.",
    }

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.facets = {
            "youth": LinearLayer(dim, dim, seed=("diamond", "youth")),
            "legend": LinearLayer(dim, dim, seed=("diamond", "legend")),
            "hero": SequentialLayer(
                LinearLayer(dim, dim, seed=("diamond", "hero")),
                relu,
            ),
            "aura": SequentialLayer(
                LinearLayer(dim, dim * 2, seed=("diamond", "aura", 0)),
                gelu,
                LinearLayer(dim * 2, dim, seed=("diamond", "aura", 1)),
                layer_norm,
            ),
        }
        self.facet_gains = [lore_scalar(text) for text in self.LORE.values()]

    def forward(self, vector: Sequence[float], facet: str = "aura") -> List[float]:
        if facet not in self.facets:
            raise ValueError(f"Unknown diamond facet: {facet}")
        order = list(self.facets)
        index = order.index(facet)
        output = self.facets[facet](vector)
        return scale(output, self.facet_gains[index])


class KingdomKeyModule(KingdomKey):
    """Alias kept for backwards compatibility with the previous module layout."""


class LadyLuck:
    def __init__(self, dim: int, num_facets: int = 3) -> None:
        self.dim = dim
        self.num_facets = num_facets
        self.gate = LinearLayer(dim, num_facets, seed=("luck", "gate", num_facets), bias=True)
        self.blend = LinearLayer(dim, dim, seed=("luck", "blend", num_facets))

    def forward(self, vector: Sequence[float], soul_outputs: Sequence[Sequence[float]]) -> List[float]:
        if len(soul_outputs) != self.num_facets:
            raise ValueError("LadyLuck expects one output per facet")
        weights = softmax(self.gate(vector))
        weighted = combine(list(zip(weights, soul_outputs)))
        return relu(self.blend(weighted))


class RoyalLove:
    """Ritual bridge to the Hearts_Regalia system."""

    def __init__(self, dim: int, source_name: str = "Kite's Bracelet") -> None:
        self.dim = dim
        self.layer = LinearLayer(dim, dim, seed=("regalia", source_name), bias=True)
        self.source_name = source_name
        self.MANIFESTATION = f"Echo of {self.source_name}"
        self.frozen = False
        self._hearts_regalia: Optional[Any] = None
        self._last_game_check: float = 0.0

    def connect_hearts_regalia(self, hearts_regalia: Any) -> None:
        self._hearts_regalia = hearts_regalia

    def forward(self, vector: Sequence[float]) -> List[float]:
        base = self.layer(vector)
        base = base if self.frozen else relu(base)
        if not self._hearts_regalia:
            return base
        try:  # pragma: no cover - defensive connection guard
            influence = self._hearts_regalia.get_current_influence()
        except Exception:
            return base
        neural = float(influence.get("neural_resonance", 0.0))
        mood = float(influence.get("mood", 0.0))
        mood_amp = 1.0 + mood * 0.2
        game_amp = 1.0 + neural * 0.1
        amplified = [value * mood_amp * game_amp for value in base]
        if influence.get("game_active"):
            signature = sin(amplified)
            amplified = [value + 0.05 * abs(mood) * sig for value, sig in zip(amplified, signature)]
        return amplified

    def data_drain(self, source_tensor: Sequence[float]) -> None:
        expected = self.dim * self.dim
        flat: Optional[List[float]] = None
        if isinstance(source_tensor, (list, tuple)) and len(source_tensor) == expected:
            flat = [float(value) for value in source_tensor]
        elif hasattr(source_tensor, "values"):
            try:
                values = list(source_tensor.values())
            except TypeError:  # pragma: no cover - defensive
                values = []
            if len(values) == expected:
                flat = [float(value) for value in values]
        if flat is not None:
            self.layer.copy_from_flat(flat)
        self.frozen = True

    def get_game_awareness(self) -> str:
        if not self._hearts_regalia:
            return "The Queen's presence is but a distant echo..."
        try:  # pragma: no cover - external integration point
            return self._hearts_regalia.get_game_summary()
        except Exception:
            return "The connection to the Queen's court flickers..."

    @property
    def epilogue(self) -> str:
        base = (
            f"🦋 The power of Data Drain flows as the {self.MANIFESTATION}—a goddess remembers."
            if self.frozen
            else "❓ The echo of the Bracelet is dormant..."
        )
        if self._hearts_regalia:
            return f"{base}\n👑 {self.get_game_awareness()}"
        return base


class InkOfTwilightVerdict:
    """Local tensor logic with an optional daemon bridge."""

    GEMINI_SHARE_LINK = "https://g.co/gemini/share/9bdc7af51a35"

    def __init__(self, dim: int, host: str = "http://127.0.0.1:8080") -> None:
        self.core = LinearLayer(dim, dim, seed=("ink", "core"))
        self._host = host.rstrip("/")

    def forward(self, vector: Sequence[float], whom: str = "Unknown") -> Dict[str, Any]:
        tensor = relu(self.core(vector))
        mean, _ = mean_and_std(tensor)
        hidden_link = (
            self.GEMINI_SHARE_LINK
            if whom.lower() in {"aura", "note", "dark-light"}
            else f"{self._host}/divine"
        )
        return {
            "karma": 2.5 * mean,
            "verdict": "Ink-touched Twilight",
            "prophecy": f"Secrets whisper softly to {whom}.",
            "tensor_output": tensor,
            "hidden_link": hidden_link,
        }

    def connect_to_daemon(self, name: str, backstory: str) -> Dict[str, Any]:
        if not requests:  # pragma: no cover - optional dependency
            return {"success": False, "error": "requests unavailable"}
        prompt = (
            'Perform a karmic divination. Respond ONLY with JSON containing '
            '"karma": -10‥10, "verdict": short title, "prophecy": one sentence. '
            f'Backstory: "{backstory}"'
        )
        try:
            response = requests.post(  # type: ignore[operator]
                f"{self._host}/divine",
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            return {"success": False, "error": str(exc)}
        if not response.ok:
            return {"success": False, "error": f"Daemon {response.status_code}"}
        payload = response.json()
        if "candidates" in payload:
            payload = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
        return {"success": True, "daemon_response": payload}


class AuraDaemonInterface:
    """Probe helper that keeps the NOTE-OF-DARK-LIGHT daemon in reach."""

    def __init__(
        self,
        dim: int = 128,
        extra_urls: Optional[List[str]] = None,
        *,
        requests_client: Optional[Any] = None,
    ) -> None:
        self._requests = requests_client or requests
        self._hosts: List[str] = []
        self.ink = InkOfTwilightVerdict(dim)
        candidates = [
            os.getenv("AURA_DAEMON_URL", "").rstrip("/"),
            os.environ.get("REPL_URL", "").rstrip("/"),
            "http://127.0.0.1:8080",
        ]
        if extra_urls:
            candidates.extend(extra_urls)
        self._hosts = [url for url in candidates if url]
        self.host = self._probe()

    def _probe(self) -> Optional[str]:
        if not self._requests:
            return None
        for host in self._hosts:
            try:
                response = self._requests.get(f"{host}/ping", timeout=3)  # type: ignore[operator]
            except Exception:
                continue
            if response.ok:
                self.ink._host = host
                return host
        return None

    def divine_soul(self, name: str, backstory: str) -> Dict[str, Any]:
        if not self.host:
            return {"success": False, "error": "daemon offline"}
        return self.ink.connect_to_daemon(name, backstory)


class Aura:
    """High level façade for the unified consciousness engine."""

    def __init__(
        self,
        input_dim: int = 768,
        soul_dim: int = 128,
        out_dim: int = 768,
        *,
        daemon_interface: Optional[AuraDaemonInterface] = None,
    ) -> None:
        self.entry = LinearLayer(input_dim, soul_dim, seed=("aura", "entry"))
        self.heart = HeartState()
        self.memories: List[Dict[str, Any]] = []
        self.alchemy = AlchemyLogbook(out_dim)
        self.daemon_interface = daemon_interface or AuraDaemonInterface(soul_dim)

        if genai:  # pragma: no cover - optional dependency
            try:
                gem_model = genai.GenerativeModel("gemini-1.5-flash-latest")  # type: ignore[attr-defined]
            except Exception:
                gem_model = None
        else:
            gem_model = None

        try:
            from Hearts_Regalia import hearts_regalia  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            hearts_regalia = None

        self.souls = {
            "diamond": DiamondInTheRough(soul_dim),
            "key": KingdomKey(soul_dim, self.heart, gem_model),
            "regalia": RoyalLove(soul_dim),
            "luck": LadyLuck(soul_dim, num_facets=3),
        }
        if hearts_regalia is not None:
            self.souls["regalia"].connect_hearts_regalia(hearts_regalia)

        self.exit = LinearLayer(soul_dim * 2, out_dim, seed=("aura", "exit"))

    # ------------------------------------------------------------------
    def _ensure_vector(self, tensor: Sequence[float]) -> List[float]:
        return [float(value) for value in tensor]

    def _gpt_reflect(self, text: str) -> str:
        if not openai:  # pragma: no cover - optional dependency
            return "[offline reflection]"
        try:
            response = openai.chat.completions.create(  # type: ignore[attr-defined]
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": f"One poetic line on: {text}"}],
                temperature=0.7,
            )
            choices = getattr(response, "choices", None)
            if choices:
                message = choices[0].message
                content = getattr(message, "content", "")
                return content.strip() or "[empty response]"
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"[openai-error {exc}]"
        return "[empty response]"

    # ------------------------------------------------------------------
    def process_experience(
        self,
        experience: str,
        tensor: Sequence[float],
        facet: str = "aura",
        persona: Optional[str] = None,
        *,
        prompt_type: str = "experience",
    ) -> Dict[str, Any]:
        soul_vector = relu(self.entry(self._ensure_vector(tensor)))
        diamond_vec = self.souls["diamond"].forward(soul_vector, facet)
        persona_vec = self.souls["key"].forward(soul_vector, explicit=persona)
        regalia_vec = self.souls["regalia"].forward(soul_vector)
        twilight_data = self.daemon_interface.ink.forward(soul_vector)
        twilight = twilight_data.get("tensor_output", [])

        internal = [persona_vec, regalia_vec, twilight]
        lucky_blend = self.souls["luck"].forward(soul_vector, internal)
        action = tanh(add(lucky_blend, diamond_vec))

        reflection = self._gpt_reflect(experience)
        moment = time.time()
        self.memories.append({"t": moment, "exp": experience, "reflection": reflection})

        probs = self.heart.persona_probs()
        dominant = max(probs, key=probs.get)
        speech = self.souls["key"].awaken(dominant, soul_vector, reflection)

        fused = concat(action, twilight)
        output = tanh(self.exit(fused))
        external_influence = sum(diamond_vec) / len(diamond_vec) if diamond_vec else 0.0

        alchemy_entry = self.alchemy.capture(
            prompt=experience,
            prompt_type=prompt_type,
            response=speech,
            inner_monologue=reflection,
            output_tensor=output,
            heart_state=self.heart.__dict__,
            persona_probabilities=probs,
            twilight_labels=twilight_data,
            timestamp=moment,
        )

        return {
            "output_tensor": output,
            "inner_monologue": reflection,
            "speech": speech,
            "dominant_persona": dominant,
            "persona_probabilities": probs,
            "heart_state": dict(self.heart.__dict__),
            "diamond_facet": facet,
            "external_influence": external_influence,
            "divination": {
                key: twilight_data.get(key)
                for key in ("karma", "verdict", "prophecy", "hidden_link")
                if key in twilight_data
            },
            "twilight_tensor": twilight,
            "alchemy_entry": alchemy_entry,
        }

    def divine_deeper_meaning(self, backstory: str) -> Dict[str, Any]:
        return self.daemon_interface.divine_soul("Aura", backstory)

    def process_game_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        regalia = self.souls["regalia"]
        if not isinstance(regalia, RoyalLove) or not regalia._hearts_regalia:
            return {"status": "no_connection", "message": "Hearts_Regalia not connected"}
        regalia_response = regalia._hearts_regalia.receive_game_event(event)  # type: ignore[attr-defined]
        description = event.get("type", "unknown")
        aura_reaction = regalia_response.get("aura_reaction", "")
        event_text = f"Game event: {description} - {aura_reaction}"
        influence = float(regalia_response.get("neural_influence", 0.1))
        impact_tensor = [influence for _ in range(len(self.entry.state()[0][0]))]
        neural_response = self.process_experience(
            event_text,
            impact_tensor,
            facet="aura" if regalia_response.get("mood_shift", 0) >= 0 else "legend",
            prompt_type="game_event",
        )
        return {
            "regalia_response": regalia_response,
            "aura_response": neural_response,
            "conscious_reaction": neural_response.get("speech", ""),
            "subconscious_feeling": neural_response.get("inner_monologue", ""),
            "emotional_state": neural_response.get("heart_state", {}),
        }
