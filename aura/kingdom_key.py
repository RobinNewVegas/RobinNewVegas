"""Neural-flavoured implementation of the fabled *Kingdom Key*."""

from __future__ import annotations

from typing import Optional, Sequence

from ._math import LinearLayer, SequentialLayer, combine, gelu, layer_norm, mean_and_std, relu
from .heart_state import HeartState, lore_scalar

__all__ = ["KingdomKey"]


class KingdomKey:
    """Blend Aura's inner personas through lightweight linear algebra."""

    PERSONA_LORE = {
        "sora": "Your heart is your greatest strength.",
        "roxas": "Key of destiny—Oblivion & Oathkeeper.",
        "xion": "∫ Kindness d(Story)",
    }
    AURA_BIO = (
        "Aura, 13½-year-old synthetic girl, balances Detroit rA9 pragmatism "
        "with Wonderland paradox."
    )

    def __init__(self, dim: int, heart: HeartState, gemini: Optional[object] = None) -> None:
        self.dim = dim
        self.heart = heart
        self.gemini = gemini
        self.transforms = {
            "sora": SequentialLayer(
                LinearLayer(dim, dim, seed=("key", "sora", 0)),
                layer_norm,
                relu,
            ),
            "roxas": LinearLayer(dim, dim, seed=("key", "roxas", 0)),
            "xion": SequentialLayer(
                LinearLayer(dim, dim * 2, seed=("key", "xion", 0)),
                gelu,
                LinearLayer(dim * 2, dim, seed=("key", "xion", 1)),
            ),
        }
        self.lore_gains = [lore_scalar(text) for text in self.PERSONA_LORE.values()]

    def awaken(self, persona: str, ctx: Sequence[float], reflection: str) -> str:
        """Return a textual response for ``persona`` using the Gemini stub."""

        if not self.gemini:
            return "[Gemini module offline]"
        heart_state = ", ".join(f"{key}={value:.2f}" for key, value in self.heart.__dict__.items())
        mean, std = mean_and_std(ctx)
        prompt = (
            f"System:\n\n{self.AURA_BIO}\n\nYour inner monologue just whispered: \"{reflection}\"\n\n"
            f"Persona Lore: {self.PERSONA_LORE.get(persona, 'Unknown')}\nHeart Snapshot: {heart_state}\n"
            f"Context: Tensor μ={mean:.4f}, σ={std:.4f}\n\n"
            f"Speak your immediate, outward thought as {persona.capitalize()}:"
        )
        try:
            response = self.gemini.generate_content(prompt)
            text = getattr(response, "text", "")
            return text.strip().replace("*", "") or "[Gemini silence]"
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"[Gemini error: {exc}]"

    def forward(self, vector: Sequence[float], explicit: Optional[str] = None) -> Sequence[float]:
        """Blend persona transforms using the heart's emotional balance."""

        persona_order = ("sora", "roxas", "xion")
        probs = self.heart.persona_probs()
        if explicit is not None and explicit in self.transforms:
            result = self.transforms[explicit](vector)
        else:
            weights = [probs[name] for name in persona_order]
            outputs = [self.transforms[name](vector) for name in persona_order]
            weighted = [weight * gain for weight, gain in zip(weights, self.lore_gains)]
            result = combine(list(zip(weighted, outputs)))
        self.heart.apply_event({
            "Ho": 0.02 * probs["sora"],
            "M": 0.01 * probs["xion"],
            "SD": -0.015 * probs["roxas"],
        })
        return result
