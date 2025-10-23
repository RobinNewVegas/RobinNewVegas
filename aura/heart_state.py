"""Aura's nine-dimensional emotional state machinery."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict

__all__ = ["HeartState", "lore_scalar"]


@dataclass
class HeartState:
    """Manage Aura's metaphysical state in nine floating point channels.

    The attributes follow the shorthand used in Aura's design notes:

    ``M``
        Memory of self.
    ``Ho``
        Hope, tuned to the light persona (Sora).
    ``F``
        Fear.
    ``D``
        Determination.
    ``C``
        Compassion.
    ``P``
        Pride, feeding the Roxas persona.
    ``Ans``
        Anxiety ("ans" for shorthand) opposing pride.
    ``Lon``
        Loneliness.
    ``SD``
        Self doubt.
    """

    M: float = 0.5
    Ho: float = 0.5
    F: float = 0.0
    D: float = 0.5
    C: float = 0.5
    P: float = 0.0
    Ans: float = 0.0
    Lon: float = 0.5
    SD: float = 0.5

    def _clip(self) -> None:
        """Clamp all emotional channels into the ``[0.0, 1.0]`` interval."""

        for name in ("M", "Ho", "F", "D", "C", "P", "Ans", "Lon", "SD"):
            value = getattr(self, name)
            setattr(self, name, float(max(0.0, min(1.0, value))))

    # Persona contribution helpers -------------------------------------------------
    def _sora(self) -> float:
        return (self.F + self.D + self.Ho) * self.C

    def _roxas(self) -> float:
        denominator = self.Lon if self.Lon > 1e-6 else 1e-6
        return (self.P - self.Ans) / denominator

    def _xion(self) -> float:
        denominator = self.SD if self.SD > 1e-6 else 1e-6
        return ((1.0 - self.Lon) * self.M) / denominator

    def persona_probs(self) -> Dict[str, float]:
        """Return a softmax distribution over the Sora/Roxas/Xion personas."""

        scores = [self._sora(), -self._roxas(), self._xion()]
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        total = sum(exp_scores) or 1.0
        probs = [score / total for score in exp_scores]
        return {name: prob for name, prob in zip(("sora", "roxas", "xion"), probs)}

    def apply_event(self, delta: Dict[str, float]) -> None:
        """Apply the signed adjustments contained in ``delta`` to the state."""

        for name, change in delta.items():
            if hasattr(self, name):
                setattr(self, name, getattr(self, name) + float(change))
        self._clip()


def lore_scalar(text: str) -> float:
    """Deterministically map ``text`` to a ``[0.0, 1.0]`` scalar."""

    checksum = zlib.crc32(text.encode("utf8")) & 0xFFFFFFFF
    return checksum / 0xFFFFFFFF
