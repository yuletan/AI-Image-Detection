"""Heads / fine-tuning / fusion (Day 2: A3 builds here)."""

from __future__ import annotations


class ClipMlpHead:
    """Placeholder so imports work on Day 0. Real 2-layer MLP lands Day 1-2."""

    def __init__(self, in_dim: int = 768):
        self.in_dim = in_dim

    def score(self, images) -> list[float]:  # Contract: Model.score -> probs[0..1]
        return [0.5 for _ in images]
