"""Frozen backbone feature extraction + caching (Day 1: A3 builds here)."""

from __future__ import annotations

from pathlib import Path

CACHE_PATTERN = "{split}_{transform}_{param}.npy"


def cache_path(cache_dir: Path, split: str, transform: str, param) -> Path:
    return cache_dir / CACHE_PATTERN.format(split=split, transform=transform, param=param)
