"""Explicit non-overlapping seed manifests for synthetic dataset splits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SPLIT_SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 99_999),
    "validation": (100_000, 109_999),
    "test": (200_000, 209_999),
    "ood": (300_000, 309_999),
}

_SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "compositional_ood": "ood",
}


def canonical_split_name(split: str) -> str:
    """Normalise common split aliases and reject unknown split names."""

    canonical = _SPLIT_ALIASES.get(split, split)
    if canonical not in SPLIT_SEED_RANGES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_SEED_RANGES)}")
    return canonical


@dataclass(frozen=True)
class SeedManifest:
    """Named immutable episode seeds for reproducible dataset construction."""

    split: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        canonical = canonical_split_name(self.split)
        object.__setattr__(self, "split", canonical)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("a seed manifest may not contain duplicates")
        lower, upper = SPLIT_SEED_RANGES[canonical]
        if any(seed < lower or seed > upper for seed in self.seeds):
            raise ValueError(f"all {canonical} seeds must be inside [{lower}, {upper}]")

    def __len__(self) -> int:
        return len(self.seeds)

    def __iter__(self) -> Iterable[int]:
        return iter(self.seeds)


def make_seed_manifest(
    split: str,
    count: int,
    *,
    offset: int = 0,
) -> SeedManifest:
    """Create a deterministic contiguous manifest inside a reserved seed range."""

    canonical = canonical_split_name(split)
    if count < 0 or offset < 0:
        raise ValueError("manifest count and offset must be nonnegative")
    lower, upper = SPLIT_SEED_RANGES[canonical]
    start = lower + offset
    stop = start + count
    if stop - 1 > upper:
        raise ValueError(
            f"requested {count} seeds at offset {offset} exceeds the reserved {canonical} range"
        )
    return SeedManifest(canonical, tuple(range(start, stop)))


def assert_disjoint_manifests(*manifests: SeedManifest) -> None:
    """Raise if any episode seed appears in more than one supplied manifest."""

    seen: set[int] = set()
    for manifest in manifests:
        overlap = seen.intersection(manifest.seeds)
        if overlap:
            raise ValueError(f"seed manifests overlap at {sorted(overlap)[:5]}")
        seen.update(manifest.seeds)
