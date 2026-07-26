"""On-the-fly deterministic dataset backed by explicit simulator seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from torch.utils.data import Dataset

from world_model.datasets.collate import collate_episodes
from world_model.datasets.splits import (
    SeedManifest,
    canonical_split_name,
    make_seed_manifest,
)
from world_model.simulator.episode import Episode, generate_episode
from world_model.simulator.sphere_world import SphereWorldConfig


def clone_nested(value: Any) -> Any:
    """Clone tensor-containing nested records before returning cached samples."""

    if hasattr(value, "clone") and callable(value.clone):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    return value


class SyntheticSphereDataset(Dataset[Episode]):
    """Generate one labelled sphere episode deterministically per manifest seed."""

    DEFAULT_LENGTHS = {
        "train": 1_024,
        "validation": 128,
        "test": 128,
        "ood": 128,
    }

    def __init__(
        self,
        config: SphereWorldConfig | Mapping[str, Any] | Any,
        *,
        split: str = "train",
        num_episodes: int | None = None,
        seeds: Sequence[int] | SeedManifest | None = None,
        seed_offset: int = 0,
        memory_cache: bool = False,
    ) -> None:
        self.split = canonical_split_name(split)
        resolved = SphereWorldConfig.from_config(config)
        self.config = (
            resolved.for_distribution("ood")
            if self.split == "ood"
            else replace(resolved, distribution="in_distribution")
        )
        if seeds is not None:
            if isinstance(seeds, SeedManifest):
                if seeds.split != self.split:
                    raise ValueError("manifest split does not match dataset split")
                self.manifest = seeds
            else:
                self.manifest = SeedManifest(self.split, tuple(int(seed) for seed in seeds))
            if num_episodes is not None and num_episodes != len(self.manifest):
                raise ValueError("num_episodes conflicts with explicit seeds")
        else:
            count = self.DEFAULT_LENGTHS[self.split] if num_episodes is None else int(num_episodes)
            self.manifest = make_seed_manifest(self.split, count, offset=seed_offset)
        self.memory_cache = bool(memory_cache)
        self._cache: dict[int, Episode] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Episode:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.memory_cache and index in self._cache:
            return clone_nested(self._cache[index])
        seed = self.manifest.seeds[index]
        episode = generate_episode(self.config, seed)
        episode["metadata"]["split"] = self.split
        if self.memory_cache:
            self._cache[index] = clone_nested(episode)
        return episode


def make_synthetic_datasets(
    config: SphereWorldConfig | Mapping[str, Any] | Any,
    *,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    ood_episodes: int = 0,
    memory_cache: bool = False,
) -> dict[str, SyntheticSphereDataset]:
    """Construct standard non-overlapping split datasets."""

    datasets = {
        "train": SyntheticSphereDataset(
            config,
            split="train",
            num_episodes=train_episodes,
            memory_cache=memory_cache,
        ),
        "validation": SyntheticSphereDataset(
            config,
            split="validation",
            num_episodes=validation_episodes,
            memory_cache=memory_cache,
        ),
        "test": SyntheticSphereDataset(
            config,
            split="test",
            num_episodes=test_episodes,
            memory_cache=memory_cache,
        ),
    }
    if ood_episodes > 0:
        datasets["ood"] = SyntheticSphereDataset(
            config,
            split="ood",
            num_episodes=ood_episodes,
            memory_cache=memory_cache,
        )
    return datasets


__all__ = [
    "SyntheticSphereDataset",
    "clone_nested",
    "collate_episodes",
    "make_synthetic_datasets",
]
