"""Synthetic episode datasets, explicit splits, collation, and local caching."""

from world_model.datasets.caching import (
    CachedEpisodeDataset,
    cache_dataset,
    load_episode,
    save_episode,
)
from world_model.datasets.collate import collate_episodes
from world_model.datasets.splits import (
    SPLIT_SEED_RANGES,
    SeedManifest,
    assert_disjoint_manifests,
    canonical_split_name,
    make_seed_manifest,
)
from world_model.datasets.synthetic import (
    SyntheticSphereDataset,
    make_synthetic_datasets,
)

__all__ = [
    "CachedEpisodeDataset",
    "SPLIT_SEED_RANGES",
    "SeedManifest",
    "SyntheticSphereDataset",
    "assert_disjoint_manifests",
    "cache_dataset",
    "canonical_split_name",
    "collate_episodes",
    "load_episode",
    "make_seed_manifest",
    "make_synthetic_datasets",
    "save_episode",
]
