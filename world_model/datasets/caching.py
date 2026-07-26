"""Optional local `.pt` episode caching with an explicit path manifest."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from world_model.simulator.episode import Episode, validate_episode


def save_episode(path: str | Path, episode: Mapping[str, Any]) -> Path:
    """Atomically save one trusted/local episode record."""

    validate_episode(episode)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(episode), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_episode(path: str | Path) -> Episode:
    """Load and validate a trusted local episode cache file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    episode = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(episode, dict):
        raise TypeError(f"cached episode at {source} is not a dictionary")
    validate_episode(episode)
    return episode


class CachedEpisodeDataset(Dataset[Episode]):
    """Read a fixed explicit list of local episode files."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("cached dataset requires at least one path")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Episode:
        return load_episode(self.paths[index])


def cache_dataset(
    dataset: Dataset[Episode],
    directory: str | Path,
    *,
    prefix: str = "episode",
) -> tuple[Path, ...]:
    """Materialise a deterministic dataset into numbered local cache files."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(len(dataset)):
        path = destination / f"{prefix}_{index:06d}.pt"
        save_episode(path, dataset[index])
        paths.append(path)
    return tuple(paths)
