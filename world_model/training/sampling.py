"""Deterministic, step-addressable training batch order."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class StepIndexedBatchSampler(Sampler[list[int]]):
    """Map every absolute training-data draw to a deterministic dataset batch.

    The mapping depends only on the dataset shape, seed, shuffle policy, and
    absolute draw index. In the common case one draw produces one optimiser
    update. Unsupported causal batches may now consume a draw without claiming
    an update; persisting that draw index still lets a resumed trainer start
    directly without replaying a shuffled epoch or serialising DataLoader
    prefetch internals.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        shuffle: bool,
        start_step: int,
        stop_step: int,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_step < 0 or stop_step < start_step:
            raise ValueError("step range must satisfy 0 <= start_step <= stop_step")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.batches_per_epoch = math.ceil(self.dataset_size / self.batch_size)

    def _draw_legacy_epoch_order(self, generator: torch.Generator) -> list[int]:
        """Draw one epoch exactly as the former shuffled DataLoader did.

        ``DataLoader(generator=...)`` consumes one random int64 for its worker
        base seed before ``RandomSampler`` draws the epoch permutation.  Keeping
        that otherwise-invisible draw here means a checkpoint written by the
        former loader can move to this step-addressable sampler without
        changing the next sample order.
        """

        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        # ``RandomSampler`` requests one full permutation and then evaluates a
        # second, empty remainder slice when ``num_samples == dataset_size``.
        # Although that second result is never yielded, it advances the shared
        # generator and therefore changes every later epoch.
        torch.randperm(self.dataset_size, generator=generator)
        return order

    def __iter__(self) -> Iterator[list[int]]:
        if self.start_step == self.stop_step:
            return
        if not self.shuffle:
            order = list(range(self.dataset_size))
            for step in range(self.start_step, self.stop_step):
                _, batch_index = divmod(step, self.batches_per_epoch)
                start = batch_index * self.batch_size
                stop = min(start + self.batch_size, self.dataset_size)
                yield order[start:stop]
            return

        first_epoch = self.start_step // self.batches_per_epoch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        order: list[int] = []
        # Reconstruct the legacy generator state at the requested epoch once;
        # subsequent epochs advance it incrementally rather than replaying the
        # complete history for every permutation.
        for _ in range(first_epoch + 1):
            order = self._draw_legacy_epoch_order(generator)
        current_epoch = first_epoch
        for step in range(self.start_step, self.stop_step):
            epoch, batch_index = divmod(step, self.batches_per_epoch)
            if epoch != current_epoch:
                order = self._draw_legacy_epoch_order(generator)
                current_epoch = epoch
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            yield order[start:stop]

    def __len__(self) -> int:
        return self.stop_step - self.start_step


__all__ = ["StepIndexedBatchSampler"]
