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


class ScenarioBalancedStepIndexedBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches with equal support from every scenario.

    ``scenario_index_by_dataset_index`` binds the sampler to the dataset's
    explicit seed manifest instead of assuming that dataset indices happen to
    equal simulator seeds. Every batch contains the same number of examples
    from each declared scenario, while each per-scenario pool is independently
    shuffled on every epoch. The absolute draw index therefore remains enough
    to reconstruct an exact continuation.
    """

    def __init__(
        self,
        *,
        scenario_index_by_dataset_index: list[int],
        scenario_count: int,
        batch_size: int,
        seed: int,
        shuffle: bool,
        start_step: int,
        stop_step: int,
    ) -> None:
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        if batch_size <= 0 or batch_size % scenario_count != 0:
            raise ValueError("batch_size must be a positive multiple of scenario_count")
        if start_step < 0 or stop_step < start_step:
            raise ValueError("step range must satisfy 0 <= start_step <= stop_step")
        if not scenario_index_by_dataset_index:
            raise ValueError("scenario assignments must not be empty")
        if any(index < 0 or index >= scenario_count for index in scenario_index_by_dataset_index):
            raise ValueError("scenario assignments must lie in [0, scenario_count)")

        pools = [list[int]() for _ in range(scenario_count)]
        for dataset_index, scenario_index in enumerate(scenario_index_by_dataset_index):
            pools[scenario_index].append(dataset_index)
        pool_sizes = {len(pool) for pool in pools}
        examples_per_scenario = batch_size // scenario_count
        if len(pool_sizes) != 1 or 0 in pool_sizes:
            raise ValueError("scenario-balanced sampling requires equal nonempty scenario pools")
        pool_size = next(iter(pool_sizes))
        if pool_size % examples_per_scenario != 0:
            raise ValueError("each scenario pool must be divisible by its per-batch example count")

        self.pools = pools
        self.scenario_count = int(scenario_count)
        self.batch_size = int(batch_size)
        self.examples_per_scenario = int(examples_per_scenario)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.batches_per_epoch = pool_size // examples_per_scenario

    def _draw_epoch_orders(self, generator: torch.Generator) -> list[list[int]]:
        if not self.shuffle:
            return [list(pool) for pool in self.pools]
        return [
            [pool[index] for index in torch.randperm(len(pool), generator=generator).tolist()]
            for pool in self.pools
        ]

    def __iter__(self) -> Iterator[list[int]]:
        if self.start_step == self.stop_step:
            return
        first_epoch = self.start_step // self.batches_per_epoch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        orders: list[list[int]] = []
        for _ in range(first_epoch + 1):
            orders = self._draw_epoch_orders(generator)
        current_epoch = first_epoch
        for step in range(self.start_step, self.stop_step):
            epoch, batch_index = divmod(step, self.batches_per_epoch)
            if epoch != current_epoch:
                orders = self._draw_epoch_orders(generator)
                current_epoch = epoch
            start = batch_index * self.examples_per_scenario
            stop = start + self.examples_per_scenario
            yield [index for order in orders for index in order[start:stop]]

    def __len__(self) -> int:
        return self.stop_step - self.start_step


__all__ = ["ScenarioBalancedStepIndexedBatchSampler", "StepIndexedBatchSampler"]
