"""Random-access six-microbatch schedule for specification 1.61."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch

from world_model.training.dynamic_set_protocol import (
    MACROCYCLE_UPDATES,
    PHYSICAL_CELL_COUNT,
    PhysicalManifestRow,
    macrocycle_cell_indices,
)


class DynamicSetMicrobatchSchedule:
    """Resolve any absolute optimiser update without replaying prior draws.

    Every update yields six tuples of four dataset indices.  The manifest must
    contain nonempty pools for all 22 physical cells.  Each pool receives an
    independent deterministic permutation per epoch, so exact resume needs
    only the absolute update index and the frozen schedule seed.  Production
    pools are equal; allowing bounded unequal pools lets the exact frozen
    64-row disposable screen use this same schedule without substituting rows.
    """

    def __init__(
        self,
        rows: Sequence[PhysicalManifestRow],
        *,
        seed: int,
    ) -> None:
        if not rows:
            raise ValueError("dynamic-set training manifest must not be empty")
        pools: dict[int, list[int]] = defaultdict(list)
        for dataset_index, row in enumerate(rows):
            if row.split != "training":
                raise ValueError("dynamic-set optimiser schedule requires the training split")
            if row.cell_index not in range(PHYSICAL_CELL_COUNT):
                raise ValueError("training row has an invalid physical cell")
            pools[row.cell_index].append(dataset_index)
        if set(pools) != set(range(PHYSICAL_CELL_COUNT)):
            raise ValueError("training manifest must support all 22 physical cells")
        self._pools = tuple(tuple(pools[index]) for index in range(PHYSICAL_CELL_COUNT))
        self._pool_sizes = tuple(len(pool) for pool in self._pools)
        if any(size == 0 for size in self._pool_sizes):
            raise ValueError("training physical-cell pools must be nonempty")
        self.seed = int(seed)
        self._permutation_cache: dict[tuple[int, int], tuple[int, ...]] = {}

    @staticmethod
    def _extra_draws_before(update_index: int, cell_index: int) -> int:
        phase = cell_index // 2
        if update_index <= phase:
            return 0
        return (update_index - 1 - phase) // MACROCYCLE_UPDATES + 1

    def _permutation(self, cell_index: int, epoch: int) -> tuple[int, ...]:
        key = (cell_index, epoch)
        cached = self._permutation_cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (self.seed + 1_000_003 * cell_index + 97_003 * epoch) & 0x7FFF_FFFF_FFFF_FFFF
        )
        # A production cell pool contains 250 twelve-row orthogonal blocks.
        # Shuffle blocks and then rows within each block independently.  Every
        # macrocycle consumes exactly twelve examples per cell, so this keeps
        # causal action regime, lifecycle, and contact geometry balanced from
        # macrocycle zero while retaining a full no-replacement random
        # permutation.  Small disposable-screen pools fall back to randperm;
        # their epochs are shorter than one physical macrocycle by design.
        pool_size = self._pool_sizes[cell_index]
        block_size = 12
        if pool_size >= block_size and pool_size % block_size == 0:
            block_order = torch.randperm(pool_size // block_size, generator=generator)
            ordered: list[int] = []
            for block_index in block_order.tolist():
                within = torch.randperm(block_size, generator=generator)
                ordered.extend(block_index * block_size + offset for offset in within.tolist())
            permutation = tuple(ordered)
        else:
            permutation = tuple(torch.randperm(pool_size, generator=generator).tolist())
        self._permutation_cache[key] = permutation
        return permutation

    def _dataset_index(self, cell_index: int, draw_index: int) -> int:
        epoch, offset = divmod(draw_index, self._pool_sizes[cell_index])
        source_offset = self._permutation(cell_index, epoch)[offset]
        return self._pools[cell_index][source_offset]

    def microbatches_for_update(self, update_index: int) -> tuple[tuple[int, ...], ...]:
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise ValueError("update_index must be a nonnegative integer")
        groups = macrocycle_cell_indices(update_index)
        prior_draws = {
            cell_index: update_index + self._extra_draws_before(update_index, cell_index)
            for cell_index in range(PHYSICAL_CELL_COUNT)
        }
        current_occurrence: dict[int, int] = defaultdict(int)
        resolved: list[tuple[int, ...]] = []
        for group in groups:
            batch: list[int] = []
            for cell_index in group:
                draw_index = prior_draws[cell_index] + current_occurrence[cell_index]
                batch.append(self._dataset_index(cell_index, draw_index))
                current_occurrence[cell_index] += 1
            resolved.append(tuple(batch))
        return tuple(resolved)


__all__ = ["DynamicSetMicrobatchSchedule"]
