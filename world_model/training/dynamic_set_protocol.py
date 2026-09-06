"""Frozen public protocol primitives for specification 1.61.

This module contains no simulator access and materialises no episode.  It
declares deterministic, disjoint manifests and the balanced optimiser-cell
schedule so source review can bind the experimental population before any
RGB-D scene is generated.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Literal

from world_model.training.qualification_core import canonical_sha256, weighted_score

SPECIFICATION_VERSION = "1.61"
SIMULATOR_VERSION = "sphere_world_v7"
PHYSICAL_CELL_COUNT = 22
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 6
EXAMPLES_PER_UPDATE = MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE
MACROCYCLE_UPDATES = 11

PhysicalSplit = Literal[
    "training",
    "development",
    "selector",
    "confirmation",
    "final_test",
    "compositional_ood",
]
PlanningSplit = Literal[
    "development",
    "selector",
    "confirmation",
    "final_test",
    "compositional_ood",
]
LifecycleSchedule = Literal["none", "birth", "removal", "remove_then_birth"]
ContactOrigin = Literal["none", "natural", "action_induced"]


@dataclass(frozen=True, order=True)
class PhysicalCell:
    object_count: int
    contact: bool
    dynamic_membership: bool

    def validate(self) -> PhysicalCell:
        if self.object_count not in range(1, 7):
            raise ValueError("physical-cell object_count must lie in [1,6]")
        if self.contact and self.object_count < 2:
            raise ValueError("one-object scenes cannot contain pair contact")
        return self


@dataclass(frozen=True)
class PhysicalManifestRow:
    split: PhysicalSplit
    ordinal: int
    seed: int
    cell_index: int
    object_count: int
    contact: bool
    dynamic_membership: bool
    lifecycle_schedule: LifecycleSchedule
    known_action: bool
    contact_origin: ContactOrigin
    action_target_rank: int | None
    action_time_stratum: int | None
    camera_stratum: int
    contact_geometry: Literal["none", "head_on", "glancing"]
    distribution: Literal["in_distribution", "compositional_ood"]


@dataclass(frozen=True)
class PlanningManifestRow:
    split: PlanningSplit
    ordinal: int
    seed: int
    object_count: int
    previously_dynamic: bool
    candidate_induced_contact: bool
    target_rank: int
    action_time_stratum: int
    camera_stratum: int
    goal_direction: int
    candidate_count: Literal[8, 32]
    minimum_normalized_winner_margin: float
    distribution: Literal["in_distribution", "compositional_ood"]


@dataclass(frozen=True)
class PlanningOracleCertificate:
    winner_index: int
    normalized_winner_margin: float
    winner_succeeds: bool

    @classmethod
    def from_costs(
        cls,
        costs: Sequence[float],
        *,
        winner_succeeds: bool,
        minimum_margin: float = 0.05,
    ) -> PlanningOracleCertificate:
        if len(costs) < 2:
            raise ValueError("planning certification requires at least two candidates")
        if not math.isfinite(minimum_margin) or minimum_margin <= 0.0:
            raise ValueError("minimum_margin must be finite and positive")
        if any(not math.isfinite(float(cost)) for cost in costs):
            raise ValueError("oracle candidate costs must be finite")
        order = sorted(range(len(costs)), key=lambda index: (float(costs[index]), index))
        best, second = order[:2]
        scale = max(abs(float(costs[second])), abs(float(costs[best])), 1.0e-12)
        margin = (float(costs[second]) - float(costs[best])) / scale
        if margin < minimum_margin:
            raise ValueError("planning task does not have the required unique winner margin")
        return cls(
            winner_index=best,
            normalized_winner_margin=margin,
            winner_succeeds=bool(winner_succeeds),
        )


PHYSICAL_SPLIT_SIZES: Mapping[PhysicalSplit, int] = {
    "training": 66_000,
    "development": 4_400,
    "selector": 2_200,
    "confirmation": 2_200,
    "final_test": 4_400,
    "compositional_ood": 4_400,
}
PHYSICAL_SPLIT_SEED_BASES: Mapping[PhysicalSplit, int] = {
    "training": 70_000_000,
    "development": 71_000_000,
    "selector": 72_000_000,
    "confirmation": 73_000_000,
    "final_test": 74_000_000,
    "compositional_ood": 75_000_000,
}
PLANNING_SPLIT_SIZES: Mapping[PlanningSplit, int] = {
    "development": 1_200,
    "selector": 600,
    "confirmation": 600,
    "final_test": 1_200,
    "compositional_ood": 1_200,
}
PLANNING_SPLIT_SEED_BASES: Mapping[PlanningSplit, int] = {
    "development": 76_000_000,
    "selector": 77_000_000,
    "confirmation": 78_000_000,
    "final_test": 79_000_000,
    "compositional_ood": 80_000_000,
}

# A strength-two GF(8) orthogonal array supplies the first 64 rows of each
# per-cardinality planning cover.  The following 36-row completions were solved
# once against the frozen factor schema.  Together they give every valid pair
# of factor values while keeping every marginal within one sample at 100 rows.
# Keeping the completion in source (rather than solving it at runtime) makes
# manifest generation dependency-free and source-hash reproducible.
_PLANNING_COVER_COMPLETIONS: Mapping[int, tuple[tuple[int, ...], ...]] = {
    1: (
        (0, 0, 0, 0, 1, 3, 0),
        (0, 0, 0, 0, 2, 4, 0),
        (0, 0, 0, 0, 4, 0, 0),
        (0, 0, 0, 0, 6, 3, 1),
        (0, 0, 0, 0, 7, 4, 1),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 2, 2, 3, 1),
        (0, 0, 0, 2, 2, 3, 1),
        (0, 0, 0, 2, 2, 3, 1),
        (0, 0, 0, 2, 3, 4, 1),
        (0, 0, 0, 2, 3, 4, 1),
        (0, 0, 0, 2, 4, 2, 0),
        (0, 0, 0, 2, 4, 2, 0),
        (0, 0, 0, 2, 6, 4, 0),
        (0, 0, 0, 3, 3, 0, 0),
        (0, 0, 0, 3, 3, 0, 0),
        (1, 0, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 6, 1, 0),
        (1, 0, 0, 0, 6, 1, 0),
        (1, 0, 0, 1, 4, 5, 1),
        (1, 0, 0, 1, 5, 5, 0),
        (1, 0, 0, 1, 5, 5, 0),
        (1, 0, 0, 1, 7, 5, 0),
        (1, 0, 0, 1, 7, 5, 0),
        (1, 0, 0, 1, 7, 5, 0),
        (1, 0, 0, 2, 0, 2, 1),
        (1, 0, 0, 3, 0, 4, 0),
        (1, 0, 0, 3, 0, 4, 0),
        (1, 0, 0, 3, 1, 4, 1),
        (1, 0, 0, 3, 2, 5, 1),
        (1, 0, 0, 3, 3, 5, 0),
        (1, 0, 0, 3, 5, 2, 1),
        (1, 0, 0, 3, 5, 2, 1),
    ),
    2: (
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 2, 0, 1),
        (0, 0, 0, 1, 5, 4, 0),
        (0, 0, 0, 2, 0, 1, 1),
        (0, 0, 0, 2, 0, 1, 1),
        (0, 0, 0, 2, 0, 1, 1),
        (0, 0, 1, 1, 6, 2, 0),
        (0, 0, 1, 1, 6, 2, 0),
        (0, 0, 1, 2, 1, 2, 0),
        (0, 1, 0, 1, 5, 5, 0),
        (0, 1, 0, 3, 4, 4, 0),
        (0, 1, 0, 3, 4, 4, 0),
        (0, 1, 0, 3, 4, 4, 0),
        (0, 1, 1, 0, 5, 4, 1),
        (0, 1, 1, 0, 5, 4, 1),
        (0, 1, 1, 3, 1, 5, 1),
        (1, 0, 1, 0, 3, 5, 0),
        (1, 0, 1, 0, 3, 5, 0),
        (1, 0, 1, 0, 3, 5, 0),
        (1, 0, 1, 2, 6, 5, 1),
        (1, 0, 1, 2, 6, 5, 1),
        (1, 0, 1, 3, 0, 2, 1),
        (1, 0, 1, 3, 0, 2, 1),
        (1, 1, 0, 0, 2, 2, 0),
        (1, 1, 0, 0, 2, 2, 0),
        (1, 1, 0, 0, 2, 2, 0),
        (1, 1, 0, 0, 2, 2, 0),
        (1, 1, 0, 2, 3, 4, 1),
        (1, 1, 0, 2, 3, 4, 1),
        (1, 1, 1, 1, 4, 5, 1),
        (1, 1, 1, 2, 7, 3, 1),
        (1, 1, 1, 3, 7, 1, 0),
        (1, 1, 1, 3, 7, 1, 0),
        (1, 1, 1, 3, 7, 1, 0),
    ),
    3: (
        (0, 0, 0, 0, 3, 5, 0),
        (0, 0, 0, 2, 7, 1, 0),
        (0, 0, 0, 3, 2, 0, 1),
        (0, 0, 0, 3, 7, 3, 1),
        (0, 0, 1, 2, 0, 5, 0),
        (0, 0, 2, 0, 2, 2, 0),
        (0, 0, 2, 0, 4, 2, 0),
        (0, 0, 2, 2, 0, 4, 0),
        (0, 0, 2, 3, 0, 0, 1),
        (0, 1, 0, 3, 0, 1, 1),
        (0, 1, 1, 1, 1, 4, 0),
        (0, 1, 1, 2, 2, 1, 1),
        (0, 1, 2, 1, 4, 4, 0),
        (0, 1, 2, 1, 4, 4, 0),
        (0, 1, 2, 1, 4, 4, 0),
        (0, 1, 2, 1, 4, 4, 0),
        (0, 1, 2, 2, 7, 4, 0),
        (0, 1, 2, 3, 5, 3, 0),
        (1, 0, 0, 0, 1, 5, 1),
        (1, 0, 0, 0, 1, 5, 1),
        (1, 0, 0, 3, 6, 5, 1),
        (1, 0, 1, 1, 6, 3, 0),
        (1, 0, 1, 3, 6, 4, 1),
        (1, 0, 2, 0, 5, 5, 1),
        (1, 0, 2, 1, 1, 0, 1),
        (1, 0, 2, 2, 3, 1, 1),
        (1, 0, 2, 2, 3, 1, 1),
        (1, 1, 0, 0, 0, 2, 0),
        (1, 1, 0, 2, 1, 2, 1),
        (1, 1, 1, 1, 2, 5, 1),
        (1, 1, 1, 1, 2, 5, 1),
        (1, 1, 1, 2, 5, 1, 1),
        (1, 1, 1, 3, 3, 3, 0),
        (1, 1, 2, 0, 6, 2, 0),
        (1, 1, 2, 0, 7, 0, 1),
        (1, 1, 2, 3, 5, 2, 0),
    ),
    4: (
        (0, 0, 0, 2, 1, 4, 1),
        (0, 0, 0, 3, 7, 1, 1),
        (0, 0, 0, 3, 7, 1, 1),
        (0, 0, 0, 3, 7, 1, 1),
        (0, 0, 0, 3, 7, 1, 1),
        (0, 0, 1, 2, 2, 1, 0),
        (0, 0, 1, 2, 2, 1, 0),
        (0, 0, 2, 2, 0, 2, 0),
        (0, 0, 3, 1, 6, 2, 0),
        (0, 0, 3, 1, 6, 2, 0),
        (0, 1, 0, 0, 2, 2, 1),
        (0, 1, 2, 0, 2, 2, 1),
        (0, 1, 2, 2, 5, 2, 0),
        (0, 1, 2, 2, 5, 2, 0),
        (0, 1, 3, 2, 6, 3, 1),
        (0, 1, 3, 3, 0, 5, 0),
        (0, 1, 3, 3, 0, 5, 0),
        (0, 1, 3, 3, 0, 5, 0),
        (1, 0, 0, 0, 1, 4, 0),
        (1, 0, 0, 0, 1, 4, 0),
        (1, 0, 1, 1, 3, 5, 1),
        (1, 0, 1, 1, 3, 5, 1),
        (1, 0, 1, 1, 3, 5, 1),
        (1, 0, 1, 1, 3, 5, 1),
        (1, 0, 2, 1, 2, 5, 1),
        (1, 0, 3, 3, 0, 1, 0),
        (1, 1, 0, 3, 3, 0, 0),
        (1, 1, 1, 0, 6, 4, 1),
        (1, 1, 1, 1, 5, 2, 1),
        (1, 1, 1, 1, 5, 2, 1),
        (1, 1, 2, 0, 4, 4, 0),
        (1, 1, 2, 0, 4, 4, 0),
        (1, 1, 2, 0, 4, 4, 0),
        (1, 1, 2, 0, 4, 4, 0),
        (1, 1, 3, 2, 1, 1, 1),
        (1, 1, 3, 2, 1, 1, 1),
    ),
    5: (
        (0, 0, 0, 0, 0, 2, 0),
        (0, 0, 1, 2, 7, 4, 0),
        (0, 0, 2, 3, 4, 2, 1),
        (0, 0, 3, 3, 5, 5, 0),
        (0, 0, 3, 3, 5, 5, 0),
        (0, 0, 3, 3, 5, 5, 0),
        (0, 0, 4, 0, 2, 4, 0),
        (0, 0, 4, 0, 2, 4, 0),
        (0, 0, 4, 0, 2, 4, 0),
        (0, 0, 4, 1, 1, 2, 0),
        (0, 0, 4, 1, 6, 0, 1),
        (0, 0, 4, 2, 0, 4, 0),
        (0, 1, 1, 2, 7, 1, 0),
        (0, 1, 1, 3, 3, 1, 1),
        (0, 1, 2, 1, 6, 1, 1),
        (0, 1, 3, 3, 3, 4, 0),
        (0, 1, 4, 2, 1, 4, 0),
        (0, 1, 4, 3, 7, 2, 1),
        (1, 0, 0, 3, 1, 1, 0),
        (1, 0, 2, 2, 5, 2, 0),
        (1, 0, 3, 0, 6, 5, 1),
        (1, 0, 3, 0, 6, 5, 1),
        (1, 0, 3, 1, 1, 5, 1),
        (1, 0, 4, 0, 4, 1, 0),
        (1, 1, 0, 1, 2, 3, 0),
        (1, 1, 0, 2, 0, 1, 1),
        (1, 1, 1, 3, 1, 4, 1),
        (1, 1, 2, 2, 2, 2, 0),
        (1, 1, 3, 0, 7, 3, 1),
        (1, 1, 3, 1, 0, 5, 1),
        (1, 1, 3, 1, 0, 5, 1),
        (1, 1, 3, 2, 4, 0, 1),
        (1, 1, 3, 2, 4, 0, 1),
        (1, 1, 4, 0, 3, 0, 1),
        (1, 1, 4, 1, 3, 3, 1),
        (1, 1, 4, 1, 3, 3, 1),
    ),
    6: (
        (0, 0, 0, 0, 0, 2, 0),
        (0, 0, 4, 0, 2, 5, 1),
        (0, 0, 4, 0, 2, 5, 1),
        (0, 0, 4, 1, 7, 3, 1),
        (0, 0, 5, 1, 2, 5, 1),
        (0, 1, 2, 0, 1, 2, 0),
        (0, 1, 2, 0, 1, 2, 0),
        (0, 1, 2, 0, 1, 2, 0),
        (0, 1, 2, 0, 1, 2, 0),
        (0, 1, 2, 3, 6, 4, 0),
        (0, 1, 2, 3, 6, 4, 0),
        (0, 1, 2, 3, 6, 4, 0),
        (0, 1, 3, 0, 7, 0, 1),
        (0, 1, 3, 1, 2, 2, 1),
        (0, 1, 4, 1, 4, 5, 1),
        (0, 1, 4, 3, 0, 1, 1),
        (0, 1, 5, 2, 0, 1, 1),
        (0, 1, 5, 2, 0, 1, 1),
        (1, 0, 1, 1, 6, 4, 1),
        (1, 0, 2, 2, 3, 1, 1),
        (1, 0, 3, 2, 0, 1, 1),
        (1, 0, 3, 2, 4, 1, 1),
        (1, 0, 3, 2, 4, 1, 1),
        (1, 0, 3, 2, 4, 1, 1),
        (1, 0, 4, 2, 7, 2, 0),
        (1, 0, 4, 2, 7, 2, 0),
        (1, 0, 5, 0, 5, 2, 1),
        (1, 0, 5, 3, 3, 4, 0),
        (1, 0, 5, 3, 3, 4, 0),
        (1, 0, 5, 3, 3, 4, 0),
        (1, 0, 5, 3, 3, 4, 0),
        (1, 1, 2, 1, 2, 1, 0),
        (1, 1, 3, 1, 5, 5, 0),
        (1, 1, 3, 1, 5, 5, 0),
        (1, 1, 3, 1, 5, 5, 0),
        (1, 1, 4, 3, 1, 5, 1),
    ),
}


def physical_cells() -> tuple[PhysicalCell, ...]:
    cells: list[PhysicalCell] = []
    for object_count in range(1, 7):
        contact_values = (False,) if object_count == 1 else (False, True)
        for contact in contact_values:
            for dynamic_membership in (False, True):
                cells.append(
                    PhysicalCell(
                        object_count=object_count,
                        contact=contact,
                        dynamic_membership=dynamic_membership,
                    ).validate()
                )
    if len(cells) != PHYSICAL_CELL_COUNT or len(set(cells)) != PHYSICAL_CELL_COUNT:
        raise RuntimeError("specification 1.61 must contain exactly 22 unique physical cells")
    return tuple(cells)


PHYSICAL_CELLS = physical_cells()


def _lifecycle_schedule(cell: PhysicalCell, cell_cycle: int) -> LifecycleSchedule:
    if not cell.dynamic_membership:
        return "none"
    # For contact cells, the coprime two-regime and three-lifecycle cycles,
    # combined with the four-row geometry cycle below, form the complete
    # 2 x 3 x 2 factorial in each aligned twelve-row training block.  The same
    # order lets the exact three-cycle screen see every lifecycle immediately.
    return ("birth", "removal", "remove_then_birth")[cell_cycle % 3]


def _contact_action_regime(cell_cycle: int) -> tuple[bool, ContactOrigin]:
    """Balance the two frozen causal regimes in every contact cell."""

    return (
        (False, "natural"),
        (True, "action_induced"),
    )[cell_cycle % 2]


def _physical_camera_stratum(
    *,
    lifecycle_schedule: LifecycleSchedule,
    known_action: bool,
    cell_cycle: int,
    compositional_ood: bool,
) -> int:
    """Compose familiar camera/action/lifecycle primitives on a held-out parity.

    In-distribution manifests reserve one camera-parity value for each
    ``(known_action, lifecycle_schedule)`` context.  OOD uses the opposite
    parity while preserving the four camera axes and every individual
    primitive.  This makes the OOD split compositionally different rather than
    merely assigning the same rows another seed namespace.
    """

    lifecycle_parity = {
        "none": 0,
        "birth": 0,
        "removal": 1,
        "remove_then_birth": 0,
    }[lifecycle_schedule]
    parity = int(known_action) ^ lifecycle_parity ^ int(compositional_ood)
    camera_axis = (cell_cycle // 2) % 4
    return 2 * camera_axis + parity


def physical_manifest(split: PhysicalSplit) -> tuple[PhysicalManifestRow, ...]:
    try:
        size = PHYSICAL_SPLIT_SIZES[split]
        seed_base = PHYSICAL_SPLIT_SEED_BASES[split]
    except KeyError as error:
        raise ValueError(f"unknown physical split {split!r}") from error
    if size % PHYSICAL_CELL_COUNT:
        raise RuntimeError("every physical manifest must balance all 22 cells exactly")
    rows: list[PhysicalManifestRow] = []
    for ordinal in range(size):
        cell_index = ordinal % PHYSICAL_CELL_COUNT
        cell_cycle = ordinal // PHYSICAL_CELL_COUNT
        cell = PHYSICAL_CELLS[cell_index]
        if cell.contact:
            known_action, contact_origin = _contact_action_regime(cell_cycle)
        else:
            known_action = bool(cell_cycle % 2)
            contact_origin = "none"
        lifecycle_schedule = _lifecycle_schedule(cell, cell_cycle)
        compositional_ood = split == "compositional_ood"
        if cell.contact:
            contact_geometry: Literal["none", "head_on", "glancing"] = (
                "head_on" if (cell_cycle // 2) % 2 == 0 else "glancing"
            )
        else:
            contact_geometry = "none"
        if known_action:
            action_factor_cycle = cell_cycle // 2
            action_target_rank = action_factor_cycle % cell.object_count
            action_time_stratum = (action_factor_cycle // max(cell.object_count, 1)) % 4
        else:
            action_target_rank = None
            action_time_stratum = None
        rows.append(
            PhysicalManifestRow(
                split=split,
                ordinal=ordinal,
                seed=seed_base + ordinal,
                cell_index=cell_index,
                object_count=cell.object_count,
                contact=cell.contact,
                dynamic_membership=cell.dynamic_membership,
                lifecycle_schedule=lifecycle_schedule,
                known_action=known_action,
                contact_origin=contact_origin,
                action_target_rank=action_target_rank,
                action_time_stratum=action_time_stratum,
                camera_stratum=_physical_camera_stratum(
                    lifecycle_schedule=lifecycle_schedule,
                    known_action=known_action,
                    cell_cycle=cell_cycle,
                    compositional_ood=compositional_ood,
                ),
                contact_geometry=contact_geometry,
                distribution=(
                    "compositional_ood" if split == "compositional_ood" else "in_distribution"
                ),
            )
        )
    return tuple(rows)


def _gf8_multiply(left: int, right: int) -> int:
    """Multiply two three-bit field elements under x^3 + x + 1."""

    result = 0
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        left <<= 1
        if left & 0b1000:
            left ^= 0b1011
    return result


@lru_cache(maxsize=6)
def _planning_base_cover(object_count: int) -> tuple[tuple[int, ...], ...]:
    """Return the frozen 100-row pairwise factor cover for one cardinality."""

    if object_count not in range(1, 7):
        raise ValueError("planning-cover object_count must lie in [1,6]")
    contact_levels = 1 if object_count == 1 else 2
    rows: list[tuple[int, ...]] = []
    for left in range(8):
        for right in range(8):
            symbols = (
                left,
                right,
                left ^ _gf8_multiply(1, right),
                left ^ _gf8_multiply(2, right),
                left ^ _gf8_multiply(3, right),
                left ^ _gf8_multiply(4, right),
                left ^ _gf8_multiply(5, right),
            )
            previously_dynamic = symbols[0] % 2
            candidate_induced_contact = symbols[1] % contact_levels
            target_rank = symbols[2] % object_count
            action_time_stratum = symbols[3] % 4
            camera_stratum = symbols[4]
            goal_axis = symbols[5] % 3
            candidate_count_index = symbols[6] % 2
            goal_sign = (
                previously_dynamic
                ^ candidate_induced_contact
                ^ (target_rank % 2)
                ^ (action_time_stratum % 2)
                ^ (camera_stratum % 2)
                ^ candidate_count_index
            )
            rows.append(
                (
                    previously_dynamic,
                    candidate_induced_contact,
                    target_rank,
                    action_time_stratum,
                    camera_stratum,
                    2 * goal_axis + goal_sign,
                    candidate_count_index,
                )
            )
    rows.extend(_PLANNING_COVER_COMPLETIONS[object_count])
    if len(rows) != 100:
        raise RuntimeError("planning base cover must contain exactly 100 rows")
    return tuple(rows)


def _planning_second_cover(
    object_count: int,
    base: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Relabel a base cover to complement all 200-row factor marginals."""

    if object_count == 3:
        # Cardinality three has different 13/12 camera residues in its solved
        # completion.  These permutations complement those residues and its
        # 34/33/33 target counts while preserving a 50/50 parity split.
        target_permutation = (2, 0, 1)
        action_permutation = (0, 1, 3, 2)
        camera_permutation = (3, 5, 6, 0, 7, 1, 2, 4)
    else:
        target_permutation = tuple(
            (target + (2 if object_count == 6 else 0)) % object_count
            for target in range(object_count)
        )
        action_permutation = (0, 1, 2, 3)
        camera_permutation = tuple(camera ^ 4 for camera in range(8))

    transformed: list[tuple[int, int, int, int, int, int, int]] = []
    for (
        previously_dynamic,
        candidate_induced_contact,
        target_rank,
        action_time_stratum,
        camera_stratum,
        _goal_direction,
        candidate_count_index,
    ) in base:
        target_rank = target_permutation[target_rank]
        action_time_stratum = action_permutation[action_time_stratum]
        camera_stratum = camera_permutation[camera_stratum]
        goal_sign = (
            previously_dynamic
            ^ candidate_induced_contact
            ^ (target_rank % 2)
            ^ (action_time_stratum % 2)
            ^ (camera_stratum % 2)
            ^ candidate_count_index
        )
        transformed.append(
            (
                previously_dynamic,
                candidate_induced_contact,
                target_rank,
                action_time_stratum,
                camera_stratum,
                goal_sign,
                candidate_count_index,
            )
        )

    # The first cover has goal counts 17/17/17/17/16/16.  Complement it to
    # balanced 200-row totals of 33/33/33/33/34/34, distributing axes round
    # robin within each parity instead of introducing an ordinal block alias.
    base_goal_counts = Counter(row[5] for row in base)
    goal_remaining = {
        goal_direction: 33 + int(goal_direction >= 4) - base_goal_counts[goal_direction]
        for goal_direction in range(6)
    }
    goal_cursor = {0: 0, 1: 0}
    result: list[tuple[int, ...]] = []
    for row in transformed:
        goal_sign = row[5]
        for _ in range(3):
            goal_axis = goal_cursor[goal_sign]
            goal_cursor[goal_sign] = (goal_axis + 1) % 3
            goal_direction = 2 * goal_axis + goal_sign
            if goal_remaining[goal_direction] > 0:
                goal_remaining[goal_direction] -= 1
                result.append((*row[:5], goal_direction, row[6]))
                break
        else:
            raise RuntimeError("second planning cover has an unbalanced goal parity")
    if any(goal_remaining.values()):
        raise RuntimeError("second planning cover did not consume every goal quota")
    return tuple(result)


def _validate_planning_factor_cover(
    object_count: int,
    rows: tuple[tuple[int, ...], ...],
) -> None:
    levels = (2, 1 if object_count == 1 else 2, object_count, 4, 8, 6, 2)
    for factor_index, level_count in enumerate(levels):
        counts = Counter(row[factor_index] for row in rows)
        if (
            set(counts) != set(range(level_count))
            or max(counts.values()) - min(counts.values()) > 1
        ):
            raise RuntimeError("planning factor cover has an unbalanced marginal")
    for left, right in combinations(range(len(levels)), 2):
        observed = {(row[left], row[right]) for row in rows}
        expected = set(product(range(levels[left]), range(levels[right])))
        if observed != expected:
            raise RuntimeError("planning factor cover is missing a valid factor pair")
    if any(
        row[5] % 2 != row[0] ^ row[1] ^ (row[2] % 2) ^ (row[3] % 2) ^ (row[4] % 2) ^ row[6]
        for row in rows
    ):
        raise RuntimeError("planning factor cover violates its held-out parity")


@lru_cache(maxsize=12)
def _planning_factor_cover(
    object_count: int,
    rows_per_cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    base = _planning_base_cover(object_count)
    if rows_per_cardinality == 100:
        rows = base
    elif rows_per_cardinality == 200:
        rows = (*base, *_planning_second_cover(object_count, base))
    else:
        raise ValueError("planning split must provide 100 or 200 rows per cardinality")
    _validate_planning_factor_cover(object_count, rows)
    return tuple(rows)


def planning_manifest(split: PlanningSplit) -> tuple[PlanningManifestRow, ...]:
    try:
        size = PLANNING_SPLIT_SIZES[split]
        seed_base = PLANNING_SPLIT_SEED_BASES[split]
    except KeyError as error:
        raise ValueError(f"unknown planning split {split!r}") from error
    if size % 6:
        raise RuntimeError("every planning manifest must balance all six cardinalities exactly")
    rows_per_cardinality = size // 6
    rows: list[PlanningManifestRow] = []
    for ordinal in range(size):
        object_count = ordinal % 6 + 1
        factor_row = _planning_factor_cover(object_count, rows_per_cardinality)[ordinal // 6]
        (
            previously_dynamic,
            candidate_induced_contact,
            target_rank,
            action_time_stratum,
            camera_stratum,
            goal_direction,
            candidate_count_index,
        ) = factor_row
        if split == "compositional_ood":
            goal_direction ^= 1
        rows.append(
            PlanningManifestRow(
                split=split,
                ordinal=ordinal,
                seed=seed_base + ordinal,
                object_count=object_count,
                previously_dynamic=bool(previously_dynamic),
                candidate_induced_contact=bool(candidate_induced_contact),
                target_rank=target_rank,
                action_time_stratum=action_time_stratum,
                camera_stratum=camera_stratum,
                goal_direction=goal_direction,
                candidate_count=8 if candidate_count_index == 0 else 32,
                minimum_normalized_winner_margin=0.05,
                distribution=(
                    "compositional_ood" if split == "compositional_ood" else "in_distribution"
                ),
            )
        )
    return tuple(rows)


def macrocycle_cell_indices(update_index: int) -> tuple[tuple[int, ...], ...]:
    """Return six B4 cell groups covering all cells on one optimiser update.

    Two cells are duplicated per update.  Across eleven consecutive updates,
    every one of the 22 cells is duplicated exactly once.
    """

    if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
        raise ValueError("update_index must be a nonnegative integer")
    phase = update_index % MACROCYCLE_UPDATES
    rotation = (update_index * 7) % PHYSICAL_CELL_COUNT
    ordered = tuple(
        (rotation + offset) % PHYSICAL_CELL_COUNT for offset in range(PHYSICAL_CELL_COUNT)
    )
    duplicates = (2 * phase, 2 * phase + 1)
    flat = (*ordered, *duplicates)
    return tuple(
        tuple(flat[start : start + MICROBATCH_SIZE])
        for start in range(0, EXAMPLES_PER_UPDATE, MICROBATCH_SIZE)
    )


SELECTION_SCORE_WEIGHTS: Mapping[str, float] = {
    "current_position": 0.15,
    "current_velocity": 0.10,
    "horizon_position": 0.20,
    "horizon_velocity": 0.10,
    "proposal_error": 0.10,
    "identity_lifecycle_error": 0.10,
    "contact_error": 0.10,
    "planning_error": 0.15,
}


def selection_score(components: Mapping[str, float]) -> float:
    return weighted_score(components, SELECTION_SCORE_WEIGHTS)


def _manifest_payload(rows: Sequence[object]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]


FROZEN_PHYSICAL_MANIFEST_SHA256: Mapping[PhysicalSplit, str] = {
    "training": "c37b0e7643e740ea6cdf65400c2bc6787a8e5038cea073c22ecff7025a3b1f06",
    "development": "d94448e0493ed73b15f960a24c8813c2419926fb445e2fab8a594119cf33d35c",
    "selector": "c17d030a8db17f8c299b8fc701f76b52a9e3357ec659d03acb9d62760055059e",
    "confirmation": "6916aad9cb40a6c3b428f87fc0bc7cc95845b14e263b5f11dcb6b07f822154a2",
    "final_test": "25df07b2d8788eea5f42e0c4cd5c2c73f4448a4a9633832bd96c43191131408a",
    "compositional_ood": ("01c55d6b2827c581b6c9832df385799a0af47ff49fb118b93f22e91058689ce1"),
}
FROZEN_PLANNING_MANIFEST_SHA256: Mapping[PlanningSplit, str] = {
    "development": "ed56bd7274841240268640609a2faed398c317da04faee930c2183f0057a3b15",
    "selector": "08f9a2280e5ee18c14245f96b48078716332b4a8fdaae4531fe846fe03892536",
    "confirmation": "08edc8073bb8c7f47e46f21669e2a35590a88e235615ad0a25d6877f25590ceb",
    "final_test": "370f72774b150ded638514e49b9e536c41cc48616e89a7e03913bd04cd6a735d",
    "compositional_ood": ("56d85ee36e9f7a1010569ae5f269aa6efb24eda9b045c95217f5420be39d2248"),
}


def validate_frozen_manifests() -> None:
    if set(FROZEN_PHYSICAL_MANIFEST_SHA256) != set(PHYSICAL_SPLIT_SIZES):
        raise RuntimeError("physical manifest digests have not been source-frozen")
    if set(FROZEN_PLANNING_MANIFEST_SHA256) != set(PLANNING_SPLIT_SIZES):
        raise RuntimeError("planning manifest digests have not been source-frozen")
    all_seeds: set[int] = set()
    for split in PHYSICAL_SPLIT_SIZES:
        rows = physical_manifest(split)
        digest = canonical_sha256(_manifest_payload(rows))
        if digest != FROZEN_PHYSICAL_MANIFEST_SHA256[split]:
            raise RuntimeError(f"physical manifest {split} differs from its source freeze")
        seeds = {row.seed for row in rows}
        if len(seeds) != len(rows) or all_seeds.intersection(seeds):
            raise RuntimeError("physical manifest seeds must be unique and disjoint")
        all_seeds.update(seeds)
    for split in PLANNING_SPLIT_SIZES:
        rows = planning_manifest(split)
        digest = canonical_sha256(_manifest_payload(rows))
        if digest != FROZEN_PLANNING_MANIFEST_SHA256[split]:
            raise RuntimeError(f"planning manifest {split} differs from its source freeze")
        seeds = {row.seed for row in rows}
        if len(seeds) != len(rows) or all_seeds.intersection(seeds):
            raise RuntimeError("planning manifest seeds must be unique and disjoint")
        all_seeds.update(seeds)


__all__ = [
    "EXAMPLES_PER_UPDATE",
    "FROZEN_PHYSICAL_MANIFEST_SHA256",
    "FROZEN_PLANNING_MANIFEST_SHA256",
    "MACROCYCLE_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "PHYSICAL_CELLS",
    "PHYSICAL_CELL_COUNT",
    "PHYSICAL_SPLIT_SEED_BASES",
    "PHYSICAL_SPLIT_SIZES",
    "PLANNING_SPLIT_SEED_BASES",
    "PLANNING_SPLIT_SIZES",
    "SELECTION_SCORE_WEIGHTS",
    "SIMULATOR_VERSION",
    "SPECIFICATION_VERSION",
    "PhysicalCell",
    "PhysicalManifestRow",
    "PlanningManifestRow",
    "PlanningOracleCertificate",
    "canonical_sha256",
    "macrocycle_cell_indices",
    "physical_cells",
    "physical_manifest",
    "planning_manifest",
    "selection_score",
    "validate_frozen_manifests",
]
