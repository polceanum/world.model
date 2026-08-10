"""Safe model-state composition for modular accuracy experiments.

Orpheus deliberately separates perception, filtering, identification, and
dynamics.  This helper lets an offline qualification keep a verified base
checkpoint while importing selected top-level modules from a donor.  It never
copies optimizer or RNG state and it validates the complete tensor schema
before producing a candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def _normalise_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(dict.fromkeys(prefix.strip().rstrip(".") for prefix in prefixes))
    if not normalised or any(not prefix for prefix in normalised):
        raise ValueError("at least one non-empty module prefix is required")
    if any(".." in prefix or prefix.startswith(".") for prefix in normalised):
        raise ValueError("module prefixes must be dotted state-dict names")
    return normalised


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def compose_model_state(
    base: Mapping[str, Tensor],
    donor: Mapping[str, Tensor],
    *,
    module_prefixes: Sequence[str],
    donor_weight: float = 1.0,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Return a schema-checked base state with selected donor modules blended in.

    Floating tensors are linearly interpolated. Non-floating buffers use the
    donor only for an exact donor weight of one; otherwise they remain at the
    base value. The returned tensors are detached clones, so neither input
    checkpoint can be mutated through the result.
    """

    prefixes = _normalise_prefixes(module_prefixes)
    if not 0.0 <= donor_weight <= 1.0:
        raise ValueError("donor_weight must lie in [0, 1]")
    if set(base) != set(donor):
        missing_from_donor = sorted(set(base) - set(donor))
        missing_from_base = sorted(set(donor) - set(base))
        raise ValueError(
            "checkpoint model-state schemas differ: "
            f"missing_from_donor={missing_from_donor}, "
            f"missing_from_base={missing_from_base}"
        )

    selected = tuple(sorted(name for name in base if _matches_prefix(name, prefixes)))
    unmatched = [
        prefix
        for prefix in prefixes
        if not any(name == prefix or name.startswith(f"{prefix}.") for name in base)
    ]
    if unmatched:
        raise ValueError(f"module prefixes matched no tensors: {unmatched}")

    composed: dict[str, Tensor] = {}
    selected_set = set(selected)
    for name, base_tensor in base.items():
        donor_tensor = donor[name]
        if base_tensor.shape != donor_tensor.shape:
            raise ValueError(f"tensor shape differs for {name!r}")
        if base_tensor.dtype != donor_tensor.dtype:
            raise ValueError(f"tensor dtype differs for {name!r}")
        if name not in selected_set or donor_weight == 0.0:
            value = base_tensor
        elif base_tensor.is_floating_point() or base_tensor.is_complex():
            value = torch.lerp(base_tensor, donor_tensor, donor_weight)
        elif donor_weight == 1.0:
            value = donor_tensor
        else:
            value = base_tensor
        composed[name] = value.detach().clone()
    return composed, selected


def compose_model_state_rows(
    base: Mapping[str, Tensor],
    donor: Mapping[str, Tensor],
    *,
    tensor_rows: Mapping[str, Sequence[int]],
    donor_weight: float = 1.0,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Blend selected leading-dimension rows from ``donor`` into ``base``.

    Row-level composition is intentionally explicit and is primarily useful
    for axis-local output-head ablations.  It cannot address scalar tensors,
    silently wrap negative indices, or mutate either source checkpoint.
    Complete state-dict schema validation is retained even though only a small
    subset of rows is selected.
    """

    if not 0.0 <= donor_weight <= 1.0:
        raise ValueError("donor_weight must lie in [0, 1]")
    if not tensor_rows:
        raise ValueError("at least one tensor row selection is required")
    if set(base) != set(donor):
        missing_from_donor = sorted(set(base) - set(donor))
        missing_from_base = sorted(set(donor) - set(base))
        raise ValueError(
            "checkpoint model-state schemas differ: "
            f"missing_from_donor={missing_from_donor}, "
            f"missing_from_base={missing_from_base}"
        )

    normalised_rows: dict[str, tuple[int, ...]] = {}
    selected: list[str] = []
    for name, rows in tensor_rows.items():
        if name not in base:
            raise ValueError(f"tensor row selection matched no tensor: {name!r}")
        base_tensor = base[name]
        donor_tensor = donor[name]
        if base_tensor.shape != donor_tensor.shape:
            raise ValueError(f"tensor shape differs for {name!r}")
        if base_tensor.dtype != donor_tensor.dtype:
            raise ValueError(f"tensor dtype differs for {name!r}")
        if base_tensor.ndim == 0:
            raise ValueError(f"tensor row selection requires a non-scalar tensor: {name!r}")
        unique_rows = tuple(dict.fromkeys(int(row) for row in rows))
        if not unique_rows:
            raise ValueError(f"tensor row selection is empty for {name!r}")
        invalid = [row for row in unique_rows if row < 0 or row >= base_tensor.shape[0]]
        if invalid:
            raise ValueError(
                f"tensor row selection is outside leading dimension for {name!r}: {invalid}"
            )
        normalised_rows[name] = unique_rows
        selected.extend(f"{name}[{row}]" for row in unique_rows)

    composed = {name: tensor.detach().clone() for name, tensor in base.items()}
    if donor_weight == 0.0:
        return composed, tuple(sorted(selected))
    for name, rows in normalised_rows.items():
        base_tensor = base[name]
        donor_tensor = donor[name]
        row_index = torch.as_tensor(rows, dtype=torch.int64, device=base_tensor.device)
        if base_tensor.is_floating_point() or base_tensor.is_complex():
            values = torch.lerp(
                base_tensor.index_select(0, row_index),
                donor_tensor.index_select(0, row_index),
                donor_weight,
            )
        elif donor_weight == 1.0:
            values = donor_tensor.index_select(0, row_index)
        else:
            values = base_tensor.index_select(0, row_index)
        composed[name].index_copy_(0, row_index, values)
    return composed, tuple(sorted(selected))


__all__ = ["compose_model_state", "compose_model_state_rows"]
