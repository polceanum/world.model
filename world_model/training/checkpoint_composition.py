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


__all__ = ["compose_model_state"]
