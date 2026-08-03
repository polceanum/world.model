"""Typed one-use propagation prepared for a specific runtime revision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TypeAlias

import torch
from torch import Tensor

from world_model.belief import WorldBelief


class PreparedPropagationError(ValueError):
    """A prepared propagation does not match the current runtime transition."""


TensorVersionPath: TypeAlias = tuple[str, ...]
TensorVersionEntry: TypeAlias = tuple[
    TensorVersionPath,
    int,
    int,
    bool,
    bool,
    bool,
]
TensorVersionSignature: TypeAlias = tuple[TensorVersionEntry, ...]


def tensor_identity_version_signature(value: object) -> TensorVersionSignature:
    """Describe nested tensors without copying or synchronizing their values.

    Tensor identity detects replacement of mutable dataclass/container fields,
    PyTorch's version counter detects in-place writes, and graph metadata
    detects ownership changes such as ``detach_()`` or ``requires_grad_()``.
    Inference-mode tensors deliberately fail closed because they do not expose
    a version counter; callers can use ``torch.no_grad()`` when preparing
    propagation.
    """

    signature: list[TensorVersionEntry] = []
    visited_containers: set[int] = set()

    def visit(item: object, path: TensorVersionPath) -> None:
        if isinstance(item, Tensor):
            is_inference = getattr(item, "is_inference", None)
            inference = bool(is_inference()) if callable(is_inference) else False
            try:
                version = int(item._version)
            except RuntimeError as exc:
                raise PreparedPropagationError(
                    "prepared propagation requires version-tracked tensors; "
                    "torch.inference_mode() is unsupported"
                ) from exc
            signature.append(
                (
                    path,
                    id(item),
                    version,
                    bool(item.requires_grad),
                    bool(item.is_leaf),
                    inference,
                )
            )
            return

        if is_dataclass(item) and not isinstance(item, type):
            identity = id(item)
            if identity in visited_containers:
                return
            visited_containers.add(identity)
            for item_field in fields(item):
                visit(getattr(item, item_field.name), (*path, f".{item_field.name}"))
            return

        if isinstance(item, Mapping):
            identity = id(item)
            if identity in visited_containers:
                return
            visited_containers.add(identity)
            ordered_items = sorted(
                item.items(),
                key=lambda pair: (
                    type(pair[0]).__module__,
                    type(pair[0]).__qualname__,
                    repr(pair[0]),
                ),
            )
            for key, nested in ordered_items:
                visit(nested, (*path, f"[{key!r}]"))
            return

        if isinstance(item, (tuple, list)):
            identity = id(item)
            if identity in visited_containers:
                return
            visited_containers.add(identity)
            for index, nested in enumerate(item):
                visit(nested, (*path, f"[{index}]"))

    visit(value, ())
    return tuple(signature)


@dataclass(frozen=True)
class PreparedPropagation:
    """One dynamics step that may be consumed by exactly one matching ingest.

    The source reference and revision bind the prediction to one persistent
    runtime state. Timestamp, device, dtype, and batch snapshots make silent
    reuse after in-place or execution-context changes fail closed.
    """

    source: WorldBelief
    prior: WorldBelief
    source_revision: int
    source_timestamp: Tensor
    target_timestamp: Tensor
    delta_time: Tensor
    source_device: torch.device
    source_dtype: torch.dtype
    source_batch_size: int
    event_logits: Tensor | None
    auxiliary: Mapping[str, Tensor]
    interval_collision_mask: Tensor | None
    source_tensor_signature: TensorVersionSignature = field(repr=False)
    result_tensor_signature: TensorVersionSignature = field(repr=False)
    dynamics_tensor_signature: TensorVersionSignature = field(repr=False)
    dynamics_training: bool
    _owner_token: object = field(repr=False, compare=False)
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def consumed(self) -> bool:
        """Whether a matching ingest has already claimed this propagation."""

        return self._consumed

    def _consume(self) -> None:
        if self._consumed:
            raise PreparedPropagationError("prepared propagation has already been consumed")
        object.__setattr__(self, "_consumed", True)


__all__ = [
    "PreparedPropagation",
    "PreparedPropagationError",
    "TensorVersionSignature",
    "tensor_identity_version_signature",
]
