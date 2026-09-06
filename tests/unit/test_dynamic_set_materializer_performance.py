"""Parity and safety tests for the bounded dynamic-set simulator fast path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import pytest
import torch
from torch import Tensor

import world_model.training.dynamic_set_materializer as materializer_module
from world_model.simulator.physics import (
    PhysicsConfig,
    SphereState,
    advance_spheres,
)
from world_model.simulator.renderer import render_spheres as uncached_render_spheres
from world_model.training.dynamic_set_physics import (
    advance_spheres_no_boundary_contacts_prevalidated,
    prevalidated_sphere_pair_cache,
)
from world_model.training.dynamic_set_protocol import PHYSICAL_CELLS, physical_manifest


def _assert_nested_bit_exact(left: object, right: object, *, path: str = "root") -> None:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        assert isinstance(left, Tensor) and isinstance(right, Tensor), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if is_dataclass(left) or is_dataclass(right):
        assert type(left) is type(right), path
        assert not isinstance(left, type)
        for field in fields(left):
            _assert_nested_bit_exact(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping), path
        assert tuple(left) == tuple(right), path
        for key in left:
            _assert_nested_bit_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right), path
        assert isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_bit_exact(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
        return
    assert left == right, path


def _single_state(*, x_position: float) -> SphereState:
    return SphereState(
        object_id=torch.tensor([0], dtype=torch.int64),
        active=torch.tensor([True]),
        position=torch.tensor([[x_position, 0.0, 0.0]], dtype=torch.float32),
        velocity=torch.zeros(1, 3, dtype=torch.float32),
        radius=torch.tensor([[0.2]], dtype=torch.float32),
        mass=torch.ones(1, 1, dtype=torch.float32),
        restitution=torch.zeros(1, 1, dtype=torch.float32),
        drag=torch.zeros(1, 1, dtype=torch.float32),
        friction=torch.zeros(1, 1, dtype=torch.float32),
        albedo=torch.ones(1, 3, dtype=torch.float32),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        angular_velocity=torch.zeros(1, 3, dtype=torch.float32),
        sleeping=torch.tensor([False]),
        sleep_counter=torch.zeros(1, dtype=torch.int64),
    )


def test_no_boundary_fast_path_rejects_contact_instead_of_fabricating_clearance() -> None:
    state = _single_state(x_position=0.8)
    config = PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )
    config.validate()
    cache = prevalidated_sphere_pair_cache(state)

    with pytest.raises(ValueError, match="reached a world boundary"):
        advance_spheres_no_boundary_contacts_prevalidated(
            state,
            1.0 / 20.0,
            config,
            pair_cache=cache,
        )


def test_fast_materializer_is_bit_exact_and_repeatable_across_all_22_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_rows = physical_manifest("development")
    pools = tuple(
        tuple(row for row in all_rows if row.cell_index == cell_index)
        for cell_index in range(len(PHYSICAL_CELLS))
    )
    selected = [pool[cell_index % len(pool)] for cell_index, pool in enumerate(pools)]
    for cell_index, action_time_stratum in enumerate(range(4)):
        selected[cell_index] = next(
            row for row in pools[cell_index] if row.action_time_stratum == action_time_stratum
        )
    selected[4] = next(
        row
        for row in pools[4]
        if row.contact_geometry == "glancing" and row.contact_origin == "action_induced"
    )
    selected[5] = next(
        row for row in pools[5] if row.contact_origin == "natural" and not row.known_action
    )
    rows = tuple(selected)
    assert {row.lifecycle_schedule for row in rows} == {
        "none",
        "birth",
        "removal",
        "remove_then_birth",
    }
    assert {row.known_action for row in rows} == {False, True}
    assert {row.contact_origin for row in rows} == {"none", "natural", "action_induced"}
    assert {row.action_time_stratum for row in rows} == {None, 0, 1, 2, 3}
    assert {row.camera_stratum for row in rows} == set(range(8))
    assert {row.contact_geometry for row in rows} == {"none", "head_on", "glancing"}
    fast_step = materializer_module.advance_spheres_no_boundary_contacts_prevalidated
    fast_render = materializer_module.render_spheres

    def reference_step(
        state: SphereState,
        dt: float,
        config: PhysicsConfig,
        *,
        external_impulse: Tensor | None = None,
        **_ignored: object,
    ) -> tuple[SphereState, object]:
        return advance_spheres(
            state,
            dt,
            config,
            external_impulse=external_impulse,
        )

    def reference_render(
        state: SphereState,
        camera: object,
        image_size: tuple[int, int],
        **kwargs: object,
    ) -> object:
        kwargs.pop("_prevalidated_cache", None)
        return uncached_render_spheres(state, camera, image_size, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        materializer_module,
        "advance_spheres_no_boundary_contacts_prevalidated",
        reference_step,
    )
    monkeypatch.setattr(materializer_module, "render_spheres", reference_render)
    reference = tuple(
        materializer_module.materialize_dynamic_set_episode(row, maximum_attempts=1) for row in rows
    )

    monkeypatch.setattr(
        materializer_module,
        "advance_spheres_no_boundary_contacts_prevalidated",
        fast_step,
    )
    monkeypatch.setattr(materializer_module, "render_spheres", fast_render)
    accelerated = tuple(
        materializer_module.materialize_dynamic_set_episode(row, maximum_attempts=1) for row in rows
    )
    repeated = tuple(
        materializer_module.materialize_dynamic_set_episode(row, maximum_attempts=1) for row in rows
    )

    assert tuple(item.attempt_count for item in accelerated) == (1,) * len(PHYSICAL_CELLS)
    for cell_index, (expected, actual, repeat) in enumerate(
        zip(reference, accelerated, repeated, strict=True)
    ):
        _assert_nested_bit_exact(
            expected,
            actual,
            path=f"physical_cell[{cell_index}].reference",
        )
        _assert_nested_bit_exact(
            actual,
            repeat,
            path=f"physical_cell[{cell_index}].repeat",
        )
