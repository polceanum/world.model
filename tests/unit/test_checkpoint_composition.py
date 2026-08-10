from __future__ import annotations

import pytest
import torch

from world_model.training.checkpoint_composition import (
    compose_model_state,
    compose_model_state_rows,
)


def test_compose_model_state_replaces_only_selected_module() -> None:
    base = {
        "dynamics.weight": torch.tensor([1.0, 2.0]),
        "updater.weight": torch.tensor([3.0]),
        "counter": torch.tensor(1),
    }
    donor = {
        "dynamics.weight": torch.tensor([5.0, 6.0]),
        "updater.weight": torch.tensor([7.0]),
        "counter": torch.tensor(2),
    }

    composed, selected = compose_model_state(
        base,
        donor,
        module_prefixes=["dynamics"],
    )

    assert selected == ("dynamics.weight",)
    torch.testing.assert_close(composed["dynamics.weight"], donor["dynamics.weight"])
    torch.testing.assert_close(composed["updater.weight"], base["updater.weight"])
    assert composed["counter"].item() == 1
    assert composed["dynamics.weight"].data_ptr() != donor["dynamics.weight"].data_ptr()


def test_compose_model_state_interpolates_floating_tensors() -> None:
    base = {"dynamics.weight": torch.tensor([0.0]), "dynamics.count": torch.tensor(1)}
    donor = {"dynamics.weight": torch.tensor([4.0]), "dynamics.count": torch.tensor(2)}

    composed, _ = compose_model_state(
        base,
        donor,
        module_prefixes=["dynamics"],
        donor_weight=0.25,
    )

    torch.testing.assert_close(composed["dynamics.weight"], torch.tensor([1.0]))
    assert composed["dynamics.count"].item() == 1


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_compose_model_state_rejects_invalid_weight(weight: float) -> None:
    state = {"dynamics.weight": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="donor_weight"):
        compose_model_state(
            state,
            state,
            module_prefixes=["dynamics"],
            donor_weight=weight,
        )


def test_compose_model_state_rejects_schema_and_prefix_mismatch() -> None:
    with pytest.raises(ValueError, match="schemas differ"):
        compose_model_state(
            {"dynamics.weight": torch.tensor([1.0])},
            {"updater.weight": torch.tensor([1.0])},
            module_prefixes=["dynamics"],
        )
    state = {"dynamics.weight": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="matched no tensors"):
        compose_model_state(state, state, module_prefixes=["updater"])


def test_compose_model_state_rows_replaces_only_selected_leading_rows() -> None:
    base = {
        "head.weight": torch.zeros(3, 2),
        "head.bias": torch.zeros(3),
        "trunk.weight": torch.zeros(2, 2),
    }
    donor = {
        "head.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2) + 1,
        "head.bias": torch.tensor([4.0, 5.0, 6.0]),
        "trunk.weight": torch.ones(2, 2),
    }

    composed, selected = compose_model_state_rows(
        base,
        donor,
        tensor_rows={"head.weight": [0, 2], "head.bias": [0, 2]},
    )

    assert selected == (
        "head.bias[0]",
        "head.bias[2]",
        "head.weight[0]",
        "head.weight[2]",
    )
    torch.testing.assert_close(composed["head.weight"][0], donor["head.weight"][0])
    torch.testing.assert_close(composed["head.weight"][1], base["head.weight"][1])
    torch.testing.assert_close(composed["head.weight"][2], donor["head.weight"][2])
    torch.testing.assert_close(composed["head.bias"], torch.tensor([4.0, 0.0, 6.0]))
    torch.testing.assert_close(composed["trunk.weight"], base["trunk.weight"])
    assert composed["head.weight"].data_ptr() != base["head.weight"].data_ptr()


def test_compose_model_state_rows_interpolates_and_rejects_invalid_rows() -> None:
    state = {"head.weight": torch.zeros(2, 2), "scalar": torch.tensor(0.0)}
    donor = {"head.weight": torch.ones(2, 2), "scalar": torch.tensor(1.0)}

    composed, _ = compose_model_state_rows(
        state,
        donor,
        tensor_rows={"head.weight": [1]},
        donor_weight=0.25,
    )
    torch.testing.assert_close(composed["head.weight"][0], torch.zeros(2))
    torch.testing.assert_close(composed["head.weight"][1], torch.full((2,), 0.25))

    with pytest.raises(ValueError, match="matched no tensor"):
        compose_model_state_rows(state, donor, tensor_rows={"missing": [0]})
    with pytest.raises(ValueError, match="non-scalar"):
        compose_model_state_rows(state, donor, tensor_rows={"scalar": [0]})
    with pytest.raises(ValueError, match="outside leading dimension"):
        compose_model_state_rows(state, donor, tensor_rows={"head.weight": [-1, 2]})
    with pytest.raises(ValueError, match="at least one tensor row"):
        compose_model_state_rows(state, donor, tensor_rows={})
