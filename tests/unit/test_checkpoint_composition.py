from __future__ import annotations

import pytest
import torch

from world_model.training.checkpoint_composition import compose_model_state


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
