from pathlib import Path

import pytest

from world_model.datasets.splits import SPLIT_SEED_RANGES
from world_model.utils.config import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_profiles_resolve_and_validate(path: Path) -> None:
    config = load_config(path)
    assert config.model.max_objects >= config.simulator.max_objects
    assert config.runtime.modality == "rgb"
    assert config.evaluation.rgb_only
    test_lower, test_upper = SPLIT_SEED_RANGES["test"]
    assert test_lower <= config.demo.seed <= test_upper
    assert config.simulator.split_validation_start == SPLIT_SEED_RANGES["validation"][0]
    assert config.simulator.split_test_start == test_lower
    assert config.simulator.split_ood_start == SPLIT_SEED_RANGES["ood"][0]


def test_dotted_override_is_typed(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=["training.steps=3", "simulator.image_size=[32, 40]"],
    )
    assert config.training.steps == 3
    assert config.simulator.image_size == (32, 40)


def test_unknown_key_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  mystery: true\n", encoding="utf-8")
    with pytest.raises(KeyError, match="mystery"):
        load_config(bad)


def test_oracle_cannot_hide_in_rgb_evaluation(tmp_path: Path) -> None:
    bad = tmp_path / "oracle.yaml"
    bad.write_text(
        "runtime:\n  modality: debug_oracle\n  enable_debug_oracle: true\n"
        "evaluation:\n  rgb_only: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RGB-only"):
        load_config(bad)
