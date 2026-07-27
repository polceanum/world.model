from datetime import datetime, timezone
from pathlib import Path

from world_model.utils.artifacts import timestamped_artifact_path


def test_artifact_directory_gets_sortable_utc_prefix() -> None:
    timestamp = datetime(2026, 7, 27, 17, 30, 45, tzinfo=timezone.utc)
    result = timestamped_artifact_path("demo_outputs/interaction-demo", now=timestamp)
    assert result == Path("demo_outputs/20260727-173045-interaction-demo")


def test_existing_timestamp_prefix_is_not_duplicated() -> None:
    path = Path("runs/20260727-173045-training")
    assert timestamped_artifact_path(path) == path


def test_naive_timestamp_is_interpreted_as_utc() -> None:
    timestamp = datetime(2026, 7, 27, 17, 30, 45)
    result = timestamped_artifact_path("runs/training", now=timestamp)
    assert result.name == "20260727-173045-training"
