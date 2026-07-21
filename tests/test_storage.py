import json
from pathlib import Path

import pytest

from exceptions import ExperimentStorageError
from helpers import make_completed_experiment
from storage import load_experiment, save_experiment


def test_save_experiment_creates_file(tmp_path: Path) -> None:
    experiment = make_completed_experiment()
    file_path = tmp_path / "experiments" / "customer_churn_v1.json"

    save_experiment(experiment, file_path)

    assert file_path.exists()
    assert file_path.is_file()


def test_save_experiment_creates_parent_directory(tmp_path: Path) -> None:
    experiment = make_completed_experiment()
    file_path = tmp_path / "nested" / "folder" / "experiment.json"

    save_experiment(experiment, file_path)

    assert file_path.parent.exists()


def test_saved_json_content_is_correct(tmp_path: Path) -> None:
    experiment = make_completed_experiment()
    file_path = tmp_path / "customer_churn_v1.json"

    save_experiment(experiment, file_path)

    with file_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    assert data["name"] == "customer_churn_v1"
    assert data["status"] == "completed"
    assert data["metrics"]["accuracy"] == 0.91
    assert data["logs"] == [
        "Training started.",
        "Training completed successfully.",
    ]
    assert data["tags"] == ["classification", "baseline"]


def test_load_experiment_restores_data(tmp_path: Path) -> None:
    original = make_completed_experiment()
    file_path = tmp_path / "customer_churn_v1.json"
    save_experiment(original, file_path)

    loaded = load_experiment(file_path)

    assert loaded.name == original.name
    assert loaded.model_name == original.model_name
    assert loaded.status == original.status
    assert loaded.metrics == original.metrics
    assert loaded.logs == original.logs
    assert loaded.tags == original.tags
    assert loaded.training_config.epochs == original.training_config.epochs
    assert loaded.dataset.file_name == original.dataset.file_name


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = make_completed_experiment()
    file_path = tmp_path / "customer_churn_v1.json"

    save_experiment(original, file_path)
    loaded = load_experiment(file_path)

    assert loaded.to_dict() == original.to_dict()


def test_load_missing_file_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.json"

    with pytest.raises(ExperimentStorageError):
        load_experiment(file_path)


def test_load_invalid_json_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.json"
    file_path.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(ExperimentStorageError):
        load_experiment(file_path)


def test_load_incomplete_data_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "incomplete.json"
    file_path.write_text('{"name": "test"}', encoding="utf-8")

    with pytest.raises(ExperimentStorageError):
        load_experiment(file_path)
