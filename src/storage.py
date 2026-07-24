import json
from pathlib import Path

from experiment import Experiment
from exceptions import ExperimentStorageError


def save_experiment(experiment: Experiment, file_path: Path) -> None:
    """Save an experiment to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as json_file:
            json.dump(experiment.to_dict(), json_file, indent=2)
    except OSError as error:
        raise ExperimentStorageError(
            f"Failed to save experiment to {file_path}."
        ) from error


def load_experiment(file_path: Path) -> Experiment:
    """Load an experiment from a JSON file."""
    if not file_path.exists():
        raise ExperimentStorageError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ExperimentStorageError(f"Path is not a file: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ExperimentStorageError(
            f"Invalid JSON in file: {file_path}"
        ) from error
    except OSError as error:
        raise ExperimentStorageError(
            f"Failed to read file: {file_path}"
        ) from error

    try:
        return Experiment.from_dict(data)
    except (KeyError, TypeError) as error:
        raise ExperimentStorageError(
            f"Invalid experiment data in file: {file_path}"
        ) from error
