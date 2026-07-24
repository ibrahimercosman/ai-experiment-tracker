# AI Experiment Tracker

Track AI/ML experiments in Python: validate configs, store metrics/logs/tags, and persist results as JSON.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Features

- Define training configuration (`TrainingConfig`) and dataset info (`DatasetInfo`)
- Create and manage `Experiment` objects
- Add metrics, logs, and tags with validation
- Update experiment status (`created`, `running`, `completed`, `failed`)
- Convert experiments to/from dictionaries
- Save and load experiments as JSON files
- Custom exceptions for validation and storage errors
- Pytest coverage for validation and storage flows

## Project Structure

```text
ai_experiment_tracker/
├── src/
│   ├── experiment.py      # Dataclasses and Experiment class
│   ├── exceptions.py      # Custom exception classes
│   ├── storage.py         # JSON save/load functions
│   └── main.py            # Example usage script
├── tests/
│   ├── test_experiment.py
│   ├── test_storage.py
│   └── helpers.py
├── data/
│   └── experiments/       # Saved experiment JSON files
├── configs/               # Reserved for future config files
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

```bash
cd ai_experiment_tracker
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from pathlib import Path

from experiment import DatasetInfo, Experiment, TrainingConfig
from storage import load_experiment, save_experiment

config = TrainingConfig(epochs=20, learning_rate=0.01)
dataset = DatasetInfo(
    file_name="customers.csv",
    target_column="churn",
    row_count=10000,
    feature_count=18,
)

experiment = Experiment(
    name="customer_churn_v1",
    model_name="random_forest",
    training_config=config,
    dataset=dataset,
)

experiment.add_metric("accuracy", 0.91)
experiment.add_log("Training completed.")
experiment.add_tag("classification")
experiment.update_status("completed")

save_experiment(experiment, Path("data/experiments/customer_churn_v1.json"))
loaded = load_experiment(Path("data/experiments/customer_churn_v1.json"))
```

## Run Example

From the project root:

```bash
PYTHONPATH=src python src/main.py
```

## Run Tests

From the project root:

```bash
python -m pytest -v
```

## Exception Hierarchy

```text
Exception
└── ExperimentError
    ├── ExperimentValidationError   # Invalid data (empty name, bad metric, etc.)
    └── ExperimentStorageError      # File/JSON errors (not found, corrupt, etc.)
```

## What This Project Demonstrates

- Object-oriented Python design
- Type hints and dataclasses
- Custom exception handling
- `pathlib` + JSON persistence
- Unit testing with pytest
- Clean repository layout for portfolio use
