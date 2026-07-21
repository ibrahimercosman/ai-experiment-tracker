# AI Experiment Tracker

A Python application for managing, validating, saving, and loading AI/ML experiment records as JSON files.

## Features

- Define training configuration (`TrainingConfig`) and dataset info (`DatasetInfo`)
- Create and manage `Experiment` objects
- Add metrics, logs, and tags with validation
- Update experiment status (`created`, `running`, `completed`, `failed`)
- Convert experiments to/from dictionaries
- Save and load experiments as JSON files
- Custom exceptions for validation and storage errors
- Full pytest test coverage

## Project Structure

```
ai_experiment_tracker/
    experiment.py      # Dataclasses and Experiment class
    exceptions.py      # Custom exception classes
    storage.py         # JSON save/load functions
    main.py            # Example usage script
    tests/
        test_experiment.py
        test_storage.py
        helpers.py         # Shared test helpers
    data/
        experiments/   # Saved JSON files
    requirements.txt
    README.md
```

## Installation

```bash
cd ai_experiment_tracker
pip install -r requirements.txt
```

## Quick Start

```python
from experiment import DatasetInfo, Experiment, TrainingConfig
from storage import load_experiment, save_experiment
from pathlib import Path

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

```bash
python main.py
```

## Run Tests

```bash
python -m pytest -v
```

## Exception Hierarchy

```
Exception
└── ExperimentError
    ├── ExperimentValidationError   # Invalid data (empty name, bad metric, etc.)
    └── ExperimentStorageError      # File/JSON errors (not found, corrupt, etc.)
```
