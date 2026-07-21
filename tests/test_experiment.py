import pytest

from experiment import DatasetInfo, Experiment, TrainingConfig
from exceptions import ExperimentValidationError
from helpers import make_dataset, make_experiment, make_training_config


def test_valid_config_stores_values() -> None:
    config = TrainingConfig(
        epochs=20,
        learning_rate=0.01,
        use_gpu=True,
        batch_size=32,
    )

    assert config.epochs == 20
    assert config.learning_rate == 0.01
    assert config.use_gpu is True
    assert config.batch_size == 32


def test_default_values() -> None:
    config = TrainingConfig(epochs=10, learning_rate=0.001)

    assert config.use_gpu is False
    assert config.batch_size is None


def test_invalid_epochs_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=0, learning_rate=0.01)


def test_negative_epochs_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=-5, learning_rate=0.01)


def test_invalid_learning_rate_zero_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=10, learning_rate=0)


def test_invalid_learning_rate_too_high_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=10, learning_rate=1.5)


def test_learning_rate_one_is_accepted() -> None:
    config = TrainingConfig(epochs=10, learning_rate=1.0)

    assert config.learning_rate == 1.0


def test_invalid_batch_size_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=10, learning_rate=0.01, batch_size=0)


def test_negative_batch_size_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        TrainingConfig(epochs=10, learning_rate=0.01, batch_size=-3)


def test_batch_size_none_is_accepted() -> None:
    config = TrainingConfig(epochs=10, learning_rate=0.01, batch_size=None)

    assert config.batch_size is None


def test_valid_dataset_stores_values() -> None:
    dataset = DatasetInfo(
        file_name="customers.csv",
        target_column="churn",
        row_count=10000,
        feature_count=18,
        description="Customer churn dataset",
        shuffle=True,
    )

    assert dataset.file_name == "customers.csv"
    assert dataset.target_column == "churn"
    assert dataset.row_count == 10000
    assert dataset.feature_count == 18
    assert dataset.description == "Customer churn dataset"
    assert dataset.shuffle is True


def test_dataset_default_values() -> None:
    dataset = DatasetInfo(
        file_name="customers.csv",
        target_column="churn",
        row_count=10000,
        feature_count=18,
    )

    assert dataset.description is None
    assert dataset.shuffle is False


def test_empty_file_name_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        DatasetInfo(
            file_name="",
            target_column="churn",
            row_count=100,
            feature_count=5,
        )


def test_whitespace_file_name_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        DatasetInfo(
            file_name="   ",
            target_column="churn",
            row_count=100,
            feature_count=5,
        )


def test_empty_target_column_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        DatasetInfo(
            file_name="customers.csv",
            target_column="",
            row_count=100,
            feature_count=5,
        )


def test_negative_row_count_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        DatasetInfo(
            file_name="customers.csv",
            target_column="churn",
            row_count=-1,
            feature_count=5,
        )


def test_negative_feature_count_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        DatasetInfo(
            file_name="customers.csv",
            target_column="churn",
            row_count=100,
            feature_count=-1,
        )


def test_zero_row_and_feature_count_are_accepted() -> None:
    dataset = DatasetInfo(
        file_name="customers.csv",
        target_column="churn",
        row_count=0,
        feature_count=0,
    )

    assert dataset.row_count == 0
    assert dataset.feature_count == 0


def test_experiment_starts_with_created_status() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    assert experiment.name == "customer_churn_v1"
    assert experiment.model_name == "random_forest"
    assert experiment.description is None
    assert experiment.status == "created"
    assert experiment.metrics == {}
    assert experiment.logs == []
    assert experiment.tags == []


def test_experiment_stores_description() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
        description="Baseline churn experiment",
    )

    assert experiment.description == "Baseline churn experiment"


def test_empty_experiment_name_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        Experiment(
            name="",
            model_name="random_forest",
            training_config=make_training_config(),
            dataset=make_dataset(),
        )


def test_whitespace_experiment_name_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        Experiment(
            name="   ",
            model_name="random_forest",
            training_config=make_training_config(),
            dataset=make_dataset(),
        )


def test_empty_model_name_raises_error() -> None:
    with pytest.raises(ExperimentValidationError):
        Experiment(
            name="customer_churn_v1",
            model_name="",
            training_config=make_training_config(),
            dataset=make_dataset(),
        )

def test_add_metric_stores_value() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_metric("accuracy", 0.91)

    assert experiment.metrics == {"accuracy": 0.91}


def test_add_multiple_metrics() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_metric("accuracy", 0.91)
    experiment.add_metric("f1_score", 0.88)
    assert len(experiment.metrics) == 2


def test_empty_metric_name_raises_error() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    with pytest.raises(ExperimentValidationError):
        experiment.add_metric("", 0.5)


def test_invalid_accuracy_raises_error() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    with pytest.raises(ExperimentValidationError):
        experiment.add_metric("accuracy", 1.5)
        


def test_loss_can_be_greater_than_one() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_metric("loss", 2.5)
    assert experiment.metrics["loss"] == 2.5



def test_add_log_stores_message() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_log("Training started")
    assert experiment.logs == ["Training started"]



def test_add_multiple_logs_keeps_order() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )
    experiment.add_log("Training started")
    experiment.add_log("Epoch 1 completed")

    assert len(experiment.logs) == 2
    assert experiment.logs[0] == "Training started"
    assert experiment.logs[1] == "Epoch 1 completed"



def test_empty_log_message_raises_error() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    with pytest.raises(ExperimentValidationError):
        experiment.add_log("")



def test_add_tag_stores_value() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_tag("classification")
    assert experiment.tags == ["classification"]



def test_add_tag_converts_to_lowercase() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_tag("Classification")
    assert experiment.tags == ["classification"]
    
    

def test_duplicate_tag_not_added_twice() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    experiment.add_tag("baseline")
    experiment.add_tag("baseline")
    assert len(experiment.tags) == 1



def test_empty_tag_raises_error() -> None:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )

    with pytest.raises(ExperimentValidationError):
        experiment.add_tag("")


def test_update_status_changes_status() -> None:
    experiment = make_experiment()

    experiment.update_status("running")

    assert experiment.status == "running"


def test_update_status_to_completed() -> None:
    experiment = make_experiment()

    experiment.update_status("running")
    experiment.update_status("completed")

    assert experiment.status == "completed"


def test_invalid_status_raises_error() -> None:
    experiment = make_experiment()

    with pytest.raises(ExperimentValidationError):
        experiment.update_status("unknown")


def test_to_dict_contains_all_keys() -> None:
    experiment = make_experiment()

    data = experiment.to_dict()

    assert "name" in data
    assert "model_name" in data
    assert "training_config" in data
    assert "dataset" in data
    assert "description" in data
    assert "status" in data
    assert "metrics" in data
    assert "logs" in data
    assert "tags" in data
    
    
def test_to_dict_stores_correct_values() -> None:
    experiment = make_experiment()
    experiment.add_metric("accuracy", 0.91)
    experiment.add_log("Training started.")
    experiment.add_tag("classification")

    data = experiment.to_dict()

    assert data["name"] == "customer_churn_v1"
    assert data["status"] == "created"
    assert data["training_config"]["epochs"] == 20
    assert data["dataset"]["file_name"] == "customers.csv"
    assert data["metrics"]["accuracy"] == 0.91
    assert data["logs"] == ["Training started."]
    assert data["tags"] == ["classification"]


def test_from_dict_restores_experiment() -> None:
    original = make_experiment()
    original.add_metric("accuracy", 0.91)
    original.add_log("Training started.")
    original.add_tag("classification")
    original.update_status("completed")

    restored = Experiment.from_dict(original.to_dict())

    assert restored.name == original.name
    assert restored.model_name == original.model_name
    assert restored.status == "completed"
    assert restored.metrics == {"accuracy": 0.91}
    assert restored.logs == ["Training started."]
    assert restored.tags == ["classification"]
    assert restored.training_config.epochs == 20
    assert restored.dataset.file_name == "customers.csv"


def test_to_dict_and_from_dict_round_trip() -> None:
    original = make_experiment()
    original.add_metric("f1_score", 0.88)

    data = original.to_dict()
    restored = Experiment.from_dict(data)

    assert restored.to_dict() == data
