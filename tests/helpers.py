from experiment import DatasetInfo, Experiment, TrainingConfig


def make_training_config() -> TrainingConfig:
    return TrainingConfig(epochs=20, learning_rate=0.01)


def make_dataset() -> DatasetInfo:
    return DatasetInfo(
        file_name="customers.csv",
        target_column="churn",
        row_count=1000,
        feature_count=10,
    )


def make_experiment() -> Experiment:
    return Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=make_training_config(),
        dataset=make_dataset(),
    )


def make_completed_experiment() -> Experiment:
    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=TrainingConfig(epochs=20, learning_rate=0.01, use_gpu=True),
        dataset=DatasetInfo(
            file_name="customers.csv",
            target_column="churn",
            row_count=10000,
            feature_count=18,
        ),
        description="Baseline churn experiment",
    )
    experiment.add_metric("accuracy", 0.91)
    experiment.add_metric("f1_score", 0.88)
    experiment.add_log("Training started.")
    experiment.add_log("Training completed successfully.")
    experiment.add_tag("classification")
    experiment.add_tag("baseline")
    experiment.update_status("completed")
    return experiment
