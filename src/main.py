"""Example usage of AI Experiment Tracker."""

from pathlib import Path

from experiment import DatasetInfo, Experiment, TrainingConfig
from storage import load_experiment, save_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    training_config = TrainingConfig(
        epochs=20,
        learning_rate=0.01,
        use_gpu=True,
    )

    dataset = DatasetInfo(
        file_name="customers.csv",
        target_column="churn",
        row_count=10000,
        feature_count=18,
    )

    experiment = Experiment(
        name="customer_churn_v1",
        model_name="random_forest",
        training_config=training_config,
        dataset=dataset,
        description="Baseline churn experiment",
    )

    experiment.add_log("Training started.")
    experiment.add_log("Dataset loaded successfully.")
    experiment.update_status("running")
    experiment.add_metric("accuracy", 0.91)
    experiment.add_metric("f1_score", 0.88)
    experiment.add_tag("classification")
    experiment.add_tag("customer-churn")
    experiment.add_log("Training completed successfully.")
    experiment.update_status("completed")

    file_path = PROJECT_ROOT / "data" / "experiments" / "customer_churn_v1.json"
    save_experiment(experiment, file_path)
    print(f"Experiment saved to: {file_path}")

    loaded = load_experiment(file_path)
    print(f"Loaded experiment: {loaded.name}")
    print(f"Status: {loaded.status}")
    print(f"Metrics: {loaded.metrics}")
    print(f"Tags: {loaded.tags}")


if __name__ == "__main__":
    main()
