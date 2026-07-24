from dataclasses import dataclass

from exceptions import ExperimentValidationError

VALID_STATUSES = frozenset({"created", "running", "completed", "failed"})
BOUNDED_METRICS = frozenset({
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
})


def _require_non_empty(value: str, field_name: str) -> str:
    if value.strip() == "":
        raise ExperimentValidationError(f"{field_name} cannot be empty.")
    return value


@dataclass
class TrainingConfig:
    """Training configuration for an AI experiment.

    Attributes:
        epochs: The number of epochs to train for.
        learning_rate: The learning rate to use for training.
        use_gpu: Whether to use GPU for training.
        batch_size: The batch size to use for training.
    """

    epochs: int
    learning_rate: float
    use_gpu: bool = False
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ExperimentValidationError("epochs must be greater than 0.")

        if self.learning_rate <= 0:
            raise ExperimentValidationError("learning_rate must be greater than 0.")

        if self.learning_rate > 1.0:
            raise ExperimentValidationError(
                "learning_rate must be less than or equal to 1.0."
            )

        if self.batch_size is not None and self.batch_size <= 0:
            raise ExperimentValidationError(
                "batch_size must be greater than 0 when provided."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrainingConfig":
        return cls(
            epochs=data["epochs"],
            learning_rate=data["learning_rate"],
            use_gpu=data.get("use_gpu", False),
            batch_size=data.get("batch_size"),
        )


@dataclass
class DatasetInfo:
    """Information about the dataset used in an AI experiment."""

    file_name: str
    target_column: str
    row_count: int
    feature_count: int
    description: str | None = None
    shuffle: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.file_name, "file_name")
        _require_non_empty(self.target_column, "target_column")

        if self.row_count < 0:
            raise ExperimentValidationError("row_count cannot be negative.")

        if self.feature_count < 0:
            raise ExperimentValidationError("feature_count cannot be negative.")

    def to_dict(self) -> dict[str, object]:
        return {
            "file_name": self.file_name,
            "target_column": self.target_column,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "description": self.description,
            "shuffle": self.shuffle,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DatasetInfo":
        return cls(
            file_name=data["file_name"],
            target_column=data["target_column"],
            row_count=data["row_count"],
            feature_count=data["feature_count"],
            description=data.get("description"),
            shuffle=data.get("shuffle", False),
        )


class Experiment:
    """Represents an AI experiment."""

    def __init__(
        self,
        name: str,
        model_name: str,
        training_config: TrainingConfig,
        dataset: DatasetInfo,
        description: str | None = None,
    ) -> None:
        self.name = _require_non_empty(name, "name")
        self.model_name = _require_non_empty(model_name, "model_name")
        self.training_config = training_config
        self.dataset = dataset
        self.description = description
        self.status = "created"
        self.metrics: dict[str, float] = {}
        self.logs: list[str] = []
        self.tags: list[str] = []

    def add_metric(self, name: str, value: float) -> None:
        _require_non_empty(name, "metric name")

        if name in BOUNDED_METRICS and not (0.0 <= value <= 1.0):
            raise ExperimentValidationError(
                f"{name} must be between 0.0 and 1.0."
            )

        self.metrics[name] = value

    def add_log(self, message: str) -> None:
        _require_non_empty(message, "log message")
        self.logs.append(message)

    def add_tag(self, tag: str) -> None:
        _require_non_empty(tag, "tag")
        normalized_tag = tag.strip().lower()

        if normalized_tag in self.tags:
            return

        self.tags.append(normalized_tag)

    def update_status(self, new_status: str) -> None:
        if new_status not in VALID_STATUSES:
            raise ExperimentValidationError(
                f"Invalid status: {new_status}. "
                f"Must be one of {VALID_STATUSES}."
            )

        self.status = new_status

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "training_config": self.training_config.to_dict(),
            "dataset": self.dataset.to_dict(),
            "description": self.description,
            "status": self.status,
            "metrics": self.metrics,
            "logs": self.logs,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Experiment":
        training_config = TrainingConfig.from_dict(data["training_config"])
        dataset = DatasetInfo.from_dict(data["dataset"])

        experiment = cls(
            name=data["name"],
            model_name=data["model_name"],
            training_config=training_config,
            dataset=dataset,
            description=data.get("description"),
        )

        experiment.status = data.get("status", "created")
        experiment.metrics = data.get("metrics", {})
        experiment.logs = data.get("logs", [])
        experiment.tags = data.get("tags", [])

        return experiment
