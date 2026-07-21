class ExperimentError(Exception):
    """Base exception for all AI Experiment Tracker errors."""

    pass


class ExperimentValidationError(ExperimentError):
    """Raised when experiment data fails validation."""

    pass


class ExperimentStorageError(ExperimentError):
    """Raised when saving or loading an experiment fails."""

    pass
