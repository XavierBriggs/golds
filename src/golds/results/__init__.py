"""Results storage and tracking module for GOLDS."""

from golds.results.schema import EvalResult, TrainingResult
from golds.results.store import ResultStore

__all__ = ["EvalResult", "TrainingResult", "ResultStore"]
