"""Coordinate chunk-wise training and evaluation."""
import tracemalloc
import sys

import numpy as np

def _estimated_size_bytes(value, seen=None):
    """Estimate retained object memory without counting shared objects twice.

    Includes Python containers, object attributes, and NumPy array storage.
    This is an estimate, not the process's total RAM usage.
    """
    if seen is None:
        seen = set()

    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    size = sys.getsizeof(value)

    if isinstance(value, np.ndarray):
        # Owning arrays include their data in getsizeof().
        # Views refer to storage owned by their base array.
        if value.base is not None:
            size += _estimated_size_bytes(value.base, seen)
        if value.dtype.hasobject:
            for item in value.flat:
                size += _estimated_size_bytes(item, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            size += _estimated_size_bytes(key, seen)
            size += _estimated_size_bytes(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            size += _estimated_size_bytes(item, seen)
    elif hasattr(value, "__dict__"):
        size += _estimated_size_bytes(value.__dict__, seen)

    return size

class StreamTrainer:
    """Train and evaluate a model or pipeline one chunk at a time."""

    def __init__(self, estimator):
        """Store an estimator that provides predict and partial_fit."""
        self.estimator = estimator
        self.history = []
        self._total_correct = 0
        self._total_scored = 0

    def fit_chunk(self, X, y):
        """Pass one labelled chunk to the estimator for training."""
        self.estimator.partial_fit(X, y)
        return self

    def score_chunk(self, X, y):
        """Return the fraction of correct predictions on a labelled chunk."""
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("y must be a non-empty one-dimensional array")

        started_tracing = not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()

        try:
            predictions = np.asarray(self.estimator.predict(X))
            prediction_peak_bytes = (
                tracemalloc.get_traced_memory()[1] if started_tracing else None
            )
        finally:
            if started_tracing:
                tracemalloc.stop()

        if predictions.shape != labels.shape:
            raise ValueError("Expected one prediction per label")

        correct = int(np.count_nonzero(predictions == labels))
        accuracy = correct / labels.size

        # Count rows so cumulative accuracy handles different chunk sizes.
        self._total_correct += correct
        self._total_scored += labels.size
        self.history.append({
            "accuracy": accuracy,
            "cumulative_accuracy": self._total_correct / self._total_scored,
            "prediction_peak_bytes": prediction_peak_bytes,
            "estimator_bytes": _estimated_size_bytes(self.estimator),
        })
        return accuracy