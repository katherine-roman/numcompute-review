"""Coordinate chunk-wise training and evaluation."""
import tracemalloc

import numpy as np

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
        })
        return accuracy