"""Classification metrics accumulated across incoming chunks."""

import numpy as np


def _labels(values, name):
    """Validate a non-empty vector of numeric or string class labels."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if values.dtype.kind not in "biufUS":
        raise TypeError(f"{name} must contain numeric or string labels")
    if values.dtype.kind in "biuf" and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite labels")
    return values


def _divide(numerator, denominator):
    """Return zero when a metric's denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


class StreamingMetrics:
    """Accumulate accuracy, precision, recall, F1, confusion matrix and AUC.

    window_size=None uses all samples; a positive integer keeps only
    the latest window_size samples, including part of an earlier chunk.

    Precision, recall and F1 are returned per class and as macro averages.
    Undefined precision, recall and F1 values use zero.

    Binary AUC requires a score for pos_label on every update. Scores
    may be probabilities or finite decision scores. AUC is NaN when
    scores are absent or the retained data lacks either binary class.

    Samples are retained to support exact AUC and rolling windows.
    Memory grows with accumulated samples unless window_size is set.
    """

    def __init__(self, window_size=None, pos_label=1):
        if window_size is not None and (
            isinstance(window_size, (bool, np.bool_))
            or not isinstance(window_size, (int, np.integer))
            or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer or None")

        positive = _labels([pos_label], "pos_label")
        self.pos_label = positive[0]
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Discard previous observations and return self."""
        self._true = None
        self._pred = None
        self._scores = None
        self._with_scores = None
        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        """Add matching label vectors and optional binary scores.

        Either supply scores on every update or omit them on every
        update until reset(). Inputs are copied into retained storage.
        """
        true = _labels(y_true_chunk, "y_true")
        pred = _labels(y_pred_chunk, "y_pred")

        if true.shape != pred.shape:
            raise ValueError("Expected one prediction per true label")

        # Prevent accidental conversion between numeric and text labels.
        text = true.dtype.kind in "US"
        if text != (pred.dtype.kind in "US"):
            raise TypeError("True and predicted label types must match")
        if self._true is not None:
            if text != (self._true.dtype.kind in "US"):
                raise TypeError("Label types must stay consistent across chunks")

        with_scores = y_score is not None
        if self._with_scores is not None:
            if with_scores != self._with_scores:
                raise ValueError("Supply scores consistently until reset()")

        scores = None
        if with_scores:
            scores = np.asarray(y_score)
            if scores.shape != true.shape:
                raise ValueError("Expected one score per true label")
            if scores.dtype.kind not in "biuf":
                raise TypeError("Scores must be real numeric values")
            scores = scores.astype(float)
            if not np.isfinite(scores).all():
                raise ValueError("Scores must be finite")
            if text != (np.asarray(self.pos_label).dtype.kind in "US"):
                raise TypeError("pos_label must match the label type")

        if self._true is None:
            combined_true = true.copy()
            combined_pred = pred.copy()
            combined_scores = scores
        else:
            combined_true = np.concatenate([self._true, true])
            combined_pred = np.concatenate([self._pred, pred])
            combined_scores = (
                np.concatenate([self._scores, scores])
                if with_scores else None
            )

        if self.window_size is not None:
            combined_true = combined_true[-self.window_size:]
            combined_pred = combined_pred[-self.window_size:]
            if with_scores:
                combined_scores = combined_scores[-self.window_size:]

        self._true = combined_true
        self._pred = combined_pred
        self._scores = combined_scores
        self._with_scores = with_scores
        return self

    def _auc(self):
        """Calculate binary AUC using average ranks for tied scores."""
        if self._scores is None:
            return float("nan")

        classes = np.unique(self._true)
        if classes.size != 2 or not np.any(classes == self.pos_label):
            return float("nan")

        positive = self._true == self.pos_label
        n_positive = np.count_nonzero(positive)
        n_negative = positive.size - n_positive

        order = np.argsort(self._scores, kind="stable")
        sorted_scores = self._scores[order]
        _, starts, counts = np.unique(
            sorted_scores, return_index=True, return_counts=True
        )

        # Ranks start at one; tied observations receive their mean rank.
        average_ranks = starts + (counts + 1) / 2.0
        sorted_ranks = np.repeat(average_ranks, counts)
        positive_rank_sum = np.sum(sorted_ranks[positive[order]])

        return float(
            (positive_rank_sum - n_positive * (n_positive + 1) / 2.0)
            / (n_positive * n_negative)
        )

    def result(self):
        """Return metrics for all retained samples.

        Confusion matrix rows are true classes; columns are predictions.
        The classes array defines their order. Returned arrays are new.
        """
        if self._true is None:
            raise RuntimeError("Call update before requesting results")

        classes, indices = np.unique(
            np.concatenate([self._true, self._pred]),
            return_inverse=True,
        )
        size = self._true.size
        n_classes = classes.size
        true_indices = indices[:size]
        pred_indices = indices[size:]

        matrix = np.bincount(
            true_indices * n_classes + pred_indices,
            minlength=n_classes**2,
        ).reshape(n_classes, n_classes)

        correct = np.diag(matrix)
        precision = _divide(correct, matrix.sum(axis=0))
        recall = _divide(correct, matrix.sum(axis=1))
        f1 = _divide(2 * precision * recall, precision + recall)

        return {
            "n_samples": int(size),
            "classes": classes,
            "confusion_matrix": matrix,
            "accuracy": float(correct.sum() / size),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "auc": self._auc(),
        }