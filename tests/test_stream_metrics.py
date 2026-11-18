"""Tests for accumulated and rolling classification metrics."""

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingMetrics


def test_accumulated_classification_metrics():
    """Metrics include observations from both chunks."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1])
    metrics.update([1, 0], [0, 0])

    result = metrics.result()

    assert result["n_samples"] == 4
    assert result["accuracy"] == pytest.approx(0.75)
    np.testing.assert_array_equal(
        result["confusion_matrix"], [[2, 0], [1, 1]]
    )
    np.testing.assert_allclose(result["precision"], [2 / 3, 1])
    np.testing.assert_allclose(result["recall"], [1, 0.5])
    np.testing.assert_allclose(result["f1"], [0.8, 2 / 3])
    assert result["macro_precision"] == pytest.approx(5 / 6)
    assert result["macro_recall"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(11 / 15)


def test_multiclass_string_labels():
    """Confusion matrix ordering follows sorted string classes."""
    metrics = StreamingMetrics()
    metrics.update(["a", "b", "c"], ["a", "c", "c"])

    result = metrics.result()

    np.testing.assert_array_equal(result["classes"], ["a", "b", "c"])
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
    )
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_undefined_precision_uses_zero():
    """A class with no predicted examples has precision zero."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 0])

    result = metrics.result()

    np.testing.assert_allclose(result["precision"], [0.5, 0.0])
    np.testing.assert_allclose(result["recall"], [1.0, 0.0])


def test_rolling_window_trims_part_of_chunk():
    """Only the latest samples remain, even across chunk boundaries."""
    metrics = StreamingMetrics(window_size=3)
    metrics.update([0, 1], [1, 1])
    metrics.update([0, 1], [0, 0])

    result = metrics.result()

    assert result["n_samples"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(
        result["confusion_matrix"], [[1, 0], [1, 1]]
    )


def test_chunk_larger_than_window():
    """An oversized chunk retains only its latest rows."""
    metrics = StreamingMetrics(window_size=2)
    metrics.update([0, 0, 1, 1], [1, 1, 1, 1])

    result = metrics.result()

    assert result["n_samples"] == 2
    assert result["accuracy"] == pytest.approx(1.0)


def test_auc_accumulates_across_chunks():
    """AUC uses scores from all received chunks."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1], y_score=[0.1, 0.9])
    metrics.update([1, 0], [0, 0], y_score=[0.4, 0.2])

    assert metrics.result()["auc"] == pytest.approx(1.0)


def test_auc_handles_tied_scores():
    """Equal positive and negative scores receive half credit."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 0], y_score=[0.5, 0.5])

    assert metrics.result()["auc"] == pytest.approx(0.5)


def test_auc_reversed_ranking():
    """Ranking the negative above the positive gives AUC zero."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [1, 0], y_score=[0.9, 0.1])

    assert metrics.result()["auc"] == pytest.approx(0.0)


def test_auc_with_string_positive_label():
    """The caller can select a string positive class."""
    metrics = StreamingMetrics(pos_label="yes")
    metrics.update(
        ["no", "yes"], ["no", "yes"], y_score=[0.1, 0.9]
    )

    assert metrics.result()["auc"] == pytest.approx(1.0)


def test_auc_uses_rolling_window():
    """AUC excludes observations that leave the window."""
    metrics = StreamingMetrics(window_size=2)
    metrics.update([0, 1], [0, 1], y_score=[0.1, 0.9])
    metrics.update([0, 1], [1, 0], y_score=[0.9, 0.1])

    assert metrics.result()["auc"] == pytest.approx(0.0)


def test_auc_undefined_without_scores_or_two_classes():
    """Unavailable AUC is reported as NaN."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1])
    assert np.isnan(metrics.result()["auc"])

    metrics.reset()
    metrics.update([1, 1], [1, 1], y_score=[0.4, 0.8])
    assert np.isnan(metrics.result()["auc"])


def test_reset_discards_previous_samples():
    """Reset starts a fresh evaluation."""
    metrics = StreamingMetrics()
    metrics.update([0], [1])
    metrics.reset()

    with pytest.raises(RuntimeError):
        metrics.result()

    metrics.update([1], [1])
    assert metrics.result()["n_samples"] == 1
    assert metrics.result()["accuracy"] == pytest.approx(1.0)


def test_invalid_update_preserves_previous_results():
    """A rejected update must not change accumulated observations."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1])

    with pytest.raises(ValueError):
        metrics.update([0, 1], [0])

    assert metrics.result()["n_samples"] == 2
    assert metrics.result()["accuracy"] == pytest.approx(1.0)


def test_scores_must_be_supplied_consistently():
    """Updates cannot mix scored and unscored observations."""
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 1], y_score=[0.1, 0.9])

    with pytest.raises(ValueError):
        metrics.update([0], [0])


def test_nonfinite_scores_are_rejected():
    """AUC scores must be finite."""
    metrics = StreamingMetrics()

    with pytest.raises(ValueError):
        metrics.update([0, 1], [0, 1], y_score=[0.1, np.nan])


def test_invalid_window_size():
    """A rolling window requires a positive integer."""
    with pytest.raises(ValueError):
        StreamingMetrics(window_size=0)

    with pytest.raises(ValueError):
        StreamingMetrics(window_size=True)