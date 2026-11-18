"""Tests for accumulated streaming statistics."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats


def test_statistics_match_combined_data():
    """Unequal chunks match NumPy statistics on the complete data."""
    X = np.array([[1.0, 10.0], [3.0, 14.0], [8.0, 18.0]])
    stats = StreamingStats()
    stats.update_stats(X[:2])
    stats.update_stats(X[2:])

    result = stats.result()

    np.testing.assert_array_equal(result["count"], [3, 3])
    np.testing.assert_allclose(result["mean"], np.mean(X, axis=0))
    np.testing.assert_allclose(
        result["variance"], np.var(X, axis=0)
    )
    assert stats.n_samples_seen_ == 3


def test_missing_values_are_ignored():
    """Each feature counts only its observed values."""
    stats = StreamingStats()
    stats.update_stats([[1.0, np.nan], [3.0, 10.0]])
    stats.update_stats([[5.0, 14.0]])

    result = stats.result()

    np.testing.assert_array_equal(result["count"], [3, 2])
    np.testing.assert_allclose(result["mean"], [3.0, 12.0])
    np.testing.assert_allclose(
        result["variance"], [8.0 / 3.0, 4.0]
    )


def test_entirely_missing_feature():
    """An unobserved feature returns zero count and NaN statistics."""
    stats = StreamingStats()
    stats.update_stats([[1.0, np.nan], [3.0, np.nan]])

    result = stats.result()

    assert result["count"][1] == 0
    assert np.isnan(result["mean"][1])
    assert np.isnan(result["variance"][1])
    assert np.isnan(stats.quantile()[1])


def test_quantiles_include_previous_chunks():
    """Exact quantiles use observations from every received chunk."""
    stats = StreamingStats()
    stats.update_stats([[1.0], [100.0], [np.nan]])
    stats.update_stats([[3.0], [5.0]])

    np.testing.assert_allclose(stats.quantile(), [4.0])
    np.testing.assert_allclose(stats.quantile(0.0), [1.0])
    np.testing.assert_allclose(stats.quantile(1.0), [100.0])


def test_histogram_includes_previous_chunks():
    """Histogram bins count values from all chunks and ignore NaN."""
    stats = StreamingStats()
    stats.update_stats([[1.0], [np.nan], [3.0]])
    stats.update_stats([[5.0], [6.0]])

    counts, edges = stats.histogram(bins=[0, 2, 4, 6])

    np.testing.assert_array_equal(counts, [1, 1, 2])
    np.testing.assert_array_equal(edges, [0, 2, 4, 6])


def test_constant_feature_has_zero_variance():
    """Repeated identical values have population variance zero."""
    stats = StreamingStats()
    stats.update_stats([[7.0], [7.0]])
    stats.update_stats([[7.0]])

    np.testing.assert_allclose(stats.result()["variance"], [0.0])


def test_reset_discards_previous_data():
    """Reset permits a fresh stream with a different feature count."""
    stats = StreamingStats()
    stats.update_stats([[100.0]])
    stats.reset()

    with pytest.raises(RuntimeError):
        stats.result()

    stats.update_stats([[2.0, 4.0]])

    np.testing.assert_allclose(stats.result()["mean"], [2.0, 4.0])
    assert stats.n_samples_seen_ == 1


def test_changed_feature_count_is_rejected():
    """The fixed-column limitation is enforced without losing data."""
    stats = StreamingStats()
    stats.update_stats([[1.0]])

    with pytest.raises(ValueError):
        stats.update_stats([[2.0, 3.0]])

    np.testing.assert_allclose(stats.result()["mean"], [1.0])
    assert stats.n_samples_seen_ == 1


def test_inputs_and_results_are_independent():
    """Caller edits cannot change retained observations or statistics."""
    X = np.array([[1.0], [3.0]])
    stats = StreamingStats()
    stats.update_stats(X)
    X[:] = 100.0

    result = stats.result()
    result["mean"][:] = -100.0

    np.testing.assert_allclose(stats.result()["mean"], [2.0])
    np.testing.assert_allclose(stats.quantile(), [2.0])


def test_queries_require_data():
    """All statistical queries require at least one chunk."""
    stats = StreamingStats()

    with pytest.raises(RuntimeError):
        stats.result()

    with pytest.raises(RuntimeError):
        stats.quantile()

    with pytest.raises(RuntimeError):
        stats.histogram()


def test_invalid_chunks_are_rejected():
    """Empty, nonnumeric, and infinite input is rejected."""
    stats = StreamingStats()

    with pytest.raises(ValueError):
        stats.update_stats([])

    with pytest.raises(TypeError):
        stats.update_stats([["text"]])

    with pytest.raises(ValueError):
        stats.update_stats([[np.inf]])


def test_invalid_query_arguments():
    """Quantiles and histogram feature indices must be valid."""
    stats = StreamingStats()
    stats.update_stats([[1.0]])

    with pytest.raises(ValueError):
        stats.quantile(1.5)

    with pytest.raises(ValueError):
        stats.histogram(feature=1)