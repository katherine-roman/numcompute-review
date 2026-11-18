"""Tests for incremental preprocessing and streaming edge cases."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def test_scaler_matches_combined_data():
    """Unequal chunks produce the same statistics as combined data."""
    X = np.array([[1.0, 4.0], [3.0, 8.0], [8.0, 12.0]])
    scaler = StandardScaler()

    scaler.partial_fit(X[:2])
    scaler.partial_fit(X[2:])

    np.testing.assert_allclose(scaler.mean_, np.mean(X, axis=0))
    np.testing.assert_allclose(scaler.scale_, np.std(X, axis=0))


def test_scaler_ignores_nan_per_feature():
    """Each feature counts only its observed values."""
    scaler = StandardScaler()
    scaler.partial_fit([[1.0, np.nan], [3.0, 10.0]])
    scaler.partial_fit([[5.0, 14.0]])

    np.testing.assert_allclose(scaler.mean_, [3.0, 12.0])
    np.testing.assert_allclose(
        scaler.scale_, [np.sqrt(8.0 / 3.0), 2.0]
    )
    assert np.isnan(scaler.transform([[np.nan, 12.0]])[0, 0])


def test_scaler_constant_feature():
    """A constant feature uses a scale of one."""
    scaler = StandardScaler()
    scaler.partial_fit([[7.0], [7.0]])

    np.testing.assert_allclose(scaler.scale_, [1.0])
    np.testing.assert_allclose(scaler.transform([[7.0]]), [[0.0]])


def test_scaler_recovers_from_missing_first_chunk():
    """An initially missing feature learns from later observations."""
    scaler = StandardScaler()
    scaler.partial_fit([[np.nan]])
    scaler.partial_fit([[4.0], [8.0]])

    np.testing.assert_allclose(scaler.mean_, [6.0])
    np.testing.assert_allclose(scaler.scale_, [2.0])


def test_scaler_fit_resets_statistics():
    """Calling fit starts a new set of statistics."""
    scaler = StandardScaler()
    scaler.partial_fit([[100.0], [200.0]])
    scaler.fit([[2.0], [4.0]])

    np.testing.assert_allclose(scaler.mean_, [3.0])
    np.testing.assert_allclose(scaler.scale_, [1.0])


def test_scaler_rejects_changed_feature_count():
    """Chunks must keep the same input width."""
    scaler = StandardScaler()
    scaler.partial_fit([[1.0]])

    with pytest.raises(ValueError):
        scaler.partial_fit([[1.0, 2.0]])


def test_imputer_mean_uses_all_observed_values():
    """Mean imputation weights observations across unequal chunks."""
    imputer = SimpleImputer()
    imputer.partial_fit([[1.0], [np.nan], [3.0]])
    imputer.partial_fit([[8.0]])

    np.testing.assert_allclose(imputer.statistics_, [4.0])
    np.testing.assert_allclose(
        imputer.transform([[np.nan], [5.0]]), [[4.0], [5.0]]
    )


def test_imputer_median_uses_previous_chunks():
    """Median is calculated from the combined observed values."""
    imputer = SimpleImputer(strategy="median")
    imputer.partial_fit([[1.0], [100.0], [np.nan]])
    imputer.partial_fit([[3.0], [5.0]])

    np.testing.assert_allclose(imputer.statistics_, [4.0])


def test_imputer_constant_fills_missing_feature():
    """Constant imputation also handles entirely missing features."""
    imputer = SimpleImputer(strategy="constant", fill_value=-2.0)
    imputer.partial_fit([[np.nan], [np.nan]])

    np.testing.assert_allclose(
        imputer.transform([[np.nan], [5.0]]), [[-2.0], [5.0]]
    )


def test_imputer_missing_feature_then_observed():
    """The zero fallback is replaced when observations arrive."""
    imputer = SimpleImputer()
    imputer.partial_fit([[np.nan]])

    np.testing.assert_allclose(imputer.statistics_, [0.0])

    imputer.partial_fit([[6.0], [10.0]])

    np.testing.assert_allclose(imputer.statistics_, [8.0])


def test_imputer_fit_resets_statistics():
    """Calling fit discards earlier observations."""
    imputer = SimpleImputer()
    imputer.partial_fit([[100.0]])
    imputer.fit([[2.0], [4.0]])

    np.testing.assert_allclose(imputer.statistics_, [3.0])


def test_imputer_rejects_invalid_strategy():
    """Unsupported strategies raise a clear error."""
    with pytest.raises(ValueError):
        SimpleImputer(strategy="invalid").partial_fit([[1.0]])


def test_encoder_adds_categories():
    """New categories are included without losing previous ones."""
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.partial_fit([["red"], ["blue"]])
    encoder.partial_fit([["green"]])

    np.testing.assert_array_equal(
        encoder.categories_[0], ["blue", "green", "red"]
    )
    np.testing.assert_array_equal(
        encoder.transform([["green"], ["red"], ["yellow"]]),
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    )


def test_encoder_updates_multiple_features():
    """Features can have different numbers of categories."""
    encoder = OneHotEncoder()
    encoder.partial_fit([["blue", "small"], ["red", "small"]])
    encoder.partial_fit([["green", "large"]])

    assert encoder.n_features_out_ == 5
    np.testing.assert_array_equal(
        encoder.transform([["blue", "large"]]),
        [[1.0, 0.0, 0.0, 1.0, 0.0]],
    )


def test_encoder_repeated_categories_keep_width():
    """Repeated categories do not create extra output columns."""
    encoder = OneHotEncoder()
    encoder.partial_fit([["blue"], ["red"]])
    encoder.partial_fit([["blue"], ["blue"]])

    assert encoder.n_features_out_ == 2


def test_encoder_unknown_category_error():
    """The error policy rejects categories not yet learned."""
    encoder = OneHotEncoder(handle_unknown="error")
    encoder.partial_fit([["blue"]])

    with pytest.raises(ValueError):
        encoder.transform([["green"]])