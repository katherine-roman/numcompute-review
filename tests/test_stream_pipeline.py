"""Tests for preprocessing and buffered training in a streaming pipeline."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import (
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute_stream.tree import DecisionTreeClassifier


def test_numeric_pipeline_handles_missing_values():
    """Impute, scale, and train using two incoming chunks."""
    pipe = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=4)),
    ])

    pipe.partial_fit([[2.0], [np.nan]], [0, 0])
    pipe.partial_fit([[10.0], [19.0]], [1, 1])

    np.testing.assert_array_equal(
        pipe.predict([[2.0], [15.0]]),
        [0, 1],
    )


def test_pipeline_waits_for_enough_training_rows():
    """Prediction is unavailable before the first buffer fills."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=4)),
    ])

    pipe.partial_fit([[2.0], [3.0]], [0, 0])

    with pytest.raises(RuntimeError):
        pipe.predict([[2.0]])


def test_predictions_stay_stable_between_retrains():
    """A new pending chunk must not change the active model's scaling."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=4)),
    ])

    pipe.partial_fit([[2.0], [3.0]], [0, 0])
    pipe.partial_fit([[10.0], [19.0]], [1, 1])
    before = pipe.predict([[4.0], [15.0]])

    pipe.partial_fit([[1000.0]], [1])

    np.testing.assert_array_equal(
        pipe.predict([[4.0], [15.0]]),
        before,
    )


def test_pipeline_retrains_after_categories_expand():
    """Retraining accepts the encoder's increased output width."""
    pipe = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("model", DecisionTreeClassifier(buffer_size=2)),
    ])

    pipe.partial_fit([["blue"], ["red"]], [0, 1])
    np.testing.assert_array_equal(
        pipe.predict([["blue"], ["red"]]),
        [0, 1],
    )

    pipe.partial_fit([["green"], ["green"]], [2, 2])
    np.testing.assert_array_equal(
        pipe.predict([["green"]]),
        [2],
    )