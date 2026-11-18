"""Tests for decision tree training and buffered updates."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def test_fit_learns_simple_boundary():
    """Separate two classes using a numeric feature."""
    tree = DecisionTreeClassifier()
    tree.fit([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1])

    np.testing.assert_array_equal(
        tree.predict([[1.5], [8.5]]), [0, 1]
    )


def test_predict_before_training():
    """Prediction requires a trained tree."""
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[1.0]])


def test_partial_fit_waits_for_buffer():
    """A buffer below its threshold does not train the first tree."""
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[1.0], [2.0]], [0, 0])

    with pytest.raises(RuntimeError):
        tree.predict([[1.0]])


def test_partial_fit_combines_chunks():
    """Training uses rows from both incoming chunks."""
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[1.0], [2.0]], [0, 0])
    tree.partial_fit([[8.0], [9.0]], [1, 1])

    np.testing.assert_array_equal(
        tree.predict([[1.0], [9.0]]), [0, 1]
    )


def test_predictions_unchanged_below_retraining_threshold():
    """Pending rows do not replace the active tree."""
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[1.0], [9.0]], [0, 1])
    tree.partial_fit([[1.0]], [1])

    np.testing.assert_array_equal(tree.predict([[1.0]]), [0])


def test_retraining_clears_previous_buffer():
    """The next full buffer replaces training on the previous buffer."""
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[1.0], [9.0]], [0, 1])
    tree.partial_fit([[1.0]], [2])
    tree.partial_fit([[9.0]], [2])

    np.testing.assert_array_equal(
        tree.predict([[1.0], [9.0]]), [2, 2]
    )


def test_oversized_chunk_uses_all_rows():
    """A chunk exceeding buffer_size is trained in full."""
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1])

    np.testing.assert_array_equal(
        tree.predict([[1.0], [9.0]]), [0, 1]
    )


def test_buffer_copies_caller_inputs():
    """Changing caller arrays must not alter pending training data."""
    tree = DecisionTreeClassifier(buffer_size=4)
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 0])
    tree.partial_fit(X, y)

    X[:] = 100.0
    y[:] = 1
    tree.partial_fit([[8.0], [9.0]], [1, 1])

    np.testing.assert_array_equal(
        tree.predict([[1.0], [9.0]]), [0, 1]
    )


def test_depth_zero_predicts_majority():
    """A tree with no splits predicts the majority class."""
    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit([[1.0], [2.0], [9.0]], [0, 0, 1])

    np.testing.assert_array_equal(tree.predict([[9.0]]), [0])


def test_min_samples_split_prevents_split():
    """Too few rows leave the root as a majority leaf."""
    tree = DecisionTreeClassifier(min_samples_split=4)
    tree.fit([[1.0], [2.0], [9.0]], [0, 0, 1])

    np.testing.assert_array_equal(tree.predict([[9.0]]), [0])


def test_constant_feature_predicts_majority():
    """Identical feature values cannot separate different labels."""
    tree = DecisionTreeClassifier()
    tree.fit([[5.0], [5.0], [5.0]], [1, 1, 0])

    np.testing.assert_array_equal(tree.predict([[5.0]]), [1])


def test_string_labels():
    """Class labels may be strings."""
    tree = DecisionTreeClassifier()
    tree.fit([[1.0], [9.0]], ["low", "high"])

    np.testing.assert_array_equal(
        tree.predict([[1.0], [9.0]]), ["low", "high"]
    )


def test_rejects_nonfinite_features():
    """Missing and infinite features must be handled before the tree."""
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit([[np.nan]], [0])

    with pytest.raises(ValueError):
        tree.fit([[np.inf]], [0])


def test_rejects_mismatched_labels():
    """Every training row needs one label."""
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[1.0], [2.0]], [0])


def test_rejects_changed_chunk_width():
    """Pending chunks must have matching feature counts."""
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[1.0]], [0])

    with pytest.raises(ValueError):
        tree.partial_fit([[2.0, 3.0]], [1])


def test_rejects_invalid_buffer_size():
    """The buffer size must be a positive integer."""
    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=0)

    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=True)
        