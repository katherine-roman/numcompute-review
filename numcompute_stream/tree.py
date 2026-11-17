import numpy as np


def _gini(y):
    """Return the Gini impurity of a non-empty 1-D sequence of class labels."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    return float(1.0 - np.sum(proportions**2))

def _split_impurity(left_y, right_y):
    """Return the size-weighted Gini impurity of two non-empty label groups."""
    left_size = len(left_y)
    right_size = len(right_y)
    total_size = left_size + right_size

    return (
        left_size * _gini(left_y) + right_size * _gini(right_y)
    ) / total_size

def _best_feature_split(values, y):
    """Find the best threshold for one feature.

    `values` and `y` are 1-D NumPy arrays with one entry per sample.
    Return (threshold, weighted impurity), with threshold set to None
    when no split improves on the original group.
    """
    unique_values = np.unique(values)
    best_threshold = None
    best_score = _gini(y)

    # Each midpoint separates two consecutive distinct feature values.
    for lower, upper in zip(unique_values[:-1], unique_values[1:]):
        threshold = lower / 2.0 + upper / 2.0
        left = values <= threshold
        if not left.any() or left.all():
            continue

        score = _split_impurity(y[left], y[~left])
        if score < best_score - 1e-12:
            best_threshold = float(threshold)
            best_score = score

    return best_threshold, best_score

def _best_feature_split_vectorised(values, y):
    """Return (threshold, weighted Gini) for one finite numeric feature.

    values and y are non-empty 1-D arrays with one entry per sample.
    Return (None, parent impurity) if no split improves the group.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_labels = y[order]
    parent_score = _gini(y)

    # A threshold can separate adjacent rows only if their values differ.
    split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
    if split_positions.size == 0:
        return None, parent_score

    # Count each class up to every candidate split in one NumPy operation.
    classes = np.unique(y)
    class_matches = sorted_labels[:, None] == classes
    cumulative_counts = np.cumsum(class_matches, axis=0)
    left_counts = cumulative_counts[split_positions]
    right_counts = cumulative_counts[-1] - left_counts

    left_sizes = split_positions + 1
    right_sizes = len(y) - left_sizes
    left_proportions = left_counts / left_sizes[:, None]
    right_proportions = right_counts / right_sizes[:, None]

    left_gini = 1.0 - np.sum(left_proportions**2, axis=1)
    right_gini = 1.0 - np.sum(right_proportions**2, axis=1)
    scores = (left_sizes * left_gini + right_sizes * right_gini) / len(y)

    best = np.argmin(scores)
    if scores[best] >= parent_score - 1e-12:
        return None, parent_score

    position = split_positions[best]
    lower = sorted_values[position]
    upper = sorted_values[position + 1]
    threshold = lower / 2.0 + upper / 2.0
    # If rounding lands on the upper value, keep the same split with <=.
    if not lower <= threshold < upper:
        threshold = lower

    return float(threshold), float(scores[best])


class DecisionTreeClassifier:
    """Classify rows using a depth-limited decision tree with Gini impurity.

    max_depth limits the number of splits on a path from the root.
    min_samples_split is the minimum number of rows needed to try a split.
    max_features limits the number of randomly selected columns per split.
    With max_features=None, every column is considered.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if max_features is not None and (
            not isinstance(max_features, (int, np.integer)) or max_features < 1
        ):
            raise ValueError("max_features must be None or a positive integer")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self._tree = None

    def fit(self, X, y):
        """Train from scratch on 2-D numeric X and 1-D class labels y.

        X must contain at least one row and one feature, with finite values.
        y must contain one label per row. Return the fitted classifier.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a 2-D array with at least one row and feature")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("y must contain finite class labels")
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError("max_features cannot exceed the number of columns in X")

        # A fixed seed makes feature sampling repeatable across fits.
        rng = np.random.default_rng(0)
        self._tree = self._build(X, y, depth=0, rng=rng)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self

    def _build(self, X, y, depth, rng):
        """Build a node by choosing a split or storing a majority label."""
        labels, counts = np.unique(y, return_counts=True)
        leaf = {"label": labels[np.argmax(counts)]}

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(labels) == 1:
            return leaf

        features = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < len(features):
            features = np.sort(rng.choice(features, self.max_features, replace=False))

        best_feature = None
        best_threshold = None
        best_score = _gini(y)

        for feature in features:
            threshold, score = _best_feature_split_vectorised(X[:, feature], y)
            if threshold is not None and score < best_score - 1e-12:
                best_feature = feature
                best_threshold = threshold
                best_score = score

        if best_feature is None:
            return leaf

        left = X[:, best_feature] <= best_threshold
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(X[left], y[left], depth + 1, rng),
            "right": self._build(X[~left], y[~left], depth + 1, rng),
        }

    def predict(self, X):
        """Return one class label per row of 2-D X using the fitted tree."""
        if self._tree is None:
            raise RuntimeError("Train the tree before calling predict()")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2-D with the same feature count used in fit()")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        predictions = np.empty(len(X), dtype=self.classes_.dtype)
        self._predict_node(self._tree, X, np.arange(len(X)), predictions)
        return predictions

    def _predict_node(self, node, X, rows, predictions):
        """Route a group of row indices through the tree."""
        if rows.size == 0:
            return
        if "label" in node:
            predictions[rows] = node["label"]
            return

        left = X[rows, node["feature"]] <= node["threshold"]
        self._predict_node(node["left"], X, rows[left], predictions)
        self._predict_node(node["right"], X, rows[~left], predictions)