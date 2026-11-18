"""Connect incremental preprocessing to a buffered decision tree."""

from copy import deepcopy

import numpy as np


class Pipeline:
    """Update named transformers and train a buffered tree on their output."""

    def __init__(self, steps):
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("Pipeline requires a final decision tree")

        self._transformers = [step for _, step in self.steps[:-1]]
        self._tree = self.steps[-1][1]
        self._active_transformers = None
        self._pending_X = []
        self._pending_y = []
        self._pending_rows = 0
        self._n_features = None

    def partial_fit(self, X_chunk, y_chunk):
        """Update the transformers and retrain when enough rows arrive."""
        X = np.asarray(X_chunk)
        y = np.asarray(y_chunk)
        if X.ndim != 2 or not all(X.shape):
            raise ValueError("X must contain rows and features")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one label per row")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError("Chunks must have the same number of input features")

        transformers = deepcopy(self._transformers)
        values = X
        for step in transformers:
            step.partial_fit(values)
            values = step.transform(values)

        total_rows = self._pending_rows + len(X)
        if total_rows >= self._tree.buffer_size:
            # Apply one set of fitted transformations to the whole training batch.
            values = np.concatenate([*self._pending_X, X], axis=0)
            labels = np.concatenate([*self._pending_y, y])
            for step in transformers:
                values = step.transform(values)

            self._tree.fit(values, labels)
            self._active_transformers = transformers
            self._pending_X.clear()
            self._pending_y.clear()
            self._pending_rows = 0
        else:
            self._pending_X.append(X.copy())
            self._pending_y.append(y.copy())
            self._pending_rows = total_rows

        self._transformers = transformers
        self._n_features = X.shape[1]
        return self

    def predict(self, X):
        """Predict using the transformations fitted when the tree last trained."""
        if self._active_transformers is None:
            raise RuntimeError("Train the pipeline on enough rows before predicting")

        values = np.asarray(X)
        for step in self._active_transformers:
            values = step.transform(values)
        return self._tree.predict(values)