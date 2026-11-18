"""Preprocessing steps for data that arrives in chunks."""

import numpy as np

from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute.preprocessing import matrix
from numcompute.preprocessing import SimpleImputer as BatchSimpleImputer
from numcompute.preprocessing import OneHotEncoder as BatchOneHotEncoder

class StandardScaler(BatchStandardScaler):
    """Scale each feature using statistics accumulated across chunks."""

    def fit(self, X):
        """Start fresh and learn statistics from one training array."""
        X = matrix(X, allow_nan=True)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("cannot fit a feature containing only NaN values")

        self._count = np.zeros(X.shape[1], dtype=np.int64)
        self._m2 = np.zeros(X.shape[1])
        self.mean_ = np.zeros(X.shape[1])
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Update the running mean and variance using one chunk."""
        X = matrix(X, allow_nan=True)
        if hasattr(self, "_count") and X.shape[1] != self._count.size:
            raise ValueError("chunk must have the same number of features")

        valid = ~np.isnan(X)
        count = valid.sum(axis=0)
        chunk_mean = np.divide(
            np.nansum(X, axis=0),
            count,
            out=np.zeros(X.shape[1]),
            where=count > 0,
        )
        centered = np.where(valid, X - chunk_mean, 0.0)
        chunk_m2 = np.sum(centered**2, axis=0)

        if not hasattr(self, "_count"):
            self._count = np.zeros_like(count)
            self._m2 = np.zeros_like(chunk_mean)
            self.mean_ = np.zeros_like(chunk_mean)

        # Merge the existing statistics with the statistics of this chunk.
        total = self._count + count
        weight = np.divide(
            count, total, out=np.zeros_like(chunk_mean), where=total > 0
        )
        delta = chunk_mean - self.mean_
        self.mean_ += delta * weight
        self._m2 += chunk_m2 + delta**2 * self._count * weight
        self._count = total

        variance = np.divide(
            self._m2, total, out=np.zeros_like(self._m2), where=total > 0
        )
        self.scale_ = np.where(variance > 0, np.sqrt(variance), 1.0)
        self.n_features_in_ = X.shape[1]
        return self

class SimpleImputer(BatchSimpleImputer):
    """Update numeric replacement values as labelled or unlabelled chunks arrive.

    Mean uses running statistics. Exact median retains previous chunks.
    Until a feature has an observed value, its replacement is zero.
    """

    def fit(self, X):
        """Reset previous statistics and learn from one training array."""
        candidate = type(self)(
            strategy=self.strategy,
            fill_value=self.fill_value,
        )
        candidate.partial_fit(X)
        self.__dict__ = candidate.__dict__
        return self

    def partial_fit(self, X):
        """Update replacement values using a two-dimensional numeric chunk."""
        if self.strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median', or 'constant'")

        X = matrix(X, allow_nan=True)
        if hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Chunks must have the same number of features")

        if self.strategy == "constant":
            # Reuse the original validation of the constant fill value.
            BatchSimpleImputer.fit(self, X)

        elif self.strategy == "mean":
            if not hasattr(self, "_running"):
                self._running = StandardScaler()

            self._running.partial_fit(X)
            self.statistics_ = self._running.mean_.copy()

        else:
            previous = getattr(self, "_median_chunks", [])
            chunks = [*previous, X.copy()]
            values = np.concatenate(chunks, axis=0)
            observed = np.any(~np.isnan(values), axis=0)
            statistics = np.zeros(X.shape[1])

            # Calculate medians only for columns with observed values.
            statistics[observed] = np.nanmedian(
                values[:, observed], axis=0
            )
            self.statistics_ = statistics
            self._median_chunks = chunks

        self.n_features_in_ = X.shape[1]
        return self

class OneHotEncoder(BatchOneHotEncoder):
    """Learn categories incrementally and return dense one-hot arrays.

    Categories are sorted within each input feature. New categories can
    increase the output width and change encoded column positions.
    Only distinct categories are retained between updates.
    """

    def partial_fit(self, X):
        """Add categories from a non-empty two-dimensional chunk."""
        X = self._check_input(X)

        if not hasattr(self, "categories_"):
            return self.fit(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, "
                f"got {X.shape[1]}"
            )

        if X.dtype.kind != self._input_kind:
            raise TypeError(
                "category dtype kind must match the training data"
            )

        # Represent every known category without retaining previous chunks.
        rows = max(len(categories) for categories in self.categories_)
        previous = np.empty(
            (rows, self.n_features_in_),
            dtype=self._categories.dtype,
        )

        # Features can have different numbers of categories.
        for feature, categories in enumerate(self.categories_):
            previous[:, feature] = categories[0]
            previous[:len(categories), feature] = categories

        combined = np.concatenate([previous, X], axis=0)
        return self.fit(combined)