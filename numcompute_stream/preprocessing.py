"""Preprocessing steps for data that arrives in chunks."""

import numpy as np

from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute.preprocessing import matrix


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