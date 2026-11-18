"""Column-wise statistics for numeric data arriving in chunks."""

import numpy as np

from numcompute_stream.preprocessing import StandardScaler


class StreamingStats:
    """Accumulate statistics for a fixed number of numeric features.

    NaN values are ignored. Infinite values are rejected.
    Variance is population variance, equivalent to NumPy's ddof=0.

    Mean and variance use running statistics. Chunks are also retained
    for exact quantiles and histograms, requiring O(total values) memory.
    Features with no observed values return NaN for mean and variance.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Discard all observations and return self."""
        self._running = StandardScaler()
        self._chunks = []
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        return self

    def update_stats(self, X_chunk):
        """Add a non-empty 2-D numeric chunk and return self.

        Each row is a sample and each column is a feature.
        All chunks must have the same number of features.
        Input arrays are copied before being retained.
        """
        X = np.asarray(X_chunk)
        if X.ndim != 2 or not all(X.shape):
            raise ValueError("X must contain rows and features")
        if X.dtype.kind not in "biuf":
            raise TypeError("X must contain real numeric values")

        X = X.astype(float, copy=True)
        if np.isinf(X).any():
            raise ValueError("X must not contain infinite values")
        if (
            self.n_features_in_ is not None
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError("Chunks must have the same feature count")

        self._running.partial_fit(X)
        self._chunks.append(X)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ += len(X)
        return self

    def _require_data(self):
        """Reject queries before receiving any observations."""
        if not self._chunks:
            raise RuntimeError("Call update_stats before requesting results")

    def result(self):
        """Return independent arrays of per-feature count, mean, and variance."""
        self._require_data()
        count = self._running._count.copy()
        mean = self._running.mean_.copy()
        mean[count == 0] = np.nan
        variance = np.divide(
            self._running._m2,
            count,
            out=np.full(count.shape, np.nan),
            where=count > 0,
        )
        return {
            "count": count,
            "mean": mean,
            "variance": variance,
        }

    def quantile(self, q=0.5):
        """Return exact per-feature quantiles, ignoring NaN.

        q must be a scalar in [0, 1]. The result has one value per
        feature. Entirely missing features return NaN.
        """
        self._require_data()
        q = np.asarray(q)
        if q.ndim != 0 or q.dtype.kind not in "biuf":
            raise TypeError("q must be a real numeric scalar")
        q = float(q)
        if not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("q must be between zero and one")

        values = np.concatenate(self._chunks, axis=0)
        observed = np.any(~np.isnan(values), axis=0)
        result = np.full(self.n_features_in_, np.nan)
        result[observed] = np.nanquantile(
            values[:, observed], q, axis=0
        )
        return result

    def histogram(self, feature=0, bins=10, value_range=None):
        """Return (counts, edges) for one feature across received chunks.

        bins accepts a positive integer or explicit increasing edges,
        following NumPy's histogram interface. NaN values are excluded.
        value_range optionally supplies the histogram's (minimum, maximum).
        """
        self._require_data()
        if (
            isinstance(feature, (bool, np.bool_))
            or not isinstance(feature, (int, np.integer))
        ):
            raise TypeError("feature must be an integer column index")
        if not 0 <= feature < self.n_features_in_:
            raise ValueError("feature index is out of range")

        values = np.concatenate(
            [chunk[:, feature] for chunk in self._chunks]
        )
        values = values[~np.isnan(values)]
        return np.histogram(values, bins=bins, range=value_range)