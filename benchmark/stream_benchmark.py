"""Compare loop-based and vectorised decision tree split searches."""

from pathlib import Path
from time import perf_counter
import sys

import numpy as np


# Allow this script to import modules from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute_stream.tree import (
    _best_feature_split,
    _best_feature_split_vectorised,
)


def median_seconds(function, values, labels, repeats=5):
    """Return median execution time after an untimed warm-up call."""
    function(values, labels)
    timings = []

    for _ in range(repeats):
        started = perf_counter()
        function(values, labels)
        timings.append(perf_counter() - started)

    return float(np.median(timings))


def main():
    """Check equivalent results and save measured benchmark timings."""
    rng = np.random.default_rng(42)
    results = []

    print("Split-search benchmark: median of five timed runs")
    print("Rows | Loop seconds | Vectorised seconds | Speedup")

    for size in [50, 100, 200]:
        values = rng.uniform(0.0, 10.0, size=size)
        labels = np.where(values < 4.0, 0, 1)

        # Add label noise so the split search evaluates imperfect groups.
        noisy = rng.random(size) < 0.1
        labels[noisy] = 1 - labels[noisy]

        loop_threshold, loop_score = _best_feature_split(
            values, labels
        )
        vector_threshold, vector_score = (
            _best_feature_split_vectorised(values, labels)
        )

        np.testing.assert_allclose(loop_score, vector_score)

        if loop_threshold is None or vector_threshold is None:
            assert loop_threshold is None and vector_threshold is None
        else:
            # Compare the actual partitions, allowing harmless rounding.
            np.testing.assert_array_equal(
                values <= loop_threshold,
                values <= vector_threshold,
            )

        loop_seconds = median_seconds(
            _best_feature_split, values, labels
        )
        vector_seconds = median_seconds(
            _best_feature_split_vectorised, values, labels
        )
        speedup = loop_seconds / vector_seconds

        results.append([
            size, loop_seconds, vector_seconds, speedup
        ])

        print(
            f"{size:4d} | {loop_seconds:.6f} | "
            f"{vector_seconds:.6f} | {speedup:.2f}x"
        )

    output_path = (
        PROJECT_ROOT / "benchmark" / "stream_split_results.csv"
    )
    np.savetxt(
        output_path,
        np.asarray(results),
        delimiter=",",
        header="rows,loop_seconds,vectorised_seconds,speedup",
        comments="",
        fmt=["%d", "%.9f", "%.9f", "%.4f"],
    )

    print("Both implementations produced matching splits.")
    print("Results saved:", output_path)


if __name__ == "__main__":
    main()