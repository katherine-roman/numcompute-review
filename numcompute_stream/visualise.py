"""Reusable plots for streaming model evaluation."""

import numpy as np
import matplotlib.pyplot as plt


def _metric_values(values):
    """Validate a non-empty vector of real metric values.

    NaN is allowed for metrics that are temporarily undefined.
    Infinite values are rejected.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Metric values must be a non-empty 1-D array")
    if values.dtype.kind not in "biuf":
        raise TypeError("Metric values must be real numbers")

    values = values.astype(float)
    if np.isinf(values).any():
        raise ValueError("Metric values must not contain infinity")
    return values


def _finish(fig, save_path, show):
    """Arrange the plot, optionally save it, and optionally display it."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_metric_over_time(
    metric_values, title, ylabel, *, save_path=None, show=True
):
    """Plot one metric across chunks numbered from one.

    NaN values appear as gaps. Return (figure, axes).
    save_path may name an image file in an existing directory.
    """
    values = _metric_values(metric_values)
    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path, show)
    return fig, ax


def compare_models(
    metric1, metric2, labels, *, save_path=None, show=True
):
    """Compare the same metric for two models over matching chunks.

    labels contains two model names. Both metric arrays must have
    the same length. Return (figure, axes).
    """
    values1 = _metric_values(metric1)
    values2 = _metric_values(metric2)
    if values1.shape != values2.shape:
        raise ValueError("Both models must have one value per matching chunk")
    if (
        isinstance(labels, str)
        or len(labels) != 2
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError("labels must contain two model names")

    chunks = np.arange(1, len(values1) + 1)
    fig, ax = plt.subplots()
    ax.plot(chunks, values1, marker="o", label=labels[0])
    ax.plot(chunks, values2, marker="s", label=labels[1])
    ax.set_title("Model comparison over time")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true, y_pred, *, save_path=None, show=True
):
    """Compare true and predicted class labels for one chunk.

    Accept matching non-empty vectors of numeric or string labels.
    Labels are mapped to shared category positions on the vertical
    axis; those positions do not imply numeric distances.
    Return (figure, axes).
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if (
        true.ndim != 1
        or true.size == 0
        or pred.shape != true.shape
    ):
        raise ValueError("Provide matching non-empty 1-D label arrays")

    if true.dtype.kind not in "biufUS" or pred.dtype.kind not in "biufUS":
        raise TypeError("Labels must be real numbers or strings")
    if (true.dtype.kind in "US") != (pred.dtype.kind in "US"):
        raise TypeError("True and predicted label types must match")

    if true.dtype.kind in "biuf":
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise ValueError("Numeric labels must be finite")

    classes, indices = np.unique(
        np.concatenate([true, pred]), return_inverse=True
    )
    samples = np.arange(1, len(true) + 1)

    fig, ax = plt.subplots()
    ax.scatter(
        samples, indices[:len(true)],
        marker="o", facecolors="none", edgecolors="tab:blue",
        s=80, label="Ground truth",
    )
    ax.scatter(
        samples, indices[len(true):],
        marker="x", color="tab:orange", label="Prediction",
    )
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes.astype(str))
    ax.set_title("Predictions vs ground truth")
    ax.set_xlabel("Sample in chunk")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path, show)
    return fig, ax