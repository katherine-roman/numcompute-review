"""Tests for reusable streaming plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_metric_plot_values_and_labels():
    """Plot the supplied values against chunk numbers."""
    fig, ax = plot_metric_over_time(
        [0.5, 0.75, 1.0], "Accuracy over time", "Accuracy",
        show=False,
    )
    try:
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), [1, 2, 3]
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [0.5, 0.75, 1.0]
        )
        assert ax.get_title() == "Accuracy over time"
        assert ax.get_ylabel() == "Accuracy"
    finally:
        plt.close(fig)


def test_metric_plot_saves_image(tmp_path):
    """Saving creates a non-empty image file."""
    path = tmp_path / "accuracy.png"
    fig, ax = plot_metric_over_time(
        [0.5, 1.0], "Accuracy", "Accuracy",
        save_path=path, show=False,
    )
    try:
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close(fig)


def test_compare_models_plots_both_series():
    """Both models retain their values and legend labels."""
    fig, ax = compare_models(
        [0.5, 0.8], [0.6, 0.9], ["Small buffer", "Large buffer"],
        show=False,
    )
    try:
        assert len(ax.lines) == 2
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [0.5, 0.8]
        )
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), [0.6, 0.9]
        )
        assert ax.lines[0].get_label() == "Small buffer"
        assert ax.lines[1].get_label() == "Large buffer"
    finally:
        plt.close(fig)


def test_prediction_plot_supports_string_labels():
    """True and predicted classes share the same category positions."""
    fig, ax = plot_predictions_vs_ground_truth(
        ["low", "high"], ["high", "high"], show=False,
    )
    try:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert labels == ["high", "low"]
        np.testing.assert_array_equal(
            ax.collections[0].get_offsets(),
            [[1, 1], [2, 0]],
        )
        np.testing.assert_array_equal(
            ax.collections[1].get_offsets(),
            [[1, 0], [2, 0]],
        )
    finally:
        plt.close(fig)


def test_metric_plot_preserves_nan_gap():
    """Undefined metric values remain gaps in the plotted series."""
    fig, ax = plot_metric_over_time(
        [0.5, np.nan, 0.9], "Metric", "Value", show=False,
    )
    try:
        assert np.isnan(ax.lines[0].get_ydata()[1])
    finally:
        plt.close(fig)


def test_comparison_rejects_mismatched_lengths():
    """Compared series must describe matching chunks."""
    with pytest.raises(ValueError):
        compare_models(
            [0.5], [0.6, 0.8], ["A", "B"], show=False
        )


def test_metric_plot_rejects_invalid_values():
    """Empty or infinite metric data is rejected."""
    with pytest.raises(ValueError):
        plot_metric_over_time([], "Metric", "Value", show=False)

    with pytest.raises(ValueError):
        plot_metric_over_time(
            [np.inf], "Metric", "Value", show=False
        )


def test_prediction_plot_rejects_mismatched_lengths():
    """Every true label needs a corresponding prediction."""
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(
            [0, 1], [0], show=False
        )