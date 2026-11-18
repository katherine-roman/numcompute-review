"""Tests for chunk evaluation and stream logging."""

import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def make_trainer():
    """Train a small tree with a clear class boundary."""
    trainer = StreamTrainer(
        DecisionTreeClassifier(buffer_size=4)
    )
    trainer.fit_chunk(
        [[2.0], [3.0], [10.0], [19.0]],
        [0, 0, 1, 1],
    )
    return trainer


def test_chunk_accuracy_and_history():
    """Scoring returns accuracy and records it."""
    trainer = make_trainer()

    accuracy = trainer.score_chunk([[4.0], [15.0]], [0, 0])

    assert accuracy == pytest.approx(0.5)
    assert len(trainer.history) == 1
    assert trainer.history[0]["accuracy"] == pytest.approx(0.5)


def test_cumulative_accuracy_weights_rows():
    """Unequal chunks are weighted by samples, not chunk count."""
    trainer = make_trainer()

    trainer.score_chunk([[4.0], [15.0]], [0, 0])
    trainer.score_chunk([[15.0]], [1])

    assert trainer.history[1]["accuracy"] == pytest.approx(1.0)
    assert trainer.history[1]["cumulative_accuracy"] == pytest.approx(
        2.0 / 3.0
    )


def test_invalid_labels_do_not_add_history():
    """Rejected scoring calls leave the evaluation history unchanged."""
    trainer = make_trainer()

    with pytest.raises(ValueError):
        trainer.score_chunk([[4.0]], [])

    with pytest.raises(ValueError):
        trainer.score_chunk([[4.0]], [[0]])

    assert trainer.history == []


def test_prediction_label_count_must_match():
    """Each prediction needs a corresponding true label."""
    trainer = make_trainer()

    with pytest.raises(ValueError):
        trainer.score_chunk([[4.0], [15.0]], [0])

    assert trainer.history == []


def test_trainer_works_with_pipeline():
    """The trainer accepts a streaming pipeline as its estimator."""
    from numcompute_stream.pipeline import Pipeline
    from numcompute_stream.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=4)),
    ])
    trainer = StreamTrainer(pipeline)

    trainer.fit_chunk([[2.0], [3.0]], [0, 0])
    trainer.fit_chunk([[10.0], [19.0]], [1, 1])

    assert trainer.score_chunk(
        [[4.0], [15.0]], [0, 1]
    ) == pytest.approx(1.0)

def test_history_records_estimated_memory():
    """Evaluation logs a positive estimate of retained estimator memory."""
    trainer = make_trainer()
    trainer.score_chunk([[4.0], [15.0]], [0, 1])

    memory = trainer.history[-1]["estimator_bytes"]

    assert isinstance(memory, int)
    assert memory > 0