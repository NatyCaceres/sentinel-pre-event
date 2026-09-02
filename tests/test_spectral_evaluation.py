import numpy as np
import pytest

from src.spectral.evaluation import (
    SpectralEvaluationError,
    calculate_binary_metrics,
    load_expected_closed_test,
)


def test_binary_metrics_match_known_example() -> None:
    metrics = calculate_binary_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.4, 0.35, 0.8],
        threshold=0.5,
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.pr_auc == pytest.approx(5.0 / 6.0)
    assert metrics.brier == pytest.approx(0.158125)


def test_binary_metrics_reject_nonfinite_scores() -> None:
    with pytest.raises(
        SpectralEvaluationError,
        match="no finitos",
    ):
        calculate_binary_metrics(
            labels=[0, 1],
            scores=[0.2, np.nan],
            threshold=0.5,
        )


def test_expected_closed_test_metadata() -> None:
    sample_count, class_counts, metrics = (
        load_expected_closed_test()
    )

    assert sample_count == 244
    assert class_counts == {
        0: 118,
        1: 126,
    }
    assert metrics["accuracy"] == pytest.approx(
        0.7868852459016393
    )
    assert metrics["roc_auc"] == pytest.approx(
        0.8654156577885391
    )
    assert metrics["pr_auc"] == pytest.approx(
        0.8748153488315155
    )
