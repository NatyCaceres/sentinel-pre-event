from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.product_catalog_loader import load_product_catalog
from src.spectral.inference import (
    LoadedSpectralModel,
    load_spectral_model,
    predict_spectral_patch,
)
from src.spectral.patch import PROJECT_ROOT, load_spectral_patch
from src.spectral.storage import resolve_tif_path


DEFAULT_FINAL_TEST_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18_8ch_v2"
    / "provenance"
    / "final_test_summary.json"
)
MODEL_DISPLAY_NAME = "ResNet18 8CH"
METRIC_NAME_MAP = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier": "Brier",
}


class SpectralEvaluationError(Exception):
    """Error asociado a la reproducción del test final congelado."""


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
        }


@dataclass(frozen=True)
class ClosedTestEvaluation:
    sample_count: int
    class_counts: dict[int, int]
    threshold: float
    metrics: BinaryClassificationMetrics
    expected_metrics: dict[str, float]
    absolute_differences: dict[str, float]
    tolerance: float

    @property
    def matches_expected(self) -> bool:
        return all(
            difference <= self.tolerance
            for difference in self.absolute_differences.values()
        )


def calculate_binary_metrics(
    labels,
    scores,
    *,
    threshold: float,
) -> BinaryClassificationMetrics:
    """Calcula las métricas binarias usadas por el protocolo."""

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)

    if (
        labels_array.ndim != 1
        or scores_array.ndim != 1
        or labels_array.size != scores_array.size
        or labels_array.size == 0
    ):
        raise SpectralEvaluationError(
            "Etiquetas y scores deben ser vectores no vacíos "
            "con la misma longitud."
        )

    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise SpectralEvaluationError(
            "La evaluación requiere ambas clases binarias {0, 1}."
        )

    if not np.isfinite(scores_array).all():
        raise SpectralEvaluationError(
            "Los scores contienen valores no finitos."
        )

    if np.any(scores_array < 0.0) or np.any(scores_array > 1.0):
        raise SpectralEvaluationError(
            "Los scores deben pertenecer al intervalo [0, 1]."
        )

    if not 0.0 <= threshold <= 1.0:
        raise SpectralEvaluationError(
            "El umbral debe pertenecer al intervalo [0, 1]."
        )

    predictions = (scores_array >= threshold).astype(np.int64)
    true_positive = int(
        np.sum((labels_array == 1) & (predictions == 1))
    )
    false_positive = int(
        np.sum((labels_array == 0) & (predictions == 1))
    )
    false_negative = int(
        np.sum((labels_array == 1) & (predictions == 0))
    )

    accuracy = float(np.mean(predictions == labels_array))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / (true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    positive_scores = scores_array[labels_array == 1]
    negative_scores = scores_array[labels_array == 0]
    pairwise_differences = (
        positive_scores[:, None] - negative_scores[None, :]
    )
    roc_auc = float(
        np.mean(
            (pairwise_differences > 0.0)
            + 0.5 * (pairwise_differences == 0.0)
        )
    )

    descending_order = np.argsort(
        scores_array,
        kind="mergesort",
    )[::-1]
    sorted_scores = scores_array[descending_order]
    sorted_labels = labels_array[descending_order]
    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[
        distinct_indices,
        sorted_labels.size - 1,
    ]
    cumulative_positive = np.cumsum(sorted_labels)[
        threshold_indices
    ]
    predicted_positive = threshold_indices + 1
    precision_curve = cumulative_positive / predicted_positive
    recall_curve = cumulative_positive / positive_scores.size
    recall_increments = np.diff(np.r_[0.0, recall_curve])
    pr_auc = float(np.sum(recall_increments * precision_curve))
    brier = float(np.mean((scores_array - labels_array) ** 2))

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=brier,
    )


def load_expected_closed_test() -> tuple[
    int,
    dict[int, int],
    dict[str, float],
]:
    """Carga el cierre experimental esperado para ResNet18."""

    with DEFAULT_FINAL_TEST_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    model_metrics = next(
        item
        for item in summary["metrics_validation_threshold"]
        if item["Modelo"] == MODEL_DISPLAY_NAME
    )
    expected_metrics = {
        internal_name: float(model_metrics[external_name])
        for internal_name, external_name in METRIC_NAME_MAP.items()
    }
    class_counts = {
        int(label): int(count)
        for label, count in summary["class_counts_test"].items()
    }

    return int(summary["n_test"]), class_counts, expected_metrics


def evaluate_closed_test(
    loaded_model: LoadedSpectralModel | None = None,
    catalog: pd.DataFrame | None = None,
    *,
    tolerance: float = 1e-6,
) -> ClosedTestEvaluation:
    """Reproduce la evaluación sobre el test final congelado."""

    if loaded_model is None:
        loaded_model = load_spectral_model()

    if catalog is None:
        catalog = load_product_catalog()

    required_columns = {
        "patch_id",
        "etiqueta",
        "subsplit",
        "tif_filename",
    }
    missing_columns = required_columns - set(catalog.columns)

    if missing_columns:
        raise SpectralEvaluationError(
            "El catálogo no contiene las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    test_catalog = catalog.loc[
        catalog["subsplit"].astype(str).str.lower() == "test"
    ].copy()
    test_catalog = test_catalog.sort_values(
        "patch_id",
        kind="mergesort",
    )
    labels = pd.to_numeric(
        test_catalog["etiqueta"],
        errors="raise",
    ).astype(int).to_numpy()
    actual_class_counts = {
        label: int(np.sum(labels == label))
        for label in (0, 1)
    }
    expected_count, expected_class_counts, expected_metrics = (
        load_expected_closed_test()
    )

    if len(test_catalog) != expected_count:
        raise SpectralEvaluationError(
            "La cantidad de casos test no coincide con el cierre "
            f"experimental: {len(test_catalog)} != {expected_count}."
        )

    if actual_class_counts != expected_class_counts:
        raise SpectralEvaluationError(
            "La distribución de clases test no coincide con "
            "el cierre experimental."
        )

    scores = []

    for row in test_catalog.itertuples(index=False):
        patch = load_spectral_patch(
            resolve_tif_path(row.tif_filename)
        )
        prediction = predict_spectral_patch(patch, loaded_model)
        scores.append(prediction.class_1_score)

    threshold = float(
        loaded_model.contract["output"]["decision_threshold"]
    )
    metrics = calculate_binary_metrics(
        labels,
        scores,
        threshold=threshold,
    )
    absolute_differences = {
        name: abs(value - expected_metrics[name])
        for name, value in metrics.as_dict().items()
    }

    return ClosedTestEvaluation(
        sample_count=len(test_catalog),
        class_counts=actual_class_counts,
        threshold=threshold,
        metrics=metrics,
        expected_metrics=expected_metrics,
        absolute_differences=absolute_differences,
        tolerance=tolerance,
    )


def main() -> int:
    print("Evaluando 244 parches del test final congelado...")
    evaluation = evaluate_closed_test()
    print(f"Casos evaluados: {evaluation.sample_count}")
    print(f"Clases: {evaluation.class_counts}")
    print(f"Umbral: {evaluation.threshold:.6f}")

    for name, value in evaluation.metrics.as_dict().items():
        expected = evaluation.expected_metrics[name]
        difference = evaluation.absolute_differences[name]
        print(
            f"{name}: {value:.12f} "
            f"(esperado {expected:.12f}, "
            f"diferencia {difference:.3e})"
        )

    result = "COINCIDE" if evaluation.matches_expected else "NO COINCIDE"
    print(f"Resultado: {result}")
    return 0 if evaluation.matches_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
