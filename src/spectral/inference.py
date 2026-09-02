from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch

from src.spectral.patch import (
    PROJECT_ROOT,
    SpectralPatch,
    validate_patch_array,
)


DEFAULT_INFERENCE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18_8ch_v2"
    / "runtime"
    / "inference_contract.json"
)


class SpectralInferenceError(Exception):
    """
    Error asociado a la preparación de un parche
    para inferencia espectral.
    """


@dataclass(frozen=True)
class LoadedSpectralModel:
    """
    Modelo validado junto con su contrato operacional.
    """

    model: torch.nn.Module
    contract: dict
    contract_path: Path
    checkpoint_path: Path
    device: torch.device


@dataclass(frozen=True)
class SpectralPrediction:
    """
    Resultado de clasificación a nivel de parche.
    """

    probabilities: tuple[float, float]
    class_1_score: float
    predicted_class: int
    threshold: float


def load_inference_contract(
    path: Path | str = DEFAULT_INFERENCE_CONTRACT_PATH,
) -> dict:
    """
    Carga el contrato portable utilizado para inferencia.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "No se encontró el contrato de inferencia: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _validate_checkpoint_metadata(
    checkpoint: dict,
    contract: dict,
) -> None:
    model_contract = contract["model"]
    input_contract = contract["input"]
    normalization = contract[
        "preprocessing"
    ]["normalization"]

    expected_values = {
        "model_name": model_contract["architecture"],
        "in_chans": model_contract["in_chans"],
        "num_classes": model_contract["num_classes"],
    }

    for key, expected in expected_values.items():
        if checkpoint.get(key) != expected:
            raise SpectralInferenceError(
                "Metadato incompatible en el checkpoint. "
                f"{key}: recibido {checkpoint.get(key)!r}; "
                f"esperado {expected!r}."
            )

    if tuple(checkpoint.get("channels", ())) != tuple(
        input_contract["channels"]
    ):
        raise SpectralInferenceError(
            "Los canales del checkpoint no coinciden "
            "con el contrato de inferencia."
        )

    checkpoint_config = checkpoint.get(
        "experiment_config",
        {},
    )

    if checkpoint_config.get("dropout_rate") != (
        model_contract["dropout_rate"]
    ):
        raise SpectralInferenceError(
            "El dropout del checkpoint no coincide "
            "con el contrato de inferencia."
        )

    if tuple(checkpoint.get("mean", ())) != tuple(
        normalization["mean"]
    ):
        raise SpectralInferenceError(
            "La media del checkpoint no coincide "
            "con el contrato de inferencia."
        )

    if tuple(checkpoint.get("std", ())) != tuple(
        normalization["std"]
    ):
        raise SpectralInferenceError(
            "La desviación estándar del checkpoint no "
            "coincide con el contrato de inferencia."
        )


def load_spectral_model(
    contract_path: Path | str = DEFAULT_INFERENCE_CONTRACT_PATH,
    *,
    device: str = "cpu",
) -> LoadedSpectralModel:
    """
    Verifica y carga el checkpoint ResNet18 para inferencia.

    La carga se restringe a pesos y estructuras simples. La
    arquitectura se crea sin descargar pesos preentrenados.
    """

    contract_path = Path(
        contract_path
    ).resolve()
    contract = load_inference_contract(
        contract_path
    )
    model_contract = contract["model"]
    checkpoint_path = (
        contract_path.parent
        / model_contract["checkpoint_relative_path"]
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "No se encontró el checkpoint del modelo: "
            f"{checkpoint_path}"
        )

    actual_sha256 = _calculate_sha256(
        checkpoint_path
    )
    expected_sha256 = model_contract[
        "checkpoint_sha256"
    ]

    if actual_sha256 != expected_sha256:
        raise SpectralInferenceError(
            "El SHA-256 del checkpoint no coincide con "
            "el contrato de inferencia."
        )

    device_object = torch.device(
        device
    )

    if (
        device_object.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise SpectralInferenceError(
            "Se solicitó CUDA, pero no está disponible."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise SpectralInferenceError(
            "No fue posible cargar el checkpoint "
            "en modo seguro."
        ) from error

    if not isinstance(checkpoint, dict):
        raise SpectralInferenceError(
            "El checkpoint no contiene una estructura válida."
        )

    _validate_checkpoint_metadata(
        checkpoint,
        contract,
    )

    model = timm.create_model(
        model_contract["architecture"],
        pretrained=False,
        in_chans=model_contract["in_chans"],
        num_classes=model_contract["num_classes"],
        drop_rate=model_contract["dropout_rate"],
    )

    try:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
    except (KeyError, RuntimeError) as error:
        raise SpectralInferenceError(
            "Los pesos no coinciden estrictamente con "
            "la arquitectura declarada."
        ) from error

    model.to(
        device_object
    )
    model.eval()

    return LoadedSpectralModel(
        model=model,
        contract=contract,
        contract_path=contract_path,
        checkpoint_path=checkpoint_path,
        device=device_object,
    )


def preprocess_spectral_patch(
    patch: SpectralPatch,
    contract_path: Path | str = DEFAULT_INFERENCE_CONTRACT_PATH,
) -> np.ndarray:
    """
    Reproduce el preprocesamiento definido en Fase 3.

    Aplica clipping por canal, imputa NaN con la media
    train-only y normaliza mediante Z-score. No modifica
    los datos originales del parche.
    """

    contract = load_inference_contract(
        contract_path
    )
    input_contract = contract["input"]
    preprocessing = contract["preprocessing"]

    expected_channels = tuple(
        input_contract["channels"]
    )

    if patch.channels != expected_channels:
        raise SpectralInferenceError(
            "El orden de canales del parche no coincide "
            "con el contrato de inferencia. "
            f"Recibido: {patch.channels}. "
            f"Esperado: {expected_channels}."
        )

    data = np.asarray(
        patch.data,
        dtype=np.float32,
    ).copy()

    validate_patch_array(
        data,
        {
            "shape": input_contract["shape"],
        },
    )

    clip_ranges = np.asarray(
        [
            preprocessing["clip_ranges"][channel]
            for channel in expected_channels
        ],
        dtype=np.float32,
    )
    lower_bounds = clip_ranges[:, 0, None, None]
    upper_bounds = clip_ranges[:, 1, None, None]

    np.clip(
        data,
        lower_bounds,
        upper_bounds,
        out=data,
    )

    normalization = preprocessing["normalization"]
    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )[:, None, None]
    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )[:, None, None]

    data = np.where(
        np.isnan(data),
        mean,
        data,
    )
    data = (
        data - mean
    ) / std

    if not np.isfinite(data).all():
        raise SpectralInferenceError(
            "El parche contiene valores no finitos "
            "después del preprocesamiento."
        )

    return np.ascontiguousarray(
        data,
        dtype=np.float32,
    )


def predict_spectral_patch(
    patch: SpectralPatch,
    loaded_model: LoadedSpectralModel,
) -> SpectralPrediction:
    """
    Clasifica un parche mediante el modelo ya cargado.

    El score informado corresponde a la probabilidad softmax
    de la clase 1 y no representa riesgo absoluto de incendio.
    """

    model_input = preprocess_spectral_patch(
        patch,
        loaded_model.contract_path,
    )
    tensor = torch.from_numpy(
        model_input
    ).unsqueeze(0).to(
        loaded_model.device
    )

    with torch.inference_mode():
        logits = loaded_model.model(
            tensor
        )

        if tuple(logits.shape) != (1, 2):
            raise SpectralInferenceError(
                "El modelo produjo una salida con forma "
                f"inesperada: {tuple(logits.shape)}."
            )

        probabilities_tensor = torch.softmax(
            logits,
            dim=1,
        )

    if not torch.isfinite(
        probabilities_tensor
    ).all():
        raise SpectralInferenceError(
            "El modelo produjo probabilidades no finitas."
        )

    probability_values = (
        probabilities_tensor[0]
        .detach()
        .cpu()
        .tolist()
    )
    probabilities = (
        float(probability_values[0]),
        float(probability_values[1]),
    )
    class_1_score = probabilities[1]
    threshold = float(
        loaded_model.contract[
            "output"
        ]["decision_threshold"]
    )
    predicted_class = int(
        class_1_score >= threshold
    )

    return SpectralPrediction(
        probabilities=probabilities,
        class_1_score=class_1_score,
        predicted_class=predicted_class,
        threshold=threshold,
    )
