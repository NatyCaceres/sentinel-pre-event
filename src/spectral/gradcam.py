"""Grad-CAM por parche para el ResNet18 8CH validado del prototipo.

No cambia pesos, gradientes de parámetros, modos ni registra hooks en el
modelo compartido. El RGB y su contraste no intervienen en este cálculo.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from timm.models.resnet import ResNet

from src.spectral.inference import (
    LoadedSpectralModel,
    SpectralInferenceError,
    preprocess_spectral_patch,
)
from src.spectral.patch import SpectralPatch


class SpectralGradCAMError(SpectralInferenceError):
    """No se pudo obtener una explicación espacial válida."""


@dataclass(frozen=True)
class SpectralGradCAM:
    heatmap: np.ndarray
    target_class: int
    target_probability: float
    native_shape: tuple[int, int]
    target_layer: str = "layer2.1"
    normalization: str = "per_patch_minmax_after_bilinear_resize"


def calculate_gradcam(
    patch: SpectralPatch,
    loaded_model: LoadedSpectralModel,
    *,
    target_class: int,
) -> SpectralGradCAM:
    """Explica el logit de una clase explícita, sin decidir con otro umbral.

Reproduce layer2[-1], media espacial de gradientes, suma ponderada,
ReLU, interpolación bilineal (align_corners=False) y min-max del notebook.
El mapa 64x64 es relativo a este parche; no es probabilidad por píxel.
Un mapa constante se devuelve como ceros, sin inventar contraste.

El recorrido explícito corresponde a timm ResNet.forward_features y
forward_head. Evita hooks globales en el modelo cacheado por Streamlit.
Las pruebas comparan este recorrido con el forward normal y el notebook.
"""
    if isinstance(target_class, bool) or not isinstance(target_class, (int, np.integer)) or target_class not in (0, 1):
        raise SpectralGradCAMError("La clase objetivo debe ser 0 o 1.")

    model = loaded_model.model
    if (
        loaded_model.contract["model"]["architecture"] != "resnet18"
        or not isinstance(model, ResNet)
        or len(model.layer2) != 2
    ):
        raise SpectralGradCAMError("Grad-CAM requiere el ResNet18 del contrato.")
    if any(module.training for module in model.modules()):
        raise SpectralGradCAMError("El modelo debe estar en modo evaluación.")

    data = preprocess_spectral_patch(patch, loaded_model.contract_path)

    # Se habilita autograd solo para esta operación, incluso si el llamador
    # está en inference_mode. No se necesita derivar el prefijo de la red.
    with torch.inference_mode(False), torch.enable_grad():
        tensor = torch.from_numpy(data).unsqueeze(0).to(loaded_model.device)
        with torch.no_grad():
            features = model.maxpool(model.act1(model.bn1(model.conv1(tensor))))
            features = model.layer2(model.layer1(features))

        activation = features.detach().requires_grad_(True)
        logits = model.forward_head(model.layer4(model.layer3(activation)))
        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
            raise SpectralGradCAMError("El modelo produjo logits inválidos.")

        # autograd.grad devuelve el gradiente local sin acumular parameter.grad.
        gradient, = torch.autograd.grad(logits[0, int(target_class)], activation)
        if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
            raise SpectralGradCAMError("Activaciones o gradientes no finitos.")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=data.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = cam[0, 0].detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(heatmap).all():
            raise SpectralGradCAMError("El mapa Grad-CAM contiene valores no finitos.")
        lower, upper = float(heatmap.min()), float(heatmap.max())
        if upper > lower:
            heatmap = (heatmap - lower) / (upper - lower)
        else:
            heatmap.fill(0.0)

        probability = float(torch.softmax(logits.detach(), dim=1)[0, int(target_class)].cpu())

    return SpectralGradCAM(
        heatmap=heatmap,
        target_class=int(target_class),
        target_probability=probability,
        native_shape=tuple(activation.shape[-2:]),
    )
