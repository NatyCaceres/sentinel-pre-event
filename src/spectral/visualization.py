from __future__ import annotations

from typing import Any

import numpy as np

from src.spectral.patch import (
    SpectralPatch,
    SpectralPatchError,
)


DEFAULT_LOWER_PERCENTILE = 2.0
DEFAULT_UPPER_PERCENTILE = 98.0

SUPPORTED_INDICES = {
    "NDMI",
    "NBR",
}


class SpectralVisualizationError(Exception):
    """
    Error asociado exclusivamente a la preparación
    visual de información espectral.
    """


def percentile_stretch(
    channel: np.ndarray,
    *,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> np.ndarray:
    """
    Aplica un contraste robusto para visualización.

    Los percentiles se calculan únicamente sobre valores
    finitos y el resultado se escala al intervalo [0, 1].

    Esta transformación es solo visual y no debe utilizarse
    como preprocessing del modelo.
    """

    if channel.ndim != 2:
        raise SpectralVisualizationError(
            "El canal utilizado para visualización "
            "debe ser bidimensional."
        )

    if not (
        0.0 <= lower_percentile
        < upper_percentile
        <= 100.0
    ):
        raise SpectralVisualizationError(
            "Los percentiles de visualización "
            "no son válidos."
        )

    channel = channel.astype(
        np.float32,
        copy=False,
    )

    finite_mask = np.isfinite(
        channel
    )

    if not finite_mask.any():
        raise SpectralVisualizationError(
            "El canal no contiene valores finitos "
            "para su visualización."
        )

    finite_values = channel[
        finite_mask
    ]

    lower_value = float(
        np.percentile(
            finite_values,
            lower_percentile,
        )
    )

    upper_value = float(
        np.percentile(
            finite_values,
            upper_percentile,
        )
    )

    display = np.zeros_like(
        channel,
        dtype=np.float32,
    )

    # Canal prácticamente constante.
    if np.isclose(
        lower_value,
        upper_value,
    ):
        return display

    display[
        finite_mask
    ] = (
        channel[finite_mask]
        - lower_value
    ) / (
        upper_value
        - lower_value
    )

    display = np.clip(
        display,
        0.0,
        1.0,
    )

    # Los píxeles inválidos se representan como negro.
    display[
        ~finite_mask
    ] = 0.0

    return display.astype(
        np.float32,
        copy=False,
    )


def prepare_rgb(
    patch: SpectralPatch,
    *,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> np.ndarray:
    """
    Construye una composición RGB natural:

        R = B4
        G = B3
        B = B2

    Devuelve un array [alto, ancho, 3] en [0, 1].
    """

    try:
        red = patch.channel("B4")
        green = patch.channel("B3")
        blue = patch.channel("B2")

    except SpectralPatchError as error:
        raise SpectralVisualizationError(
            "No fue posible construir RGB porque "
            "faltan bandas visibles requeridas."
        ) from error

    red_display = percentile_stretch(
        red,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    green_display = percentile_stretch(
        green,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    blue_display = percentile_stretch(
        blue,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    rgb = np.stack(
        [
            red_display,
            green_display,
            blue_display,
        ],
        axis=-1,
    )

    return rgb.astype(
        np.float32,
        copy=False,
    )


def prepare_index(
    patch: SpectralPatch,
    index_name: str,
) -> np.ndarray:
    """
    Obtiene NDMI o NBR conservando sus valores científicos.

    Los valores finitos se limitan al rango físico esperado
    [-1, 1]. Los NaN se mantienen para poder enmascararlos
    posteriormente en la visualización.
    """

    index_name = str(
        index_name
    ).upper()

    if index_name not in SUPPORTED_INDICES:
        raise SpectralVisualizationError(
            f"Índice no soportado: {index_name}"
        )

    try:
        index = patch.channel(
            index_name
        )

    except SpectralPatchError as error:
        raise SpectralVisualizationError(
            f"No fue posible obtener {index_name}."
        ) from error

    index = index.astype(
        np.float32,
        copy=True,
    )

    finite_mask = np.isfinite(
        index
    )

    index[
        finite_mask
    ] = np.clip(
        index[finite_mask],
        -1.0,
        1.0,
    )

    return index


def calculate_channel_statistics(
    channel: np.ndarray,
) -> dict[str, Any]:
    """
    Calcula estadísticas descriptivas utilizando
    únicamente píxeles finitos.
    """

    if channel.ndim != 2:
        raise SpectralVisualizationError(
            "Las estadísticas requieren "
            "un canal bidimensional."
        )

    finite_mask = np.isfinite(
        channel
    )

    total_pixels = int(
        channel.size
    )

    valid_pixels = int(
        finite_mask.sum()
    )

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "valid_percentage": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    values = channel[
        finite_mask
    ].astype(
        np.float64
    )

    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_percentage": (
            valid_pixels
            / total_pixels
            * 100.0
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
    }