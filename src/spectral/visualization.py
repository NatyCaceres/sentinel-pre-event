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

VIRIDIS_COLORS = (
    (0.267004, 0.004874, 0.329415),
    (0.278826, 0.175490, 0.483397),
    (0.229739, 0.322361, 0.545706),
    (0.172719, 0.448791, 0.557885),
    (0.127568, 0.566949, 0.550556),
    (0.157851, 0.683765, 0.501686),
    (0.369214, 0.788888, 0.382914),
    (0.678489, 0.863742, 0.189503),
    (0.993248, 0.906157, 0.143936),
)


class SpectralVisualizationError(Exception):
    """
    Error asociado exclusivamente a la preparación
    visual de información espectral.
    """


def colorize_gradcam(heatmap: np.ndarray) -> np.ndarray:
    """Escala visual fija violeta–amarillo para intensidades relativas [0, 1]."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SpectralVisualizationError("El mapa Grad-CAM debe ser bidimensional y finito.")
    if values.size == 0 or values.min() < 0 or values.max() > 1:
        raise SpectralVisualizationError("El mapa Grad-CAM debe estar entre 0 y 1.")
    # Paleta explícita y versionada: no depende de librerías gráficas externas.
    colors = np.asarray([
        (0.001, 0.000, 0.014), (0.113, 0.065, 0.277),
        (0.317, 0.072, 0.485), (0.512, 0.148, 0.508),
        (0.716, 0.215, 0.475), (0.902, 0.365, 0.384),
        (0.987, 0.536, 0.382), (0.997, 0.733, 0.505),
        (0.987, 0.991, 0.750),
    ], dtype=np.float32)
    positions = np.linspace(0, 1, len(colors))
    return np.stack([
        np.interp(values, positions, colors[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.float32)


def prepare_gradcam_overlay(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.46,
) -> np.ndarray:
    """Superpone el mapa sin modificar ni renormalizar las entradas."""
    colors = colorize_gradcam(heatmap)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.shape != colors.shape or not np.isfinite(rgb).all():
        raise SpectralVisualizationError("RGB y Grad-CAM deben tener dimensiones compatibles.")
    if not 0 <= alpha <= 1 or np.any((rgb < 0) | (rgb > 1)):
        raise SpectralVisualizationError("RGB y opacidad deben estar entre 0 y 1.")
    return ((1 - alpha) * rgb + alpha * colors).astype(np.float32)


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


def prepare_false_color(
    patch: SpectralPatch,
    *,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> np.ndarray:
    """
    Construye una composición de falso color para vegetación:

        R = B8 (infrarrojo cercano)
        G = B4 (rojo)
        B = B3 (verde)

    Devuelve un array [alto, ancho, 3] en [0, 1].
    """

    try:
        near_infrared = patch.channel("B8")
        red = patch.channel("B4")
        green = patch.channel("B3")

    except SpectralPatchError as error:
        raise SpectralVisualizationError(
            "No fue posible construir falso color porque "
            "faltan bandas requeridas."
        ) from error

    false_color = np.stack(
        [
            percentile_stretch(
                near_infrared,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            percentile_stretch(
                red,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            percentile_stretch(
                green,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
        ],
        axis=-1,
    )

    return false_color.astype(
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


def prepare_index_display(
    patch: SpectralPatch,
    index_name: str,
) -> np.ndarray:
    """
    Colorea NDMI o NBR con una escala visual fija [-1, 1].

    La transformación genera únicamente una imagen RGB. Los
    valores científicos devueltos por prepare_index no cambian.
    Los píxeles no finitos se representan en negro.
    """

    index_name = str(index_name).upper()
    index = prepare_index(
        patch,
        index_name,
    )
    finite_mask = np.isfinite(index)
    display = np.zeros(
        (*index.shape, 3),
        dtype=np.float32,
    )

    if not finite_mask.any():
        raise SpectralVisualizationError(
            f"{index_name} no contiene valores finitos "
            "para su visualización."
        )

    normalized = (
        index[finite_mask] + 1.0
    ) / 2.0
    viridis = np.asarray(
        VIRIDIS_COLORS,
        dtype=np.float32,
    )
    viridis_positions = np.linspace(
        0.0,
        1.0,
        len(VIRIDIS_COLORS),
        dtype=np.float32,
    )
    colors = np.empty(
        (normalized.size, 3),
        dtype=np.float32,
    )

    for channel_index in range(3):
        colors[:, channel_index] = np.interp(
            normalized,
            viridis_positions,
            viridis[:, channel_index],
        )

    display[finite_mask] = colors

    return np.clip(
        display,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


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
