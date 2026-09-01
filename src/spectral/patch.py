from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "spectral_contract.json"
)


class SpectralPatchError(Exception):
    """
    Error asociado a la carga o validación
    de un parche Sentinel-2.
    """


@dataclass(frozen=True)
class SpectralPatch:
    """
    Representación validada de un parche multicanal.
    """

    data: np.ndarray
    channels: tuple[str, ...]
    path: Path

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def channel(
        self,
        name: str,
    ) -> np.ndarray:
        """
        Obtiene un canal por nombre.
        """

        try:
            index = self.channels.index(name)
        except ValueError as error:
            raise SpectralPatchError(
                f"Canal no disponible: {name}"
            ) from error

        return self.data[index].copy()


def load_spectral_contract(
    path: Path | str = DEFAULT_CONTRACT_PATH,
) -> dict:
    """
    Carga el contrato espectral utilizado
    por Fase 2 y el modelo.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el contrato espectral: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_patch_array(
    data: np.ndarray,
    contract: dict,
) -> None:
    """
    Comprueba estructura y contenido básico
    de un parche espectral.
    """

    expected_shape = tuple(
        contract["shape"]
    )

    if data.shape != expected_shape:
        raise SpectralPatchError(
            "Shape inválido. "
            f"Recibido: {data.shape}. "
            f"Esperado: {expected_shape}."
        )

    if data.ndim != 3:
        raise SpectralPatchError(
            "El parche debe tener tres dimensiones "
            "[canales, alto, ancho]."
        )

    if not np.issubdtype(
        data.dtype,
        np.number,
    ):
        raise SpectralPatchError(
            f"Tipo de datos no numérico: {data.dtype}"
        )

    if np.isinf(data).any():
        raise SpectralPatchError(
            "El parche contiene valores infinitos."
        )


def load_spectral_patch(
    path: Path | str,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> SpectralPatch:
    """
    Carga un GeoTIFF de 8 canales y devuelve
    una representación validada.

    El array se convierte a float32 para mantener
    el contrato utilizado durante el modelado.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el GeoTIFF: {path}"
        )

    contract = load_spectral_contract(
        contract_path
    )

    with rasterio.open(path) as dataset:
        data = dataset.read()

    data = data.astype(
        np.float32,
        copy=False,
    )

    validate_patch_array(
        data,
        contract,
    )

    channels = tuple(
        contract["channels"]
    )

    return SpectralPatch(
        data=data,
        channels=channels,
        path=path,
    )