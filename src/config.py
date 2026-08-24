from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"


def load_project_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Carga la configuración general del producto.

    Parameters
    ----------
    config_path:
        Ruta al archivo JSON de configuración.

    Returns
    -------
    dict
        Configuración validada del proyecto.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.

    ValueError
        Si la configuración no cumple el contrato esperado.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    validate_project_config(config)

    return config


def validate_project_config(config: dict[str, Any]) -> None:
    """
    Valida los elementos mínimos del contrato técnico.
    """

    required_sections = {
        "project",
        "data_contract",
        "paths",
    }

    missing_sections = required_sections - config.keys()

    if missing_sections:
        raise ValueError(
            "Faltan secciones obligatorias en la configuración: "
            f"{sorted(missing_sections)}"
        )

    contract = config["data_contract"]

    required_contract_fields = {
        "channels",
        "input_shape",
        "dtype",
        "label_column",
        "group_column",
        "split_column",
        "negative_class",
        "positive_class",
    }

    missing_fields = required_contract_fields - contract.keys()

    if missing_fields:
        raise ValueError(
            "Faltan campos obligatorios en data_contract: "
            f"{sorted(missing_fields)}"
        )

    channels = contract["channels"]
    input_shape = contract["input_shape"]

    if not isinstance(channels, list) or not channels:
        raise ValueError(
            "'channels' debe ser una lista no vacía."
        )

    if len(channels) != len(set(channels)):
        raise ValueError(
            "No pueden existir canales duplicados."
        )

    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 3
    ):
        raise ValueError(
            "'input_shape' debe tener el formato "
            "[canales, alto, ancho]."
        )

    if input_shape[0] != len(channels):
        raise ValueError(
            "La cantidad de canales no coincide con "
            "la primera dimensión de input_shape."
        )

    if contract["positive_class"] == contract["negative_class"]:
        raise ValueError(
            "Las clases positiva y negativa deben ser diferentes."
        )


def get_project_path(relative_path: str) -> Path:
    """
    Convierte una ruta relativa del proyecto en una ruta absoluta.
    """

    return PROJECT_ROOT / relative_path


if __name__ == "__main__":
    project_config = load_project_config()

    contract = project_config["data_contract"]

    print("Configuración válida.")
    print(f"Proyecto: {project_config['project']['name']}")
    print(f"Versión: {project_config['project']['version']}")
    print(f"Canales: {contract['channels']}")
    print(f"Shape esperado: {contract['input_shape']}")
    print(f"Tipo de dato: {contract['dtype']}")