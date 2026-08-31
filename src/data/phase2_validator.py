from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PHASE2_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase2"
    / "metadata_modelado_protocolo_v1.csv"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase2_schema.json"
)


class Phase2ValidationError(Exception):
    """
    Error asociado a la validación de metadata de Fase 2.
    """


def load_schema(
    path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict:
    """
    Carga el contrato de validación de Fase 2.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el schema de Fase 2: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_phase2_metadata(
    dataframe: pd.DataFrame,
    schema: dict,
) -> list[str]:
    """
    Valida estructura, identidad, disponibilidad
    y consistencia básica de la metadata de Fase 2.
    """

    errors: list[str] = []

    required_columns = set(
        schema["required_columns"]
    )

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        errors.append(
            "Faltan columnas obligatorias: "
            f"{sorted(missing_columns)}"
        )

        return errors

    # --------------------------------------------------------
    # IDENTIDAD
    # --------------------------------------------------------

    if schema["identity"]["patch_id_unique"]:

        if dataframe["patch_id"].duplicated().any():
            errors.append(
                "patch_id contiene valores duplicados."
            )

    allowed_labels = set(
        schema["identity"]["allowed_labels"]
    )

    labels = set(
        dataframe["etiqueta"]
        .dropna()
        .astype(int)
        .unique()
    )

    if not labels.issubset(
        allowed_labels
    ):
        errors.append(
            "Se encontraron etiquetas no permitidas: "
            f"{sorted(labels - allowed_labels)}"
        )

    allowed_seasons = set(
        schema["identity"]["allowed_seasons"]
    )

    seasons = set(
        dataframe["season_year"]
        .dropna()
        .astype(int)
        .unique()
    )

    if not seasons.issubset(
        allowed_seasons
    ):
        errors.append(
            "Se encontraron temporadas no permitidas: "
            f"{sorted(seasons - allowed_seasons)}"
        )

    allowed_split = set(
        schema["identity"]["allowed_split"]
    )

    splits = set(
        dataframe["split"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not splits.issubset(
        allowed_split
    ):
        errors.append(
            "Se encontraron valores split no permitidos: "
            f"{sorted(splits - allowed_split)}"
        )

    allowed_subsplit = set(
        schema["identity"]["allowed_subsplit"]
    )

    subsplits = set(
        dataframe["subsplit"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not subsplits.issubset(
        allowed_subsplit
    ):
        errors.append(
            "Se encontraron valores subsplit no permitidos: "
            f"{sorted(subsplits - allowed_subsplit)}"
        )

    # --------------------------------------------------------
    # DISPONIBILIDAD FINAL
    # --------------------------------------------------------

    if not dataframe["existe_tif"].fillna(False).all():
        errors.append(
            "Existen registros sin TIFF disponible."
        )

    if not (
        dataframe["archivo_existe_drive"]
        .fillna(False)
        .all()
    ):
        errors.append(
            "Existen registros cuyo archivo final "
            "no aparece disponible en Drive."
        )

    for column in schema[
        "availability_contract"
    ]["required_non_null"]:

        if dataframe[column].isna().any():
            errors.append(
                f"La columna '{column}' contiene "
                "valores nulos."
            )

    # --------------------------------------------------------
    # CONSISTENCIA INTERNA
    # --------------------------------------------------------

    if not (
        dataframe["patch_id"]
        == dataframe["patch_id_export"]
    ).all():
        errors.append(
            "patch_id no coincide con patch_id_export."
        )

    if not (
        dataframe["split"]
        == dataframe["split_export"]
    ).all():
        errors.append(
            "split no coincide con split_export."
        )

    if not (
        dataframe["archivo_model_8ch"]
        == dataframe["archivo_model8_esperado"]
    ).all():
        errors.append(
            "archivo_model_8ch no coincide con "
            "archivo_model8_esperado."
        )

    route_basename = (
        dataframe["ruta_model_8ch"]
        .astype(str)
        .map(
            lambda value: Path(value).name
        )
    )

    if not (
        route_basename
        == dataframe["archivo_model8_esperado"]
    ).all():
        errors.append(
            "El nombre final de ruta_model_8ch "
            "no coincide con archivo_model8_esperado."
        )

    return errors


def main() -> None:
    if not DEFAULT_PHASE2_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la metadata de Fase 2: "
            f"{DEFAULT_PHASE2_PATH}"
        )

    dataframe = pd.read_csv(
        DEFAULT_PHASE2_PATH
    )

    schema = load_schema()

    errors = validate_phase2_metadata(
        dataframe,
        schema,
    )

    print()
    print("=" * 60)
    print("VALIDACIÓN METADATA — FASE 2")
    print("=" * 60)

    print(
        f"Registros: {len(dataframe)}"
    )

    print(
        "Patches únicos: "
        f"{dataframe['patch_id'].nunique()}"
    )

    print(
        "Grupos únicos: "
        f"{dataframe['group_id'].nunique()}"
    )

    print(
        "Clase 0: "
        f"{int((dataframe['etiqueta'] == 0).sum())}"
    )

    print(
        "Clase 1: "
        f"{int((dataframe['etiqueta'] == 1).sum())}"
    )

    print(
        "TIFF disponibles: "
        f"{int(dataframe['existe_tif'].sum())}"
    )

    print("-" * 60)

    if errors:

        print("ESTADO: INVÁLIDO")

        for error in errors:
            print(
                f"[ERROR] {error}"
            )

        raise SystemExit(1)

    print("ESTADO: VÁLIDO")
    print(
        "La metadata cumple el contrato "
        "definido para Fase 2."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()