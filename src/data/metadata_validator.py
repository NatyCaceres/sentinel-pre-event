from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "config" / "phase1_schema.json"
)


@dataclass(slots=True)
class ValidationReport:
    """
    Resultado estructurado de una validación de metadata.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Indica si la metadata superó todas las validaciones."""
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def load_phase1_schema(
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """
    Carga el contrato de validación de la metadata de Fase 1.
    """

    path = Path(schema_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el schema de Fase 1: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    return schema


def load_metadata(csv_path: Path | str) -> pd.DataFrame:
    """
    Carga un CSV de metadata sin aplicar transformaciones silenciosas.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de metadata: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Se esperaba un archivo CSV y se recibió: {path.suffix}"
        )

    return pd.read_csv(path)


def _validate_required_columns(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    required_columns = schema["required_columns"]

    missing_columns = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        report.add_error(
            "Faltan columnas obligatorias: "
            f"{missing_columns}"
        )
        return

    null_counts = df[required_columns].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    for column, count in columns_with_nulls.items():
        report.add_error(
            f"La columna '{column}' contiene "
            f"{int(count)} valores nulos."
        )


def _validate_identifiers(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    identifier_config = schema["identifier_columns"]

    for column, rules in identifier_config.items():

        if column not in df.columns:
            continue

        if not rules.get("nullable", True):
            null_count = int(df[column].isna().sum())

            if null_count > 0:
                report.add_error(
                    f"El identificador '{column}' contiene "
                    f"{null_count} valores nulos."
                )

        if rules.get("unique", False):
            duplicate_mask = df[column].duplicated(
                keep=False
            )

            if duplicate_mask.any():
                duplicates = (
                    df.loc[duplicate_mask, column]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                report.add_error(
                    f"El identificador '{column}' debe ser único. "
                    f"Se detectaron {len(duplicates)} valores duplicados."
                )


def _validate_classes(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    class_config = schema["classes"]

    column = class_config["column"]
    allowed_values = set(class_config["allowed_values"])

    if column not in df.columns:
        return

    numeric_labels = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    invalid_numeric = numeric_labels.isna()

    if invalid_numeric.any():
        report.add_error(
            f"La columna '{column}' contiene "
            f"{int(invalid_numeric.sum())} etiquetas no numéricas."
        )

    observed_values = set(
        numeric_labels.dropna().unique().tolist()
    )

    invalid_values = observed_values - allowed_values

    if invalid_values:
        report.add_error(
            f"La columna '{column}' contiene clases no permitidas: "
            f"{sorted(invalid_values)}. "
            f"Clases válidas: {sorted(allowed_values)}."
        )

    missing_classes = allowed_values - observed_values

    if missing_classes:
        report.add_error(
            "No están presentes todas las clases requeridas. "
            f"Faltan: {sorted(missing_classes)}."
        )


def _validate_coordinates(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    coordinate_config = schema["coordinates"]

    lat_column = coordinate_config["latitude_column"]
    lon_column = coordinate_config["longitude_column"]

    if (
        lat_column not in df.columns
        or lon_column not in df.columns
    ):
        return

    latitude = pd.to_numeric(
        df[lat_column],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df[lon_column],
        errors="coerce",
    )

    invalid_lat = latitude.isna()
    invalid_lon = longitude.isna()

    if invalid_lat.any():
        report.add_error(
            f"Se detectaron {int(invalid_lat.sum())} "
            "latitudes no numéricas."
        )

    if invalid_lon.any():
        report.add_error(
            f"Se detectaron {int(invalid_lon.sum())} "
            "longitudes no numéricas."
        )

    study_area = coordinate_config["study_area"]

    valid_lat = latitude.dropna()
    valid_lon = longitude.dropna()

    outside_lat = ~valid_lat.between(
        study_area["lat_min"],
        study_area["lat_max"],
        inclusive="both",
    )

    outside_lon = ~valid_lon.between(
        study_area["lon_min"],
        study_area["lon_max"],
        inclusive="both",
    )

    if outside_lat.any():
        report.add_error(
            f"Se detectaron {int(outside_lat.sum())} "
            "registros fuera del rango de latitud del estudio."
        )

    if outside_lon.any():
        report.add_error(
            f"Se detectaron {int(outside_lon.sum())} "
            "registros fuera del rango de longitud del estudio."
        )


def _parse_dates(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> dict[str, pd.Series]:
    parsed_dates: dict[str, pd.Series] = {}

    for column in schema["dates"]["columns"]:

        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        invalid_count = int(parsed.isna().sum())

        if invalid_count > 0:
            report.add_error(
                f"La columna '{column}' contiene "
                f"{invalid_count} fechas inválidas."
            )

        parsed_dates[column] = parsed

    return parsed_dates


def _validate_temporal_logic(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    dates = _parse_dates(df, schema, report)

    required = {
        "fecha_referencia",
        "fecha_objetivo_pre_evento",
        "gee_inicio",
        "gee_fin",
    }

    if not required.issubset(dates):
        return

    reference = dates["fecha_referencia"]
    target = dates["fecha_objetivo_pre_evento"]
    gee_start = dates["gee_inicio"]
    gee_end = dates["gee_fin"]

    valid_mask = (
        reference.notna()
        & target.notna()
        & gee_start.notna()
        & gee_end.notna()
    )

    if not valid_mask.any():
        return

    reference = reference[valid_mask]
    target = target[valid_mask]
    gee_start = gee_start[valid_mask]
    gee_end = gee_end[valid_mask]

    invalid_pre_event = target >= reference

    if invalid_pre_event.any():
        report.add_error(
            f"{int(invalid_pre_event.sum())} registros tienen "
            "una fecha objetivo pre-evento igual o posterior "
            "a la fecha de referencia."
        )

    invalid_window_order = gee_start >= gee_end

    if invalid_window_order.any():
        report.add_error(
            f"{int(invalid_window_order.sum())} registros tienen "
            "una ventana GEE temporalmente inválida."
        )

    target_outside_window = (
        (target < gee_start)
        | (target >= gee_end)
    )

    if target_outside_window.any():
        report.add_error(
            f"{int(target_outside_window.sum())} registros tienen "
            "la fecha objetivo fuera de su ventana GEE."
        )

    delta_days = (reference - target).dt.days

    minimum_days = schema["dates"]["pre_event_min_days"]
    maximum_days = schema["dates"]["pre_event_max_days"]

    invalid_delta = (
        (delta_days < minimum_days)
        | (delta_days > maximum_days)
    )

    if invalid_delta.any():
        report.add_error(
            f"{int(invalid_delta.sum())} registros no cumplen "
            f"el intervalo pre-evento esperado "
            f"({minimum_days}-{maximum_days} días)."
        )


def _validate_split(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    split_config = schema["split"]

    column = split_config["column"]

    if column not in df.columns:
        return

    allowed_values = set(
        split_config["allowed_values"]
    )

    observed_values = set(
        df[column].dropna().unique().tolist()
    )

    invalid_values = observed_values - allowed_values

    if invalid_values:
        report.add_error(
            f"El split contiene valores no permitidos: "
            f"{sorted(invalid_values)}."
        )

    missing_splits = allowed_values - observed_values

    if missing_splits:
        report.add_error(
            f"Faltan particiones requeridas: "
            f"{sorted(missing_splits)}."
        )

    label_column = schema["classes"]["column"]
    required_classes = set(
        schema["classes"]["allowed_values"]
    )

    if label_column in df.columns:
        for split_name in allowed_values:

            split_rows = df[df[column] == split_name]

            observed_classes = set(
                pd.to_numeric(
                    split_rows[label_column],
                    errors="coerce",
                )
                .dropna()
                .unique()
                .tolist()
            )

            missing_classes = (
                required_classes - observed_classes
            )

            if missing_classes:
                report.add_error(
                    f"El split '{split_name}' no contiene "
                    f"todas las clases. Faltan: "
                    f"{sorted(missing_classes)}."
                )

    for leakage_key in split_config["leakage_keys"]:

        if leakage_key not in df.columns:
            continue

        train_values = set(
            df.loc[
                df[column] == "train",
                leakage_key,
            ]
            .dropna()
            .astype(str)
        )

        test_values = set(
            df.loc[
                df[column] == "test",
                leakage_key,
            ]
            .dropna()
            .astype(str)
        )

        overlap = train_values & test_values

        if overlap:
            report.add_error(
                f"Fuga de información detectada en "
                f"'{leakage_key}': {len(overlap)} valores "
                "aparecen tanto en train como en test."
            )


def _validate_weights(
    df: pd.DataFrame,
    schema: dict[str, Any],
    report: ValidationReport,
) -> None:
    weight_config = schema["weights"]

    column = weight_config["column"]

    if column not in df.columns:
        return

    weights = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    invalid_numeric = weights.isna()

    if invalid_numeric.any():
        report.add_error(
            f"La columna '{column}' contiene "
            f"{int(invalid_numeric.sum())} pesos no numéricos."
        )

    valid_weights = weights.dropna()

    minimum = weight_config["minimum_exclusive"]
    maximum = weight_config["maximum_inclusive"]

    invalid_range = (
        (valid_weights <= minimum)
        | (valid_weights > maximum)
    )

    if invalid_range.any():
        report.add_error(
            f"{int(invalid_range.sum())} valores de "
            f"'{column}' están fuera del rango "
            f"({minimum}, {maximum}]."
        )

    if "group_id" in df.columns:
        temporary = pd.DataFrame(
            {
                "group_id": df["group_id"],
                "weight": weights,
            }
        ).dropna()

        group_weight_sum = (
            temporary
            .groupby("group_id")["weight"]
            .sum()
        )

        inconsistent_groups = (
            (group_weight_sum - 1.0).abs() > 1e-6
        )

        if inconsistent_groups.any():
            report.add_warning(
                f"{int(inconsistent_groups.sum())} grupos tienen "
                "una suma de event_weight diferente de 1. "
                "Revisar si corresponde a la metodología "
                "multiparche utilizada."
            )


def _build_stats(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    report.stats["rows"] = int(len(df))
    report.stats["columns"] = int(len(df.columns))

    if "patch_id" in df.columns:
        report.stats["unique_patches"] = int(
            df["patch_id"].nunique()
        )

    if "group_id" in df.columns:
        report.stats["unique_groups"] = int(
            df["group_id"].nunique()
        )

        group_sizes = df.groupby("group_id").size()

        report.stats["multipatch_groups"] = int(
            (group_sizes > 1).sum()
        )

    if "etiqueta" in df.columns:
        report.stats["class_distribution"] = (
            df["etiqueta"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )

    if "split" in df.columns:
        report.stats["split_distribution"] = (
            df["split"]
            .value_counts(dropna=False)
            .to_dict()
        )

    if "season_year" in df.columns:
        seasons = pd.to_numeric(
            df["season_year"],
            errors="coerce",
        ).dropna()

        if not seasons.empty:
            report.stats["season_min"] = int(
                seasons.min()
            )
            report.stats["season_max"] = int(
                seasons.max()
            )


def validate_phase1_metadata(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> ValidationReport:
    """
    Ejecuta todas las reglas de validación de Fase 1.
    """

    report = ValidationReport()

    _validate_required_columns(
        df,
        schema,
        report,
    )

    # Si faltan columnas obligatorias, varias validaciones
    # posteriores no podrían ejecutarse correctamente.
    if report.errors:
        return report

    _validate_identifiers(
        df,
        schema,
        report,
    )

    _validate_classes(
        df,
        schema,
        report,
    )

    _validate_coordinates(
        df,
        schema,
        report,
    )

    _validate_temporal_logic(
        df,
        schema,
        report,
    )

    _validate_split(
        df,
        schema,
        report,
    )

    _validate_weights(
        df,
        schema,
        report,
    )

    _build_stats(
        df,
        report,
    )

    return report


def validate_phase1_file(
    csv_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> ValidationReport:
    """
    Carga y valida un archivo CSV completo de Fase 1.
    """

    schema = load_phase1_schema(schema_path)
    dataframe = load_metadata(csv_path)

    return validate_phase1_metadata(
        dataframe,
        schema,
    )


def print_validation_report(
    report: ValidationReport,
) -> None:
    """
    Imprime un informe legible de la validación.
    """

    print()
    print("=" * 60)
    print("VALIDACIÓN DE METADATA — FASE 1")
    print("=" * 60)

    if report.is_valid:
        print("ESTADO: VÁLIDO")
    else:
        print("ESTADO: INVÁLIDO")

    print()

    if report.stats:
        print("RESUMEN")
        print("-" * 60)

        for key, value in report.stats.items():
            print(f"{key}: {value}")

        print()

    if report.errors:
        print("ERRORES")
        print("-" * 60)

        for error in report.errors:
            print(f"[ERROR] {error}")

        print()

    if report.warnings:
        print("ADVERTENCIAS")
        print("-" * 60)

        for warning in report.warnings:
            print(f"[WARN] {warning}")

        print()

    if report.is_valid:
        print(
            "La metadata cumple el contrato técnico "
            "definido para Fase 1."
        )
    else:
        print(
            "La metadata NO debe ser utilizada por el producto "
            "hasta corregir los errores anteriores."
        )

    print("=" * 60)


def main() -> None:
    """
    Punto de entrada para validación desde terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Valida un CSV limpio correspondiente "
            "a la Fase 1 del proyecto."
        )
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Ruta al CSV de Fase 1.",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Ruta alternativa al schema JSON.",
    )

    args = parser.parse_args()

    report = validate_phase1_file(
        csv_path=args.csv_path,
        schema_path=args.schema,
    )

    print_validation_report(report)

    if not report.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()