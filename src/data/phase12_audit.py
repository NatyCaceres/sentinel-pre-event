from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.phase1_geographic import (
    load_phase1_geographic,
)
from src.data.phase2_validator import (
    DEFAULT_PHASE2_PATH,
)


FLOAT_TOLERANCE = 1e-8

IDENTITY_COLUMNS = [
    "group_id",
    "etiqueta",
    "season_year",
    "split",
]

DATE_COLUMNS = [
    "fecha_referencia",
    "fecha_objetivo_pre_evento",
]

FLOAT_COLUMNS = [
    "lat_centro",
    "lon_centro",
    "event_weight",
]

SPECIAL_PATCH_ID = "P_2025_NEG_00610"


class Phase12AuditError(Exception):
    """
    Error asociado a la auditoría de correspondencia
    entre Fase 1 y Fase 2.
    """


def load_phase2_metadata(
    path: Path | str = DEFAULT_PHASE2_PATH,
) -> pd.DataFrame:
    """
    Carga la metadata validada de Fase 2.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la metadata de Fase 2: {path}"
        )

    return pd.read_csv(path)


def compare_exact_column(
    merged: pd.DataFrame,
    column: str,
) -> int:
    """
    Cuenta diferencias exactas entre una columna
    proveniente de Fase 1 y Fase 2.
    """

    left = merged[f"{column}_f1"]
    right = merged[f"{column}_f2"]

    equal = (
        left.eq(right)
        | (
            left.isna()
            & right.isna()
        )
    )

    return int((~equal).sum())


def compare_date_column(
    merged: pd.DataFrame,
    column: str,
) -> int:
    """
    Compara fechas normalizándolas previamente.
    """

    left = pd.to_datetime(
        merged[f"{column}_f1"],
        errors="coerce",
    )

    right = pd.to_datetime(
        merged[f"{column}_f2"],
        errors="coerce",
    )

    equal = (
        left.eq(right)
        | (
            left.isna()
            & right.isna()
        )
    )

    return int((~equal).sum())


def compare_float_column(
    merged: pd.DataFrame,
    column: str,
    tolerance: float = FLOAT_TOLERANCE,
) -> int:
    """
    Compara valores numéricos utilizando tolerancia
    para evitar falsos desacuerdos por serialización.
    """

    left = pd.to_numeric(
        merged[f"{column}_f1"],
        errors="coerce",
    )

    right = pd.to_numeric(
        merged[f"{column}_f2"],
        errors="coerce",
    )

    equal = np.isclose(
        left,
        right,
        rtol=0.0,
        atol=tolerance,
        equal_nan=True,
    )

    return int((~equal).sum())


def audit_phase1_phase2(
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
) -> dict:
    """
    Audita la relación entre los artefactos actuales
    de Fase 1 y Fase 2 sin generar todavía el catálogo.
    """

    if phase1["patch_id"].duplicated().any():
        raise Phase12AuditError(
            "Fase 1 contiene patch_id duplicados."
        )

    if phase2["patch_id"].duplicated().any():
        raise Phase12AuditError(
            "Fase 2 contiene patch_id duplicados."
        )

    phase1_ids = set(
        phase1["patch_id"].astype(str)
    )

    phase2_ids = set(
        phase2["patch_id"].astype(str)
    )

    f1_only = sorted(
        phase1_ids - phase2_ids
    )

    f2_only = sorted(
        phase2_ids - phase1_ids
    )

    common_ids = (
        phase1_ids & phase2_ids
    )

    comparison_columns = (
        ["patch_id"]
        + IDENTITY_COLUMNS
        + DATE_COLUMNS
        + FLOAT_COLUMNS
    )

    phase1_compare = phase1[
        comparison_columns
    ].copy()

    phase2_compare = phase2[
        comparison_columns
    ].copy()

    merged = phase1_compare.merge(
        phase2_compare,
        on="patch_id",
        how="inner",
        suffixes=("_f1", "_f2"),
        validate="one_to_one",
    )

    differences: dict[str, int] = {}

    for column in IDENTITY_COLUMNS:
        differences[column] = (
            compare_exact_column(
                merged,
                column,
            )
        )

    for column in DATE_COLUMNS:
        differences[column] = (
            compare_date_column(
                merged,
                column,
            )
        )

    for column in FLOAT_COLUMNS:
        differences[column] = (
            compare_float_column(
                merged,
                column,
            )
        )

    # --------------------------------------------------------
    # CONTEXTO ADMINISTRATIVO DE LOS PATCHES DE FASE 2
    # --------------------------------------------------------

    phase2_geographic = phase1[
        phase1["patch_id"].astype(str).isin(
            phase2_ids
        )
    ].copy()

    administrative_counts = (
        phase2_geographic[
            "administrative_status"
        ]
        .value_counts(dropna=False)
        .to_dict()
    )

    special_patch_in_phase1 = (
        SPECIAL_PATCH_ID in phase1_ids
    )

    special_patch_in_phase2 = (
        SPECIAL_PATCH_ID in phase2_ids
    )

    return {
        "phase1_rows": int(len(phase1)),
        "phase2_rows": int(len(phase2)),
        "phase1_unique_patches": int(
            phase1["patch_id"].nunique()
        ),
        "phase2_unique_patches": int(
            phase2["patch_id"].nunique()
        ),
        "common_patches": int(
            len(common_ids)
        ),
        "phase1_only_count": int(
            len(f1_only)
        ),
        "phase2_only_count": int(
            len(f2_only)
        ),
        "phase1_only_ids": f1_only,
        "phase2_only_ids": f2_only,
        "differences": differences,
        "administrative_counts_phase2": {
            str(key): int(value)
            for key, value
            in administrative_counts.items()
        },
        "special_patch": {
            "patch_id": SPECIAL_PATCH_ID,
            "in_phase1": special_patch_in_phase1,
            "in_phase2": special_patch_in_phase2,
        },
    }


def audit_is_compatible(
    report: dict,
) -> bool:
    """
    Determina si Fase 2 puede considerarse un subconjunto
    consistente de la Fase 1 actual.
    """

    if report["phase2_only_count"] != 0:
        return False

    if any(
        count != 0
        for count in report[
            "differences"
        ].values()
    ):
        return False

    return True


def main() -> None:
    phase1 = load_phase1_geographic()

    phase2 = load_phase2_metadata()

    report = audit_phase1_phase2(
        phase1,
        phase2,
    )

    print()
    print("=" * 65)
    print("AUDITORÍA DE CORRESPONDENCIA — FASE 1 ↔ FASE 2")
    print("=" * 65)

    print(
        f"Fase 1: {report['phase1_rows']} registros"
    )

    print(
        f"Fase 2: {report['phase2_rows']} registros"
    )

    print(
        f"Patch comunes: {report['common_patches']}"
    )

    print(
        "Solo en Fase 1: "
        f"{report['phase1_only_count']}"
    )

    print(
        "Solo en Fase 2: "
        f"{report['phase2_only_count']}"
    )

    print()
    print("CONSISTENCIA DE CAMPOS")
    print("-" * 65)

    for column, count in report[
        "differences"
    ].items():

        print(
            f"{column:<30} diferencias: {count}"
        )

    print()
    print("COBERTURA ADMINISTRATIVA EN FASE 2")
    print("-" * 65)

    for status, count in report[
        "administrative_counts_phase2"
    ].items():

        print(
            f"{status:<20} {count}"
        )

    special = report["special_patch"]

    print()
    print("CASO ESPECIAL")
    print("-" * 65)

    print(
        f"{special['patch_id']}"
    )

    print(
        f"Presente en Fase 1: "
        f"{special['in_phase1']}"
    )

    print(
        f"Presente en Fase 2: "
        f"{special['in_phase2']}"
    )

    print()
    print("=" * 65)

    if audit_is_compatible(report):

        print("ESTADO: COMPATIBLE")

        print(
            "Fase 2 es un subconjunto consistente "
            "de la Fase 1 actual."
        )

    else:

        print("ESTADO: REVISAR")

        print(
            "Se detectaron diferencias que deben "
            "resolverse antes de generar el catálogo."
        )

    print("=" * 65)

    if report["phase2_only_count"] > 0:

        print()
        print("PATCHES PRESENTES SOLO EN FASE 2")

        for patch_id in report[
            "phase2_only_ids"
        ]:
            print(patch_id)


if __name__ == "__main__":
    main()