from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "chile_dpa_2023"
    / "raw"
    / "DPA_2023"
    / "COMUNAS"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "chile_dpa_2023"
    / "dpa_manifest.json"
)

SHAPEFILE_BASE_NAME = "COMUNAS_v1"

REQUIRED_EXTENSIONS = [
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
    ".cpg",
]

EXPECTED_COLUMNS = [
    "CUT_REG",
    "CUT_PROV",
    "CUT_COM",
    "REGION",
    "PROVINCIA",
    "COMUNA",
    "SUPERFICIE",
    "geometry",
]

EXPECTED_CRS = "EPSG:5360"

EXPECTED_RECORDS = 345


class BoundaryValidationError(Exception):
    """
    Error asociado a la validación de la DPA.
    """


def calculate_sha256(path: Path | str) -> str:
    """
    Calcula SHA-256 leyendo el archivo por bloques.
    """

    path = Path(path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest().upper()


def get_required_files(
    boundary_dir: Path | str = DEFAULT_BOUNDARY_DIR,
) -> dict[str, Path]:
    """
    Obtiene los componentes requeridos del shapefile.
    """

    boundary_dir = Path(boundary_dir)

    files = {
        extension: (
            boundary_dir
            / f"{SHAPEFILE_BASE_NAME}{extension}"
        )
        for extension in REQUIRED_EXTENSIONS
    }

    missing = [
        str(path)
        for path in files.values()
        if not path.exists()
    ]

    if missing:
        raise BoundaryValidationError(
            "Faltan componentes de la capa DPA: "
            f"{missing}"
        )

    return files


def inspect_boundary_source(
    boundary_dir: Path | str = DEFAULT_BOUNDARY_DIR,
) -> dict[str, Any]:
    """
    Inspecciona estructura, CRS, geometrías y hashes
    de la capa comunal utilizada por el producto.
    """

    files = get_required_files(boundary_dir)

    communes = gpd.read_file(
        files[".shp"]
    )

    crs = (
        communes.crs.to_string()
        if communes.crs is not None
        else None
    )

    empty_geometries = int(
        communes.geometry.is_empty.sum()
    )

    null_geometries = int(
        communes.geometry.isna().sum()
    )

    invalid_geometries = int(
        (~communes.geometry.is_valid).sum()
    )

    hashes = {
        path.name: calculate_sha256(path)
        for path in files.values()
    }

    return {
        "layer": SHAPEFILE_BASE_NAME,
        "records": int(len(communes)),
        "crs": crs,
        "columns": communes.columns.tolist(),
        "empty_geometries": empty_geometries,
        "null_geometries": null_geometries,
        "invalid_geometries": invalid_geometries,
        "files": hashes,
    }


def validate_boundary_structure(
    inspection: dict[str, Any],
) -> list[str]:
    """
    Valida las características estructurales esperadas.
    """

    errors: list[str] = []

    if inspection["records"] != EXPECTED_RECORDS:
        errors.append(
            "Cantidad inesperada de comunas: "
            f"{inspection['records']} "
            f"(esperado: {EXPECTED_RECORDS})."
        )

    if inspection["crs"] != EXPECTED_CRS:
        errors.append(
            "CRS inesperado: "
            f"{inspection['crs']} "
            f"(esperado: {EXPECTED_CRS})."
        )

    if inspection["columns"] != EXPECTED_COLUMNS:
        errors.append(
            "Las columnas de la capa no coinciden "
            "con el contrato esperado."
        )

    if inspection["empty_geometries"] > 0:
        errors.append(
            "La capa contiene geometrías vacías."
        )

    if inspection["null_geometries"] > 0:
        errors.append(
            "La capa contiene geometrías nulas."
        )

    if inspection["invalid_geometries"] > 0:
        errors.append(
            "La capa contiene geometrías inválidas."
        )

    return errors


def build_boundary_manifest(
    boundary_dir: Path | str = DEFAULT_BOUNDARY_DIR,
) -> dict[str, Any]:
    """
    Construye el manifest de la versión local de la DPA.
    """

    inspection = inspect_boundary_source(
        boundary_dir
    )

    errors = validate_boundary_structure(
        inspection
    )

    if errors:
        raise BoundaryValidationError(
            "La DPA no cumple el contrato estructural: "
            + "; ".join(errors)
        )

    return {
        "dataset": "División Política Administrativa 2023",
        "layer": "COMUNAS_v1",
        "expected_records": EXPECTED_RECORDS,
        "expected_crs": EXPECTED_CRS,
        "expected_columns": EXPECTED_COLUMNS,
        "source_fingerprint": inspection,
    }


def save_boundary_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> Path:
    """
    Guarda el manifest reproducible de la DPA.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def validate_against_manifest(
    boundary_dir: Path | str = DEFAULT_BOUNDARY_DIR,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> list[str]:
    """
    Compara la DPA local con la versión registrada
    previamente en el manifest.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No se encontró el manifest DPA: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    current = inspect_boundary_source(
        boundary_dir
    )

    errors = validate_boundary_structure(
        current
    )

    expected_hashes = (
        manifest["source_fingerprint"]["files"]
    )

    current_hashes = current["files"]

    for file_name, expected_hash in expected_hashes.items():

        current_hash = current_hashes.get(
            file_name
        )

        if current_hash != expected_hash:
            errors.append(
                f"SHA-256 diferente para "
                f"'{file_name}'."
            )

    return errors


def main() -> None:
    if DEFAULT_MANIFEST_PATH.exists():

        errors = validate_against_manifest()

        print()
        print("=" * 60)
        print("VALIDACIÓN DPA 2023")
        print("=" * 60)

        if errors:
            print("ESTADO: INVÁLIDO")

            for error in errors:
                print(f"[ERROR] {error}")

            raise SystemExit(1)

        print("ESTADO: VÁLIDO")
        print("La DPA coincide con la versión registrada.")
        print("=" * 60)

    else:
        manifest = build_boundary_manifest()

        path = save_boundary_manifest(
            manifest
        )

        info = manifest["source_fingerprint"]

        print()
        print("=" * 60)
        print("REGISTRO INICIAL DPA 2023")
        print("=" * 60)
        print(f"Registros: {info['records']}")
        print(f"CRS: {info['crs']}")
        print(
            "Geometrías inválidas: "
            f"{info['invalid_geometries']}"
        )
        print(f"Manifest creado: {path}")
        print("=" * 60)


if __name__ == "__main__":
    main()