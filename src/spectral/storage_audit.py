from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError

from src.data.product_catalog import (
    DEFAULT_CATALOG_MANIFEST_PATH,
    DEFAULT_CATALOG_PATH,
    calculate_sha256,
)
from src.data.product_catalog_loader import (
    load_product_catalog,
)
from src.spectral.patch import (
    DEFAULT_CONTRACT_PATH,
    SpectralPatchError,
    load_spectral_contract,
    load_spectral_patch,
)
from src.spectral.storage import (
    get_patch_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STORAGE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "product"
    / "spectral_storage_manifest.json"
)

SUPPORTED_TIFF_EXTENSIONS = {
    ".tif",
    ".tiff",
}


class SpectralStorageAuditError(Exception):
    """
    Error asociado a la auditoría del almacenamiento
    de parches Sentinel-2.
    """


def _sorted_names(
    names,
) -> list[str]:
    """
    Ordena nombres de forma estable e independiente
    de diferencias entre mayúsculas y minúsculas.
    """

    return sorted(
        names,
        key=lambda value: (
            str(value).casefold(),
            str(value),
        ),
    )


def _validate_catalog_filenames(
    catalog: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Extrae tif_filename y registra valores que no representan
    nombres portables de archivos GeoTIFF.
    """

    if "tif_filename" not in catalog.columns:
        raise SpectralStorageAuditError(
            "El catálogo no contiene la columna tif_filename."
        )

    valid_names: list[str] = []
    invalid_entries: list[dict[str, Any]] = []

    for row_index, value in catalog["tif_filename"].items():

        if pd.isna(value):
            invalid_entries.append(
                {
                    "row_index": int(row_index),
                    "value": None,
                    "reason": "nombre ausente",
                }
            )
            continue

        filename = str(value)
        path = Path(filename)

        if (
            not filename
            or path.name != filename
            or path.suffix.lower()
            not in SUPPORTED_TIFF_EXTENSIONS
        ):
            invalid_entries.append(
                {
                    "row_index": int(row_index),
                    "value": filename,
                    "reason": (
                        "tif_filename debe ser un nombre portable "
                        "con extensión .tif o .tiff"
                    ),
                }
            )
            continue

        valid_names.append(filename)

    return valid_names, invalid_entries


def list_stored_tiffs(
    patch_directory: Path | str,
) -> list[Path]:
    """
    Lista únicamente GeoTIFF ubicados directamente en el
    directorio operacional utilizado por storage.py.
    """

    patch_directory = Path(patch_directory)

    if not patch_directory.exists():
        raise FileNotFoundError(
            "No se encontró el directorio de GeoTIFF: "
            f"{patch_directory}"
        )

    if not patch_directory.is_dir():
        raise SpectralStorageAuditError(
            "La ruta de almacenamiento no es un directorio: "
            f"{patch_directory}"
        )

    return sorted(
        (
            path
            for path in patch_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_TIFF_EXTENSIONS
            )
        ),
        key=lambda path: (
            path.name.casefold(),
            path.name,
        ),
    )


def validate_stored_tif(
    path: Path | str,
    *,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Valida metadatos físicos y contenido de un GeoTIFF.

    El dtype se inspecciona antes de load_spectral_patch porque
    esa función convierte el array leído a float32.
    """

    path = Path(path)

    if contract is None:
        contract = load_spectral_contract(
            contract_path
        )

    expected_shape = tuple(
        int(value)
        for value in contract["shape"]
    )
    expected_dtype = str(
        contract["dtype"]
    )

    result: dict[str, Any] = {
        "filename": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": calculate_sha256(path),
        "count": None,
        "height": None,
        "width": None,
        "dtypes": [],
        "valid": False,
        "errors": [],
    }

    try:
        with rasterio.open(path) as dataset:
            result["count"] = int(dataset.count)
            result["height"] = int(dataset.height)
            result["width"] = int(dataset.width)
            result["dtypes"] = list(
                dataset.dtypes
            )

    except (RasterioIOError, OSError, ValueError) as error:
        result["errors"].append(
            "No fue posible abrir el GeoTIFF: "
            f"{error}"
        )
        return result

    expected_count, expected_height, expected_width = (
        expected_shape
    )

    if result["count"] != expected_count:
        result["errors"].append(
            "Cantidad de canales inválida. "
            f"Recibida: {result['count']}. "
            f"Esperada: {expected_count}."
        )

    if (
        result["height"] != expected_height
        or result["width"] != expected_width
    ):
        result["errors"].append(
            "Dimensiones espaciales inválidas. "
            "Recibidas: "
            f"{result['height']}x{result['width']}. "
            "Esperadas: "
            f"{expected_height}x{expected_width}."
        )

    if (
        not result["dtypes"]
        or any(
            dtype != expected_dtype
            for dtype in result["dtypes"]
        )
    ):
        result["errors"].append(
            "dtype físico inválido. "
            f"Recibido: {result['dtypes']}. "
            f"Esperado en todos los canales: {expected_dtype}."
        )

    try:
        load_spectral_patch(
            path,
            contract_path=contract_path,
        )

    except (
        FileNotFoundError,
        OSError,
        RasterioIOError,
        SpectralPatchError,
        ValueError,
    ) as error:
        message = str(error)

        if message not in result["errors"]:
            result["errors"].append(message)

    result["valid"] = not result["errors"]

    return result


def audit_spectral_storage(
    *,
    catalog: pd.DataFrame | None = None,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    catalog_manifest_path: Path | str = (
        DEFAULT_CATALOG_MANIFEST_PATH
    ),
    patch_directory: Path | str | None = None,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """
    Compara el catálogo con el almacenamiento y valida todos
    los GeoTIFF encontrados sin detenerse en el primer error.
    """

    if catalog is None:
        catalog = load_product_catalog(
            catalog_path=catalog_path,
            manifest_path=catalog_manifest_path,
        )

    if patch_directory is None:
        patch_directory = get_patch_directory()

    patch_directory = Path(
        patch_directory
    ).resolve()

    contract = load_spectral_contract(
        contract_path
    )

    expected_names, invalid_catalog_entries = (
        _validate_catalog_filenames(catalog)
    )

    names_by_portable_key: dict[str, list[str]] = {}

    for filename in expected_names:
        names_by_portable_key.setdefault(
            filename.casefold(),
            [],
        ).append(filename)

    duplicate_names = []

    for names in names_by_portable_key.values():
        if len(names) <= 1:
            continue

        variants = _sorted_names(
            set(names)
        )
        duplicate = {
            "filename": variants[0],
            "count": int(len(names)),
        }

        if len(variants) > 1:
            duplicate["variants"] = variants

        duplicate_names.append(
            duplicate
        )

    duplicate_names = sorted(
        duplicate_names,
        key=lambda item: (
            item["filename"].casefold(),
            item["filename"],
        ),
    )

    expected_unique_names = set(
        expected_names
    )

    stored_paths = list_stored_tiffs(
        patch_directory
    )
    stored_names = {
        path.name
        for path in stored_paths
    }

    missing_names = _sorted_names(
        expected_unique_names - stored_names
    )
    extra_names = _sorted_names(
        stored_names - expected_unique_names
    )
    matched_names = _sorted_names(
        expected_unique_names & stored_names
    )

    physical_results = [
        validate_stored_tif(
            path,
            contract_path=contract_path,
            contract=contract,
        )
        for path in stored_paths
    ]

    invalid_tiffs = [
        {
            "filename": result["filename"],
            "errors": result["errors"],
        }
        for result in physical_results
        if not result["valid"]
    ]

    summary = {
        "expected_tiffs": int(
            len(expected_names)
        ),
        "expected_unique_tiffs": int(
            len(expected_unique_names)
        ),
        "tiffs_found": int(
            len(stored_paths)
        ),
        "matched_tiffs": int(
            len(matched_names)
        ),
        "missing_tiffs": int(
            len(missing_names)
        ),
        "extra_tiffs": int(
            len(extra_names)
        ),
        "duplicate_names": int(
            len(duplicate_names)
        ),
        "invalid_catalog_entries": int(
            len(invalid_catalog_entries)
        ),
        "physically_valid_tiffs": int(
            sum(
                result["valid"]
                for result in physical_results
            )
        ),
        "physically_invalid_tiffs": int(
            len(invalid_tiffs)
        ),
    }

    report = {
        "patch_directory": str(
            patch_directory
        ),
        "summary": summary,
        "missing": missing_names,
        "extras": extra_names,
        "duplicates": duplicate_names,
        "invalid_catalog_entries": (
            invalid_catalog_entries
        ),
        "invalid_tiffs": invalid_tiffs,
        "files": physical_results,
    }

    report["is_valid"] = audit_is_valid(
        report
    )

    return report


def audit_is_valid(
    report: dict[str, Any],
) -> bool:
    """
    Indica si el almacenamiento coincide exactamente con el
    catálogo y todos los GeoTIFF cumplen el contrato.
    """

    summary = report["summary"]

    return all(
        (
            summary["expected_tiffs"]
            == summary["expected_unique_tiffs"],
            summary["expected_unique_tiffs"]
            == summary["tiffs_found"],
            summary["missing_tiffs"] == 0,
            summary["extra_tiffs"] == 0,
            summary["duplicate_names"] == 0,
            summary["invalid_catalog_entries"] == 0,
            summary["physically_invalid_tiffs"] == 0,
            summary["physically_valid_tiffs"]
            == summary["tiffs_found"],
        )
    )


def build_storage_manifest(
    report: dict[str, Any],
    *,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """
    Construye un manifest determinista para un almacenamiento
    previamente auditado y completamente válido.
    """

    if not audit_is_valid(report):
        raise SpectralStorageAuditError(
            "No se puede generar el manifest porque la "
            "auditoría del almacenamiento es inválida."
        )

    catalog_path = Path(catalog_path)
    contract_path = Path(contract_path)

    contract = load_spectral_contract(
        contract_path
    )

    files = [
        {
            "filename": result["filename"],
            "sha256": result["sha256"],
            "size_bytes": result["size_bytes"],
            "count": result["count"],
            "height": result["height"],
            "width": result["width"],
            "dtype": contract["dtype"],
        }
        for result in report["files"]
    ]

    files = sorted(
        files,
        key=lambda item: (
            item["filename"].casefold(),
            item["filename"],
        ),
    )

    inventory_payload = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    inventory_sha256 = hashlib.sha256(
        inventory_payload
    ).hexdigest().upper()

    summary = report["summary"]

    return {
        "manifest_version": "1.0.0",
        "artifact_type": "sentinel_8ch_storage",
        "storage_id": (
            "sentinel_8ch_storage_"
            f"{inventory_sha256[:8].lower()}"
        ),
        "catalog": {
            "file_name": catalog_path.name,
            "sha256": calculate_sha256(
                catalog_path
            ),
        },
        "spectral_contract": {
            "file_name": contract_path.name,
            "sha256": calculate_sha256(
                contract_path
            ),
            "channels": list(
                contract["channels"]
            ),
            "shape": list(
                contract["shape"]
            ),
            "dtype": contract["dtype"],
        },
        "statistics": {
            "expected_tiffs": summary[
                "expected_tiffs"
            ],
            "stored_tiffs": summary[
                "tiffs_found"
            ],
            "valid_tiffs": summary[
                "physically_valid_tiffs"
            ],
            "total_bytes": int(
                sum(
                    item["size_bytes"]
                    for item in files
                )
            ),
        },
        "inventory_sha256": inventory_sha256,
        "files": files,
    }


def save_storage_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = (
        DEFAULT_STORAGE_MANIFEST_PATH
    ),
) -> Path:
    """
    Guarda el manifest con formato estable y sin datos locales
    como rutas absolutas o fechas de ejecución.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )
        file.write("\n")

    return output_path


def print_audit_report(
    report: dict[str, Any],
) -> None:
    """
    Imprime un resumen legible de la auditoría.
    """

    summary = report["summary"]

    print()
    print("=" * 65)
    print("AUDITORÍA DE ALMACENAMIENTO SENTINEL-2")
    print("=" * 65)
    print(
        f"TIFF esperados: {summary['expected_tiffs']}"
    )
    print(
        f"TIFF encontrados: {summary['tiffs_found']}"
    )
    print(
        f"Coincidencias: {summary['matched_tiffs']}"
    )
    print(
        f"Faltantes: {summary['missing_tiffs']}"
    )
    print(
        f"Extras: {summary['extra_tiffs']}"
    )
    print(
        "Nombres duplicados: "
        f"{summary['duplicate_names']}"
    )
    print(
        "TIFF físicamente válidos: "
        f"{summary['physically_valid_tiffs']}"
    )
    print(
        "TIFF físicamente inválidos: "
        f"{summary['physically_invalid_tiffs']}"
    )
    print(
        "Resultado: "
        + (
            "VÁLIDO"
            if report["is_valid"]
            else "INVÁLIDO"
        )
    )
    print("=" * 65)

    for label, values in (
        ("Faltantes", report["missing"]),
        ("Extras", report["extras"]),
    ):
        if values:
            print()
            print(f"{label}:")
            for value in values:
                print(f"- {value}")

    if report["duplicates"]:
        print()
        print("Nombres duplicados:")
        for item in report["duplicates"]:
            variants = item.get(
                "variants"
            )
            variant_text = (
                f" ({', '.join(variants)})"
                if variants
                else ""
            )
            print(
                f"- {item['filename']}: "
                f"{item['count']} apariciones"
                f"{variant_text}"
            )

    if report["invalid_tiffs"]:
        print()
        print("TIFF inválidos:")
        for item in report["invalid_tiffs"]:
            print(f"- {item['filename']}")
            for error in item["errors"]:
                print(f"  · {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audita los GeoTIFF Sentinel-2 frente al "
            "catálogo del producto."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
    )
    parser.add_argument(
        "--catalog-manifest",
        type=Path,
        default=DEFAULT_CATALOG_MANIFEST_PATH,
    )
    parser.add_argument(
        "--patch-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Genera el manifest únicamente si la auditoría "
            "es completamente válida."
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_STORAGE_MANIFEST_PATH,
    )

    arguments = parser.parse_args()

    report = audit_spectral_storage(
        catalog_path=arguments.catalog,
        catalog_manifest_path=(
            arguments.catalog_manifest
        ),
        patch_directory=arguments.patch_directory,
        contract_path=arguments.contract,
    )

    print_audit_report(
        report
    )

    if not report["is_valid"]:
        raise SystemExit(1)

    if arguments.write_manifest:
        manifest = build_storage_manifest(
            report,
            catalog_path=arguments.catalog,
            contract_path=arguments.contract,
        )
        manifest_path = save_storage_manifest(
            manifest,
            arguments.manifest_output,
        )
        print(
            f"Manifest generado: {manifest_path}"
        )


if __name__ == "__main__":
    main()
