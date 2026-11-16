"""Genera un ZIP determinista y su manifest sin publicar ni renombrar originales."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.product_catalog import DEFAULT_CATALOG_PATH, DEFAULT_CATALOG_MANIFEST_PATH, calculate_sha256
from src.data.product_catalog_loader import load_product_catalog, select_final_test_catalog
from src.deployment.assets import CHECKPOINT, PATCH_PREFIX, MANIFEST_PATH, verify_archive
from src.spectral.inference import DEFAULT_INFERENCE_CONTRACT_PATH, load_inference_contract
from src.spectral.patch import DEFAULT_CONTRACT_PATH
from src.spectral.storage import resolve_tif_path
from src.spectral.storage_audit import DEFAULT_STORAGE_MANIFEST_PATH, validate_stored_tif


def record(path, name):
    return {"path": name, "size_bytes": path.stat().st_size, "sha256": calculate_sha256(path).lower()}


def build_bundle():
    catalog = select_final_test_catalog(load_product_catalog())
    contract = load_inference_contract()
    inventory = json.loads(DEFAULT_STORAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if calculate_sha256(DEFAULT_CATALOG_PATH).lower() != inventory["catalog"]["sha256"].lower():
        raise ValueError("El catálogo no coincide con la auditoría de almacenamiento.")
    existing = {entry["filename"]: entry for entry in inventory["files"]}
    files = []
    paths = {}
    for name in sorted(catalog["tif_filename"]):
        path = resolve_tif_path(name)
        checked = validate_stored_tif(path)
        if not checked["valid"] or checked["sha256"].lower() != existing[name]["sha256"].lower():
            raise ValueError(f"TIFF inválido o distinto del auditado: {name}")
        relative = PATCH_PREFIX + name
        files.append(record(path, relative))
        paths[relative] = path
    checkpoint = (DEFAULT_INFERENCE_CONTRACT_PATH.parent / contract["model"]["checkpoint_relative_path"]).resolve()
    if checkpoint != (ROOT / CHECKPOINT).resolve() or calculate_sha256(checkpoint).lower() != contract["model"]["checkpoint_sha256"]:
        raise ValueError("El checkpoint no coincide con el contrato de inferencia.")
    files.append(record(checkpoint, CHECKPOINT))
    paths[CHECKPOINT] = checkpoint
    files.sort(key=lambda entry: entry["path"])
    output = ROOT / ".cloud"
    output.mkdir(exist_ok=True)
    target = output / "sentinel-test-resnet18-v1.zip"
    with tempfile.TemporaryDirectory(prefix="build-", dir=output) as temporary:
        archive_path = Path(temporary) / target.name
        # Sin compresión: independiente de la versión de zlib. Fecha, permisos
        # y orden fijos; los mismos bytes producen el mismo SHA-256.
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for entry in files:
                info = zipfile.ZipInfo(entry["path"], date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                data = paths[entry["path"]].read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Un archivo cambió durante la preparación.")
                archive.writestr(info, data)
        manifest = {
            "schema_version": 1,
            "dataset_split": "test",
            "patch_count": len(catalog),
            "bundle": record(archive_path, target.name),
            "sources": [record(path, path.relative_to(ROOT).as_posix()) for path in (
                DEFAULT_CATALOG_PATH, DEFAULT_CATALOG_MANIFEST_PATH,
                DEFAULT_CONTRACT_PATH, DEFAULT_INFERENCE_CONTRACT_PATH,
            )],
            "files": files,
        }
        verify_archive(archive_path, manifest)
        os.replace(archive_path, target)
        manifest_temp = Path(temporary) / "manifest.json"
        manifest_temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(manifest_temp, MANIFEST_PATH)
    print(f"ZIP: {target}")
    print(f"Contenido: {len(catalog)} TIFF + 1 checkpoint; {target.stat().st_size / 1024**2:.2f} MiB")
    print(f"SHA-256: {manifest['bundle']['sha256']}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Paquete verificado. No se publicó ningún archivo.")
    return manifest


if __name__ == "__main__":
    build_bundle()
