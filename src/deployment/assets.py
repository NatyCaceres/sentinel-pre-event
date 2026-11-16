"""Distribución verificada del modelo y TIFF; sin dependencias nuevas."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from threading import RLock
import urllib.parse
import urllib.request
import zipfile

from src.data.product_catalog import calculate_sha256

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "config/cloud_bundle.json"
CHECKPOINT = "artifacts/models/resnet18_8ch_v2/best_resnet18_8ch_v2.pt"
PATCH_PREFIX = "data/patches/sentinel_8ch/"
_LOCK = RLock()


class CloudAssetError(Exception):
    """El paquete remoto no es compatible con este despliegue."""


def checked_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or path.is_absolute() or ".." in path.parts:
        raise CloudAssetError("El manifest contiene una ruta no permitida.")
    resolved = (Path(root) / relative).resolve()
    if not resolved.is_relative_to(Path(root).resolve()) or resolved == Path(root).resolve():
        raise CloudAssetError("La ruta sale de la carpeta del proyecto.")
    return resolved


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise CloudAssetError("Manifest de despliegue inválido.")
    paths = []
    for entry in manifest["files"]:
        name = entry["path"]
        valid_patch = name.startswith(PATCH_PREFIX) and PurePosixPath(name).parent.as_posix() == PATCH_PREFIX.rstrip("/") and name.lower().endswith(".tif")
        if name != CHECKPOINT and not valid_patch:
            raise CloudAssetError("El paquete solo puede contener el checkpoint y TIFF.")
        checked_path(ROOT, name)
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or entry["size_bytes"] <= 0:
            raise CloudAssetError("Hash o tamaño de archivo inválido.")
        paths.append(name)
    if len(paths) != len(set(paths)) or paths.count(CHECKPOINT) != 1:
        raise CloudAssetError("Inventario duplicado o checkpoint ausente.")
    if manifest.get("dataset_split") != "test" or manifest.get("patch_count") != len(paths) - 1:
        raise CloudAssetError("La partición o cantidad de parches del paquete no coincide.")
    bundle = manifest["bundle"]
    if not re.fullmatch(r"[0-9a-f]{64}", bundle["sha256"]) or bundle["size_bytes"] <= 0:
        raise CloudAssetError("Hash o tamaño del ZIP inválido.")


def file_matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["size_bytes"] and calculate_sha256(path).lower() == entry["sha256"]


def verify_sources(manifest, root):
    for entry in manifest["sources"]:
        if not file_matches(checked_path(root, entry["path"]), entry):
            raise CloudAssetError("El catálogo o los contratos cambiaron. Regenera el paquete de despliegue.")


def verify_archive(archive_path, manifest):
    """Verifica inventario, tamaño y bytes antes de escribir archivos destino."""
    validate_manifest(manifest)
    if not file_matches(Path(archive_path), manifest["bundle"]):
        raise CloudAssetError("El tamaño o SHA-256 del ZIP no coincide.")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise CloudAssetError("El ZIP contiene archivos faltantes, extras o duplicados.")
        for member in members:
            entry = expected[member.filename]
            if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) or member.file_size != entry["size_bytes"]:
                raise CloudAssetError("Tipo o tamaño de entrada ZIP inválido.")
            digest = hashlib.sha256()
            with archive.open(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise CloudAssetError("Un archivo dentro del ZIP no coincide con su SHA-256.")


def install_archive(archive_path, manifest, root=ROOT):
    """Extrae a una carpeta temporal y sustituye solo archivos verificados."""
    verify_sources(manifest, root)
    verify_archive(archive_path, manifest)
    cache = checked_path(root, ".cloud")
    cache.mkdir(parents=True, exist_ok=True)
    targets = {entry["path"]: checked_path(root, entry["path"]) for entry in manifest["files"]}
    with tempfile.TemporaryDirectory(prefix="extract-", dir=cache) as temporary:
        with zipfile.ZipFile(archive_path) as archive:
            for entry in manifest["files"]:
                target = checked_path(temporary, entry["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry["path"]) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        for entry in manifest["files"]:
            target = targets[entry["path"]]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(checked_path(temporary, entry["path"]), target)


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise CloudAssetError("La descarga debe mantenerse en HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(url, target, size_bytes):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise CloudAssetError("Configura una URL HTTPS de descarga directa.")
    opener = urllib.request.build_opener(_HTTPSRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "Sentinel-Pre-Event-MVP"})
    try:
        with opener.open(request, timeout=60) as response, Path(target).open("wb") as output:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > size_bytes:
                    raise CloudAssetError("La descarga supera el tamaño esperado.")
                output.write(block)
            if received != size_bytes:
                raise CloudAssetError("La descarga está incompleta.")
    except CloudAssetError:
        raise
    except Exception:
        # No exponer URL firmada, token o detalles de la petición en la interfaz.
        raise CloudAssetError("No se pudo descargar el paquete. Revisa la URL y su acceso.") from None


def ensure_cloud_assets(url="", *, manifest_path=MANIFEST_PATH, root=ROOT):
    """Se invoca una vez por proceso mediante cache_resource del arranque cloud."""
    with _LOCK:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        validate_manifest(manifest)
        verify_sources(manifest, root)
        if all(file_matches(checked_path(root, entry["path"]), entry) for entry in manifest["files"]):
            return
        if not url:
            raise CloudAssetError("Falta configurar CLOUD_BUNDLE_URL en los Secrets de Streamlit.")
        cache = checked_path(root, ".cloud")
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as temporary:
            archive = Path(temporary) / "bundle.zip"
            download_archive(url, archive, manifest["bundle"]["size_bytes"])
            install_archive(archive, manifest, root)
