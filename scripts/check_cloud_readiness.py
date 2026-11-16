"""Comprobaciones previas al despliegue, sin instalar ni escribir artefactos.

--packages: consulta metadatos oficiales para Linux x86_64 / CPython 3.13.
--runtime: verifica archivos locales y mide tres inferencias y Grad-CAM.
Estas comprobaciones no sustituyen una instalación y prueba real en Linux.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import email
import json
import platform
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def check_packages():
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
    tags = set(cpython_tags((3, 13), abis=["cp313"], platforms=platforms))
    tags.update(compatible_tags((3, 13), interpreter="cp313", platforms=platforms))
    requirements = [
        Requirement(line.strip())
        for line in (ROOT / "app/requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    def inspect(req):
        name = canonicalize_name(req.name)
        if req.url:
            url = req.url.split("#", 1)[0]
            filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[1])
            _, version, _, wheel_tags = parse_wheel_filename(filename)
            if not tags.intersection(wheel_tags) or "#sha256=" not in req.url:
                raise ValueError(f"Wheel o hash incompatible: {name}")
            metadata = email.message_from_bytes(fetch(url + ".metadata"))
            if metadata["Version"] != str(version):
                raise ValueError(f"Metadatos incompatibles: {name}")
            return name, str(version), metadata.get_all("Requires-Dist", [])
        pins = list(req.specifier)
        if len(pins) != 1 or pins[0].operator != "==":
            raise ValueError(f"Se requiere versión exacta: {name}")
        version = pins[0].version
        metadata = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
        if not SpecifierSet(metadata["info"].get("requires_python") or "").contains("3.13.15"):
            raise ValueError(f"Python incompatible: {name}=={version}")
        compatible = []
        for file in metadata["urls"]:
            if file["filename"].endswith(".whl") and not file.get("yanked"):
                _, _, _, wheel_tags = parse_wheel_filename(file["filename"])
                if tags.intersection(wheel_tags):
                    compatible.append(file)
        if not compatible:
            raise ValueError(f"No hay wheel compatible con Linux/CPython 3.13: {name}=={version}")
        return name, version, metadata["info"].get("requires_dist") or []

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(inspect, requirements))
    versions = {name: version for name, version, _ in records}
    metadata = {name: deps for name, _, deps in records}
    environment = default_environment()
    environment.update(
        sys_platform="linux", platform_system="Linux", os_name="posix",
        platform_machine="x86_64", python_version="3.13", python_full_version="3.13.15",
        implementation_name="cpython", platform_python_implementation="CPython",
    )
    # Incluye dependencias activadas por extras y sus restricciones de versión.
    pending = [(canonicalize_name(req.name), frozenset(req.extras)) for req in requirements]
    visited = set()
    while pending:
        name, extras = pending.pop()
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        for raw in metadata[name]:
            dep = Requirement(raw)
            if dep.marker and not any(dep.marker.evaluate(dict(environment, extra=e)) for e in ({""} | set(extras))):
                continue
            dependency = canonicalize_name(dep.name)
            if dependency not in versions:
                raise ValueError(f"Dependencia sin fijar: {name} requiere {dep}")
            if not dep.specifier.contains(versions[dependency]):
                raise ValueError(f"Conflicto: {name} requiere {dep}; fijado {versions[dependency]}")
            pending.append((dependency, frozenset(dep.extras)))
    print(f"Paquetes: {len(records)} versiones con wheel Linux compatible y restricciones coherentes.")
    print("PyTorch/torchvision: wheels CPU con SHA-256 fijado. No se descargaron wheels.")
    print("Pendiente: instalación real en Linux y verificación de dependencias nativas.")


def peak_memory_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def check_runtime():
    start = time.perf_counter()
    # Incluye las bibliotecas principales de la interfaz en la medición.
    import streamlit
    import streamlit_folium
    import numpy as np
    from src.data.product_catalog_loader import load_product_catalog, select_final_test_catalog
    from src.spectral.inference import load_spectral_model, predict_spectral_patch
    from src.spectral.gradcam import calculate_gradcam
    from src.spectral.patch import load_spectral_patch
    from src.spectral.storage import resolve_tif_path

    catalog = select_final_test_catalog(load_product_catalog())
    paths = [resolve_tif_path(name) for name in catalog["tif_filename"]]
    patch_bytes = sum(path.stat().st_size for path in paths)
    print(f"Archivos test disponibles: {len(paths)} ({patch_bytes / 1024**2:.2f} MiB).")
    model = load_spectral_model()
    print(f"Checkpoint verificado: {model.checkpoint_path.stat().st_size / 1024**2:.2f} MiB.")
    print(f"Importaciones, catálogo y carga del modelo: {time.perf_counter() - start:.2f} s.")
    for name in (
        "model8__test__neg__P_2018_NEG_00015.tif",
        "model8__test__pos__P_2019_POS_C0079_01.tif",
        "model8__test__neg__P_2018_NEG_00014.tif",
    ):
        patch = load_spectral_patch(resolve_tif_path(name))
        start = time.perf_counter()
        prediction = predict_spectral_patch(patch, model)
        inference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = calculate_gradcam(patch, model, target_class=prediction.predicted_class)
        gradcam_seconds = time.perf_counter() - start
        if not np.isfinite(result.heatmap).all() or abs(result.target_probability - prediction.probabilities[result.target_class]) > 1e-6:
            raise ValueError(f"Grad-CAM incompatible con inferencia: {name}")
        if predict_spectral_patch(patch, model) != prediction:
            raise ValueError(f"La predicción cambió: {name}")
        print(f"{name}: inferencia {inference_seconds:.3f} s; Grad-CAM {gradcam_seconds:.3f} s; COINCIDE.")
    print(f"Pico de memoria residente del proceso: {peak_memory_mib():.1f} MiB.")
    print(f"Medición local: {platform.system()}, Python {platform.python_version()}, CPU, uso secuencial.")
    print("No representa el tiempo de arranque ni la memoria multiusuario de Community Cloud.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", action="store_true", help="Consulta metadatos oficiales por Internet.")
    parser.add_argument("--runtime", action="store_true", help="Mide inferencia local con el modelo y TIFF existentes.")
    args = parser.parse_args()
    if not (args.packages or args.runtime):
        parser.error("Selecciona --packages, --runtime o ambos.")
    try:
        if args.packages:
            check_packages()
        if args.runtime:
            check_runtime()
    except Exception as error:
        print(f"COMPROBACIÓN FALLIDA: {error}", file=sys.stderr)
        return 1
    print("Comprobaciones solicitadas: OK. Esto todavía no es un despliegue publicado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
