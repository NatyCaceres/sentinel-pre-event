import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import warnings
import zipfile

import pytest

from src.deployment import assets


def test_versioned_sources_preserve_integrity_with_linux_checkout(tmp_path):
    """Un clon sin autocrlf debe conservar los hashes usados en producción."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("La comprobación de checkout requiere Git.")
    manifest = json.loads(assets.MANIFEST_PATH.read_text(encoding="utf-8"))
    target = tmp_path / "checkout"
    target.mkdir()
    checkout = subprocess.run(
        [
            git, "--no-optional-locks", "-C", str(assets.ROOT),
            "-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "-c", "core.longpaths=true",
            "checkout-index", f"--prefix={target.as_posix()}/", "--",
            *[entry["path"] for entry in manifest["sources"]],
        ],
        capture_output=True, text=True,
    )
    assert checkout.returncode == 0, checkout.stderr
    assets.verify_sources(manifest, target)


def entry(name, data):
    return {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "config").mkdir()
    source = b'contract-v1'
    (root / "config/contract.json").write_bytes(source)
    contents = {assets.CHECKPOINT: b"test-weights", assets.PATCH_PREFIX + "sample.tif": b"test-tiff"}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, data in contents.items():
            output.writestr(name, data)
    manifest = {
        "schema_version": 1, "dataset_split": "test", "patch_count": 1,
        "sources": [entry("config/contract.json", source)],
        "files": [entry(name, data) for name, data in contents.items()],
        "bundle": entry(archive.name, archive.read_bytes()),
    }
    return root, archive, manifest, contents


def test_verified_install_preserves_names_and_unrelated_files(bundle):
    root, archive, manifest, contents = bundle
    (root / "keep.txt").write_text("keep")
    assets.install_archive(archive, manifest, root)
    for name, data in contents.items():
        assert (root / name).read_bytes() == data
    assert (root / "keep.txt").read_text() == "keep"
    assert list((root / ".cloud").iterdir()) == []


def test_corrupt_zip_does_not_change_existing_files(bundle):
    root, archive, manifest, contents = bundle
    target = root / assets.CHECKPOINT
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    archive.write_bytes(archive.read_bytes()[:-1])
    with pytest.raises(assets.CloudAssetError, match="SHA-256"):
        assets.install_archive(archive, manifest, root)
    assert target.read_bytes() == b"existing"
    assert not (root / assets.PATCH_PREFIX).exists()


@pytest.mark.parametrize("mode", ["extra", "missing", "duplicate", "changed", "traversal", "symlink"])
def test_zip_contents_are_checked_before_install(bundle, mode):
    root, archive, manifest, contents = bundle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w") as output:
            for name, data in contents.items():
                if mode == "missing" and name.endswith(".tif"):
                    continue
                if mode == "symlink" and name.endswith(".tif"):
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = 0o120777 << 16
                    output.writestr(info, data)
                else:
                    output.writestr(name, b"X" * len(data) if mode == "changed" else data)
            if mode in ("extra", "traversal"):
                output.writestr("extra.txt" if mode == "extra" else "../outside.txt", b"bad")
            if mode == "duplicate":
                output.writestr(assets.CHECKPOINT, contents[assets.CHECKPOINT])
    manifest["bundle"] = entry(archive.name, archive.read_bytes())
    with pytest.raises(assets.CloudAssetError):
        assets.install_archive(archive, manifest, root)
    assert not (root / assets.CHECKPOINT).exists()
    assert not (root.parent / "outside.txt").exists()


def test_changed_contract_blocks_install(bundle):
    root, archive, manifest, _ = bundle
    (root / "config/contract.json").write_bytes(b"contract-v2")
    with pytest.raises(assets.CloudAssetError, match="contratos cambiaron"):
        assets.install_archive(archive, manifest, root)


def test_download_once_and_repair_corruption(bundle, monkeypatch):
    root, archive, manifest, contents = bundle
    manifest_path = root / "config/cloud_bundle.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def download(url, target, expected_size):
        calls.append(url)
        shutil.copyfile(archive, target)

    monkeypatch.setattr(assets, "download_archive", download)
    for _ in range(2):
        assets.ensure_cloud_assets("https://example.org/bundle.zip", root=root, manifest_path=manifest_path)
    assert len(calls) == 1
    (root / assets.CHECKPOINT).write_bytes(b"damaged")
    assets.ensure_cloud_assets("https://example.org/bundle.zip", root=root, manifest_path=manifest_path)
    assert len(calls) == 2
    assert (root / assets.CHECKPOINT).read_bytes() == contents[assets.CHECKPOINT]
    assert list((root / ".cloud").iterdir()) == []


def test_missing_url_is_actionable(bundle):
    root, archive, manifest, _ = bundle
    manifest_path = root / "config/cloud_bundle.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(assets.CloudAssetError, match="CLOUD_BUNDLE_URL"):
        assets.ensure_cloud_assets(root=root, manifest_path=manifest_path)


@pytest.mark.parametrize("url", ["http://example.org/data.zip", "file:///tmp/data.zip", "https://user:password@example.org/data.zip"])
def test_download_requires_https_without_credentials(tmp_path, url):
    with pytest.raises(assets.CloudAssetError, match="HTTPS"):
        assets.download_archive(url, tmp_path / "download.zip", 10)


@pytest.mark.parametrize("data", [b"short", b"this is too long"])
def test_download_rejects_wrong_size(tmp_path, monkeypatch, data):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(data)
    monkeypatch.setattr(assets.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(assets.CloudAssetError):
        assets.download_archive("https://example.org/data.zip", tmp_path / "download.zip", 10)


def test_download_failure_removes_temporary_files(bundle, monkeypatch):
    root, archive, manifest, _ = bundle
    manifest_path = root / "config/cloud_bundle.json"
    manifest_path.write_text(json.dumps(manifest))

    def fail(url, target, expected_size):
        Path(target).write_bytes(b"partial")
        raise assets.CloudAssetError("Descarga interrumpida")

    monkeypatch.setattr(assets, "download_archive", fail)
    with pytest.raises(assets.CloudAssetError):
        assets.ensure_cloud_assets("https://example.org/bundle.zip", root=root, manifest_path=manifest_path)
    assert list((root / ".cloud").iterdir()) == []
    assert not (root / assets.CHECKPOINT).exists()
