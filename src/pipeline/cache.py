import hashlib
import json
from datetime import datetime
from pathlib import Path

CACHE_VERSION = 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def _gather_directory_metadata(root: Path) -> list[dict]:
    if not root.exists():
        return []

    files = [p for p in root.iterdir() if p.is_file()]
    return sorted([_file_metadata(p) for p in files], key=lambda item: item["path"])


def compute_manifest(config_path: str, raw_dir: str) -> dict:
    config_text = Path(config_path).read_text(encoding="utf-8")
    raw_path = Path(raw_dir)
    return {
        "version": CACHE_VERSION,
        "config_hash": sha256_text(config_text),
        "raw_files": _gather_directory_metadata(raw_path),
    }


def load_manifest(path: str) -> dict | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(path: str, manifest: dict) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def outputs_exist(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def is_cache_valid(manifest_path: str, config_path: str, raw_dir: str, expected_outputs: list[str]) -> bool:
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return False
    if manifest.get("version") != CACHE_VERSION:
        return False

    current = compute_manifest(config_path, raw_dir)
    if manifest.get("config_hash") != current.get("config_hash"):
        return False
    if manifest.get("raw_files") != current.get("raw_files"):
        return False
    return outputs_exist(expected_outputs)


def write_cache_manifest(manifest_path: str, config_path: str, raw_dir: str, expected_outputs: list[str]) -> None:
    manifest = compute_manifest(config_path, raw_dir)
    manifest["outputs"] = [str(Path(p).resolve()) for p in expected_outputs]
    manifest["generated_at"] = datetime.now().isoformat()
    save_manifest(manifest_path, manifest)
