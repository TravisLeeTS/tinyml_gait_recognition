from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def download_first_available(urls: list[str], destination: Path, timeout: int = 60) -> str:
    """Download from the first reachable URL and return the URL used."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return "existing-file"

    errors: list[str] = []
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "tinyml-har-repro/1.0"})
            with urlopen(request, timeout=timeout) as response:
                if getattr(response, "status", 200) >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
                with destination.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if destination.exists():
                destination.unlink()
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


def extract_archive(archive_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination_dir)
        return
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(destination_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")
