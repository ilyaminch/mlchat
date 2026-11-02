"""Собрать эталонный wheel из неизменённого ZIP во временном каталоге.

Это инструмент совместимости MLChat. В репозиторий не копируются исходники backend;
оригинальный архив не меняется. Требует установленных build и hatchling.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REFERENCE_SHA256 = "b6860f90057a450665039feb8fd1043106f91952884b20a973e985c5a67f0aa7"
PREFIX = "backend-main/backend/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".artifacts/backend"))
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != REFERENCE_SHA256:
        parser.error("ZIP отличается от проверенного backend. Нужна новая проверка совместимости.")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module_hashes = {}
    with tempfile.TemporaryDirectory(prefix="mlchat-backend-") as folder:
        root = Path(folder).resolve()
        with ZipFile(args.archive) as archive:
            for name in archive.namelist():
                if not name.startswith(PREFIX) or name.endswith("/"):
                    continue
                relative = PurePosixPath(name.removeprefix(PREFIX))
                if relative != PurePosixPath("pyproject.toml") and not (
                    relative.parts[0] == "app" and relative.suffix == ".py"
                ):
                    continue
                destination = root.joinpath(*relative.parts).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("Недопустимый путь в архиве")
                data = archive.read(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                module_hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
            cwd=root,
            check=True,
        )
    wheel = output / "gis_assistant_backend-0.1.0-py3-none-any.whl"
    with wheel.open("rb") as stream:
        wheel_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "package": "gis-assistant-backend",
        "version": "0.1.0",
        "archive_sha256": digest,
        "wheel": wheel.name,
        "wheel_sha256": wheel_hash,
        "source_sha256": module_hashes,
    }
    (output / "reference-build.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Backend wheel: {wheel.name}; SHA-256: {wheel_hash}")


if __name__ == "__main__":
    main()
