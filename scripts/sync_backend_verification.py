"""Populate ignored local verification data from backend, preserving acceptance hashes."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("objects.geojson", "regions.geojson", "sources.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", type=Path, default=ROOT.parent / "backend")
    args = parser.parse_args()
    backend = args.backend_dir.resolve()
    reference = json.loads((backend / "data/verification/source-reference.json").read_text())
    # Acceptance expectations belong to MLChat; never regenerate them from GIS/model output.
    acceptance_path = ROOT / "fixtures/mlchat/dataset-reference.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.is_file() else None
    contents = {}
    for name in NAMES:
        key = f"data/verification/{name}"
        content = (backend / key).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if reference["files"].get(key) != digest:
            raise ValueError(f"Backend source checksum mismatch: {name}")
        if acceptance is not None and key in acceptance["files"]:
            if acceptance["files"][key] != digest:
                raise ValueError(f"MLChat acceptance dataset differs: {name}; review expectations")
        contents[name] = content
    # Validate all inputs before copying. No network, model execution or database writes.
    destination = ROOT / "data/verification"
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (destination / name).write_bytes(content)
    print("Synced 28+4 verification sources from backend; acceptance hashes preserved.")
    if acceptance is None:
        print("MLChat acceptance fixtures are not installed locally; no acceptance claim made.")


if __name__ == "__main__":
    main()
