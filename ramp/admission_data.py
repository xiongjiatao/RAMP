"""Manifest-authoritative dataset admission for formal first-paper suites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_manifest_paths(root: Path, manifest: Path) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for entry in payload["entries"]:
        if entry["status"] != "VALID":
            continue
        path = root / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"dataset admission hash mismatch: {path}")
        paths.append(path.resolve())
    return sorted(paths)


def resolve_admission_suite(
    repository_root: str | Path, suite_name: str
) -> dict[str, list[Path]]:
    """Resolve a frozen suite without admitting quarantined manifest entries."""

    repository_root = Path(repository_root).resolve()
    contract = json.loads(
        (repository_root / "configs/fjsp_admission_suites.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        suite: dict[str, Any] = contract["suites"][suite_name]
    except KeyError as exc:
        raise ValueError(f"unknown admission suite: {suite_name}") from exc
    if suite["type"] != "manifest_partition":
        raise ValueError(f"unsupported admission suite type: {suite['type']}")
    train_root = repository_root / suite["train_root"]
    reference_root = repository_root / suite["reference_root"]
    train = _valid_manifest_paths(
        train_root, repository_root / suite["train_manifest"]
    )
    reference = _valid_manifest_paths(
        reference_root, repository_root / suite["reference_manifest"]
    )
    modulus = int(suite["reference_partition"]["modulus"])
    validation_remainder = int(
        suite["reference_partition"]["validation_remainder"]
    )
    validation = [p for index, p in enumerate(reference) if index % modulus == validation_remainder]
    test = [p for index, p in enumerate(reference) if index % modulus != validation_remainder]
    result = {"train": train, "validation": validation, "test": test}
    expected = suite["expected_counts"]
    actual = {name: len(paths) for name, paths in result.items()}
    if actual != expected:
        raise ValueError(f"admission suite count mismatch: {actual} != {expected}")
    if any(
        set(result[left]) & set(result[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("admission suite paths overlap")
    return result
