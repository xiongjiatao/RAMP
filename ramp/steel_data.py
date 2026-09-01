"""Validated Steel-FJSP bundle loader for the first-paper data interface."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_utils import load_data_from_single_file


@dataclass(frozen=True)
class SteelSplitMetadata:
    split: str | None
    status: str
    manifest_path: Path


@dataclass(frozen=True)
class SteelInstanceBundle:
    instance_path: Path
    variant: str
    job_lengths: np.ndarray
    nominal_processing_times: np.ndarray
    machine_mapping: tuple[dict[str, str], ...]
    job_metadata: tuple[dict[str, str], ...]
    stochastic_profile: dict[tuple[int, int, int], dict[str, str]]
    transport_matrix: np.ndarray
    split_metadata: SteelSplitMetadata
    manifest_record: dict[str, str]
    dataset_manifest_hash: str
    exact_transport_consumed: bool = False


_ROUTES = {
    "单LF": ("KR", "BOF", "LF", "CC"),
    "单RH": ("KR", "BOF", "RH", "CC"),
    "双精炼": ("KR", "BOF", "LF", "RH", "CC"),
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bundle_root(instance: Path) -> Path:
    for parent in (instance.parent, *instance.parents):
        if (parent / "metadata" / "instance_manifest.csv").is_file():
            return parent
    raise ValueError(f"{instance} is not inside a Steel_FJSP_Real_v1 bundle")


def _relative(instance: Path, root: Path) -> str:
    return instance.resolve().relative_to(root.resolve()).as_posix()


def _manifest_hash(root: Path, instance: Path, split_manifest: Path) -> str:
    digest = hashlib.sha256()
    files = (
        root / "metadata" / "source_manifest.json",
        root / "metadata" / "instance_manifest.csv",
        root / "metadata" / "machine_mapping.csv",
        root / "metadata" / "jobs_expanded.csv",
        root / "metadata" / "transport_matrix.csv",
        root / "stochastic_profiles" / "operation_machine_profiles.csv",
        split_manifest,
        instance,
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _load_transport(path: Path, machine_names: list[str]) -> np.ndarray:
    rows = _csv_rows(path)
    if [row["from_machine"] for row in rows] != machine_names:
        raise ValueError("transport matrix row order does not match machine mapping")
    matrix = np.full((len(machine_names), len(machine_names)), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, name in enumerate(machine_names):
            value = row[name].strip()
            if value:
                matrix[row_index, column_index] = float(value)
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("transport matrix diagonal must be zero")
    return matrix


def validate_steel_split(root: str | Path) -> dict[str, Any]:
    """Validate the frozen 8/1/2 job- and template-disjoint first-paper split."""

    root = Path(root).resolve()
    split_path = root / "splits" / "first_paper_10x15_split.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    instances = {
        row["relative_path"]: row
        for row in _csv_rows(root / "metadata" / "instance_manifest.csv")
    }
    jobs = {
        row["job_id"]: row
        for row in _csv_rows(root / "metadata" / "jobs_expanded.csv")
    }
    seen_jobs: dict[str, str] = {}
    seen_templates: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for relative in payload[split]:
            record = instances[relative]
            for job_id in record["source_job_ids"].split(","):
                job_id = job_id.strip()
                if job_id in seen_jobs and seen_jobs[job_id] != split:
                    raise ValueError(f"job {job_id} leaks across Steel splits")
                template = jobs[job_id]["template_id"]
                if template in seen_templates and seen_templates[template] != split:
                    raise ValueError(f"template {template} leaks across Steel splits")
                seen_jobs[job_id] = split
                seen_templates[template] = split
    quarantine = set(payload["quarantined"])
    if any(path in quarantine for split in ("train", "validation", "test") for path in payload[split]):
        raise ValueError("a quarantined Steel instance appears in an admitted split")
    return payload


def load_steel_instance_bundle(
    path: str | Path, *, validate: bool = True
) -> SteelInstanceBundle:
    """Load one `.fjs` and every indexed Steel sidecar without changing physics."""

    instance = Path(path).resolve()
    if instance.suffix.lower() != ".fjs" or not instance.is_file():
        raise ValueError("load_steel_instance_bundle requires an existing .fjs path")
    root = _bundle_root(instance)
    relative = _relative(instance, root)
    records = _csv_rows(root / "metadata" / "instance_manifest.csv")
    matches = [row for row in records if row["relative_path"] == relative]
    if len(matches) != 1:
        raise ValueError(f"instance manifest has {len(matches)} matches for {relative}")
    record = matches[0]
    job_lengths, nominal = load_data_from_single_file(str(instance), strict=True)
    if validate and (
        len(job_lengths) != int(record["job_count"])
        or nominal.shape != (int(record["operation_count"]), int(record["machine_count"]))
    ):
        raise ValueError("Steel .fjs shape disagrees with instance manifest")

    machines = tuple(_csv_rows(root / "metadata" / "machine_mapping.csv"))
    ids = [int(row["machine_id"]) for row in machines]
    if ids != list(range(1, 16)) or nominal.shape[1] != 15:
        raise ValueError("Steel machine mapping must be exactly IDs 1..15")
    machine_names = [row["machine_name"] for row in machines]
    source_ids = [value.strip() for value in record["source_job_ids"].split(",")]
    all_jobs = {
        row["job_id"]: row
        for row in _csv_rows(root / "metadata" / "jobs_expanded.csv")
    }
    selected_jobs = tuple(all_jobs[job_id] for job_id in source_ids)
    profiles = _csv_rows(
        root / "stochastic_profiles" / "operation_machine_profiles.csv"
    )
    profile_map: dict[tuple[int, int, int], dict[str, str]] = {}
    global_to_local = {job_id: local for local, job_id in enumerate(source_ids)}
    for row in profiles:
        if row["variant"] != record["variant"] or row["job_id"] not in global_to_local:
            continue
        key = (
            global_to_local[row["job_id"]],
            int(row["operation_index"]) - 1,
            int(row["machine_id"]) - 1,
        )
        if key in profile_map:
            if profile_map[key] != row:
                raise ValueError(f"conflicting stochastic profile key {key}")
            continue
        profile_map[key] = row

    operation_offset = 0
    for local_job, (job, length) in enumerate(zip(selected_jobs, job_lengths)):
        stages: list[str] = []
        for operation in range(int(length)):
            feasible = np.flatnonzero(nominal[operation_offset + operation] > 0)
            operation_stages = set()
            for machine in feasible:
                key = (local_job, operation, int(machine))
                if key not in profile_map:
                    raise ValueError(f"missing stochastic profile {key}")
                profile = profile_map[key]
                if int(profile["fjs_processing_time_integer"]) != int(
                    nominal[operation_offset + operation, machine]
                ):
                    raise ValueError(f"stochastic profile processing time mismatch {key}")
                operation_stages.add(profile["stage"])
            if len(operation_stages) != 1:
                raise ValueError(f"operation {local_job}:{operation} spans multiple stages")
            stages.append(operation_stages.pop())
        expected_route = _ROUTES.get(job["route"])
        if expected_route is None or tuple(stages) != expected_route:
            raise ValueError(
                f"route mismatch for job {job['job_id']}: {tuple(stages)} vs {expected_route}"
            )
        operation_offset += int(length)

    split_path = root / "splits" / "first_paper_10x15_split.json"
    split_payload = validate_steel_split(root) if validate else json.loads(
        split_path.read_text(encoding="utf-8")
    )
    split_name = next(
        (name for name in ("train", "validation", "test") if relative in split_payload[name]),
        None,
    )
    status = (
        "QUARANTINED_SPLIT_BOUNDARY_LEAKAGE"
        if relative in split_payload["quarantined"]
        else "VALID"
    )
    return SteelInstanceBundle(
        instance_path=instance,
        variant=record["variant"],
        job_lengths=job_lengths.astype(np.int64, copy=False),
        nominal_processing_times=nominal.astype(np.float32, copy=False),
        machine_mapping=machines,
        job_metadata=selected_jobs,
        stochastic_profile=profile_map,
        transport_matrix=_load_transport(
            root / "metadata" / "transport_matrix.csv", machine_names
        ),
        split_metadata=SteelSplitMetadata(split_name, status, split_path),
        manifest_record=record,
        dataset_manifest_hash=_manifest_hash(root, instance, split_path),
    )
