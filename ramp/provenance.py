"""Recursive active-code and formal-input provenance authority."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Iterable


# This RAMP fork deliberately quarantines the standalone ATMSL quality-gate
# runners.  The active ATMSL library imported by train_ramp.py remains in the
# recursive graph; quarantined historical entrypoints must not be asserted as
# root-level production files.
DEFAULT_ENTRYPOINTS = (
    "train_ramp.py",
    "generate_overlay.py",
)


@dataclass(frozen=True)
class ReproducibilityManifest:
    schema: str
    entrypoints: tuple[str, ...]
    active_python: dict[str, str]
    supporting_files: dict[str, str]
    config_schemas: dict[str, str]
    unresolved_dynamic_imports: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_index(project: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in project.rglob("*.py"):
        relative = path.relative_to(project)
        if any(part in {"tests", "docs", ".runtime", "local_mod_backup_20260710_2255"}
               for part in relative.parts):
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            result[".".join(parts)] = relative.as_posix()
    return result


def _resolve(module: str, modules: dict[str, str]) -> str | None:
    candidate = module
    while candidate:
        if candidate in modules:
            return modules[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def _active_import_graph(
    project: Path, entrypoints: Iterable[str]
) -> tuple[set[str], set[str]]:
    modules = _module_index(project)
    reachable: set[str] = set()
    unresolved: set[str] = set()
    queue = list(entrypoints)
    while queue:
        relative = queue.pop()
        if relative in reachable:
            continue
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"active entry/import does not exist: {relative}")
        reachable.add(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        package = list(Path(relative).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    prefix = package[: max(0, len(package) - node.level + 1)]
                    module = ".".join(prefix + ([module] if module else []))
                names = [module]
                names.extend(
                    f"{module}.{alias.name}" for alias in node.names if module
                )
            for name in names:
                resolved = _resolve(name, modules)
                if resolved is not None and resolved not in reachable:
                    queue.append(resolved)
            if isinstance(node, ast.Call):
                called = node.func
                dynamic = (
                    isinstance(called, ast.Name) and called.id == "__import__"
                ) or (
                    isinstance(called, ast.Attribute)
                    and called.attr == "import_module"
                )
                if dynamic:
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        name = node.args[0].value
                        resolved = _resolve(name, modules)
                        if resolved is not None:
                            queue.append(resolved)
                    else:
                        unresolved.add(f"{relative}:{getattr(node, 'lineno', '?')}")
    return reachable, unresolved


def _schema_hash(type_: type) -> str:
    rows = []
    for field in fields(type_):
        if field.default is not MISSING:
            default = repr(field.default)
        elif field.default_factory is not MISSING:
            default = repr(field.default_factory())
        else:
            default = "<required>"
        rows.append((field.name, str(field.type), default))
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _supporting_paths(project: Path) -> set[Path]:
    paths = {
        project / "configs" / "paper_ramp.json",
        project / "configs" / "fjsp_admission_suites.json",
    }
    formal = project / "configs" / "paper_ramp.json"
    if formal.is_file():
        payload = json.loads(formal.read_text(encoding="utf-8"))
        for suite in payload.get("dataset_suites", []):
            for key in ("train_dirs", "validation_dirs", "test_dirs"):
                for relative in suite.get(key, []):
                    directory = project / relative
                    if directory.is_dir():
                        paths.update(directory.rglob("*.fjs"))
    return {path for path in paths if path.is_file()}


def build_reproducibility_manifest(
    root: str | Path | None = None,
    *,
    entrypoints: Iterable[str] = DEFAULT_ENTRYPOINTS,
) -> ReproducibilityManifest:
    project = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    entries = tuple(entrypoints)
    active, unresolved = _active_import_graph(project, entries)
    active_hashes = {path: _sha256(project / path) for path in sorted(active)}
    support_hashes = {
        path.relative_to(project).as_posix(): _sha256(path)
        for path in sorted(_supporting_paths(project))
    }
    from ramp.config import RAMPConfig, HealthOverlayConfig, ObjectiveConfig
    from ramp.atmsl import ATMSLConfig
    from ramp.ppo import RAMPPPOConfig
    from model.ramp_core import RAMPModelConfig

    schemas = {
        type_.__name__: _schema_hash(type_)
        for type_ in (
            HealthOverlayConfig,
            ObjectiveConfig,
            RAMPConfig,
            RAMPModelConfig,
            RAMPPPOConfig,
            ATMSLConfig,
        )
    }
    authority = {
        "schema": "ramp_reproducibility_manifest_v1",
        "entrypoints": entries,
        "active_python": active_hashes,
        "supporting_files": support_hashes,
        "config_schemas": schemas,
        "unresolved_dynamic_imports": sorted(unresolved),
    }
    digest = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ReproducibilityManifest(
        schema=authority["schema"],
        entrypoints=entries,
        active_python=active_hashes,
        supporting_files=support_hashes,
        config_schemas=schemas,
        unresolved_dynamic_imports=tuple(sorted(unresolved)),
        digest=digest,
    )


def production_source_hash(root: str | Path | None = None) -> str:
    """Backward-compatible name for the complete reproducibility digest."""

    manifest = build_reproducibility_manifest(root)
    if manifest.unresolved_dynamic_imports:
        raise RuntimeError(
            "unresolved active dynamic imports: "
            + ", ".join(manifest.unresolved_dynamic_imports)
        )
    return manifest.digest
