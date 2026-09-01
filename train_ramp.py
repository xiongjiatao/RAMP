"""Paper-level RAMP train/validation/test entry point.

Formal runs use real multi-instance datasets and physical GPU 0--6. The
embedded 2x2 instance exists only behind ``--smoke`` and is labeled as such in
every artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from ramp import RAMPConfig, ObjectiveConfig, RAMPScenarioEnv
from ramp.config import FORMAL_PHYSICAL_GPUS
from ramp.atmsl import (
    ATMSLConfig,
    ATMSLScheduler,
    ATMSLStage,
    corrected_rewards,
    identity_scenario_support,
    migrate_scheduler_config_for_suffix,
    select_tail_preserving_representatives,
    weighted_scenario_mean,
    weighted_upper_tail_cvar,
)
from ramp.admission_data import resolve_admission_suite
from ramp.checkpoint import (
    RESUME_CONTRACT,
    load_checkpoint,
    restore_random_state,
    restore_training_session,
    restore_trainer,
    save_checkpoint,
)
from ramp.ppo import RAMPPPO, RAMPPPOConfig, RAMPRolloutBuffer
from ramp.overlay import HealthOverlay
from ramp.metrics import (
    INVALID_EVALUATION_STATUS,
    PAPER_METRIC_SCHEMA,
    PAPER_RUN_METRIC_SCHEMA,
    VALID_EVALUATION_STATUS,
    validate_paper_result_row,
)
from ramp.provenance import build_reproducibility_manifest, production_source_hash
from ramp.profiling import ThroughputProfiler, tensor_bytes
from ramp.route_audit import ROUTE_AUDIT_SCHEMA, RouteAuditAccumulator
from ramp.state import new_boundary_events
from ramp.steel_data import load_steel_instance_bundle
from ramp.experiments import METHODS, configure_environment, configure_model, get_method
from data_utils import load_data_from_single_file
from model.ramp_core import RAMPModelConfig
from model.policy_factory import build_policy


@dataclass(frozen=True)
class InstanceRecord:
    path: str
    job_lengths: np.ndarray
    nominal: np.ndarray
    split: str
    health_overlay_path: str | None = None

    @property
    def signature(self) -> tuple[int, int, int]:
        return (
            int(self.job_lengths.shape[0]),
            int(self.nominal.shape[0]),
            int(self.nominal.shape[1]),
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize the active batch independently of the dataset iterator."""

        return {
            "path": self.path,
            "job_lengths": np.asarray(self.job_lengths).copy(),
            "nominal": np.asarray(self.nominal).copy(),
            "split": self.split,
            "health_overlay_path": self.health_overlay_path,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "InstanceRecord":
        return cls(
            path=str(payload["path"]),
            job_lengths=np.asarray(payload["job_lengths"]).copy(),
            nominal=np.asarray(payload["nominal"]).copy(),
            split=str(payload["split"]),
            health_overlay_path=(
                None
                if payload.get("health_overlay_path") is None
                else str(payload["health_overlay_path"])
            ),
        )


class StatefulInstanceSampler:
    """Random, checkpointable batching across heterogeneous instance sizes."""

    FORMAT = "RAMP instance sampler v1"

    def __init__(self, records: list[InstanceRecord], *, seed: int):
        if not records:
            raise ValueError("training dataset is empty")
        self.records = records
        self.rng = np.random.default_rng(seed)
        self.groups: dict[tuple[int, int, int], list[int]] = {}
        for index, record in enumerate(records):
            self.groups.setdefault(record.signature, []).append(index)
        self.signatures = sorted(self.groups)
        self.draw_count = 0

    def sample(self, batch_size: int) -> list[InstanceRecord]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        signature = self.signatures[int(self.rng.integers(len(self.signatures)))]
        candidates = self.groups[signature]
        indices = self.rng.choice(candidates, size=batch_size, replace=True)
        self.draw_count += 1
        return [self.records[int(index)] for index in indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "bit_generator_state": self.rng.bit_generator.state,
            "draw_count": self.draw_count,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != self.FORMAT:
            raise ValueError("unsupported instance-sampler state")
        self.rng.bit_generator.state = payload["bit_generator_state"]
        self.draw_count = int(payload["draw_count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate RAMP")
    parser.add_argument("--instance", type=Path, default=None)
    parser.add_argument(
        "--health-overlay",
        type=Path,
        default=None,
        help="validated sidecar for a single --instance evaluation",
    )
    parser.add_argument("--train-dir", type=Path, action="append", default=[])
    parser.add_argument("--validation-dir", type=Path, action="append", default=[])
    parser.add_argument("--test-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--steel-suite",
        choices=("main", "sensitivity", "large_scale_zero_shot"),
        default=None,
        help="load the admitted Steel 8/1/2 suite from configs/steel_fjsp_suites.json",
    )
    parser.add_argument(
        "--admission-suite",
        choices=("sd3_valid",),
        default=None,
        help="load a manifest-authoritative nominal FJSP suite",
    )
    parser.add_argument(
        "--setting",
        choices=("H0", "H1"),
        default="H1",
        help="paper regime: H0=healthy, H1=stochastic health and maintenance",
    )
    parser.add_argument("--method", choices=tuple(METHODS), default="ramp")
    parser.add_argument("--num-scenarios", type=int, default=32)
    parser.add_argument("--reward-scenarios", type=int, default=128)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument(
        "--max-updates-this-invocation",
        type=int,
        default=None,
        help="bounded preemption gate; preserves the total ATMSL schedule budget",
    )
    parser.add_argument(
        "--atmsl-config",
        type=Path,
        default=None,
        help="enable the checkpointable three-stage ATMSL production runner",
    )
    parser.add_argument(
        "--allow-atmsl-v2-1-suffix-migration",
        action="store_true",
        help=(
            "admit only the frozen v2.0-to-v2.1 update-701 full-fidelity "
            "boundary migration from a checkpoint at or before update 700"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ppo-update-epochs",
        type=int,
        default=None,
        help="P2 research override; the first-paper default remains four epochs",
    )
    parser.add_argument(
        "--ppo-minibatch-size",
        type=int,
        default=None,
        help="P2 research override; use the rollout size for full-batch PPO",
    )
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--test-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--physical-gpu", type=int, choices=FORMAL_PHYSICAL_GPUS, default=None
    )
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument(
        "--validation-selection-metric",
        choices=("cost", "phi"),
        default="cost",
        help=(
            "metric used to select best.pt from validation checkpoints; "
            "the historical default is cost"
        ),
    )
    parser.add_argument(
        "--validation-stochastic-samples",
        type=int,
        default=1,
        help=(
            "number of stochastic validation rollouts per instance; set to zero "
            "for a strictly Greedy convergence trace"
        ),
    )
    parser.add_argument(
        "--kappa-r",
        type=float,
        default=None,
        help=(
            "RAMP risk-logit bias kappa_R; when omitted, use the method default"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help=(
            "RAMP empirical-survival intervention threshold; when omitted, "
            "use the method default"
        ),
    )
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="run the complete fixed update budget; best checkpoint remains diagnostic",
    )
    parser.add_argument(
        "--defer-final-test",
        action="store_true",
        help="skip final test during an intermediate resumability/stability gate",
    )
    parser.add_argument(
        "--quality-gate-mode",
        action="store_true",
        help="retain invalid evaluation rows as completion diagnostics without aborting",
    )
    parser.add_argument("--stochastic-eval-samples", type=int, default=8)
    parser.add_argument("--processing-distribution", choices=("lognormal", "beta", "mixture"), default="lognormal")
    parser.add_argument("--degradation-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--gamma-shape-multiplier", type=float, default=1.0)
    parser.add_argument("--gamma-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--initial-health-multiplier", type=float, default=1.0)
    parser.add_argument("--cm-cost-ratio-multiplier", type=float, default=1.0)
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument(
        "--trusted-tensor-fastpath",
        action="store_true",
        help=(
            "P2 research mode: skip repeated GPU scalar assertions after the "
            "environment state contract has been independently validated"
        ),
    )
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--last-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--session-checkpoint-interval",
        type=int,
        default=0,
        help="save an exact resumable session every N rollout transitions (0 disables)",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--raw-results", type=Path, default=None)
    parser.add_argument(
        "--route-audit-out",
        type=Path,
        default=None,
        help=(
            "write read-only RAMP route counts/rates collected from the same "
            "single forward pass used to select each evaluation action"
        ),
    )
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--throughput-profile",
        type=Path,
        default=None,
        help="write opt-in phase/CUDA/transfer profiling JSON",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--entry-verification",
        action="store_true",
        help=(
            "label a short real-dataset GPU run as infrastructure verification; "
            "its outputs are excluded from paper tables"
        ),
    )
    return parser.parse_args()


def _enforce_compute_policy(args: argparse.Namespace) -> torch.device:
    """Reject formal execution unless it is bound to an admitted physical GPU."""

    if args.smoke and args.entry_verification:
        raise ValueError("--smoke and --entry-verification are mutually exclusive")
    if args.smoke:
        return torch.device(args.device)
    if args.physical_gpu not in FORMAL_PHYSICAL_GPUS:
        raise ValueError(
            "formal train/validation/test requires --physical-gpu 0--6"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly the requested physical GPU before launch"
        )
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("formal execution must use --device cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the requested physical GPU is not visible as one CUDA device")
    return torch.device("cuda:0")


def _smoke_record(split: str) -> InstanceRecord:
    return InstanceRecord(
        path=f"synthetic://2x2/{split}",
        job_lengths=np.array([2, 2], dtype=np.int64),
        nominal=np.array(
            [[3, 5], [4, 6], [2, 6], [5, 3]], dtype=np.float32
        ),
        split=split,
    )


def _read_instance(
    path: Path, split: str, health_overlay: Path | None = None
) -> InstanceRecord:
    if "Steel_FJSP_Real_v1" in path.resolve().parts:
        steel = load_steel_instance_bundle(path)
        if steel.split_metadata.status != "VALID":
            raise ValueError(
                f"Steel instance is not admitted: {steel.split_metadata.status}"
            )
        job_lengths, nominal = steel.job_lengths, steel.nominal_processing_times
    else:
        job_lengths, nominal = load_data_from_single_file(str(path))
    if len(job_lengths) == 0:
        raise FileNotFoundError(path)
    return InstanceRecord(
        path=str(path.resolve()),
        job_lengths=np.asarray(job_lengths, dtype=np.int64),
        nominal=np.asarray(nominal, dtype=np.float32),
        split=split,
        health_overlay_path=(
            str(health_overlay.resolve()) if health_overlay is not None else None
        ),
    )


def _discover(directories: Iterable[Path], split: str) -> list[InstanceRecord]:
    paths: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(sorted(directory.rglob("*.fjs")))
    return [_read_instance(path, split) for path in sorted(set(paths))]


def load_splits(args: argparse.Namespace) -> dict[str, list[InstanceRecord]]:
    """Load strict, path-disjoint train/validation/test partitions."""

    admission_suite = getattr(args, "admission_suite", None)
    if args.steel_suite is not None and admission_suite is not None:
        raise ValueError("--steel-suite and --admission-suite are mutually exclusive")
    if args.smoke:
        if args.steel_suite is not None or admission_suite is not None:
            raise ValueError("--smoke cannot be combined with a dataset suite")
        return {
            "train": [_smoke_record("train")],
            "validation": [_smoke_record("validation")],
            "test": [_smoke_record("test")],
        }
    if args.instance is not None:
        if args.steel_suite is not None or admission_suite is not None:
            raise ValueError("--instance cannot be combined with a dataset suite")
        if not args.evaluate_only:
            raise ValueError("--instance is evaluation-only; formal training needs split dirs")
        if args.health_overlay is not None and not args.health_overlay.is_dir():
            raise FileNotFoundError(args.health_overlay)
        record = _read_instance(args.instance, "test", args.health_overlay)
        return {"train": [], "validation": [], "test": [record]}
    if args.health_overlay is not None:
        raise ValueError("--health-overlay is supported only with --instance")
    if admission_suite is not None:
        if args.train_dir or args.validation_dir or args.test_dir:
            raise ValueError("--admission-suite cannot be combined with split dirs")
        paths = resolve_admission_suite(Path(__file__).resolve().parent, admission_suite)
        return {
            split: [_read_instance(path, split) for path in selected]
            for split, selected in paths.items()
        }
    if args.steel_suite is not None:
        if args.train_dir or args.validation_dir or args.test_dir:
            raise ValueError("--steel-suite cannot be combined with explicit split dirs")
        root = Path(__file__).resolve().parent
        suite = json.loads(
            (root / "configs/steel_fjsp_suites.json").read_text(encoding="utf-8")
        )
        dataset_root = root / suite["dataset_root"]
        split_payload = json.loads(
            (dataset_root / suite["main"]["split_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        selected_suite = suite[args.steel_suite]
        variant = selected_suite["variant"]
        if args.steel_suite == "large_scale_zero_shot":
            zero_shot = json.loads(
                (dataset_root / selected_suite["test_manifest"]).read_text(
                    encoding="utf-8"
                )
            )
            test_paths: list[Path] = []
            for entry in zero_shot["test"]:
                path = dataset_root / entry["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(f"Steel zero-shot hash mismatch: {path}")
                test_paths.append(path)
            split_payload = {
                "train": split_payload["train"],
                "validation": split_payload["validation"],
                "test": [path.relative_to(dataset_root).as_posix() for path in test_paths],
            }
        result: dict[str, list[InstanceRecord]] = {}
        for split in ("train", "validation", "test"):
            paths = []
            for relative in split_payload[split]:
                selected = relative.replace("process_only_median", variant)
                paths.append(dataset_root / selected)
            result[split] = [_read_instance(path, split) for path in paths]
        return result
    splits = {
        "train": _discover(args.train_dir, "train"),
        "validation": _discover(args.validation_dir, "validation"),
        "test": _discover(args.test_dir, "test"),
    }
    if not args.evaluate_only and (not splits["train"] or not splits["validation"]):
        raise ValueError("formal training requires nonempty train and validation splits")
    if not splits["test"]:
        raise ValueError("formal execution requires a nonempty test split")
    path_sets = {
        name: {record.path for record in records} for name, records in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = path_sets[left] & path_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} split overlap: {sorted(overlap)[:3]}")
    return splits


def _stack(records: list[InstanceRecord]) -> tuple[np.ndarray, np.ndarray]:
    if not records or len({record.signature for record in records}) != 1:
        raise ValueError("a vectorized batch must contain one size signature")
    return (
        np.stack([record.job_lengths for record in records]),
        np.stack([record.nominal for record in records]),
    )


def _configs(
    args: argparse.Namespace, payload: dict[str, Any] | None
) -> tuple[RAMPConfig, RAMPModelConfig, RAMPPPOConfig, int]:
    if payload is None:
        state_scenarios = 4 if args.smoke else args.num_scenarios
        method = get_method(args.method)
        if method.policy_family != "neural":
            raise ValueError("threshold heuristic is evaluated by experiment_ramp.py")
        if args.method in {
            "ramp_core",
            "ramp_core_exact",
            "ramp",
            "ramp_without_proactive_pm",
            "production_only_auto_cm",
        } and args.setting != "H1":
            method = replace(method, setting=args.setting)
        env_config = configure_environment(
            method,
            num_scenarios=state_scenarios,
            seed=args.train_seed,
            epsilon_use=0.25 if args.smoke else 0.05,
            processing_distribution=args.processing_distribution,
            degradation_rate_multiplier=args.degradation_rate_multiplier,
            gamma_shape_multiplier=args.gamma_shape_multiplier,
            gamma_scale_multiplier=args.gamma_scale_multiplier,
            initial_health_multiplier=args.initial_health_multiplier,
            cm_cost_ratio_multiplier=args.cm_cost_ratio_multiplier,
        )
        model_config = configure_model(method, smoke=args.smoke)
        if args.kappa_r is not None:
            if not math.isfinite(args.kappa_r) or args.kappa_r < 0:
                raise ValueError("--kappa-r must be finite and nonnegative")
            model_config = replace(model_config, risk_logit_coefficient=float(args.kappa_r))
        if args.tau is not None:
            if not math.isfinite(args.tau) or not 0 < args.tau <= 1:
                raise ValueError("--tau must be finite and in (0, 1]")
            model_config = replace(
                model_config, survival_threshold=float(args.tau)
            )
        if args.trusted_tensor_fastpath:
            model_config = replace(model_config, runtime_tensor_validation=False)
        ppo_config = RAMPPPOConfig(
            update_epochs=(
                int(args.ppo_update_epochs)
                if args.ppo_update_epochs is not None
                else (1 if args.smoke else 4)
            ),
            minibatch_size=(
                int(args.ppo_minibatch_size)
                if args.ppo_minibatch_size is not None
                else (16 if args.smoke else 128)
            ),
            gamma=1.0,
        )
        return env_config, model_config, ppo_config, 0
    environment_values = dict(payload["environment_config"])
    environment_values["objective"] = ObjectiveConfig(
        **environment_values["objective"]
    )
    model_field_names = {field.name for field in fields(RAMPModelConfig)}
    model_values = {
        key: value
        for key, value in payload["model_config"].items()
        if key in model_field_names
    }
    return (
        RAMPConfig(**environment_values),
        RAMPModelConfig(**model_values),
        RAMPPPOConfig(**payload["ppo_config"]),
        int(payload["completed_update"]),
    )


def _make_env(
    records: list[InstanceRecord],
    env_config: RAMPConfig,
    *,
    reward_scenarios: int,
    reward_seed: int,
    device: torch.device,
    profiler: ThroughputProfiler | None = None,
) -> RAMPScenarioEnv:
    jobs, nominal = _stack(records)
    overlay = None
    overlay_paths = {record.health_overlay_path for record in records if record.health_overlay_path}
    if overlay_paths:
        if len(records) != 1 or len(overlay_paths) != 1:
            raise ValueError("health overlay sidecars require a single-instance batch")
        overlay = HealthOverlay.load(next(iter(overlay_paths)), device=device)
        overlay.validate(torch.as_tensor(nominal, device=device))
        if overlay.num_scenarios != env_config.num_scenarios:
            raise ValueError(
                "health overlay scenario count must match --num-scenarios"
            )
    return RAMPScenarioEnv(
        jobs,
        nominal,
        state_config=replace_seed(env_config, reward_seed - 1_000_000),
        reward_num_scenarios=reward_scenarios,
        state_overlay=overlay,
        reward_seed=reward_seed,
        device=device,
        profiler=profiler,
    )


def replace_seed(config: RAMPConfig, seed: int) -> RAMPConfig:
    values = asdict(config)
    values["objective"] = config.objective
    values["scenario_seed"] = int(seed)
    return RAMPConfig(**values)


def _load_atmsl_config(path: Path) -> ATMSLConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("production_protocol", payload)
    allowed = {field.name for field in fields(ATMSLConfig)}
    return ATMSLConfig(**{key: value for key, value in values.items() if key in allowed})


def _parameter_group_digest(module: torch.nn.Module, group: str) -> str:
    """Stable digest proving that an ATMSL update changed a parameter group."""

    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(module.named_parameters()):
        if group not in name.lower():
            continue
        matched += 1
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    if matched == 0:
        raise ValueError(f"policy has no {group!r} parameter group")
    return digest.hexdigest()


def _rollout(
    env: RAMPScenarioEnv,
    trainer: RAMPPPO,
    *,
    deterministic: bool,
    buffer: RAMPRolloutBuffer | None = None,
    reset: bool = True,
    progress: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[
        [RAMPRolloutBuffer, dict[str, Any]], None
    ] | None = None,
) -> tuple[RAMPRolloutBuffer, dict[str, float]]:
    """Collect or resume one complete episode without losing pending transitions."""

    buffer = RAMPRolloutBuffer() if buffer is None else buffer
    if reset:
        if len(buffer):
            raise ValueError("a fresh rollout cannot start with a non-empty buffer")
        with trainer.profiler.phase("rollout.environment_reset"):
            state = env.reset()
    else:
        state = env.state
    progress_state: dict[str, Any] = {
        "action_type_counts": [0.0, 0.0, 0.0],
        "valid_count": 0,
        "terminated_events": 0,
        "truncated_events": 0,
        "failure_events": 0,
        "pm_events": 0,
        "cm_events": 0,
        "all_masked_events": 0,
        **dict(progress or {}),
    }
    counter_device = state.action_mask_tensor.device
    action_type_counts = torch.tensor(
        progress_state["action_type_counts"],
        dtype=torch.float32,
        device=counter_device,
    )
    valid_count = torch.tensor(
        int(progress_state["valid_count"]), dtype=torch.float32, device=counter_device
    )
    terminated_events = torch.tensor(
        int(progress_state["terminated_events"]), dtype=torch.float32, device=counter_device
    )
    truncated_events = torch.tensor(
        int(progress_state["truncated_events"]), dtype=torch.float32, device=counter_device
    )
    failure_events = torch.tensor(
        int(progress_state["failure_events"]), dtype=torch.float32, device=counter_device
    )
    pm_events = torch.tensor(
        int(progress_state["pm_events"]), dtype=torch.float32, device=counter_device
    )
    cm_events = torch.tensor(
        int(progress_state["cm_events"]), dtype=torch.float32, device=counter_device
    )
    all_masked_events = torch.tensor(
        int(progress_state["all_masked_events"]),
        dtype=torch.float32,
        device=counter_device,
    )
    while not (env.observed_state.terminated | env.observed_state.truncated).all():
        active_before = ~(state.terminated_tensor | state.truncated_tensor)
        terminated_before = state.terminated_tensor.clone()
        truncated_before = state.truncated_tensor.clone()
        failure_before = env.observed_state.failure_count.sum()
        pm_before = env.observed_state.pm_count.sum()
        cm_before = env.observed_state.cm_count.sum()
        all_masked_events += (
            active_before & state.action_mask_tensor.all(dim=1)
        ).sum()
        with trainer.profiler.phase("rollout.model_forward_and_action_sampling"):
            actions, log_probs, values = trainer.act(state, deterministic=deterministic)
        # For an ordinary continuing transition V(s_{t+1}) is exactly the
        # value returned by the next behavior-policy call.  Fill the pending
        # previous slot here instead of running a duplicate forward after
        # every environment step.  A true truncation still receives an
        # explicit bootstrap below; natural termination never bootstraps.
        if len(buffer):
            previous_continues = (
                buffer.valid_transition_mask[-1]
                & ~buffer.terminated[-1]
                & ~buffer.truncated[-1]
            )
            buffer.next_values[-1] = torch.where(
                previous_continues,
                values.detach().cpu(),
                buffer.next_values[-1],
            )
        with trainer.profiler.phase("rollout.environment_step"):
            next_state, reward, terminated, truncated, info = env.step(
                actions, return_tensors=True
            )
        next_values = torch.zeros_like(values, device="cpu")
        truncated_tensor = truncated.bool()
        if bool((truncated_tensor & active_before).any()):
            with trainer.profiler.phase("rollout.truncation_bootstrap_forward"):
                with torch.no_grad():
                    truncation_values = trainer.policy_old(
                        next_state.to(trainer.device)
                    ).value.cpu()
            next_values = torch.where(
                (truncated_tensor & active_before).cpu(),
                truncation_values,
                next_values,
            )
        with trainer.profiler.phase("rollout.buffer_clone_to_cpu"):
            trainer.profiler.transfer("gpu_to_cpu_bytes", tensor_bytes(state))
            buffer.add(
                state,
                actions,
                log_probs,
                values,
                reward,
                terminated,
                truncated,
                active_before,
                next_values,
            )
        decoded_types = info["action_type"]
        for action_type in range(3):
            action_type_counts[action_type] += (
                (decoded_types == action_type) & active_before
            ).sum()
        valid_count += active_before.sum()
        terminated_events += ((~terminated_before) & terminated & active_before).sum()
        truncated_events += ((~truncated_before) & truncated & active_before).sum()
        failure_events += env.observed_state.failure_count.sum() - failure_before
        pm_events += env.observed_state.pm_count.sum() - pm_before
        cm_events += env.observed_state.cm_count.sum() - cm_before
        state = next_state
        if checkpoint_callback is not None:
            progress_state.update(
                {
                    "action_type_counts": action_type_counts.cpu().tolist(),
                    "valid_count": int(valid_count.cpu()),
                    "terminated_events": int(terminated_events.cpu()),
                    "truncated_events": int(truncated_events.cpu()),
                    "failure_events": int(failure_events.cpu()),
                    "pm_events": int(pm_events.cpu()),
                    "cm_events": int(cm_events.cpu()),
                    "all_masked_events": int(all_masked_events.cpu()),
                    "transition_index": len(buffer),
                    "rollout_complete": False,
                }
            )
            checkpoint_callback(buffer, dict(progress_state))
    metrics = env.metrics()
    valid_count_value = int(valid_count.cpu())
    denominator = max(valid_count_value, 1)
    diagnostics = {
        "production_action_ratio": float(action_type_counts[0] / denominator),
        "pm_action_ratio": float(action_type_counts[1] / denominator),
        "cm_action_ratio": float(action_type_counts[2] / denominator),
        "terminated_ratio": int(terminated_events.cpu()) / max(env.batch_size, 1),
        "truncated_ratio": int(truncated_events.cpu()) / max(env.batch_size, 1),
        "failure_event_count": float(failure_events.cpu()),
        "pm_event_count": float(pm_events.cpu()),
        "cm_event_count": float(cm_events.cpu()),
        "all_masked_event_count": float(all_masked_events.cpu()),
        "failure_rate": float(metrics["failure_probability"].mean()),
        "expected_cost": float(metrics["mean_total_cost"].mean()),
        "cvar_cost": float(metrics["cvar_0_95_total_cost"].mean()),
    }
    progress_state["rollout_complete"] = True
    return buffer, diagnostics


def _serialize_paper_metrics(
    env: RAMPScenarioEnv,
    metrics: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    """Serialize a paper row only after the schedule passes hard admission."""

    state = env.observed_state
    total_operations = int(state.op_scheduled[0].numel())
    completed_operations = int(state.op_scheduled[0].sum().item())
    unfinished_operations = total_operations - completed_operations
    terminated = bool(state.terminated[0].item())
    truncated = bool(state.truncated[0].item())
    completed_schedule = unfinished_operations == 0
    valid = completed_schedule and terminated and not truncated
    admission = {
        "completed_schedule": completed_schedule,
        "terminated": terminated,
        "truncated": truncated,
        "total_operation_count": total_operations,
        "unfinished_operation_count": unfinished_operations,
        "completion_ratio": (
            completed_operations / total_operations if total_operations else 1.0
        ),
        "evaluation_valid": valid,
        "evaluation_status": (
            VALID_EVALUATION_STATUS if valid else INVALID_EVALUATION_STATUS
        ),
    }
    if not valid:
        return {
            **admission,
            "diagnostic_observed_calendar_horizon": float(
                state.current_makespan.detach().float().mean().cpu()
            ),
        }
    if metrics is None:
        return admission

    scalar_names = (
        "mean_total_cost", "cvar_0_95_total_cost", "pm_cost", "cm_cost",
        "failure_probability", "failure_count", "planned_downtime",
        "unplanned_downtime",
    )
    serialized = {
        name: float(metrics[name].detach().float().mean().cpu())
        for name in scalar_names
    }
    serialized["expected_reward_scenario_makespan"] = float(
        metrics["expected_makespan"].detach().float().mean().cpu()
    )
    serialized["observed_calendar_horizon"] = float(
        metrics["calendar_horizon"].detach().float().mean().cpu()
    )
    serialized["raw_weighted_total_cost"] = float(
        metrics["raw_weighted_total_cost"].detach().float().mean().cpu()
    )
    serialized["physical_availability"] = float(
        metrics["physical_availability"].detach().float().mean().cpu()
    )
    serialized["production_utilization"] = float(
        metrics["production_utilization"].detach().float().mean().cpu()
    )
    serialized["behavior_audit"] = {
        "observed_pm_count": int(state.pm_count[0].sum().item()),
        "observed_cm_count": int(state.cm_count[0].sum().item()),
        "observed_failure_count": float(state.failure_count[0].sum().item()),
        "observed_unplanned_downtime": float(
            state.unplanned_downtime[0].sum().item()
        ),
        "observed_pm_cost": float(state.pm_cost_total[0].item()),
        "observed_cm_cost": float(state.cm_cost_total[0].item()),
    }
    serialized.update(admission)
    serialized["paper_metric_schema"] = PAPER_METRIC_SCHEMA
    serialized["physical_availability_by_machine"] = (
        metrics["physical_availability"].detach().cpu().flatten().tolist()
    )
    serialized["production_utilization_by_machine"] = (
        metrics["production_utilization"].detach().cpu().flatten().tolist()
    )
    components = env.scenario_cost_components().detach().cpu()[0]
    totals = env.scenario_total_cost().detach().cpu()[0]
    component_names = (
        "makespan",
        "pm_cost",
        "cm_cost",
        "unplanned_downtime",
        "failure_count",
    )
    serialized["scenario_results"] = [
        {
            "scenario_id": scenario_id,
            **{
                name: float(components[scenario_id, index])
                for index, name in enumerate(component_names)
            },
            "total_cost": float(totals[scenario_id]),
        }
        for scenario_id in range(components.shape[0])
    ]
    return serialized


@torch.no_grad()
def evaluate_records(
    records: list[InstanceRecord],
    trainer: RAMPPPO,
    env_config: RAMPConfig,
    *,
    reward_scenarios: int,
    seed: int,
    stochastic_samples: int,
    limit: int,
    device: torch.device,
    profiler: ThroughputProfiler | None = None,
    route_audit: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = records if limit <= 0 else records[:limit]
    if not selected:
        return {
            "greedy_cost": None,
            "greedy_cvar": None,
            "greedy_phi": None,
            "sampling_cost": None,
            "sampling_cvar": None,
            "sampling_phi": None,
            "valid_rows": 0,
            "invalid_rows": 0,
        }, []
    raw: list[dict[str, Any]] = []
    greedy_costs: list[float] = []
    greedy_cvars: list[float] = []
    greedy_phis: list[float] = []
    sampling_costs: list[float] = []
    sampling_cvars: list[float] = []
    sampling_phis: list[float] = []
    total_rollouts = len(selected) * (1 + stochastic_samples)
    rollout_idx = 0
    eval_start = time.perf_counter()
    for index, record in enumerate(selected):
        instance_name = Path(record.path).stem
        print(f"[{index+1}/{len(selected)}] {instance_name}", flush=True)
        mode_costs: dict[str, list[float]] = {"greedy": [], "sampling": []}
        mode_cvars: dict[str, list[float]] = {"greedy": [], "sampling": []}
        mode_phis: dict[str, list[float]] = {"greedy": [], "sampling": []}
        for mode, repeats in (("greedy", 1), ("sampling", stochastic_samples)):
            for repeat in range(repeats):
                rollout_idx += 1
                elapsed = time.perf_counter() - eval_start
                eta = elapsed / rollout_idx * (total_rollouts - rollout_idx) if rollout_idx > 0 else 0
                print(f"  rollout {rollout_idx}/{total_rollouts} | {mode} rep={repeat} | elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
                run_seed = seed + index * 10_000 + repeat
                env = _make_env(
                    [record],
                    env_config,
                    reward_scenarios=reward_scenarios,
                    reward_seed=run_seed + 1_000_000,
                    device=device,
                    profiler=profiler,
                )
                state = env.reset()
                route_counter = (
                    RouteAuditAccumulator(
                        jobs=int(state.dynamic_pair_mask_tensor.shape[1]),
                        machines=int(state.dynamic_pair_mask_tensor.shape[2]),
                    )
                    if route_audit
                    else None
                )
                route_action_trace: list[int] = []
                start = time.perf_counter()
                while not (
                    env.observed_state.terminated | env.observed_state.truncated
                ).all():
                    if route_counter is None:
                        action, _, _ = trainer.act(
                            state, deterministic=mode == "greedy"
                        )
                    else:
                        action, _, _, policy_output = trainer.act_with_output(
                            state, deterministic=mode == "greedy"
                        )
                        backoff_mask = getattr(
                            env, "chance_constraint_backoff_mask", None
                        )
                        if backoff_mask is None:
                            backoff_row = torch.zeros(
                                state.dynamic_pair_mask_tensor.shape[0],
                                device=state.dynamic_pair_mask_tensor.device,
                                dtype=torch.bool,
                            )
                        else:
                            backoff_row = backoff_mask.flatten(1).any(dim=1)
                        route_counter.observe(
                            policy_output,
                            empty_safety_set_backoff_row=backoff_row,
                        )
                        route_action_trace.extend(
                            int(value)
                            for value in action.detach().cpu().flatten()
                        )
                    state, _, _, _, _ = env.step(action)
                inference = time.perf_counter() - start
                # Admission is deliberately evaluated before env.metrics(): an
                # incomplete schedule must not even compute paper makespan/cost.
                paper_metrics = _serialize_paper_metrics(env, None)
                if paper_metrics["evaluation_valid"]:
                    paper_metrics = _serialize_paper_metrics(env, env.metrics())
                row = {
                        "split": record.split,
                        "instance": record.path,
                        "mode": mode,
                        "repeat": repeat,
                        "seed": run_seed,
                        **paper_metrics,
                        "inference_time": inference,
                        "smoke": record.path.startswith("synthetic://"),
                    }
                if route_counter is not None:
                    row["route_audit"] = {
                        **route_counter.to_dict(),
                        "action_count": len(route_action_trace),
                        "action_trace_sha256": hashlib.sha256(
                            json.dumps(route_action_trace).encode("utf-8")
                        ).hexdigest(),
                    }
                if row["evaluation_valid"]:
                    cost = float(row["mean_total_cost"])
                    cvar = float(row["cvar_0_95_total_cost"])
                    phi = cost + env_config.objective.cvar_beta * cvar
                    mode_costs[mode].append(cost)
                    mode_cvars[mode].append(cvar)
                    mode_phis[mode].append(phi)
                    validate_paper_result_row(row)
                    print(f"    ✓ {mode} rep={repeat} cost={cost:.2f} time={inference:.1f}s", flush=True)
                else:
                    print(f"    ✗ {mode} rep={repeat} INVALID", flush=True)
                raw.append(row)
        if mode_costs["greedy"]:
            greedy_costs.append(mode_costs["greedy"][0])
            greedy_cvars.append(mode_cvars["greedy"][0])
            greedy_phis.append(mode_phis["greedy"][0])
        if mode_costs["sampling"]:
            sampling_costs.append(float(np.mean(mode_costs["sampling"])))
            sampling_cvars.append(float(np.mean(mode_cvars["sampling"])))
            sampling_phis.append(float(np.mean(mode_phis["sampling"])))
    valid_rows = sum(bool(row["evaluation_valid"]) for row in raw)
    return {
        "greedy_cost": float(np.mean(greedy_costs)) if greedy_costs else None,
        "greedy_cvar": float(np.mean(greedy_cvars)) if greedy_cvars else None,
        "greedy_phi": float(np.mean(greedy_phis)) if greedy_phis else None,
        "sampling_cost": float(np.mean(sampling_costs)) if sampling_costs else None,
        "sampling_cvar": float(np.mean(sampling_cvars)) if sampling_cvars else None,
        "sampling_phi": float(np.mean(sampling_phis)) if sampling_phis else None,
        "valid_rows": valid_rows,
        "invalid_rows": len(raw) - valid_rows,
    }, raw


def _aggregate_route_audit_rows(
    rows: list[dict[str, Any]],
    *,
    training_seed: int,
    model_config: RAMPModelConfig,
) -> dict[str, Any]:
    """Build one-seed micro aggregates while retaining rollout provenance."""

    audited = [row for row in rows if row.get("route_audit") is not None]
    if len(audited) != len(rows):
        raise ValueError("route audit output requested but some rows are unaudited")
    grouped: dict[tuple[str, str], RouteAuditAccumulator] = {}
    count_fields = tuple(RouteAuditAccumulator.__dataclass_fields__)[2:]
    for row in audited:
        payload = row["route_audit"]
        key = (str(payload["scale"]), str(row["mode"]))
        counter = grouped.setdefault(
            key,
            RouteAuditAccumulator(
                jobs=int(payload["jobs"]),
                machines=int(payload["machines"]),
            ),
        )
        for name in count_fields:
            setattr(
                counter,
                name,
                getattr(counter, name) + int(payload["counts"][name]),
            )
    return {
        "schema": ROUTE_AUDIT_SCHEMA,
        "training_seed": int(training_seed),
        "routing_contract": {
            "aggregation": "any_candidate_witness",
            "survival_threshold": model_config.survival_threshold,
        },
        "rollout_count": len(audited),
        "aggregates": [
            {
                "mode": mode,
                **counter.to_dict(),
            }
            for (scale, mode), counter in sorted(grouped.items())
        ],
        "rollouts": [
            {
                "instance": row["instance"],
                "mode": row["mode"],
                "repeat": row["repeat"],
                "evaluation_seed": row["seed"],
                "evaluation_valid": row["evaluation_valid"],
                "audit": row["route_audit"],
            }
            for row in audited
        ],
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert CLI/config values into stable JSON primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path | None, payload: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def parameter_accounting(policy: torch.nn.Module) -> dict[str, int]:
    """Return reproducible trainable parameter counts by scientific role."""

    groups = {"encoder": 0, "actor": 0, "critic": 0, "other": 0}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        count = int(parameter.numel())
        if "critic" in name:
            group = "critic"
        elif "actor" in name or "type_gate" in name:
            group = "actor"
        elif any(
            token in name
            for token in (
                "encoder", "processor", "projection", "attention",
                "backbone", "fusion", "inducing", "substitution",
            )
        ):
            group = "encoder"
        else:
            group = "other"
        groups[group] += count
    groups["trainable_total"] = sum(groups.values())
    return groups


def main() -> None:
    args = parse_args()
    if args.session_checkpoint_interval < 0:
        raise ValueError("--session-checkpoint-interval must be nonnegative")
    if args.session_checkpoint_interval and not (args.checkpoint or args.last_checkpoint):
        raise ValueError(
            "mid-rollout checkpointing requires --checkpoint or --last-checkpoint"
        )
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    args.validation_seed = (
        args.seed + 100_000 if args.validation_seed is None else args.validation_seed
    )
    args.test_seed = args.seed + 200_000 if args.test_seed is None else args.test_seed
    device = _enforce_compute_policy(args)
    profiler = ThroughputProfiler(
        enabled=args.throughput_profile is not None, device=device
    )
    if not args.smoke and not args.evaluate_only:
        if args.best_checkpoint is None or args.last_checkpoint is None:
            raise ValueError("formal training requires both best and last checkpoint paths")
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)
    splits = load_splits(args)
    payload = load_checkpoint(args.resume, map_location=device) if args.resume else None
    env_config, model_config, ppo_config, completed_updates = _configs(args, payload)
    checkpoint_env_config = env_config
    atmsl_scheduler: ATMSLScheduler | None = None
    if args.atmsl_config is not None:
        if args.method != "ramp" or args.setting != "H1":
            raise ValueError("ATMSL production training requires RAMP under H1")
        atmsl_config = _load_atmsl_config(args.atmsl_config)
        extension = (payload or {}).get("training_extension_state", {})
        if extension.get("atmsl_scheduler") is not None:
            atmsl_scheduler = ATMSLScheduler.from_state_dict(
                extension["atmsl_scheduler"]
            )
            if atmsl_scheduler.total_updates != args.updates:
                raise ValueError("resumed ATMSL total update budget differs")
            if atmsl_scheduler.config != atmsl_config:
                if not args.allow_atmsl_v2_1_suffix_migration:
                    raise ValueError(
                        "resumed ATMSL config differs; an explicit admitted migration is required"
                    )
                atmsl_scheduler = migrate_scheduler_config_for_suffix(
                    atmsl_scheduler, atmsl_config
                )
        else:
            atmsl_scheduler = ATMSLScheduler(atmsl_config, args.updates)
        # Architecture and optimizer remain continuous across fidelity stages;
        # This is the full H1 physical authority used by stages B/C and eval.
        env_config = replace(
            env_config,
            num_scenarios=atmsl_config.full_state_scenarios,
            maintenance_actions=True,
            preventive_maintenance_actions=True,
            corrective_maintenance_actions=True,
        )
    selected_method = get_method(args.method)
    reward_scenarios = 6 if args.smoke else (
        selected_method.reward_scenario_count or args.reward_scenarios
    )
    checkpoint_reward_scenarios = reward_scenarios
    if payload is not None and payload.get("resume_contract") == RESUME_CONTRACT:
        checkpoint_reward_scenarios = int(
            payload["rollout_cursor"]["reward_scenarios"]
        )
    if atmsl_scheduler is not None:
        # Pending-session reconstruction uses its saved compact fidelity, but
        # validation/test are never allowed to inherit that approximation.
        reward_scenarios = atmsl_scheduler.config.full_reward_scenarios
    trainer = RAMPPPO(build_policy(model_config), ppo_config, device=device)
    trainer.profiler = profiler

    def atmsl_extension_state() -> dict[str, Any]:
        return (
            {}
            if atmsl_scheduler is None
            else {"atmsl_scheduler": atmsl_scheduler.state_dict()}
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_counts = parameter_accounting(trainer.policy)
    sampler = (
        StatefulInstanceSampler(splits["train"], seed=args.train_seed)
        if splits["train"]
        else None
    )
    last_env: RAMPScenarioEnv | None = None
    pending_records: list[InstanceRecord] | None = None
    pending_buffer: RAMPRolloutBuffer | None = None
    pending_cursor: dict[str, Any] | None = None
    if payload is not None:
        if args.evaluate_only:
            restore_trainer(trainer, payload, load_optimizer=False)
            restore_random_state(payload)
        else:
            if payload.get("resume_contract") != RESUME_CONTRACT:
                raise ValueError(
                    "training resume requires a v4 FULL_SESSION_CHECKPOINT; "
                    "legacy/update-boundary files are evaluation-only"
                )
            if sampler is None:
                raise ValueError("training resume requires a dataset sampler")
            pending_records = [
                InstanceRecord.from_state_dict(record)
                for record in payload["active_records"]
            ]
            if not pending_records:
                raise ValueError("full-session checkpoint lacks the active batch")
            pending_cursor = dict(payload["rollout_cursor"])
            last_env = _make_env(
                pending_records,
                checkpoint_env_config,
                reward_scenarios=checkpoint_reward_scenarios,
                reward_seed=int(pending_cursor["reward_seed"]),
                device=device,
                profiler=profiler,
            )
            pending_buffer = RAMPRolloutBuffer()
            pending_cursor = restore_training_session(
                trainer,
                last_env,
                pending_buffer,
                payload,
                dataset_sampler=sampler,
            )

    reproducibility_manifest = build_reproducibility_manifest()
    configuration = {
        "entry_point": "train_ramp.py",
        "formal": not args.smoke,
        "entry_verification": args.entry_verification,
        "smoke_results_forbidden_in_paper_tables": True,
        "entry_verification_results_forbidden_in_paper_tables": True,
        "physical_gpu": args.physical_gpu,
        "args": _jsonable(vars(args)),
        "environment": asdict(env_config),
        "model": asdict(model_config),
        "ppo": asdict(ppo_config),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "source_hash": production_source_hash(),
        "reproducibility_manifest": reproducibility_manifest.to_dict(),
        "parameter_counts": parameter_counts,
        "critic_semantics": "expected remaining risk-adjusted return under persistent scenario trajectories",
        "atmsl": (
            None
            if atmsl_scheduler is None
            else {
                "method": "Adaptive Tail-Preserving Multi-Fidelity Scenario Learning",
                "protocol": asdict(atmsl_scheduler.config),
                "final_evaluation_fidelity": {
                    "state_scenarios": atmsl_scheduler.config.full_state_scenarios,
                    "reward_scenarios": atmsl_scheduler.config.full_reward_scenarios,
                },
            }
        ),
    }
    config_path = args.config_out or (
        args.log.with_suffix(".config.json") if args.log is not None else None
    )
    _write_json(config_path, configuration)

    update_logs: list[dict[str, Any]] = []
    # Keep the historical minimum-Cost field for compatibility, while using a
    # separate selector value so the sensitivity protocol can select on the
    # complete mean--CVaR objective without changing the main experiments.
    best_validation = float("inf")
    best_validation_selection = float("inf")
    selected_validation_cost: float | None = None
    selected_validation_cvar: float | None = None
    selected_validation_phi: float | None = None
    stale_validations = 0
    updates_to_run = 0 if args.evaluate_only else (1 if args.smoke else max(args.updates - completed_updates, 0))
    if atmsl_scheduler is not None and not args.evaluate_only and not args.smoke:
        if updates_to_run < 0:
            raise ValueError("checkpoint exceeds requested ATMSL update budget")
    if args.max_updates_this_invocation is not None:
        if args.max_updates_this_invocation < 1:
            raise ValueError("--max-updates-this-invocation must be positive")
        updates_to_run = min(updates_to_run, args.max_updates_this_invocation)
    training_start = time.perf_counter()
    rollout_transition_count = int(
        (payload or {}).get("metadata", {}).get("rollout_transitions", 0)
    )
    for _ in range(updates_to_run):
        full_update_start = time.perf_counter()
        assert sampler is not None
        atmsl_plan = None if atmsl_scheduler is None else atmsl_scheduler.plan()
        atmsl_exact_full_fidelity = False
        update_env_config = env_config
        update_reward_scenarios = reward_scenarios
        if atmsl_plan is not None:
            atmsl_exact_full_fidelity = atmsl_plan.uses_exact_full_fidelity(
                atmsl_scheduler.config
            )
            update_env_config = replace(
                env_config,
                num_scenarios=atmsl_plan.state_scenarios,
                maintenance_actions=not atmsl_plan.production_only,
                preventive_maintenance_actions=not atmsl_plan.production_only,
                corrective_maintenance_actions=not atmsl_plan.production_only,
            )
            update_reward_scenarios = atmsl_plan.reward_scenarios
            if atmsl_plan.stage != atmsl_scheduler.stage:
                # Weights and Adam moments are continuous. Only the frozen
                # behavior policy is hard-synchronized at a stage boundary.
                trainer.sync_old_policy()
            trainer.config = replace(
                trainer.config,
                update_epochs=atmsl_plan.ppo_epochs,
            )
        resuming_pending_rollout = bool(
            pending_cursor is not None
            and not bool(pending_cursor.get("rollout_complete", False))
        )
        if resuming_pending_rollout:
            assert pending_records is not None
            assert pending_buffer is not None
            assert last_env is not None
            records = pending_records
            buffer = pending_buffer
            reward_seed = int(pending_cursor["reward_seed"])
            progress = pending_cursor
        else:
            records = sampler.sample(1 if args.smoke else args.batch_size)
            reward_seed = args.train_seed + 1_000_000 + completed_updates * 10_000
            last_env = _make_env(
                records,
                update_env_config,
                reward_scenarios=update_reward_scenarios,
                reward_seed=reward_seed,
                device=device,
                profiler=profiler,
            )
            buffer = RAMPRolloutBuffer()
            progress = None
            if atmsl_plan is not None and not atmsl_exact_full_fidelity:
                # Full-fidelity ATMSL updates are the exact P2 authority, not a
                # permutation of all archived representatives.  Reordering the
                # semantic IDs changes the keyed noise paths seen by each tensor
                # slot and therefore changes the PPO update even when K == S.
                # Tail-preserving support is used only on genuinely compact axes.
                if (
                    atmsl_plan.state_scenarios
                    == atmsl_scheduler.config.full_state_scenarios
                ):
                    state_support = identity_scenario_support(
                        atmsl_plan.state_scenarios
                    )
                else:
                    state_support = atmsl_scheduler.representative_support(
                        atmsl_plan.state_scenarios
                    )
                if (
                    atmsl_plan.reward_scenarios
                    == atmsl_scheduler.config.full_reward_scenarios
                ):
                    reward_support = identity_scenario_support(
                        atmsl_plan.reward_scenarios
                    )
                else:
                    reward_support = atmsl_scheduler.representative_support(
                        atmsl_plan.reward_scenarios
                    )
                state_ids = state_support.scenario_ids
                reward_ids = reward_support.scenario_ids
                if not atmsl_scheduler.config.paired_semantic_ids_enabled:
                    state_ids = torch.arange(atmsl_plan.state_scenarios)
                    reward_ids = torch.arange(atmsl_plan.reward_scenarios)
                last_env.configure_atmsl_scenario_support(
                    state_scenario_ids=state_ids,
                    reward_scenario_ids=reward_ids,
                    reward_weights=reward_support.weights,
                    weighted_cvar_enabled=atmsl_scheduler.config.weighted_cvar_enabled,
                )

        def save_pending_session(
            current_buffer: RAMPRolloutBuffer,
            rollout_progress: dict[str, Any],
        ) -> None:
            interval = int(args.session_checkpoint_interval)
            if interval == 0 or len(current_buffer) % interval:
                return
            session_path = args.checkpoint or args.last_checkpoint
            assert session_path is not None and last_env is not None
            cursor = {
                **rollout_progress,
                "completed_updates": completed_updates,
                "transition_index": len(current_buffer),
                "rollout_complete": False,
                "reward_seed": reward_seed,
                "reward_scenarios": update_reward_scenarios,
            }
            save_checkpoint(
                session_path,
                trainer,
                model_config=model_config,
                environment_config=update_env_config,
                ppo_config=trainer.config,
                update=completed_updates,
                environment=last_env,
                rollout_buffer=current_buffer,
                rollout_cursor=cursor,
                active_records=[record.state_dict() for record in records],
                dataset_sampler_state=sampler.state_dict(),
                full_config=configuration,
                reproducibility_manifest=reproducibility_manifest.to_dict(),
                metadata={
                    "kind": "mid_rollout",
                    "parameter_counts": parameter_counts,
                    "rollout_transitions": rollout_transition_count,
                },
                training_extension_state=atmsl_extension_state(),
            )

        core_update_start = time.perf_counter()
        buffer, rollout_diagnostics = _rollout(
            last_env,
            trainer,
            deterministic=args.deterministic_rollout,
            buffer=buffer,
            # Exact full-fidelity ATMSL uses the ordinary P2 reset/noise
            # authority. Compact stages have already reset while binding their
            # representative semantic support above.
            reset=(
                not resuming_pending_rollout
                and (atmsl_plan is None or atmsl_exact_full_fidelity)
            ),
            progress=progress,
            checkpoint_callback=(
                save_pending_session if args.session_checkpoint_interval else None
            ),
        )
        if atmsl_plan is not None and atmsl_plan.full_batch_ppo:
            valid_count_for_batch = int(
                torch.stack(buffer.valid_transition_mask).sum().item()
            )
            trainer.config = replace(
                trainer.config, minibatch_size=max(valid_count_for_batch, 1)
            )
        elif atmsl_plan is not None:
            trainer.config = replace(
                trainer.config,
                minibatch_size=atmsl_scheduler.config.full_ppo_minibatch_size,
            )
        track_parameter_changes = atmsl_plan is not None or args.quality_gate_mode
        actor_digest_before = (
            _parameter_group_digest(trainer.policy, "actor")
            if track_parameter_changes else None
        )
        critic_digest_before = (
            _parameter_group_digest(trainer.policy, "critic")
            if track_parameter_changes else None
        )
        ppo_diagnostics = trainer.update(buffer)
        if track_parameter_changes:
            actor_digest_after = _parameter_group_digest(trainer.policy, "actor")
            critic_digest_after = _parameter_group_digest(trainer.policy, "critic")
            ppo_diagnostics.update({
                "actor_parameter_digest_before": actor_digest_before,
                "actor_parameter_digest_after": actor_digest_after,
                "actor_parameters_changed": actor_digest_before != actor_digest_after,
                "critic_parameter_digest_before": critic_digest_before,
                "critic_parameter_digest_after": critic_digest_after,
                "critic_parameters_changed": critic_digest_before != critic_digest_after,
            })
        core_update_seconds = time.perf_counter() - core_update_start
        rollout_transition_count += int(ppo_diagnostics["valid_transition_count"])
        completed_updates += 1
        atmsl_diagnostics: dict[str, Any] = {}
        if atmsl_plan is not None:
            atmsl_scheduler.complete_update(atmsl_plan)
            atmsl_diagnostics = {
                "atmsl_stage": atmsl_plan.stage.value,
                "atmsl_state_scenarios": atmsl_plan.state_scenarios,
                "atmsl_reward_scenarios": atmsl_plan.reward_scenarios,
                "atmsl_paired_correction": atmsl_plan.paired_correction,
                "atmsl_forced_fallback": atmsl_plan.forced_fallback,
            }
            if atmsl_plan.paired_correction:
                full_totals = last_env.scenario_total_cost().detach().cpu()
                full_components = last_env.scenario_cost_components().detach().cpu()
                representative_count = min(
                    atmsl_scheduler.config.joint_reward_scenarios,
                    full_totals.shape[1],
                )
                representatives = select_tail_preserving_representatives(
                    full_totals,
                    full_components,
                    representative_count,
                    alpha=env_config.objective.cvar_alpha,
                    preserve_tail=atmsl_scheduler.config.tail_preservation_enabled,
                    preserve_extreme_events=atmsl_scheduler.config.extreme_event_anchors_enabled,
                )
                if not atmsl_scheduler.config.probability_weights_enabled:
                    representatives = type(representatives)(
                        representatives.scenario_ids,
                        torch.full_like(representatives.weights, 1.0 / representative_count),
                        representatives.assignment,
                        representatives.tail_coverage,
                    )
                selected = full_totals[:, representatives.scenario_ids]
                low_phi = weighted_scenario_mean(selected, representatives.weights)
                low_cvar_weights = representatives.weights
                if not atmsl_scheduler.config.weighted_cvar_enabled:
                    low_cvar_weights = torch.ones_like(representatives.weights)
                low_phi = low_phi + env_config.objective.cvar_beta * weighted_upper_tail_cvar(
                    selected, low_cvar_weights, env_config.objective.cvar_alpha
                )
                uniform = torch.full(
                    (full_totals.shape[1],), 1.0 / full_totals.shape[1]
                )
                full_phi = weighted_scenario_mean(full_totals, uniform)
                full_phi = full_phi + env_config.objective.cvar_beta * weighted_upper_tail_cvar(
                    full_totals, uniform, env_config.objective.cvar_alpha
                )
                _, correction = corrected_rewards(
                    -low_phi,
                    -full_phi,
                    (
                        atmsl_scheduler.config.correction_lambda
                        if atmsl_scheduler.config.control_variate_enabled
                        else 0.0
                    ),
                )
                fallback = atmsl_scheduler.observe_correction(
                    relative_residual=correction["correction_relative_residual"],
                    tail_coverage=representatives.tail_coverage,
                    representatives=representatives,
                    full_total_cost=full_totals,
                    full_cost_components=full_components,
                )
                atmsl_diagnostics.update(
                    {
                        **correction,
                        "tail_coverage": representatives.tail_coverage,
                        "representative_ids": representatives.scenario_ids.tolist(),
                        "representative_weights": representatives.weights.tolist(),
                        "quality_fallback_activated": fallback,
                        "correction_semantics": (
                            "paired full trajectories versus their weighted compact support; "
                            "lambda=1 trains on the exact full-fidelity reward"
                        ),
                    }
                )
        boundary_cursor = {
            "completed_updates": completed_updates,
            "transition_index": len(buffer),
            "rollout_complete": True,
            "reward_seed": reward_seed,
            "reward_scenarios": update_reward_scenarios,
        }
        pending_records = None
        pending_buffer = None
        pending_cursor = None
        validation_cost: float | None = None
        validation_cvar: float | None = None
        validation_phi: float | None = None
        validation_valid_rows: int | None = None
        validation_invalid_rows: int | None = None
        validation_seconds = 0.0
        checkpoint_seconds = 0.0
        if completed_updates % args.validation_interval == 0:
            validation_start = time.perf_counter()
            validation, _ = evaluate_records(
                splits["validation"],
                trainer,
                env_config,
                reward_scenarios=reward_scenarios,
                seed=args.validation_seed,
                stochastic_samples=args.validation_stochastic_samples,
                limit=args.validation_limit,
                device=device,
                profiler=profiler,
            )
            validation_seconds = time.perf_counter() - validation_start
            validation_cost = validation["greedy_cost"]
            validation_cvar = validation["greedy_cvar"]
            validation_phi = validation["greedy_phi"]
            validation_valid_rows = int(validation["valid_rows"])
            validation_invalid_rows = int(validation["invalid_rows"])
            if (
                validation_cost is not None
                and validation_cost < best_validation
            ):
                best_validation = validation_cost
            selection_value = (
                validation_cost
                if args.validation_selection_metric == "cost"
                else validation_phi
            )
            if (
                selection_value is not None
                and selection_value < best_validation_selection
            ):
                best_validation_selection = selection_value
                selected_validation_cost = validation_cost
                selected_validation_cvar = validation_cvar
                selected_validation_phi = validation_phi
                stale_validations = 0
                if args.best_checkpoint is not None:
                    checkpoint_start = time.perf_counter()
                    save_checkpoint(
                        args.best_checkpoint,
                        trainer,
                        model_config=model_config,
                        environment_config=update_env_config,
                        ppo_config=trainer.config,
                        update=completed_updates,
                        environment=last_env,
                        rollout_buffer=buffer,
                        rollout_cursor=boundary_cursor,
                        active_records=[record.state_dict() for record in records],
                        dataset_sampler_state=sampler.state_dict(),
                        full_config=configuration,
                        reproducibility_manifest=reproducibility_manifest.to_dict(),
                        metadata={
                            "kind": "best",
                            "validation_cost": validation_cost,
                            "validation_cvar": validation_cvar,
                            "validation_phi": validation_phi,
                            "validation_selection_metric": args.validation_selection_metric,
                            "validation_selection_value": selection_value,
                            "validation_valid_rows": validation_valid_rows,
                            "validation_invalid_rows": validation_invalid_rows,
                            "parameter_counts": parameter_counts,
                            "rollout_transitions": rollout_transition_count,
                        },
                        training_extension_state=atmsl_extension_state(),
                    )
                    checkpoint_seconds += time.perf_counter() - checkpoint_start
            elif validation_cost is not None:
                stale_validations += 1
        last_path = args.last_checkpoint or args.checkpoint
        if last_path is not None:
            checkpoint_start = time.perf_counter()
            save_checkpoint(
                last_path,
                trainer,
                model_config=model_config,
                environment_config=update_env_config,
                ppo_config=trainer.config,
                update=completed_updates,
                environment=last_env,
                rollout_buffer=buffer,
                rollout_cursor=boundary_cursor,
                active_records=[record.state_dict() for record in records],
                dataset_sampler_state=sampler.state_dict(),
                full_config=configuration,
                reproducibility_manifest=reproducibility_manifest.to_dict(),
                metadata={
                    "kind": "last",
                    "parameter_counts": parameter_counts,
                    "rollout_transitions": rollout_transition_count,
                },
                training_extension_state=atmsl_extension_state(),
            )
            checkpoint_seconds += time.perf_counter() - checkpoint_start
        diagnostics = {
            "update": completed_updates,
            **ppo_diagnostics,
            **rollout_diagnostics,
            **atmsl_diagnostics,
            "validation_cost": validation_cost,
            "validation_cvar": validation_cvar,
            "validation_phi": validation_phi,
            "validation_valid_rows": validation_valid_rows,
            "validation_invalid_rows": validation_invalid_rows,
            "core_rollout_ppo_seconds": core_update_seconds,
            "validation_seconds": validation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "full_update_seconds": time.perf_counter() - full_update_start,
            "cumulative_rollout_transitions": rollout_transition_count,
            "wall_clock_training_time": time.perf_counter() - training_start,
        }
        update_logs.append(diagnostics)
        print(json.dumps(diagnostics, sort_keys=True), flush=True)
        if (
            not args.disable_early_stopping
            and stale_validations >= args.early_stopping_patience
        ):
            break

    if args.defer_final_test:
        test_summary = {
            "greedy_cost": None,
            "sampling_cost": None,
            "valid_rows": 0,
            "invalid_rows": 0,
            "deferred": True,
        }
        raw_results = []
    else:
        test_summary, raw_results = evaluate_records(
            splits["test"],
            trainer,
            env_config,
            reward_scenarios=reward_scenarios,
            seed=args.test_seed,
            stochastic_samples=(1 if args.smoke else args.stochastic_eval_samples),
            limit=(1 if args.smoke else args.test_limit),
            device=device,
            profiler=profiler,
            route_audit=args.route_audit_out is not None,
        )
    for row in raw_results:
        row["entry_verification"] = bool(args.entry_verification)
    route_audit_payload = None
    if args.route_audit_out is not None:
        route_audit_payload = _aggregate_route_audit_rows(
            raw_results,
            training_seed=args.train_seed,
            model_config=model_config,
        )
    # Route provenance is emitted only through its versioned sidecar.  Keep
    # the established formal raw-results schema byte-structure compatible.
    raw_results_for_paper = [
        {key: value for key, value in row.items() if key != "route_audit"}
        for row in raw_results
    ]
    _write_json(args.raw_results, raw_results_for_paper)
    _write_json(args.route_audit_out, route_audit_payload)
    if args.smoke:
        execution_status = "SMOKE_ONLY"
    elif args.entry_verification:
        execution_status = "FORMAL_ENTRY_VERIFICATION"
    else:
        execution_status = "FORMAL_EXECUTION"
    training_time = time.perf_counter() - training_start
    result = {
        "paper_run_metric_schema": PAPER_RUN_METRIC_SCHEMA,
        "status": execution_status,
        "completed_updates": completed_updates,
        "state_scenarios": env_config.num_scenarios,
        "reward_scenarios": reward_scenarios,
        "best_validation_cost": (
            best_validation if math.isfinite(best_validation) else None
        ),
        "validation_selection_metric": args.validation_selection_metric,
        "selected_validation_cost": selected_validation_cost,
        "selected_validation_cvar": selected_validation_cvar,
        "selected_validation_phi": selected_validation_phi,
        "updates": update_logs,
        "loss": (
            {
                key: value
                for key, value in update_logs[-1].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if update_logs else {}
        ),
        "test": test_summary,
        "raw_result_count": len(raw_results),
        "training_time": training_time,
        "parameter_count": parameter_counts["trainable_total"],
        "parameter_counts": parameter_counts,
        "rollout_transitions": rollout_transition_count,
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
        },
    }
    _write_json(args.log, result)
    if args.throughput_profile is not None:
        profile_payload = profiler.to_dict()
        profile_payload["source_hash"] = configuration["source_hash"]
        profile_payload["completed_updates"] = completed_updates
        _write_json(args.throughput_profile, profile_payload)
    print(json.dumps(result, indent=2))
    if (
        not args.smoke
        and not args.entry_verification
        and not args.quality_gate_mode
        and int(test_summary["invalid_rows"]) > 0
    ):
        raise RuntimeError(
            "formal evaluation produced INVALID_INCOMPLETE_SCHEDULE rows; "
            "diagnostics were written but paper-result admission failed"
        )


if __name__ == "__main__":
    main()
