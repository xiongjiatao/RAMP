"""Versioned full-session RAMP checkpoint save and restore."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from ramp.config import RAMPConfig
from ramp.ppo import RAMPPPO, RAMPPPOConfig, RAMPRolloutBuffer
from model.ramp_core import RAMPModelConfig


CHECKPOINT_FORMAT = "RAMP checkpoint v4"
UPDATE_BOUNDARY_FORMAT = "RAMP checkpoint v3"
LEGACY_CHECKPOINT_FORMAT = "RAMP checkpoint v2"
RESUME_CONTRACT = "FULL_SESSION_CHECKPOINT"
WEIGHTS_ONLY_CONTRACT = "UPDATE_BOUNDARY"


class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, payload: dict[str, Any]) -> None: ...


def _rng_state() -> dict[str, Any]:
    """Capture every stochastic authority used by training and evaluation."""

    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python_random": random.getstate(),
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    # ``load_checkpoint(map_location='cuda')`` also relocates serialized CPU
    # ByteTensors. PyTorch RNG setters require host ByteTensor state even when
    # restoring CUDA generators.
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu"]).cpu())
    cuda_states = payload.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state).cpu() for state in cuda_states]
        )
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python_random"])


def restore_random_state(payload: Mapping[str, Any]) -> None:
    """Publicly restore CPU/CUDA/NumPy/Python RNGs from a checkpoint."""

    _restore_rng_state(payload["rng_state"])


def save_checkpoint(
    path: str | Path,
    trainer: RAMPPPO,
    *,
    model_config: RAMPModelConfig,
    environment_config: RAMPConfig,
    ppo_config: RAMPPPOConfig,
    update: int,
    environment: Stateful | None = None,
    rollout_buffer: RAMPRolloutBuffer | None = None,
    rollout_cursor: Mapping[str, Any] | None = None,
    active_records: list[Mapping[str, Any]] | None = None,
    dataset_sampler_state: Mapping[str, Any] | None = None,
    full_config: Mapping[str, Any] | None = None,
    reproducibility_manifest: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    training_extension_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically serialize either a full training session or a weight-only probe.

    Production training always supplies ``environment``, ``rollout_buffer`` and
    ``rollout_cursor`` together.  Omitting all three retains a small
    update-boundary payload for evaluation/legacy unit callers, but that payload
    is deliberately not accepted by :func:`restore_training_session`.
    """

    session_values = (environment, rollout_buffer, rollout_cursor)
    full_session = all(value is not None for value in session_values)
    if any(value is not None for value in session_values) and not full_session:
        raise ValueError(
            "environment, rollout_buffer, and rollout_cursor must be supplied together"
        )
    cursor = dict(rollout_cursor or {})
    if full_session:
        transition_index = int(cursor.get("transition_index", -1))
        if transition_index != len(rollout_buffer):  # type: ignore[arg-type]
            raise ValueError("rollout cursor does not match pending buffer length")
        if int(cursor.get("completed_updates", -1)) != int(update):
            raise ValueError("rollout cursor completed-update count is inconsistent")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "resume_contract": RESUME_CONTRACT if full_session else WEIGHTS_ONLY_CONTRACT,
        "full_session_resume_supported": full_session,
        "mid_rollout_resume_supported": full_session,
        "completed_update": int(update),
        "next_update": int(update) + 1,
        "model": trainer.policy.state_dict(),
        "policy_old": trainer.policy_old.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "rng_state": _rng_state(),
        "dataset_sampler_state": dict(dataset_sampler_state or {}),
        "model_config": asdict(model_config),
        "environment_config": asdict(environment_config),
        "ppo_config": asdict(ppo_config),
        "full_config": dict(full_config or {}),
        "reproducibility_manifest": dict(reproducibility_manifest or {}),
        "training_extension_state": dict(training_extension_state or {}),
        "metadata": {
            "scenario_semantics": "persistent full-episode stochastic health trajectories",
            "critic_semantics": "expected remaining risk-adjusted return under persistent scenario trajectories",
            "objective_semantics": "episode-level dimensionless mean-plus-CVaR total cost; empirical fractional-boundary upper tail",
            "recourse_semantics": "common primary action with fully costed scenario-dependent feasibility recourse",
            "resume_semantics": (
                "EXACT_FULL_SESSION_RESUME"
                if full_session
                else "MODEL_TRAINING_STATE_ONLY"
            ),
            **dict(metadata or {}),
        },
    }
    if full_session:
        payload.update(
            {
                "environment_state": environment.state_dict(),  # type: ignore[union-attr]
                "rollout_buffer_state": rollout_buffer.state_dict(),  # type: ignore[union-attr]
                "rollout_cursor": cursor,
                "active_records": [dict(record) for record in (active_records or [])],
            }
        )
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a v4 session or migrate v2/v3 payloads for evaluation-only use."""

    path = Path(path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.6
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") in {LEGACY_CHECKPOINT_FORMAT, UPDATE_BOUNDARY_FORMAT}:
        payload = dict(payload)
        payload["legacy_format"] = payload["format"]
        payload["resume_contract"] = "LEGACY_EVALUATION_ONLY"
        completed = payload.get("completed_update", payload.get("completed_updates", 0))
        payload["completed_update"] = int(completed)
        payload["next_update"] = int(completed) + 1
        payload.setdefault("full_config", {})
        payload.setdefault("reproducibility_manifest", {})
        payload.setdefault("training_extension_state", {})
        payload["full_session_resume_supported"] = False
        payload["mid_rollout_resume_supported"] = False
        return payload
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    # v4 checkpoints created before optional method extensions remain exact
    # core-session checkpoints with an empty extension namespace.
    payload.setdefault("training_extension_state", {})
    required = {
        "resume_contract",
        "completed_update",
        "next_update",
        "model",
        "policy_old",
        "optimizer",
        "rng_state",
        "dataset_sampler_state",
        "model_config",
        "environment_config",
        "ppo_config",
        "full_config",
        "reproducibility_manifest",
        "training_extension_state",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    if int(payload["next_update"]) != int(payload["completed_update"]) + 1:
        raise ValueError("checkpoint update cursor is inconsistent")
    if payload["resume_contract"] == RESUME_CONTRACT:
        session_required = {
            "environment_state",
            "rollout_buffer_state",
            "rollout_cursor",
            "active_records",
        }
        missing_session = session_required - payload.keys()
        if missing_session:
            raise ValueError(f"full-session checkpoint lacks: {sorted(missing_session)}")
        cursor = payload["rollout_cursor"]
        if int(cursor["completed_updates"]) != int(payload["completed_update"]):
            raise ValueError("session/update cursor mismatch")
        states = payload["rollout_buffer_state"].get("states", [])
        if int(cursor["transition_index"]) != len(states):
            raise ValueError("session/rollout cursor mismatch")
    elif payload["resume_contract"] != WEIGHTS_ONLY_CONTRACT:
        raise ValueError("unsupported v4 resume contract")
    return payload


def restore_trainer(
    trainer: RAMPPPO,
    payload: Mapping[str, Any],
    *,
    load_optimizer: bool = True,
) -> int:
    """Restore policy/behavior policy/optimizer and return update count."""

    if load_optimizer and payload.get("resume_contract") == "LEGACY_EVALUATION_ONLY":
        raise ValueError(
            "legacy checkpoints are evaluation-only; training requires a v4 checkpoint"
        )
    trainer.policy.load_state_dict(payload["model"], strict=True)
    trainer.policy_old.load_state_dict(payload["policy_old"], strict=True)
    if load_optimizer:
        trainer.optimizer.load_state_dict(payload["optimizer"])
    return int(payload["completed_update"])


def restore_training_session(
    trainer: RAMPPPO,
    environment: Stateful,
    rollout_buffer: RAMPRolloutBuffer,
    payload: Mapping[str, Any],
    *,
    dataset_sampler: Stateful | None = None,
) -> dict[str, Any]:
    """Restore every training authority and return the exact rollout cursor.

    RNG restoration is intentionally last: callers may construct the compatible
    environment, policy and sampler first without perturbing the continuation.
    """

    if payload.get("resume_contract") != RESUME_CONTRACT:
        raise ValueError("strict training continuation requires FULL_SESSION_CHECKPOINT")
    restore_trainer(trainer, payload, load_optimizer=True)
    environment.load_state_dict(payload["environment_state"])
    rollout_buffer.load_state_dict(payload["rollout_buffer_state"])
    if dataset_sampler is not None and payload["dataset_sampler_state"]:
        dataset_sampler.load_state_dict(payload["dataset_sampler_state"])
    restore_random_state(payload)
    return dict(payload["rollout_cursor"])
