"""RAMP adapter for the clean-room FIFO/MOR/SPT/MWKR baselines.

The selector in :mod:`pdr_baselines` deliberately depends only on a small
deterministic PDRState contract.  This module is the repository-specific
bridge to the observed-state RAMP environment.  In particular, it never
passes forecast tensors, health scenarios, reward-bank values, or policy
features into a PDR priority rule.

The paper exposes two physical regimes, ``H0`` (healthy) and ``H1``
(stochastic health and maintenance).  This adapter keeps those names at its
public boundary.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch

from ramp import RAMPConfig, RAMPEnv
from ramp.overlay import HealthOverlay
from pdr_baselines import (
    PDRState,
    PlanningScore,
    PDRCompatibleEnv,
    RuleName,
    _stable_seed,
    select_ramp_compatible_action,
)


PaperRegime = Literal["H0", "H1"]


def pdr_config_for_paper_regime(
    regime: PaperRegime,
    *,
    state_scenarios: int = 32,
    scenario_seed: int = 400,
    epsilon_use: float = 0.05,
) -> RAMPConfig:
    """Return the PDR physical configuration for a manuscript regime.

    PDRs never receive proactive PM or forecast/scenario masks; under active
    H1 they retain only the common mandatory-CM recovery after an observed
    failure.
    """

    name = str(regime).upper()
    if name == "H0":
        base = RAMPConfig.from_paper_regime(
            "H0",
            num_scenarios=state_scenarios,
            scenario_seed=scenario_seed,
            epsilon_use=epsilon_use,
        )
        return replace(
            base,
            preventive_maintenance_actions=False,
            corrective_maintenance_actions=False,
            scenario_safety_mask=False,
            scenario_recourse=False,
            chance_constraint_empty_set_backoff=True,
        )
    if name == "H1":
        base = RAMPConfig.from_paper_regime(
            "H1",
            num_scenarios=state_scenarios,
            scenario_seed=scenario_seed,
            epsilon_use=epsilon_use,
        )
        return replace(
            base,
            preventive_maintenance_actions=False,
            corrective_maintenance_actions=True,
            scenario_safety_mask=False,
            chance_constraint_empty_set_backoff=True,
        )
    raise ValueError(f"unsupported manuscript PDR regime: {regime!r}")


def h1_pdr_config(
    *,
    state_scenarios: int = 32,
    scenario_seed: int = 400,
    epsilon_use: float = 0.05,
) -> RAMPConfig:
    """Return the active H1 protocol configuration for PDR evaluation.

    PDR priorities are deterministic.  The scenario-safety mask is therefore
    disabled for this adapter; otherwise a PDR would silently become a
    health-aware rule because the mask uses future scenario survival. CM is
    retained and PM is disabled.
    """

    return pdr_config_for_paper_regime(
        "H1",
        state_scenarios=state_scenarios,
        scenario_seed=scenario_seed,
        epsilon_use=epsilon_use,
    )


def _as_numpy(value: torch.Tensor | np.ndarray, *, dtype: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _pdr_state_for_row(
    env: RAMPEnv, row: int
) -> PDRState:
    """Build the deterministic PDR view for one row of a batch environment."""

    state = env.observed_state
    nominal = env.nominal_processing_times[row]
    candidate = state.candidate[row].long()
    unfinished = ~state.job_finished[row]
    proc_time = nominal.index_select(0, candidate)
    feasible = (proc_time > 0) & unfinished[:, None]

    op_compatible = nominal > 0
    op_mean = nominal.sum(dim=1) / op_compatible.sum(dim=1).clamp_min(1)
    remaining_ops = torch.zeros_like(candidate, dtype=torch.float32)
    remaining_work = torch.zeros_like(candidate, dtype=torch.float32)
    for job in range(env.number_of_jobs):
        current = int(candidate[job])
        last = int(env.job_last_op[row, job])
        if bool(unfinished[job]):
            remaining_ops[job] = float(last - current + 1)
            remaining_work[job] = op_mean[current : last + 1].sum()

    return PDRState(
        unfinished_job=_as_numpy(unfinished, dtype=bool),
        job_ready_time=_as_numpy(state.observed_job_ready_time[row], dtype=float),
        machine_ready_time=_as_numpy(
            state.observed_machine_ready_time[row], dtype=float
        ),
        feasible_pair=_as_numpy(feasible, dtype=bool),
        proc_time=_as_numpy(proc_time, dtype=float),
        remaining_ops=_as_numpy(remaining_ops, dtype=float),
        remaining_work=_as_numpy(remaining_work, dtype=float),
    )


class RAMPBatchOnePDREnv:
    """Expose one RAMP environment through ``PDRCompatibleEnv``.

    The underlying RAMP environment is batch-oriented.  PDR evaluation is
    intentionally one instance at a time, so this wrapper enforces batch size
    one and converts the repository's ``ActionCodec``/termination protocol to
    the clean-room interface.
    """

    def __init__(self, env: RAMPEnv):
        if int(env.batch_size) != 1:
            raise ValueError("RAMP PDR adapter requires a batch-size-one environment")
        self.env = env

    @property
    def done(self) -> bool:
        state = self.env.observed_state
        return bool((state.terminated | state.truncated).all().item())

    @property
    def device(self) -> torch.device:
        return self.env.device

    def pdr_state(self) -> PDRState:
        """Build the deterministic current-candidate view used by PDRs."""
        return _pdr_state_for_row(self.env, 0)

    def failed_machine_mask(self) -> np.ndarray:
        return _as_numpy(self.env.observed_state.observed_machine_status[0], dtype=bool)

    def remaining_compatible_work(self) -> np.ndarray:
        """Nominal unfinished work compatible with each machine.

        This is used only to choose among simultaneous mandatory CM actions.
        It contains no health, scenario, CVaR, or reward-bank quantity.
        """

        state = self.env.observed_state
        nominal = self.env.nominal_processing_times[0]
        unscheduled = ~state.op_scheduled[0]
        return _as_numpy(
            (nominal * unscheduled[:, None].to(nominal.dtype)).sum(dim=0),
            dtype=float,
        )

    def encode_production(self, job: int, machine: int) -> int:
        return self.env.codec.production(job, machine)

    def encode_cm(self, machine: int) -> int:
        return self.env.codec.cm(machine)

    def step(self, action: int):
        action_tensor = torch.tensor([int(action)], dtype=torch.long, device=self.device)
        return self.env.step(action_tensor, return_tensors=True)

    def metrics(self) -> dict[str, float]:
        raw = self.env.metrics()
        return {
            key: float(value.detach().cpu().reshape(-1)[0].item())
            for key, value in raw.items()
            if isinstance(value, torch.Tensor) and value.numel() >= 1
        }


class _PDRBatchRowView:
    """One-row view used only while selecting a batch of PDR candidates."""

    def __init__(self, pool: "RAMPBatchPDRPool", row: int):
        self.pool = pool
        self.row = int(row)

    def pdr_state(self) -> PDRState:
        return _pdr_state_for_row(self.pool.env, self.row)

    def failed_machine_mask(self) -> np.ndarray:
        return _as_numpy(
            self.pool.env.observed_state.observed_machine_status[self.row],
            dtype=bool,
        )

    def remaining_compatible_work(self) -> np.ndarray:
        state = self.pool.env.observed_state
        nominal = self.pool.env.nominal_processing_times[self.row]
        unscheduled = ~state.op_scheduled[self.row]
        return _as_numpy(
            (nominal * unscheduled[:, None].to(nominal.dtype)).sum(dim=0),
            dtype=float,
        )

    def encode_production(self, job: int, machine: int) -> int:
        return self.pool.env.codec.production(job, machine)

    def encode_cm(self, machine: int) -> int:
        return self.pool.env.codec.cm(machine)


class RAMPBatchPDRPool:
    """Batch transition host for independent PDR rollouts.

    Every row has its own tie-breaking generator and action trace, while all
    rows share one vectorized physical transition.  This changes execution
    throughput only; it does not share a schedule or a random stream between
    PDR candidates.
    """

    def __init__(self, env: RAMPEnv):
        self.env = env

    @property
    def batch_size(self) -> int:
        return int(self.env.batch_size)

    @property
    def device(self) -> torch.device:
        return self.env.device

    @property
    def active(self) -> np.ndarray:
        state = self.env.observed_state
        return _as_numpy(~(state.terminated | state.truncated), dtype=bool)

    def row_view(self, row: int) -> PDRCompatibleEnv:
        return _PDRBatchRowView(self, row)

    def step(self, actions: Sequence[int]) -> None:
        if len(actions) != self.batch_size:
            raise ValueError("batch PDR action count does not match batch size")
        tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.env.step(tensor, return_tensors=True)


def make_pdr_pool(
    job_lengths: Sequence[torch.Tensor | np.ndarray],
    nominal_processing_times: Sequence[torch.Tensor | np.ndarray],
    *,
    regime: PaperRegime,
    overlay: HealthOverlay | None = None,
    config: RAMPConfig | None = None,
    reward_scenarios: int = 1,
    reward_seed: int = 1_400_000,
    device: torch.device | str = "cpu",
) -> RAMPBatchPDRPool:
    """Construct a vectorized PDR host for equal-shaped instance rows."""

    if len(job_lengths) != len(nominal_processing_times) or not job_lengths:
        raise ValueError("PDR pool requires a non-empty aligned batch")
    jobs = torch.stack([torch.as_tensor(value, dtype=torch.long) for value in job_lengths])
    nominal = torch.stack(
        [torch.as_tensor(value, dtype=torch.float32) for value in nominal_processing_times]
    )
    env = RAMPEnv(
        jobs,
        nominal,
        overlay=overlay,
        config=config or pdr_config_for_paper_regime(regime),
        reward_num_scenarios=reward_scenarios,
        reward_seed=reward_seed,
        device=device,
    )
    return RAMPBatchPDRPool(env)


def make_h1_pdr_env(
    job_lengths: torch.Tensor | np.ndarray,
    nominal_processing_times: torch.Tensor | np.ndarray,
    *,
    overlay: HealthOverlay | None = None,
    config: RAMPConfig | None = None,
    reward_scenarios: int = 128,
    reward_seed: int = 1_400_000,
    device: torch.device | str = "cpu",
) -> RAMPBatchOnePDREnv:
    """Construct the common H1 stochastic-health environment for one instance."""

    env = RAMPEnv(
        job_lengths,
        nominal_processing_times,
        overlay=overlay,
        config=config or h1_pdr_config(),
        reward_num_scenarios=reward_scenarios,
        reward_seed=reward_seed,
        device=device,
    )
    return RAMPBatchOnePDREnv(env)


def make_pdr_env(
    job_lengths: torch.Tensor | np.ndarray,
    nominal_processing_times: torch.Tensor | np.ndarray,
    *,
    regime: PaperRegime,
    overlay: HealthOverlay | None = None,
    config: RAMPConfig | None = None,
    reward_scenarios: int = 128,
    reward_seed: int = 1_400_000,
    device: torch.device | str = "cpu",
) -> RAMPBatchOnePDREnv:
    """Construct one batch-one PDR environment for manuscript H0 or H1."""

    return make_h1_pdr_env(
        job_lengths,
        nominal_processing_times,
        overlay=overlay,
        config=config or pdr_config_for_paper_regime(regime),
        reward_scenarios=reward_scenarios,
        reward_seed=reward_seed,
        device=device,
    )


def score_completed_pdr_env(env: RAMPBatchOnePDREnv) -> PlanningScore:
    """Convert the common evaluator metrics into ``Cost + 0.5 CVaR``."""

    if not env.done or bool(env.env.observed_state.truncated.any().item()):
        raise RuntimeError("cannot score an incomplete or truncated PDR rollout")
    metrics = env.metrics()
    return PlanningScore(
        mean_cost=metrics["mean_total_cost"],
        cvar95=metrics["cvar_0_95_total_cost"],
    )


def replay_actions(
    env: RAMPBatchOnePDREnv,
    actions: Sequence[int],
    *,
    require_terminal: bool = True,
) -> None:
    """Replay a selected action sequence under a fresh common environment."""

    for action in actions:
        if env.done:
            raise RuntimeError("action sequence contains steps after termination")
        env.step(int(action))
    if require_terminal and not env.done:
        raise RuntimeError("action sequence did not terminate the PDR environment")


def make_planning_evaluator(
    env_factory: Callable[[], RAMPBatchOnePDREnv],
) -> Callable[[Sequence[int]], PlanningScore]:
    """Build a fresh-environment evaluator for a PDR candidate schedule.

    The factory should use ``reward_scenarios=32`` for the Best-of-16 planning
    phase.  Final reporting should evaluate the selected candidate separately
    with ``reward_scenarios=128``; no reward-bank values are used to construct
    the PDR priority state.
    """

    def evaluate(actions: Sequence[int]) -> PlanningScore:
        env = env_factory()
        replay_actions(env, actions)
        return score_completed_pdr_env(env)

    return evaluate


__all__ = [
    "RAMPBatchOnePDREnv",
    "RAMPBatchPDRPool",
    "h1_pdr_config",
    "make_pdr_pool",
    "make_pdr_env",
    "make_h1_pdr_env",
    "make_planning_evaluator",
    "pdr_config_for_paper_regime",
    "replay_actions",
    "score_completed_pdr_env",
]
