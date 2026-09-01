"""
Clean-room PDR baselines for RAMP-style stochastic FJSP experiments.

Rule semantics follow the published DAN / SPM-DAN baselines:
FIFO, MOR (MOPNR), SPT, MWKR.

This file intentionally does not depend on the original repositories.
It provides a strict, auditable interface that can be mapped onto a
RAMP environment without changing the rule semantics.

Required RAMP-side adapter contract
-----------------------------------
env.pdr_state() -> PDRState
env.failed_machine_mask() -> np.ndarray[bool], shape [M]
env.remaining_compatible_work() -> np.ndarray[float], shape [M]
env.encode_production(job: int, machine: int) -> int
env.encode_cm(machine: int) -> int
env.step(action: int) -> any
env.done -> bool

For the four classical PDRs:
- proactive PM is never selected;
- if an observed machine is failed, mandatory CM is executed first;
- CM machine choice follows RAMP's deterministic rule:
  max remaining compatible unfinished work, with seeded uniform tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Protocol, Sequence
import hashlib
import math
import numpy as np

RuleName = Literal["FIFO", "MOR", "SPT", "MWKR"]

# Frozen audit metadata.  The selectors below are clean-room code; these
# commits record the source semantics that the paper baseline names refer to.
PDR_BASELINE_PROTOCOL_V1 = {
    "protocol": "PDR_BASELINE_PROTOCOL_V1",
    "source_repositories": {
        "DAN": "wrqccc/FJSP-DRL@2cf81b13f5044451e78cf780f8fb3e7eeac054c1",
        "SPM-DAN": "official-main@1b92d3d934adc146688792d76ac8c1c682e335f8",
        "shared_common_utils_blob": "d94bcab27a4468e253a7d973182acb16bb587917",
    },
    "rules": ("FIFO", "MOR", "SPT", "MWKR"),
    "tie_break": "uniform seeded numpy Generator; stable keyed substreams",
    "priority_information": "deterministic nominal processing and schedule clocks only",
    "maintenance": {
        "proactive_pm": False,
        "observed_failure": "mandatory_cm",
        "simultaneous_cm": "max_remaining_compatible_work",
    },
    "decoding": {
        "greedy": "one seeded rule rollout",
        "best_of": 16,
        "planning_scenarios": 32,
        "reward_scenarios": 128,
        "planning_objective": "mean_cost + 0.5 * CVaR95",
    },
}


@dataclass(frozen=True)
class PDRState:
    """
    Decision-epoch state required by the four PDRs.

    All processing-time information supplied here should be deterministic
    nominal/median information. Do not use reward-bank outcomes or sampled
    future health scenarios in PDR priority calculations.

    Shapes:
        unfinished_job:        [J] bool
        job_ready_time:        [J] float
        machine_ready_time:    [M] float
        feasible_pair:         [J, M] bool
        proc_time:             [J, M] float
        remaining_ops:         [J] int/float
        remaining_work:        [J] float
    """
    unfinished_job: np.ndarray
    job_ready_time: np.ndarray
    machine_ready_time: np.ndarray
    feasible_pair: np.ndarray
    proc_time: np.ndarray
    remaining_ops: np.ndarray
    remaining_work: np.ndarray

    def validate(self) -> None:
        J = self.unfinished_job.shape[0]
        M = self.machine_ready_time.shape[0]

        assert self.unfinished_job.shape == (J,)
        assert self.job_ready_time.shape == (J,)
        assert self.remaining_ops.shape == (J,)
        assert self.remaining_work.shape == (J,)
        assert self.feasible_pair.shape == (J, M)
        assert self.proc_time.shape == (J, M)

        if not np.any(self.unfinished_job):
            return

        bad = self.feasible_pair & ~np.isfinite(self.proc_time)
        if np.any(bad):
            raise ValueError("Feasible production pairs must have finite processing time.")

        if np.any(self.feasible_pair & (self.proc_time <= 0)):
            raise ValueError("Feasible production pairs must have strictly positive processing time.")


class PDRCompatibleEnv(Protocol):
    done: bool

    def pdr_state(self) -> PDRState: ...
    def failed_machine_mask(self) -> np.ndarray: ...
    def remaining_compatible_work(self) -> np.ndarray: ...
    def encode_production(self, job: int, machine: int) -> int: ...
    def encode_cm(self, machine: int) -> int: ...
    def step(self, action: int): ...


def _stable_seed(*keys: object, base_seed: int = 0) -> int:
    """
    Stable cross-process seed. Avoid Python's randomized hash().
    """
    payload = "|".join(map(str, (base_seed, *keys))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _uniform_tie_choice(indices: np.ndarray, rng: np.random.Generator) -> int:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise RuntimeError("Empty tie set.")
    return int(indices[rng.integers(indices.size)])


def _argmin_ties(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)

    best = np.min(values[valid])
    return np.flatnonzero(valid & np.isclose(values, best, rtol=0.0, atol=1e-12))


def _argmax_ties(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)

    best = np.max(values[valid])
    return np.flatnonzero(valid & np.isclose(values, best, rtol=0.0, atol=1e-12))


def _decision_time(s: PDRState) -> float:
    """
    Earliest time at which any feasible job-machine pair can start.
    """
    pair_start = np.maximum(
        s.job_ready_time[:, None],
        s.machine_ready_time[None, :],
    )
    masked = np.where(
        s.feasible_pair & s.unfinished_job[:, None],
        pair_start,
        np.inf,
    )
    t = float(np.min(masked))
    if not np.isfinite(t):
        raise RuntimeError("No feasible production pair remains.")
    return t


def _available_jobs_at_decision_epoch(s: PDRState) -> np.ndarray:
    """
    Source-aligned candidate filtering:
    a job is eligible if unfinished and its current operation is ready
    no later than the next scheduling epoch.
    """
    t = _decision_time(s)
    available = s.unfinished_job & (s.job_ready_time <= t + 1e-12)
    available &= np.any(s.feasible_pair, axis=1)
    jobs = np.flatnonzero(available)
    if jobs.size == 0:
        raise RuntimeError("No available job at current decision epoch.")
    return jobs


def _machine_for_selected_job(
    s: PDRState,
    job: int,
    rng: np.random.Generator,
) -> int:
    """
    DAN source semantics for MOR/MWKR machine choice.

    If all compatible machines become free after the selected job is ready,
    choose an earliest-free compatible machine.
    Otherwise choose uniformly among compatible machines that are already
    free by the selected job's ready time.
    """
    machines = np.flatnonzero(s.feasible_pair[job])
    if machines.size == 0:
        raise RuntimeError(f"Job {job} has no compatible machine.")

    m_ready = s.machine_ready_time[machines]
    j_ready = float(s.job_ready_time[job])

    already_free = machines[m_ready <= j_ready + 1e-12]
    if already_free.size > 0:
        return _uniform_tie_choice(already_free, rng)

    earliest_local = _argmin_ties(m_ready)
    earliest_machines = machines[earliest_local]
    return _uniform_tie_choice(earliest_machines, rng)


def select_fifo(s: PDRState, rng: np.random.Generator) -> tuple[int, int]:
    """
    FIFO:
      1. earliest-ready candidate job;
      2. earliest-ready compatible machine;
      3. uniform random tie-breaking.
    """
    s.validate()
    jobs = _available_jobs_at_decision_epoch(s)

    local = _argmin_ties(s.job_ready_time[jobs])
    job = _uniform_tie_choice(jobs[local], rng)

    machines = np.flatnonzero(s.feasible_pair[job])
    local_m = _argmin_ties(s.machine_ready_time[machines])
    machine = _uniform_tie_choice(machines[local_m], rng)
    return job, machine


def select_mor(s: PDRState, rng: np.random.Generator) -> tuple[int, int]:
    """
    MOR / MOPNR:
      1. among available jobs, maximize number of remaining operations;
      2. choose a compatible machine using the immediate-processing rule;
      3. uniform random tie-breaking.
    """
    s.validate()
    jobs = _available_jobs_at_decision_epoch(s)

    local = _argmax_ties(s.remaining_ops[jobs])
    job = _uniform_tie_choice(jobs[local], rng)
    machine = _machine_for_selected_job(s, job, rng)
    return job, machine


def select_spt(s: PDRState, rng: np.random.Generator) -> tuple[int, int]:
    """
    SPT:
      choose the feasible candidate operation-machine pair with minimum
      deterministic nominal/median processing time at the current decision epoch.
    """
    s.validate()
    t = _decision_time(s)

    pair_start = np.maximum(
        s.job_ready_time[:, None],
        s.machine_ready_time[None, :],
    )
    valid = (
        s.feasible_pair
        & s.unfinished_job[:, None]
        & (pair_start <= t + 1e-12)
    )

    score = np.where(valid, s.proc_time, np.inf)
    best = float(np.min(score))
    pairs = np.argwhere(valid & np.isclose(score, best, rtol=0.0, atol=1e-12))
    if pairs.shape[0] == 0:
        raise RuntimeError("SPT found no feasible pair.")

    idx = int(rng.integers(pairs.shape[0]))
    job, machine = map(int, pairs[idx])
    return job, machine


def select_mwkr(s: PDRState, rng: np.random.Generator) -> tuple[int, int]:
    """
    MWKR:
      1. among available jobs, maximize remaining work;
      2. remaining work is the sum of deterministic mean processing times
         of the current operation and all unscheduled successors;
      3. machine selection follows the same immediate-processing rule as MOR.
    """
    s.validate()
    jobs = _available_jobs_at_decision_epoch(s)

    local = _argmax_ties(s.remaining_work[jobs])
    job = _uniform_tie_choice(jobs[local], rng)
    machine = _machine_for_selected_job(s, job, rng)
    return job, machine


_SELECTOR = {
    "FIFO": select_fifo,
    "MOR": select_mor,
    "SPT": select_spt,
    "MWKR": select_mwkr,
}


def select_pdr_pair(
    rule: RuleName,
    state: PDRState,
    rng: np.random.Generator,
) -> tuple[int, int]:
    try:
        selector = _SELECTOR[rule]
    except KeyError as e:
        raise ValueError(f"Unknown PDR rule: {rule}") from e
    return selector(state, rng)


def select_ramp_compatible_action(
    env: PDRCompatibleEnv,
    rule: RuleName,
    rng: np.random.Generator,
) -> int:
    """
    RAMP-aligned wrapper.

    Maintenance semantics:
    - observed failure -> mandatory CM;
    - no observed failure -> classical PDR production action;
    - PDR never invokes proactive PM.

    This isolates the comparison to production dispatch logic while retaining
    the same physical transition model and mandatory failure recovery.
    """
    failed = np.asarray(env.failed_machine_mask(), dtype=bool).reshape(-1)

    if np.any(failed):
        work = np.asarray(env.remaining_compatible_work(), dtype=np.float64).reshape(-1)
        if work.shape != failed.shape:
            raise ValueError("remaining_compatible_work shape mismatch.")

        candidates = _argmax_ties(work, valid=failed)
        machine = _uniform_tie_choice(candidates, rng)
        return int(env.encode_cm(machine))

    state = env.pdr_state()
    job, machine = select_pdr_pair(rule, state, rng)
    return int(env.encode_production(job, machine))


def run_pdr_rollout(
    env: PDRCompatibleEnv,
    rule: RuleName,
    *,
    base_seed: int,
    instance_id: object,
    rollout_id: int = 0,
) -> list[int]:
    """
    Execute one complete seeded PDR rollout and return the action sequence.

    Seed is keyed by method, instance and rollout, ensuring reproducibility and
    preventing accidental dependence on execution order.
    """
    seed = _stable_seed(rule, instance_id, rollout_id, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    actions: list[int] = []
    guard = 0
    while not env.done:
        action = select_ramp_compatible_action(env, rule, rng)
        env.step(action)
        actions.append(action)

        guard += 1
        if guard > 1_000_000:
            raise RuntimeError("Rollout guard exceeded; check termination logic.")

    return actions


@dataclass(frozen=True)
class PlanningScore:
    mean_cost: float
    cvar95: float

    @property
    def phi(self) -> float:
        return self.mean_cost + 0.5 * self.cvar95


def best_of_k_pdr(
    *,
    rule: RuleName,
    k: int,
    base_seed: int,
    instance_id: object,
    env_factory: Callable[[], PDRCompatibleEnv],
    planning_evaluator: Callable[[Sequence[int]], PlanningScore],
) -> tuple[list[int], PlanningScore, int]:
    """
    Generate K complete PDR schedules using independent tie-breaking substreams,
    score each on the SAME independent planning bank, and return the minimum-phi
    schedule.

    The planning_evaluator must use S_plan=32 and must not use reward-bank
    scenarios. The selected schedule should then be evaluated separately on the
    common S_r=128 reward bank.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    candidates: list[tuple[float, int, list[int], PlanningScore]] = []

    for rollout_id in range(k):
        env = env_factory()
        actions = run_pdr_rollout(
            env,
            rule,
            base_seed=base_seed,
            instance_id=instance_id,
            rollout_id=rollout_id,
        )
        score = planning_evaluator(actions)
        candidates.append((score.phi, rollout_id, actions, score))

    # Deterministic secondary key: rollout_id.
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, rollout_id, actions, score = candidates[0]
    return actions, score, rollout_id
