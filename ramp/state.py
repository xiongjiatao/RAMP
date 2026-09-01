"""Observed scheduling authority, action codec, and forecast tensor state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any, Iterable

import torch


class ActionType(IntEnum):
    PRODUCTION = 0
    PREVENTIVE_MAINTENANCE = 1
    CORRECTIVE_MAINTENANCE = 2


@dataclass(frozen=True)
class DecodedActions:
    action_type: torch.Tensor
    job: torch.Tensor
    machine: torch.Tensor


class ActionCodec:
    """Flatten ``J*M`` production, ``M`` PM, and ``M`` CM actions."""

    def __init__(self, number_of_jobs: int, number_of_machines: int):
        self.number_of_jobs = int(number_of_jobs)
        self.number_of_machines = int(number_of_machines)
        self.production_count = self.number_of_jobs * self.number_of_machines
        self.pm_offset = self.production_count
        self.cm_offset = self.production_count + self.number_of_machines
        self.total_actions = self.production_count + 2 * self.number_of_machines

    def production(self, job: int, machine: int) -> int:
        return int(job) * self.number_of_machines + int(machine)

    def pm(self, machine: int) -> int:
        return self.pm_offset + int(machine)

    def cm(self, machine: int) -> int:
        return self.cm_offset + int(machine)

    def decode(self, actions: torch.Tensor) -> DecodedActions:
        actions = actions.long()
        if torch.any((actions < 0) | (actions >= self.total_actions)):
            raise ValueError("action index outside the joint action space")
        production = actions < self.pm_offset
        preventive = (actions >= self.pm_offset) & (actions < self.cm_offset)
        action_type = torch.full_like(
            actions, int(ActionType.CORRECTIVE_MAINTENANCE)
        )
        action_type[production] = int(ActionType.PRODUCTION)
        action_type[preventive] = int(ActionType.PREVENTIVE_MAINTENANCE)
        job = torch.where(
            production,
            actions // self.number_of_machines,
            -torch.ones_like(actions),
        )
        machine = torch.where(
            production,
            actions % self.number_of_machines,
            torch.where(
                preventive,
                actions - self.pm_offset,
                actions - self.cm_offset,
            ),
        )
        return DecodedActions(action_type=action_type, job=job, machine=machine)


@dataclass
class ObservedShopState:
    """The single current state observed by the online scheduler.

    No tensor in this class has a scenario axis. Forecast state and reward
    scenarios must be reconstructed from this authority after every action.
    """

    candidate: torch.Tensor
    job_finished: torch.Tensor
    op_scheduled: torch.Tensor
    observed_health: torch.Tensor
    observed_machine_status: torch.Tensor
    observed_job_ready_time: torch.Tensor
    observed_machine_ready_time: torch.Tensor
    op_completion_time: torch.Tensor
    pm_count: torch.Tensor
    cm_count: torch.Tensor
    maintenance_decision_count: torch.Tensor
    production_count: torch.Tensor
    pm_cost_total: torch.Tensor
    cm_cost_total: torch.Tensor
    maintenance_cost: torch.Tensor
    failure_count: torch.Tensor
    productive_processing_time: torch.Tensor
    available_idle_time: torch.Tensor
    preventive_maintenance_time: torch.Tensor
    corrective_maintenance_time: torch.Tensor
    diagnosis_delay_time: torch.Tensor
    failed_waiting_time: torch.Tensor
    unplanned_downtime: torch.Tensor
    failed_since_time: torch.Tensor
    failed_wait_accounted_until: torch.Tensor
    current_makespan: torch.Tensor
    decision_count: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    action_history: list[list[int]]

    def clone(self, device: torch.device | str | None = None) -> "ObservedShopState":
        """Deep-clone tensors and action history for exact checkpointing."""

        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                cloned = value.detach().clone()
                values[field.name] = cloned if device is None else cloned.to(device)
            else:
                values[field.name] = deepcopy(value)
        return ObservedShopState(**values)

    def state_dict(self) -> dict[str, Any]:
        """Return a device-independent, serialization-safe observed state."""

        result: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
            )
        return result

    @classmethod
    def from_state_dict(
        cls, payload: dict[str, Any], device: torch.device | str = "cpu"
    ) -> "ObservedShopState":
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.name in payload:
                value = payload[field.name]
            elif field.name == "failed_wait_accounted_until":
                # State-v1 migration: historical checkpoints had only the
                # detection time, which is the correct initial accounting cursor.
                value = payload["failed_since_time"].clone()
            else:
                raise KeyError(f"observed-state payload lacks {field.name}")
            values[field.name] = (
                value.to(device) if isinstance(value, torch.Tensor) else deepcopy(value)
            )
        return cls(**values)


@dataclass
class ScenarioTrajectoryState:
    """Persistent stochastic scheduling trajectories with stable identity.

    Every tensor starts with ``[B,S]``.  Unlike a one-step forecast, an instance
    of this class owns the health, failure status, schedule clocks, operation
    completion, maintenance history, and accumulated cost of each scenario for
    the whole episode.  The state and reward banks use separate instances and
    separate noise namespaces.
    """

    episode_ids: torch.Tensor
    scenario_ids: torch.Tensor
    noise_namespace: str
    candidate: torch.Tensor
    job_finished: torch.Tensor
    operation_scheduled: torch.Tensor
    health: torch.Tensor
    machine_status: torch.Tensor
    job_ready_time: torch.Tensor
    machine_ready_time: torch.Tensor
    operation_completion_time: torch.Tensor
    pm_count: torch.Tensor
    cm_count: torch.Tensor
    maintenance_decision_count: torch.Tensor
    production_count: torch.Tensor
    pm_cost: torch.Tensor
    cm_cost: torch.Tensor
    maintenance_cost: torch.Tensor
    productive_time: torch.Tensor
    available_idle_time: torch.Tensor
    pm_time: torch.Tensor
    cm_time: torch.Tensor
    diagnosis_time: torch.Tensor
    failed_waiting_time: torch.Tensor
    failure_count: torch.Tensor
    unplanned_downtime: torch.Tensor
    failed_since_time: torch.Tensor
    failed_wait_accounted_until: torch.Tensor
    current_makespan: torch.Tensor
    decision_count: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    scenario_cost_components: torch.Tensor
    action_history: list[list[list[int]]]

    @classmethod
    def from_observed(
        cls,
        observed: ObservedShopState,
        scenario_count: int,
        *,
        episode_id: int,
        noise_namespace: str,
    ) -> "ScenarioTrajectoryState":
        """Create scenario identities from one common initial observed state."""

        if scenario_count < 1:
            raise ValueError("scenario_count must be positive")
        b = observed.candidate.shape[0]

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand((-1, scenario_count) + value.shape[1:]).clone()

        device = observed.candidate.device
        dtype = observed.current_makespan.dtype
        return cls(
            episode_ids=torch.full((b, scenario_count), int(episode_id), dtype=torch.long, device=device),
            scenario_ids=torch.arange(scenario_count, device=device).expand(b, -1).clone(),
            noise_namespace=str(noise_namespace),
            candidate=repeat(observed.candidate),
            job_finished=repeat(observed.job_finished),
            operation_scheduled=repeat(observed.op_scheduled),
            health=repeat(observed.observed_health),
            machine_status=repeat(observed.observed_machine_status),
            job_ready_time=repeat(observed.observed_job_ready_time),
            machine_ready_time=repeat(observed.observed_machine_ready_time),
            operation_completion_time=repeat(observed.op_completion_time),
            pm_count=repeat(observed.pm_count),
            cm_count=repeat(observed.cm_count),
            maintenance_decision_count=repeat(observed.maintenance_decision_count),
            production_count=repeat(observed.production_count),
            pm_cost=repeat(observed.pm_cost_total),
            cm_cost=repeat(observed.cm_cost_total),
            maintenance_cost=repeat(observed.maintenance_cost),
            productive_time=repeat(observed.productive_processing_time),
            available_idle_time=repeat(observed.available_idle_time),
            pm_time=repeat(observed.preventive_maintenance_time),
            cm_time=repeat(observed.corrective_maintenance_time),
            diagnosis_time=repeat(observed.diagnosis_delay_time),
            failed_waiting_time=repeat(observed.failed_waiting_time),
            failure_count=repeat(observed.failure_count),
            unplanned_downtime=repeat(observed.unplanned_downtime),
            failed_since_time=repeat(observed.failed_since_time),
            failed_wait_accounted_until=repeat(observed.failed_wait_accounted_until),
            current_makespan=repeat(observed.current_makespan),
            decision_count=repeat(observed.decision_count),
            terminated=repeat(observed.terminated),
            truncated=repeat(observed.truncated),
            scenario_cost_components=torch.zeros((b, scenario_count, 5), dtype=dtype, device=device),
            action_history=[[[] for _ in range(scenario_count)] for _ in range(b)],
        )

    @property
    def batch_size(self) -> int:
        return int(self.health.shape[0])

    @property
    def scenario_count(self) -> int:
        return int(self.health.shape[1])

    def as_observed(self, scenario: int) -> ObservedShopState:
        """Expose one scenario slice to the shared physical kernel."""

        s = int(scenario)
        return ObservedShopState(
            candidate=self.candidate[:, s].clone(),
            job_finished=self.job_finished[:, s].clone(),
            op_scheduled=self.operation_scheduled[:, s].clone(),
            observed_health=self.health[:, s].clone(),
            observed_machine_status=self.machine_status[:, s].clone(),
            observed_job_ready_time=self.job_ready_time[:, s].clone(),
            observed_machine_ready_time=self.machine_ready_time[:, s].clone(),
            op_completion_time=self.operation_completion_time[:, s].clone(),
            pm_count=self.pm_count[:, s].clone(),
            cm_count=self.cm_count[:, s].clone(),
            maintenance_decision_count=self.maintenance_decision_count[:, s].clone(),
            production_count=self.production_count[:, s].clone(),
            pm_cost_total=self.pm_cost[:, s].clone(),
            cm_cost_total=self.cm_cost[:, s].clone(),
            maintenance_cost=self.maintenance_cost[:, s].clone(),
            failure_count=self.failure_count[:, s].clone(),
            productive_processing_time=self.productive_time[:, s].clone(),
            available_idle_time=self.available_idle_time[:, s].clone(),
            preventive_maintenance_time=self.pm_time[:, s].clone(),
            corrective_maintenance_time=self.cm_time[:, s].clone(),
            diagnosis_delay_time=self.diagnosis_time[:, s].clone(),
            failed_waiting_time=self.failed_waiting_time[:, s].clone(),
            unplanned_downtime=self.unplanned_downtime[:, s].clone(),
            failed_since_time=self.failed_since_time[:, s].clone(),
            failed_wait_accounted_until=self.failed_wait_accounted_until[:, s].clone(),
            current_makespan=self.current_makespan[:, s].clone(),
            decision_count=self.decision_count[:, s].clone(),
            terminated=self.terminated[:, s].clone(),
            truncated=self.truncated[:, s].clone(),
            action_history=[list(self.action_history[row][s]) for row in range(self.batch_size)],
        )

    def update_from_observed(self, scenario: int, state: ObservedShopState) -> None:
        """Commit one kernel-transitioned slice without changing scenario identity."""

        s = int(scenario)
        mapping = {
            "candidate": "candidate",
            "job_finished": "job_finished",
            "operation_scheduled": "op_scheduled",
            "health": "observed_health",
            "machine_status": "observed_machine_status",
            "job_ready_time": "observed_job_ready_time",
            "machine_ready_time": "observed_machine_ready_time",
            "operation_completion_time": "op_completion_time",
            "pm_count": "pm_count",
            "cm_count": "cm_count",
            "maintenance_decision_count": "maintenance_decision_count",
            "production_count": "production_count",
            "pm_cost": "pm_cost_total",
            "cm_cost": "cm_cost_total",
            "maintenance_cost": "maintenance_cost",
            "productive_time": "productive_processing_time",
            "available_idle_time": "available_idle_time",
            "pm_time": "preventive_maintenance_time",
            "cm_time": "corrective_maintenance_time",
            "diagnosis_time": "diagnosis_delay_time",
            "failed_waiting_time": "failed_waiting_time",
            "failure_count": "failure_count",
            "unplanned_downtime": "unplanned_downtime",
            "failed_since_time": "failed_since_time",
            "failed_wait_accounted_until": "failed_wait_accounted_until",
            "current_makespan": "current_makespan",
            "decision_count": "decision_count",
            "terminated": "terminated",
            "truncated": "truncated",
        }
        for target, source in mapping.items():
            getattr(self, target)[:, s].copy_(getattr(state, source))
        for row in range(self.batch_size):
            self.action_history[row][s] = list(state.action_history[row])

    def clone(self, device: torch.device | str | None = None) -> "ScenarioTrajectoryState":
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                copy = value.detach().clone()
                values[field.name] = copy if device is None else copy.to(device)
            else:
                values[field.name] = deepcopy(value)
        return ScenarioTrajectoryState(**values)

    def state_dict(self) -> dict[str, Any]:
        return {
            field.name: (
                getattr(self, field.name).detach().cpu().clone()
                if isinstance(getattr(self, field.name), torch.Tensor)
                else deepcopy(getattr(self, field.name))
            )
            for field in fields(self)
        }

    @classmethod
    def from_state_dict(
        cls, payload: dict[str, Any], device: torch.device | str = "cpu"
    ) -> "ScenarioTrajectoryState":
        return cls(
            **{
                field.name: (
                    payload[field.name].to(device)
                    if isinstance(payload[field.name], torch.Tensor)
                    else deepcopy(payload[field.name])
                )
                for field in fields(cls)
            }
        )


def new_boundary_events(
    terminated_before: torch.Tensor,
    terminated_after: torch.Tensor,
    truncated_before: torch.Tensor,
    truncated_after: torch.Tensor,
) -> dict[str, int]:
    """Count boundary events on their false-to-true edge exactly once."""

    return {
        "terminated": int((terminated_after & ~terminated_before).sum().item()),
        "truncated": int((truncated_after & ~truncated_before).sum().item()),
    }


@dataclass
class ForecastScenarioBatch:
    """Counterfactual result of one common action in every future scenario.

    ``root_*`` tensors are the scenario-free observed state repeated only for
    evaluation.  ``post_*`` tensors are produced by applying the same action
    through the shared physical transition kernel with scenario-specific
    future noise.  No tensor in this object is allowed to mutate the observed
    scheduling authority.
    """

    action: torch.Tensor
    root_health: torch.Tensor
    root_machine_status: torch.Tensor
    post_health: torch.Tensor
    post_machine_status: torch.Tensor
    post_job_ready_time: torch.Tensor
    post_machine_ready_time: torch.Tensor
    duration: torch.Tensor
    degradation_increment: torch.Tensor
    maintenance_cost_increment: torch.Tensor
    failure_crossing: torch.Tensor

    def to(self, device: torch.device | str) -> "ForecastScenarioBatch":
        """Move a forecast result without changing its axis contract."""

        return ForecastScenarioBatch(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )

    def detach_clone(
        self, device: torch.device | str | None = None
    ) -> "ForecastScenarioBatch":
        """Clone a forecast result for evidence or checkpoint payloads."""

        values: dict[str, torch.Tensor] = {}
        for field in fields(self):
            value = getattr(self, field.name).detach().clone()
            values[field.name] = value if device is None else value.to(device)
        return ForecastScenarioBatch(**values)


@dataclass
class RAMPEnvState:
    """Forecast tensors; every mask uses ``True = invalid``."""

    fea_j_tensor: torch.Tensor
    op_mask_tensor: torch.Tensor
    fea_m_tensor: torch.Tensor
    mch_mask_tensor: torch.Tensor
    dynamic_pair_mask_tensor: torch.Tensor
    comp_idx_tensor: torch.Tensor
    candidate_tensor: torch.Tensor
    fea_pairs_tensor: torch.Tensor
    health_m_tensor: torch.Tensor
    health_pair_tensor: torch.Tensor
    health_pair_mask_tensor: torch.Tensor
    pm_mask_tensor: torch.Tensor
    cm_mask_tensor: torch.Tensor
    action_mask_tensor: torch.Tensor
    scenario_invalid_mask_tensor: torch.Tensor
    machine_risk_tensor: torch.Tensor
    pair_risk_tensor: torch.Tensor
    scenario_current_health_tensor: torch.Tensor
    production_candidate_scenarios_tensor: torch.Tensor
    pm_candidate_scenarios_tensor: torch.Tensor
    cm_candidate_scenarios_tensor: torch.Tensor
    observed_operation_tensor: torch.Tensor
    observed_machine_tensor: torch.Tensor
    observed_pair_tensor: torch.Tensor
    observed_global_tensor: torch.Tensor
    all_expected_delta_tensor: torch.Tensor
    all_survival_tensor: torch.Tensor
    failure_level_tensor: torch.Tensor
    compatibility_tensor: torch.Tensor
    terminated_tensor: torch.Tensor
    truncated_tensor: torch.Tensor

    def to(self, device: torch.device | str) -> "RAMPEnvState":
        return RAMPEnvState(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )

    def detach_clone(self, device: torch.device | str | None = None) -> "RAMPEnvState":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name).detach().clone()
            values[field.name] = value if device is None else value.to(device)
        return RAMPEnvState(**values)

    def index_select(self, indices: torch.Tensor) -> "RAMPEnvState":
        return RAMPEnvState(
            **{
                field.name: getattr(self, field.name).index_select(0, indices)
                for field in fields(self)
            }
        )

    @classmethod
    def cat(cls, states: Iterable["RAMPEnvState"]) -> "RAMPEnvState":
        state_list = list(states)
        if not state_list:
            raise ValueError("cannot concatenate an empty state list")
        return cls(
            **{
                field.name: torch.cat(
                    [getattr(state, field.name) for state in state_list], dim=0
                )
                for field in fields(cls)
            }
        )

    def permute_machines(self, permutation: torch.Tensor) -> "RAMPEnvState":
        """Return the exact machine-relabelled state used for equivariance tests."""

        permutation = permutation.to(self.fea_m_tensor.device).long()
        batch, jobs, machines = self.dynamic_pair_mask_tensor.shape
        if permutation.shape != (machines,):
            raise ValueError("machine permutation has the wrong length")
        production = self.action_mask_tensor[:, : jobs * machines].reshape(
            batch, jobs, machines
        )
        pm = self.action_mask_tensor[:, jobs * machines : jobs * machines + machines]
        cm = self.action_mask_tensor[:, jobs * machines + machines :]
        values = {field.name: getattr(self, field.name).clone() for field in fields(self)}
        for name in ("fea_m_tensor", "health_m_tensor", "machine_risk_tensor"):
            values[name] = getattr(self, name)[:, :, permutation]
        values["mch_mask_tensor"] = self.mch_mask_tensor[:, permutation][:, :, permutation]
        values["dynamic_pair_mask_tensor"] = self.dynamic_pair_mask_tensor[:, :, permutation]
        values["comp_idx_tensor"] = self.comp_idx_tensor[:, permutation][:, :, permutation]
        for name in (
            "fea_pairs_tensor",
            "health_pair_tensor",
            "health_pair_mask_tensor",
            "pair_risk_tensor",
            "production_candidate_scenarios_tensor",
        ):
            values[name] = getattr(self, name)[:, :, :, permutation]
        for name in (
            "pm_candidate_scenarios_tensor",
            "cm_candidate_scenarios_tensor",
        ):
            values[name] = getattr(self, name)[:, :, permutation]
        values["observed_machine_tensor"] = self.observed_machine_tensor[:, permutation]
        values["observed_pair_tensor"] = self.observed_pair_tensor[:, :, permutation]
        values["all_expected_delta_tensor"] = self.all_expected_delta_tensor[:, :, :, permutation]
        values["all_survival_tensor"] = self.all_survival_tensor[:, :, :, permutation]
        values["failure_level_tensor"] = self.failure_level_tensor[:, permutation]
        values["compatibility_tensor"] = self.compatibility_tensor[:, :, permutation]
        values["pm_mask_tensor"] = self.pm_mask_tensor[:, :, permutation]
        values["cm_mask_tensor"] = self.cm_mask_tensor[:, :, permutation]
        values["scenario_current_health_tensor"] = self.scenario_current_health_tensor[
            :, :, permutation
        ]
        values["action_mask_tensor"] = torch.cat(
            (
                production[:, :, permutation].reshape(batch, jobs * machines),
                pm[:, permutation],
                cm[:, permutation],
            ),
            dim=1,
        )
        return RAMPEnvState(**values)
