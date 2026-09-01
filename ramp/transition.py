"""Single physical transition authority for RAMP environments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RAMPConfig
from .noise import TrajectoryNoiseBank, sample_degradation_noise
from .overlay import HealthOverlay
from .state import (
    ActionCodec,
    ActionType,
    ForecastScenarioBatch,
    ObservedShopState,
    ScenarioTrajectoryState,
)


def health_time_factor(
    health: torch.Tensor,
    failure_level: torch.Tensor,
    eta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Multiplicative duration inflation from observed degradation."""

    normalized = (health / failure_level).clamp_min(0)
    return 1.0 + eta * normalized.pow(gamma)


def effective_processing_time(
    nominal: torch.Tensor,
    health: torch.Tensor,
    failure_level: torch.Tensor,
    eta: torch.Tensor,
    gamma: torch.Tensor,
    processing_noise: torch.Tensor,
) -> torch.Tensor:
    """Compute nominal × health factor × future processing noise."""

    return nominal * health_time_factor(health, failure_level, eta, gamma) * processing_noise


def expected_degradation_increment(
    alpha: torch.Tensor,
    load: torch.Tensor,
    load_sensitivity: torch.Tensor,
    effective_duration: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Mean increment of the load-dependent Gamma degradation process."""

    shape = alpha * load.clamp_min(0).pow(load_sensitivity) * effective_duration.clamp_min(0)
    return shape * theta


def sample_degradation_increment(
    alpha: torch.Tensor,
    load: torch.Tensor,
    load_sensitivity: torch.Tensor,
    effective_duration: torch.Tensor,
    theta: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Compatibility sampler used by analytic physics unit tests."""

    concentration = (
        alpha * load.clamp_min(0).pow(load_sensitivity) * effective_duration.clamp_min(0)
    )
    return sample_degradation_noise(concentration, theta, seed=seed)


def restore_health(
    health: torch.Tensor,
    rho: torch.Tensor,
    residual_noise: torch.Tensor,
    failure_level: torch.Tensor,
) -> torch.Tensor:
    """Apply imperfect restoration while keeping degradation below failure."""

    return (rho * health + residual_noise).clamp(min=0).minimum(
        failure_level - torch.finfo(health.dtype).eps
    )


@dataclass(frozen=True)
class ProspectiveProduction:
    """One scenario's future duration and health transition for a pair."""

    duration: torch.Tensor
    expected_delta: torch.Tensor
    sampled_delta: torch.Tensor
    survival: torch.Tensor


@dataclass(frozen=True)
class PrimaryActionTransition:
    """Auditable result of one primary decision plus deterministic recourse.

    A recourse maintenance event is part of the physical transition but is not
    a second policy decision.  The state counters/timelines remain the source
    of truth; these deltas are returned so environments and oracles can expose
    identical transition evidence without reimplementing the rules.
    """

    action: int
    action_type: ActionType
    machine: int
    job: int
    recourse_action_type: ActionType | None
    primary_duration: torch.Tensor
    recourse_duration: torch.Tensor
    total_elapsed: torch.Tensor
    degradation_increment: torch.Tensor
    maintenance_cost_increment: torch.Tensor
    failure_crossing: torch.Tensor
    survival: torch.Tensor


class RAMPTransitionKernel:
    """Only authority allowed to mutate observed scheduling physics.

    Full and no-feature environments compose this kernel. Forecast scenarios
    use :meth:`prospective_production`; only :meth:`apply_production`,
    :meth:`apply_preventive_maintenance`, and
    :meth:`apply_corrective_maintenance` mutate observed state.
    """

    def __init__(
        self,
        *,
        job_first_op: torch.Tensor,
        job_last_op: torch.Tensor,
        nominal_processing_times: torch.Tensor,
        overlay: HealthOverlay,
        config: RAMPConfig,
        observed_noise_bank: TrajectoryNoiseBank,
        episode_id: int,
    ):
        self.job_first_op = job_first_op
        self.job_last_op = job_last_op
        self.nominal_processing_times = nominal_processing_times
        self.compatible = nominal_processing_times > 0
        self.overlay = overlay
        self.config = config
        self.observed_noise_bank = observed_noise_bank
        self.episode_id = int(episode_id)

    def set_episode(self, episode_id: int) -> None:
        self.episode_id = int(episode_id)

    def apply_primary_actions_to_trajectories(
        self,
        trajectories: ScenarioTrajectoryState,
        *,
        actions: torch.Tensor,
        action_codec: ActionCodec,
        noise_bank: TrajectoryNoiseBank,
    ) -> ForecastScenarioBatch:
        """Fused selected-action transition over persistent ``[B,S]`` state.

        This is an execution backend for the same physical authority as
        :meth:`apply_primary_action_with_recourse`.  It updates only the
        selected job/machine/operation fields, while globally accrued failed
        waiting time remains vectorized across machines.
        """

        device = trajectories.health.device
        dtype = trajectories.health.dtype
        actions = actions.to(device=device, dtype=torch.long)
        batch, scenarios, machines = trajectories.health.shape
        jobs = trajectories.candidate.shape[2]
        operations_total = trajectories.operation_scheduled.shape[2]
        if actions.shape != (batch,):
            raise ValueError("actions must have shape [B]")
        decoded = action_codec.decode(actions)
        machine = decoded.machine[:, None].expand(-1, scenarios)
        machine_index = machine[:, :, None]
        job = decoded.job.clamp_min(0)[:, None].expand(-1, scenarios)
        job_index = job[:, :, None]
        action_type = decoded.action_type[:, None].expand(-1, scenarios)
        active = ~(trajectories.terminated | trajectories.truncated)
        production = active & (action_type == int(ActionType.PRODUCTION))
        preventive = active & (
            action_type == int(ActionType.PREVENTIVE_MAINTENANCE)
        )
        corrective = active & (
            action_type == int(ActionType.CORRECTIVE_MAINTENANCE)
        )

        def gather_machine(value: torch.Tensor) -> torch.Tensor:
            return value.gather(2, machine_index).squeeze(2)

        def scatter_machine(value: torch.Tensor, selected: torch.Tensor) -> None:
            value.scatter_(2, machine_index, selected[:, :, None])

        def selected_parameter(value: torch.Tensor) -> torch.Tensor:
            return value.gather(1, decoded.machine[:, None]).expand(-1, scenarios)

        root_health = trajectories.health.clone()
        root_status = trajectories.machine_status.clone()
        ready_before = gather_machine(trajectories.machine_ready_time).clone()
        health_before = gather_machine(trajectories.health).clone()
        cost_before = trajectories.maintenance_cost.clone()
        failure_before = gather_machine(trajectories.failure_count).clone()
        was_failed = gather_machine(trajectories.machine_status)

        if not self.config.scenario_recourse:
            incompatible = active & (
                ((production | preventive) & was_failed)
                | (corrective & ~was_failed)
            )
            if bool(incompatible.any()):
                raise ValueError(
                    "primary action is outside the no-recourse intersection-feasible set"
                )
        recourse = self.config.scenario_recourse & was_failed & (
            production | preventive
        )
        cm_event = corrective | recourse
        pm_event = preventive & ~was_failed
        maintenance_event = cm_event | pm_event

        def accrue_failed_waiting(mask: torch.Tensor, horizon: torch.Tensor) -> None:
            failed = trajectories.machine_status & (
                trajectories.failed_wait_accounted_until >= 0
            )
            affected = mask[:, :, None] & failed
            delta = (
                horizon[:, :, None] - trajectories.failed_wait_accounted_until
            ).clamp_min(0)
            trajectories.failed_waiting_time.add_(
                torch.where(affected, delta, torch.zeros_like(delta))
            )
            trajectories.failed_wait_accounted_until.copy_(
                torch.where(
                    affected,
                    horizon[:, :, None].expand_as(
                        trajectories.failed_wait_accounted_until
                    ),
                    trajectories.failed_wait_accounted_until,
                )
            )

        scenario_ids = trajectories.scenario_ids
        cm_count = gather_machine(trajectories.cm_count)
        pm_count = gather_machine(trajectories.pm_count)
        maintenance_count = torch.where(cm_event, cm_count, pm_count)
        maintenance_std = selected_parameter(
            self.overlay.maintenance_noise_std
        )
        residual = noise_bank.maintenance_noise_selected(
            episode_id=self.episode_id,
            scenario_ids=scenario_ids,
            machine_ids=machine,
            counts=maintenance_count,
            maintenance_is_cm=cm_event,
            std=maintenance_std,
            active=maintenance_event,
            device=device,
            dtype=dtype,
        )
        failure_level = selected_parameter(self.overlay.failure_level)

        # CM recourse or primary CM.
        selected_ready = gather_machine(trajectories.machine_ready_time)
        cm_start = torch.where(
            was_failed,
            torch.maximum(selected_ready, trajectories.current_makespan),
            selected_ready,
        )
        accrue_failed_waiting(cm_event & was_failed, cm_start)
        cm_duration = selected_parameter(self.overlay.cm_duration)
        cm_cost = selected_parameter(self.overlay.cm_cost)
        cm_rho = selected_parameter(self.overlay.cm_rho)
        cm_ready = cm_start + cm_duration
        scatter_machine(
            trajectories.machine_ready_time,
            torch.where(cm_event, cm_ready, selected_ready),
        )
        selected_cm_time = gather_machine(trajectories.cm_time)
        scatter_machine(
            trajectories.cm_time,
            selected_cm_time + torch.where(cm_event, cm_duration, 0.0),
        )
        trajectories.cm_cost.add_(torch.where(cm_event, cm_cost, 0.0))
        scatter_machine(
            trajectories.cm_count,
            cm_count + cm_event.to(cm_count.dtype),
        )
        trajectories.maintenance_decision_count.add_(cm_event.long())
        current_health = gather_machine(trajectories.health)
        cm_post = restore_health(current_health, cm_rho, residual, failure_level)
        scatter_machine(
            trajectories.health,
            torch.where(cm_event, cm_post, current_health),
        )
        scatter_machine(
            trajectories.machine_status,
            torch.where(cm_event, torch.zeros_like(was_failed), was_failed),
        )
        selected_failed_since = gather_machine(trajectories.failed_since_time)
        selected_accounted = gather_machine(
            trajectories.failed_wait_accounted_until
        )
        scatter_machine(
            trajectories.failed_since_time,
            torch.where(cm_event, -torch.ones_like(selected_failed_since), selected_failed_since),
        )
        scatter_machine(
            trajectories.failed_wait_accounted_until,
            torch.where(cm_event, -torch.ones_like(selected_accounted), selected_accounted),
        )
        trajectories.current_makespan.copy_(
            torch.where(
                cm_event,
                torch.maximum(trajectories.current_makespan, cm_ready),
                trajectories.current_makespan,
            )
        )
        cm_horizon = torch.maximum(
            trajectories.current_makespan,
            trajectories.machine_ready_time.amax(dim=2),
        )
        accrue_failed_waiting(cm_event, cm_horizon)

        # Primary PM on a healthy machine.
        selected_ready = gather_machine(trajectories.machine_ready_time)
        pm_duration = selected_parameter(self.overlay.pm_duration)
        pm_cost = selected_parameter(self.overlay.pm_cost)
        pm_rho = selected_parameter(self.overlay.pm_rho)
        pm_ready = selected_ready + pm_duration
        scatter_machine(
            trajectories.machine_ready_time,
            torch.where(pm_event, pm_ready, selected_ready),
        )
        selected_pm_time = gather_machine(trajectories.pm_time)
        scatter_machine(
            trajectories.pm_time,
            selected_pm_time + torch.where(pm_event, pm_duration, 0.0),
        )
        trajectories.pm_cost.add_(torch.where(pm_event, pm_cost, 0.0))
        scatter_machine(
            trajectories.pm_count,
            pm_count + pm_event.to(pm_count.dtype),
        )
        trajectories.maintenance_decision_count.add_(pm_event.long())
        current_health = gather_machine(trajectories.health)
        pm_post = restore_health(current_health, pm_rho, residual, failure_level)
        scatter_machine(
            trajectories.health,
            torch.where(pm_event, pm_post, current_health),
        )
        trajectories.current_makespan.copy_(
            torch.where(
                pm_event,
                torch.maximum(trajectories.current_makespan, pm_ready),
                trajectories.current_makespan,
            )
        )
        pm_horizon = torch.maximum(
            trajectories.current_makespan,
            trajectories.machine_ready_time.amax(dim=2),
        )
        accrue_failed_waiting(pm_event, pm_horizon)

        # Production follows any required CM recourse.
        operation = trajectories.candidate.gather(2, job_index).squeeze(2)
        batch_index = torch.arange(batch, device=device)[:, None].expand(-1, scenarios)
        nominal = self.nominal_processing_times[batch_index, operation, machine]
        compatible = self.compatible[batch_index, operation, machine]
        invalid_production = production & (
            trajectories.job_finished.gather(2, job_index).squeeze(2)
            | ~compatible
        )
        if bool(invalid_production.any()):
            first = invalid_production.nonzero(as_tuple=False)[0]
            row = int(first[0])
            scenario = int(first[1])
            raise ValueError(
                "invalid selected production action in trajectory batch: "
                f"batch={row} scenario={scenario} "
                f"action={int(actions[row])} job={int(job[row, scenario])} "
                f"operation={int(operation[row, scenario])} "
                f"candidate={int(trajectories.candidate[row, scenario, job[row, scenario]])} "
                f"job_finished={bool(trajectories.job_finished[row, scenario, job[row, scenario]])} "
                f"compatible={bool(compatible[row, scenario])} "
                f"production_count={int(trajectories.production_count[row, scenario])} "
                f"decision_count={int(trajectories.decision_count[row, scenario])} "
                f"terminated={bool(trajectories.terminated[row, scenario])} "
                f"truncated={bool(trajectories.truncated[row, scenario])}"
            )
        if self.config.exogenous_processing_noise:
            if noise_bank.namespace == "observed":
                processing_noise = noise_bank.processing_noise_selected(
                    episode_id=self.episode_id,
                    scenario_ids=scenario_ids,
                    operation_ids=operation,
                    machine_ids=machine,
                    cov=selected_parameter(self.overlay.processing_cov),
                    distribution=self.config.processing_distribution,
                    active=production,
                    device=device,
                    dtype=dtype,
                )
            else:
                processing_noise = noise_bank.processing_noise_grid(
                    episode_id=self.episode_id,
                    scenario_count=scenarios,
                    operation_count=operations_total,
                    machine_count=machines,
                    cov=self.overlay.processing_cov,
                    compatible=self.compatible,
                    distribution=self.config.processing_distribution,
                    device=device,
                    dtype=dtype,
                )[batch_index, scenario_ids, operation, machine]
        else:
            processing_noise = torch.ones_like(nominal)
        current_health = gather_machine(trajectories.health)
        if self.config.health_dependent_processing_time:
            duration = effective_processing_time(
                nominal,
                current_health,
                failure_level,
                selected_parameter(self.overlay.eta),
                selected_parameter(self.overlay.health_time_gamma),
                processing_noise,
            )
        else:
            duration = nominal * processing_noise
        duration = torch.where(production, duration, torch.zeros_like(duration))
        if self.config.action_conditioned_degradation:
            load = self.overlay.loads[batch_index, operation, machine]
            alpha = selected_parameter(self.overlay.alpha)
            sensitivity = selected_parameter(self.overlay.load_sensitivity)
            theta = selected_parameter(self.overlay.theta)
            concentration = (
                alpha * load.clamp_min(0).pow(sensitivity) * duration.clamp_min(0)
            )
            sampled_delta = noise_bank.degradation_noise_selected(
                episode_id=self.episode_id,
                scenario_ids=scenario_ids,
                operation_ids=operation,
                machine_ids=machine,
                concentration=concentration,
                scale=theta,
                active=production,
                operation_count=operations_total,
                machine_count=machines,
                compatible=self.compatible,
            )
        else:
            sampled_delta = torch.zeros_like(duration)
        selected_job_ready = trajectories.job_ready_time.gather(
            2, job_index
        ).squeeze(2)
        selected_ready = gather_machine(trajectories.machine_ready_time)
        completion = torch.maximum(selected_job_ready, selected_ready) + duration
        trajectories.job_ready_time.scatter_(
            2,
            job_index,
            torch.where(production, completion, selected_job_ready)[:, :, None],
        )
        scatter_machine(
            trajectories.machine_ready_time,
            torch.where(production, completion, selected_ready),
        )
        op_index = operation[:, :, None]
        selected_completion = trajectories.operation_completion_time.gather(
            2, op_index
        ).squeeze(2)
        trajectories.operation_completion_time.scatter_(
            2,
            op_index,
            torch.where(production, completion, selected_completion)[:, :, None],
        )
        selected_productive = gather_machine(trajectories.productive_time)
        scatter_machine(
            trajectories.productive_time,
            selected_productive + duration,
        )
        post_health = current_health + sampled_delta
        scatter_machine(
            trajectories.health,
            torch.where(production, post_health, current_health),
        )
        selected_scheduled = trajectories.operation_scheduled.gather(
            2, op_index
        ).squeeze(2)
        trajectories.operation_scheduled.scatter_(
            2, op_index, (selected_scheduled | production)[:, :, None]
        )
        trajectories.production_count.add_(production.long())
        last_operation = self.job_last_op.gather(
            1, decoded.job.clamp_min(0)[:, None]
        ).expand(-1, scenarios)
        finishes_job = production & (operation == last_operation)
        selected_finished = trajectories.job_finished.gather(
            2, job_index
        ).squeeze(2)
        trajectories.job_finished.scatter_(
            2, job_index, (selected_finished | finishes_job)[:, :, None]
        )
        selected_candidate = trajectories.candidate.gather(
            2, job_index
        ).squeeze(2)
        trajectories.candidate.scatter_(
            2,
            job_index,
            (
                selected_candidate
                + (production & ~finishes_job).to(selected_candidate.dtype)
            )[:, :, None],
        )
        crossing = production & (post_health >= failure_level)
        selected_status = gather_machine(trajectories.machine_status)
        scatter_machine(
            trajectories.machine_status, selected_status | crossing
        )
        selected_failures = gather_machine(trajectories.failure_count)
        scatter_machine(
            trajectories.failure_count,
            selected_failures + crossing.to(selected_failures.dtype),
        )
        delay = torch.as_tensor(
            self.config.failure_diagnosis_delay, device=device, dtype=dtype
        )
        selected_diagnosis = gather_machine(trajectories.diagnosis_time)
        scatter_machine(
            trajectories.diagnosis_time,
            selected_diagnosis + crossing.to(dtype) * delay,
        )
        selected_ready = gather_machine(trajectories.machine_ready_time)
        selected_ready = selected_ready + crossing.to(dtype) * delay
        scatter_machine(trajectories.machine_ready_time, selected_ready)
        selected_failed_since = gather_machine(trajectories.failed_since_time)
        selected_accounted = gather_machine(
            trajectories.failed_wait_accounted_until
        )
        scatter_machine(
            trajectories.failed_since_time,
            torch.where(crossing, selected_ready, selected_failed_since),
        )
        scatter_machine(
            trajectories.failed_wait_accounted_until,
            torch.where(crossing, selected_ready, selected_accounted),
        )
        trajectories.current_makespan.copy_(
            torch.where(
                production,
                torch.maximum(trajectories.current_makespan, selected_ready),
                trajectories.current_makespan,
            )
        )
        production_horizon = torch.maximum(
            trajectories.current_makespan,
            trajectories.machine_ready_time.amax(dim=2),
        )
        accrue_failed_waiting(production, production_horizon)

        trajectories.maintenance_cost.copy_(
            trajectories.pm_cost + trajectories.cm_cost
        )
        trajectories.unplanned_downtime.copy_(
            trajectories.diagnosis_time
            + trajectories.cm_time
            + trajectories.failed_waiting_time
        )
        trajectories.decision_count.add_(active.long())
        trajectories.terminated.logical_or_(
            active & trajectories.job_finished.all(dim=2)
        )
        max_decisions = (
            operations_total + self.config.max_maintenance_decisions
            if self.config.max_decisions is None
            else self.config.max_decisions
        )
        trajectories.truncated.logical_or_(
            active
            & ~trajectories.terminated
            & (trajectories.decision_count >= max_decisions)
        )
        active_cpu = active.detach().cpu().tolist()
        action_cpu = actions.detach().cpu().tolist()
        for row in range(batch):
            for scenario in range(scenarios):
                if active_cpu[row][scenario]:
                    trajectories.action_history[row][scenario].append(action_cpu[row])

        final_health = gather_machine(trajectories.health)
        degradation_increment = torch.where(
            production, sampled_delta, final_health - health_before
        )
        return ForecastScenarioBatch(
            action=actions.clone(),
            root_health=root_health,
            root_machine_status=root_status,
            post_health=trajectories.health.clone(),
            post_machine_status=trajectories.machine_status.clone(),
            post_job_ready_time=trajectories.job_ready_time.clone(),
            post_machine_ready_time=trajectories.machine_ready_time.clone(),
            duration=gather_machine(trajectories.machine_ready_time) - ready_before,
            degradation_increment=degradation_increment,
            maintenance_cost_increment=trajectories.maintenance_cost - cost_before,
            failure_crossing=(
                gather_machine(trajectories.failure_count) > failure_before
            ),
        )

    def apply_primary_action_with_recourse(
        self,
        state: ObservedShopState,
        *,
        batch_index: int,
        action: int,
        action_codec: ActionCodec,
        noise_bank: TrajectoryNoiseBank | None = None,
        scenario_id: int = -1,
    ) -> PrimaryActionTransition:
        """Apply the common primary action with the fixed feasibility recourse.

        The rule is shared by observed/state/reward environment transitions
        and both bounded oracles:

        * failed + production -> CM, then production;
        * failed + PM -> CM only;
        * healthy + CM -> deterministic overhaul (needed when another scenario
          makes the same primary CM action feasible).

        Recourse consumes real maintenance time/cost/counters.  Only the
        primary action is appended to ``action_history`` and increments
        ``decision_count``.
        """

        batch = int(batch_index)
        if bool(state.terminated[batch] or state.truncated[batch]):
            raise ValueError("cannot transition an inactive batch row")
        action_value = int(action)
        decoded = action_codec.decode(
            torch.tensor([action_value], device=state.candidate.device)
        )
        action_type = ActionType(int(decoded.action_type[0]))
        machine = int(decoded.machine[0])
        job = int(decoded.job[0])
        bank = self.observed_noise_bank if noise_bank is None else noise_bank
        if not self.config.scenario_recourse:
            failed = bool(state.observed_machine_status[batch, machine])
            incompatible = (
                (action_type in {ActionType.PRODUCTION, ActionType.PREVENTIVE_MAINTENANCE} and failed)
                or (action_type == ActionType.CORRECTIVE_MAINTENANCE and not failed)
            )
            if incompatible:
                raise ValueError(
                    "primary action is outside the no-recourse intersection-feasible set"
                )

        ready_before = state.observed_machine_ready_time[batch, machine].clone()
        health_before = state.observed_health[batch, machine].clone()
        cost_before = state.maintenance_cost[batch].clone()
        cm_time_before = state.corrective_maintenance_time[batch, machine].clone()
        failure_before = state.failure_count[batch, machine].clone()
        recourse_type: ActionType | None = None
        primary_duration = torch.zeros_like(ready_before)
        degradation = torch.zeros_like(ready_before)

        if action_type == ActionType.PRODUCTION:
            if bool(state.observed_machine_status[batch, machine]):
                self.apply_corrective_maintenance(
                    state,
                    batch_index=batch,
                    machine=machine,
                    noise_bank=bank,
                    scenario_id=scenario_id,
                )
                recourse_type = ActionType.CORRECTIVE_MAINTENANCE
            production = self.apply_production(
                state,
                batch_index=batch,
                job=job,
                machine=machine,
                noise_bank=bank,
                scenario_id=scenario_id,
            )
            primary_duration = production.duration
            degradation = production.sampled_delta
        elif action_type == ActionType.PREVENTIVE_MAINTENANCE:
            if bool(state.observed_machine_status[batch, machine]):
                self.apply_corrective_maintenance(
                    state,
                    batch_index=batch,
                    machine=machine,
                    noise_bank=bank,
                    scenario_id=scenario_id,
                )
                recourse_type = ActionType.CORRECTIVE_MAINTENANCE
            else:
                ready = state.observed_machine_ready_time[batch, machine].clone()
                self.apply_preventive_maintenance(
                    state,
                    batch_index=batch,
                    machine=machine,
                    noise_bank=bank,
                    scenario_id=scenario_id,
                )
                primary_duration = (
                    state.observed_machine_ready_time[batch, machine] - ready
                )
            degradation = state.observed_health[batch, machine] - health_before
        else:
            ready = state.observed_machine_ready_time[batch, machine].clone()
            self.apply_corrective_maintenance(
                state,
                batch_index=batch,
                machine=machine,
                noise_bank=bank,
                scenario_id=scenario_id,
                allow_healthy_overhaul=True,
            )
            primary_duration = state.observed_machine_ready_time[batch, machine] - ready
            degradation = state.observed_health[batch, machine] - health_before

        state.action_history[batch].append(action_value)
        state.decision_count[batch] += 1
        state.terminated[batch] |= (
            state.job_finished[batch].all()
        )
        max_decisions = (
            self.nominal_processing_times.shape[1]
            + self.config.max_maintenance_decisions
            if self.config.max_decisions is None
            else self.config.max_decisions
        )
        state.truncated[batch] |= bool(
            not state.terminated[batch]
            and state.decision_count[batch] >= max_decisions
        )
        recourse_duration = (
            state.corrective_maintenance_time[batch, machine] - cm_time_before
            if recourse_type is not None
            else torch.zeros_like(primary_duration)
        )
        return PrimaryActionTransition(
            action=action_value,
            action_type=action_type,
            machine=machine,
            job=job,
            recourse_action_type=recourse_type,
            primary_duration=primary_duration,
            recourse_duration=recourse_duration,
            total_elapsed=(
                state.observed_machine_ready_time[batch, machine] - ready_before
            ),
            degradation_increment=degradation,
            maintenance_cost_increment=state.maintenance_cost[batch] - cost_before,
            failure_crossing=state.failure_count[batch, machine] > failure_before,
            survival=~state.observed_machine_status[batch, machine],
        )

    @staticmethod
    def accrue_failed_waiting(
        state: ObservedShopState,
        *,
        batch_index: int | None = None,
        horizon: torch.Tensor | float | None = None,
    ) -> None:
        """Accrue open post-diagnosis failure intervals exactly once.

        A failed machine is unavailable from the end of its diagnosis delay
        until CM starts or the current calendar horizon is reached.  The
        explicit cursor makes repeated metric calls idempotent and prevents
        this interval from being mislabeled as available idle time.
        """

        if horizon is None:
            targets = torch.maximum(
                state.current_makespan,
                state.observed_machine_ready_time.amax(dim=1),
            )
        else:
            targets = torch.as_tensor(
                horizon,
                device=state.current_makespan.device,
                dtype=state.current_makespan.dtype,
            )
            if targets.ndim == 0:
                targets = targets.expand_as(state.current_makespan)
        selected_rows = torch.ones(
            state.current_makespan.shape[0],
            dtype=torch.bool,
            device=state.current_makespan.device,
        )
        if batch_index is not None:
            selected_rows = (
                torch.arange(
                    state.current_makespan.shape[0],
                    device=state.current_makespan.device,
                )
                == int(batch_index)
            )
        failed = (
            state.observed_machine_status
            & (state.failed_wait_accounted_until >= 0)
            & selected_rows[:, None]
        )
        delta = (
            targets[:, None] - state.failed_wait_accounted_until
        ).clamp_min(0)
        # Functional replacement is valid for inference-created tensors both
        # inside and outside torch.inference_mode.
        state.failed_waiting_time = torch.where(
            failed,
            state.failed_waiting_time + delta,
            state.failed_waiting_time,
        )
        state.failed_wait_accounted_until = torch.where(
            failed,
            targets[:, None].expand_as(state.failed_wait_accounted_until),
            state.failed_wait_accounted_until,
        )
        affected_rows = failed.any(dim=1)
        updated_unplanned = (
            state.diagnosis_delay_time
            + state.corrective_maintenance_time
            + state.failed_waiting_time
        )
        state.unplanned_downtime = torch.where(
            affected_rows[:, None],
            updated_unplanned,
            state.unplanned_downtime,
        )

    def _duration(
        self,
        *,
        batch: int,
        operation: int,
        machine: int,
        health: torch.Tensor,
        noise_bank: TrajectoryNoiseBank,
        scenario_id: int,
    ) -> torch.Tensor:
        nominal = self.nominal_processing_times[batch, operation, machine]
        if nominal <= 0:
            raise ValueError("incompatible operation-machine pair")
        processing_noise = (
            noise_bank.processing_noise(
                episode_id=self.episode_id,
                scenario_id=scenario_id,
                operation_id=operation,
                machine_id=machine,
                distribution=self.config.processing_distribution,
                cov=float(self.overlay.processing_cov[batch, machine]),
                device=nominal.device,
                dtype=nominal.dtype,
            )
            if self.config.exogenous_processing_noise
            else torch.ones_like(nominal)
        )
        if self.config.health_dependent_processing_time:
            return effective_processing_time(
                nominal,
                health,
                self.overlay.failure_level[batch, machine],
                self.overlay.eta[batch, machine],
                self.overlay.health_time_gamma[batch, machine],
                processing_noise,
            )
        return nominal * processing_noise

    def prospective_production(
        self,
        state: ObservedShopState,
        *,
        batch_index: int,
        scenario_id: int,
        operation: int,
        machine: int,
        noise_bank: TrajectoryNoiseBank,
    ) -> ProspectiveProduction:
        """Evaluate one future pair from the common observed current health."""

        batch = int(batch_index)
        health = state.observed_health[batch, machine]
        duration = self._duration(
            batch=batch,
            operation=operation,
            machine=machine,
            health=health,
            noise_bank=noise_bank,
            scenario_id=scenario_id,
        )
        if self.config.action_conditioned_degradation:
            alpha = self.overlay.alpha[batch, machine]
            load = self.overlay.loads[batch, operation, machine]
            sensitivity = self.overlay.load_sensitivity[batch, machine]
            theta = self.overlay.theta[batch, machine]
            expected = expected_degradation_increment(
                alpha, load, sensitivity, duration, theta
            )
            concentration = alpha * load.clamp_min(0).pow(sensitivity) * duration.clamp_min(0)
            sampled = noise_bank.degradation_noise(
                episode_id=self.episode_id,
                scenario_id=scenario_id,
                operation_id=operation,
                machine_id=machine,
                concentration=concentration,
                scale=theta,
                device=duration.device,
                dtype=duration.dtype,
            )
        else:
            expected = torch.zeros_like(duration)
            sampled = torch.zeros_like(duration)
        survival = health + sampled < self.overlay.failure_level[batch, machine]
        return ProspectiveProduction(duration, expected, sampled, survival)

    def apply_production(
        self,
        state: ObservedShopState,
        *,
        batch_index: int,
        job: int,
        machine: int,
        noise_bank: TrajectoryNoiseBank | None = None,
        scenario_id: int = -1,
    ) -> ProspectiveProduction:
        """Execute a full operation then inspect for threshold crossing.

        There is no mid-operation interruption. If the completed operation
        crosses the failure level, its completion remains recorded and the
        diagnosis delay immediately advances the machine clock.
        """

        batch = int(batch_index)
        if state.terminated[batch] or state.truncated[batch]:
            raise ValueError("cannot transition an inactive batch row")
        if state.observed_machine_status[batch, machine]:
            raise ValueError("production cannot use an observed failed machine")
        if state.job_finished[batch, job]:
            raise ValueError("production cannot use a finished job")
        operation = int(state.candidate[batch, job])
        if not self.compatible[batch, operation, machine]:
            raise ValueError("incompatible operation-machine pair")
        transition = self.prospective_production(
            state,
            batch_index=batch,
            scenario_id=scenario_id,
            operation=operation,
            machine=machine,
            noise_bank=self.observed_noise_bank if noise_bank is None else noise_bank,
        )
        start = torch.maximum(
            state.observed_job_ready_time[batch, job],
            state.observed_machine_ready_time[batch, machine],
        )
        completion = start + transition.duration
        state.observed_job_ready_time[batch, job] = completion
        state.observed_machine_ready_time[batch, machine] = completion
        state.op_completion_time[batch, operation] = completion
        state.productive_processing_time[batch, machine] += transition.duration
        state.observed_health[batch, machine] += transition.sampled_delta
        state.op_scheduled[batch, operation] = True
        state.production_count[batch] += 1
        if operation == int(self.job_last_op[batch, job]):
            state.job_finished[batch, job] = True
        else:
            state.candidate[batch, job] += 1

        failure_level = self.overlay.failure_level[batch, machine]
        if state.observed_health[batch, machine] >= failure_level:
            state.observed_machine_status[batch, machine] = True
            state.failure_count[batch, machine] += 1
            delay = torch.as_tensor(
                self.config.failure_diagnosis_delay,
                device=completion.device,
                dtype=completion.dtype,
            )
            state.diagnosis_delay_time[batch, machine] += delay
            state.observed_machine_ready_time[batch, machine] += delay
            state.failed_since_time[batch, machine] = state.observed_machine_ready_time[
                batch, machine
            ]
            state.failed_wait_accounted_until[batch, machine] = (
                state.observed_machine_ready_time[batch, machine]
            )
        state.unplanned_downtime[batch] = (
            state.diagnosis_delay_time[batch]
            + state.corrective_maintenance_time[batch]
            + state.failed_waiting_time[batch]
        )
        state.current_makespan[batch] = torch.maximum(
            state.current_makespan[batch],
            state.observed_machine_ready_time[batch, machine],
        )
        self.accrue_failed_waiting(state)
        return transition

    def _has_remaining_compatible_operation(
        self, state: ObservedShopState, batch: int, machine: int
    ) -> bool:
        remaining = ~state.op_scheduled[batch]
        return bool((remaining & self.compatible[batch, :, machine]).any())

    def apply_preventive_maintenance(
        self,
        state: ObservedShopState,
        *,
        batch_index: int,
        machine: int,
        noise_bank: TrajectoryNoiseBank | None = None,
        scenario_id: int = -1,
    ) -> None:
        """Apply imperfect PM to an observed available, still-useful machine."""

        batch = int(batch_index)
        if state.observed_machine_status[batch, machine]:
            raise ValueError("PM cannot act on an observed failed machine")
        if not self._has_remaining_compatible_operation(state, batch, machine):
            raise ValueError("PM requires a remaining compatible operation")
        count = int(state.pm_count[batch, machine])
        bank = self.observed_noise_bank if noise_bank is None else noise_bank
        residual = bank.maintenance_noise(
            episode_id=self.episode_id,
            scenario_id=scenario_id,
            machine_id=machine,
            maintenance_count=count,
            maintenance_type="PM",
            std=float(self.overlay.maintenance_noise_std[batch, machine]),
            device=state.observed_health.device,
            dtype=state.observed_health.dtype,
        )
        duration = self.overlay.pm_duration[batch, machine]
        state.observed_machine_ready_time[batch, machine] += duration
        state.preventive_maintenance_time[batch, machine] += duration
        state.pm_cost_total[batch] += self.overlay.pm_cost[batch, machine]
        state.pm_count[batch, machine] += 1
        state.maintenance_decision_count[batch] += 1
        state.observed_health[batch, machine] = restore_health(
            state.observed_health[batch, machine],
            self.overlay.pm_rho[batch, machine],
            residual,
            self.overlay.failure_level[batch, machine],
        )
        state.maintenance_cost[batch] = (
            state.pm_cost_total[batch] + state.cm_cost_total[batch]
        )
        state.current_makespan[batch] = torch.maximum(
            state.current_makespan[batch],
            state.observed_machine_ready_time[batch, machine],
        )
        self.accrue_failed_waiting(state)

    def apply_corrective_maintenance(
        self,
        state: ObservedShopState,
        *,
        batch_index: int,
        machine: int,
        noise_bank: TrajectoryNoiseBank | None = None,
        scenario_id: int = -1,
        allow_healthy_overhaul: bool = False,
    ) -> None:
        """Restore an observed failed machine and advance the CM timeline."""

        batch = int(batch_index)
        was_failed = bool(state.observed_machine_status[batch, machine])
        if not was_failed and not allow_healthy_overhaul:
            raise ValueError("CM is legal only for an observed failure")
        count = int(state.cm_count[batch, machine])
        bank = self.observed_noise_bank if noise_bank is None else noise_bank
        residual = bank.maintenance_noise(
            episode_id=self.episode_id,
            scenario_id=scenario_id,
            machine_id=machine,
            maintenance_count=count,
            maintenance_type="CM",
            std=float(self.overlay.maintenance_noise_std[batch, machine]),
            device=state.observed_health.device,
            dtype=state.observed_health.dtype,
        )
        ready = state.observed_machine_ready_time[batch, machine]
        start = (
            torch.maximum(ready, state.current_makespan[batch])
            if was_failed
            else ready
        )
        if was_failed:
            self.accrue_failed_waiting(state, batch_index=batch, horizon=start)
        duration = self.overlay.cm_duration[batch, machine]
        state.observed_machine_ready_time[batch, machine] = start + duration
        state.corrective_maintenance_time[batch, machine] += duration
        state.cm_cost_total[batch] += self.overlay.cm_cost[batch, machine]
        state.cm_count[batch, machine] += 1
        state.maintenance_decision_count[batch] += 1
        state.observed_health[batch, machine] = restore_health(
            state.observed_health[batch, machine],
            self.overlay.cm_rho[batch, machine],
            residual,
            self.overlay.failure_level[batch, machine],
        )
        state.observed_machine_status[batch, machine] = False
        state.failed_since_time[batch, machine] = -1.0
        state.failed_wait_accounted_until[batch, machine] = -1.0
        state.maintenance_cost[batch] = (
            state.pm_cost_total[batch] + state.cm_cost_total[batch]
        )
        state.unplanned_downtime[batch] = (
            state.diagnosis_delay_time[batch]
            + state.corrective_maintenance_time[batch]
            + state.failed_waiting_time[batch]
        )
        state.current_makespan[batch] = torch.maximum(
            state.current_makespan[batch],
            state.observed_machine_ready_time[batch, machine],
        )
        self.accrue_failed_waiting(state)
