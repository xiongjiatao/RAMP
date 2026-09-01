"""Observed-state RAMP environment with future trajectory scenarios."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

import numpy as np
import torch

from .config import RAMPConfig
from .noise import TrajectoryNoiseBank
from .overlay import HealthOverlay, build_health_overlay
from .profiling import ThroughputProfiler, profiled, tensor_bytes
from .risk import build_machine_risk, build_pair_risk
from .state import (
    RAMPEnvState,
    ActionCodec,
    ActionType,
    ForecastScenarioBatch,
    ObservedShopState,
    ScenarioTrajectoryState,
)
from .transition import (
    RAMPTransitionKernel,
    expected_degradation_increment,
    health_time_factor,
    restore_health,
)
from .atmsl import weighted_scenario_mean, weighted_upper_tail_cvar


def upper_tail_cvar(
    values: torch.Tensor,
    alpha: float,
    scenario_invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Finite-sample upper-tail CVaR with fractional boundary mass.

    Each empirical scenario has probability ``1/S``.  The worst
    ``(1-alpha)*S`` samples are integrated exactly: complete samples receive
    unit mass and the quantile-boundary sample receives the remaining
    fractional mass.  When the tail contains less than one sample this equals
    the empirical maximum.
    """

    if values.ndim != 2:
        raise ValueError("CVaR values must be [B,S]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must be in [0,1)")
    if values.shape[1] < 1:
        raise ValueError("CVaR requires at least one scenario")
    invalid = (
        torch.zeros_like(values, dtype=torch.bool)
        if scenario_invalid_mask is None
        else scenario_invalid_mask
    )
    if invalid.shape != values.shape or invalid.dtype != torch.bool:
        raise ValueError("scenario_invalid_mask must be boolean [B,S]")
    results: list[torch.Tensor] = []
    for row in range(values.shape[0]):
        valid_values = values[row, ~invalid[row]]
        scenarios = int(valid_values.numel())
        if scenarios == 0:
            raise ValueError("CVaR requires one valid scenario per batch row")
        tail_mass = (1.0 - float(alpha)) * scenarios
        full = int(math.floor(tail_mass))
        fraction = tail_mass - full
        ordered = valid_values.sort(descending=True).values
        numerator = ordered[:full].sum() if full else ordered.new_zeros(())
        if fraction > 1e-12 and full < scenarios:
            numerator = numerator + fraction * ordered[full]
        results.append(numerator / tail_mass)
    return torch.stack(results)


def empirical_scenario_mean(
    values: torch.Tensor,
    scenario_invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over the same per-row valid empirical scenario support as CVaR."""

    if values.ndim != 2:
        raise ValueError("scenario values must be [B,S]")
    invalid = (
        torch.zeros_like(values, dtype=torch.bool)
        if scenario_invalid_mask is None
        else scenario_invalid_mask
    )
    if invalid.shape != values.shape or invalid.dtype != torch.bool:
        raise ValueError("scenario_invalid_mask must be boolean [B,S]")
    valid = ~invalid
    if (~valid.any(dim=1)).any():
        raise ValueError("mean requires one valid scenario per batch row")
    return values.masked_fill(invalid, 0.0).sum(dim=1) / valid.sum(dim=1)


class RAMPEnvironmentCore:
    """Shared environment core; all physical mutation delegates to one kernel."""

    def __init__(
        self,
        job_lengths: torch.Tensor | np.ndarray,
        nominal_processing_times: torch.Tensor | np.ndarray,
        *,
        overlay: HealthOverlay | None,
        config: RAMPConfig,
        reward_num_scenarios: int | None,
        reward_seed: int | None,
        device: torch.device | str,
        profiler: ThroughputProfiler | None = None,
    ):
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.profiler = profiler or ThroughputProfiler(device=self.device)
        self.job_lengths = torch.as_tensor(job_lengths, dtype=torch.long, device=self.device)
        self.nominal_processing_times = torch.as_tensor(
            nominal_processing_times, dtype=torch.float32, device=self.device
        )
        if self.job_lengths.ndim == 1:
            self.job_lengths = self.job_lengths.unsqueeze(0)
        if self.nominal_processing_times.ndim == 2:
            self.nominal_processing_times = self.nominal_processing_times.unsqueeze(0)
        if self.job_lengths.ndim != 2 or self.nominal_processing_times.ndim != 3:
            raise ValueError("job lengths must be [B,J] and processing times [B,N,M]")
        self.batch_size, self.number_of_ops, self.number_of_machines = (
            self.nominal_processing_times.shape
        )
        self.number_of_jobs = int(self.job_lengths.shape[1])
        if self.job_lengths.shape[0] != self.batch_size:
            raise ValueError("job lengths and processing times have different batch sizes")
        if torch.any(self.job_lengths.sum(dim=1) != self.number_of_ops):
            raise ValueError("each row of job lengths must sum to N")
        if torch.any(self.nominal_processing_times < 0):
            raise ValueError("nominal processing times must be nonnegative")
        if torch.any((self.nominal_processing_times > 0).sum(dim=2) == 0):
            raise ValueError("every operation needs a compatible machine")

        self.overlay = (
            overlay.to(self.device).clone()
            if overlay is not None
            else build_health_overlay(
                self.nominal_processing_times,
                config.num_scenarios,
                seed=config.scenario_seed,
                device=self.device,
            )
        )
        self.overlay.alpha.mul_(
            config.degradation_rate_multiplier * config.gamma_shape_multiplier
        )
        self.overlay.theta.mul_(config.gamma_scale_multiplier)
        self.overlay.initial_health.mul_(config.initial_health_multiplier)
        self.overlay.initial_health.copy_(
            torch.minimum(
                self.overlay.initial_health,
                self.overlay.failure_level - 1e-4,
            )
        )
        self.overlay.cm_cost.mul_(config.cm_cost_ratio_multiplier)
        self.overlay.validate(self.nominal_processing_times)
        if self.overlay.batch_size != self.batch_size:
            raise ValueError("overlay batch size does not match nominal instances")
        if self.overlay.num_scenarios != config.num_scenarios:
            raise ValueError("overlay and state scenario counts differ")
        self.num_scenarios = int(config.num_scenarios)
        self.reward_num_scenarios = int(
            reward_num_scenarios or config.num_scenarios
        )
        if self.reward_num_scenarios < 1:
            raise ValueError("reward_num_scenarios must be positive")
        self.reward_scenario_weights = torch.full(
            (self.batch_size, self.reward_num_scenarios),
            1.0 / self.reward_num_scenarios,
            dtype=self.nominal_processing_times.dtype,
            device=self.device,
        )
        self.reward_weighted_cvar_enabled = True
        reward_seed = (
            config.scenario_seed + 1_000_000
            if reward_seed is None
            else int(reward_seed)
        )
        observed_seed = (
            config.scenario_seed - 1
            if config.observed_seed is None
            else int(config.observed_seed)
        )
        self.observed_noise_bank = TrajectoryNoiseBank(
            seed=observed_seed,
            namespace="observed",
            seed_source=self.overlay.scenario_seeds[:, :1],
        )
        self.state_noise_bank = TrajectoryNoiseBank(
            seed=config.scenario_seed,
            namespace="state",
            seed_source=self.overlay.scenario_seeds,
        )
        reward_seed_source = self.overlay.scenario_seeds[:,
            torch.arange(self.reward_num_scenarios, device=self.device)
            % self.overlay.scenario_seeds.shape[1]
        ]
        self.reward_noise_bank = TrajectoryNoiseBank(
            seed=reward_seed,
            namespace="reward",
            seed_source=reward_seed_source,
        )
        self.compatible = self.nominal_processing_times > 0
        self.codec = ActionCodec(self.number_of_jobs, self.number_of_machines)
        self._build_job_index()
        self.objective_scales = self._build_objective_scales()
        self.time_scale = self.objective_scales[:, 0]
        self.reset_count = -1
        self.episode_id = -1
        self.transition_kernel = RAMPTransitionKernel(
            job_first_op=self.job_first_op,
            job_last_op=self.job_last_op,
            nominal_processing_times=self.nominal_processing_times,
            overlay=self.overlay,
            config=self.config,
            observed_noise_bank=self.observed_noise_bank,
            episode_id=0,
        )
        self._scenario_cpu_kernel = RAMPTransitionKernel(
            job_first_op=self.job_first_op.cpu(),
            job_last_op=self.job_last_op.cpu(),
            nominal_processing_times=self.nominal_processing_times.cpu(),
            overlay=self.overlay.to("cpu"),
            config=self.config,
            observed_noise_bank=self.state_noise_bank,
            episode_id=0,
        )
        self.reset()

    def _build_objective_scales(self) -> torch.Tensor:
        """Return fixed per-instance scales for a dimensionless total cost.

        The time scale is the maximum of the workload and longest-job lower
        bounds under nominal compatible processing times.  One complete
        machine-wise PM/CM cycle defines the corresponding cost scales;
        downtime uses the same calendar scale and failures use machine count.
        These scales are fixed at environment construction and therefore do
        not leak trajectory outcomes into the objective.
        """

        minimum = self.nominal_processing_times.masked_fill(
            ~self.compatible, float("inf")
        ).amin(dim=2)
        workload_bound = minimum.sum(dim=1) / max(self.number_of_machines, 1)
        job_bound = torch.zeros(self.batch_size, device=self.device)
        for batch in range(self.batch_size):
            for job in range(self.number_of_jobs):
                first = int(self.job_first_op[batch, job])
                last = int(self.job_last_op[batch, job]) + 1
                job_bound[batch] = torch.maximum(
                    job_bound[batch], minimum[batch, first:last].sum()
                )
        time_scale = torch.maximum(workload_bound, job_bound).clamp_min(1.0)
        pm_scale = self.overlay.pm_cost.sum(dim=1).clamp_min(1.0)
        cm_scale = self.overlay.cm_cost.sum(dim=1).clamp_min(1.0)
        failure_scale = torch.full_like(time_scale, float(self.number_of_machines))
        return torch.stack(
            (time_scale, pm_scale, cm_scale, time_scale, failure_scale), dim=1
        )

    def _build_job_index(self) -> None:
        first = torch.cat(
            (
                torch.zeros(
                    (self.batch_size, 1), dtype=torch.long, device=self.device
                ),
                self.job_lengths.cumsum(dim=1)[:, :-1],
            ),
            dim=1,
        )
        self.job_first_op = first
        self.job_last_op = first + self.job_lengths - 1
        self.job_of_op = torch.empty(
            (self.batch_size, self.number_of_ops),
            dtype=torch.long,
            device=self.device,
        )
        self.position_in_job = torch.empty_like(self.job_of_op)
        for batch in range(self.batch_size):
            for job in range(self.number_of_jobs):
                start = int(self.job_first_op[batch, job])
                end = int(self.job_last_op[batch, job]) + 1
                self.job_of_op[batch, start:end] = job
                self.position_in_job[batch, start:end] = torch.arange(
                    end - start, device=self.device
                )
        self.op_mask = torch.zeros(
            (self.batch_size, self.number_of_ops, 3),
            dtype=torch.bool,
            device=self.device,
        )
        batch_index = torch.arange(self.batch_size, device=self.device)[:, None]
        self.op_mask[batch_index, self.job_first_op, 0] = True
        self.op_mask[batch_index, self.job_last_op, 2] = True

    def _new_observed_state(self) -> ObservedShopState:
        batch, jobs, ops, machines = (
            self.batch_size,
            self.number_of_jobs,
            self.number_of_ops,
            self.number_of_machines,
        )
        zeros_bm = torch.zeros((batch, machines), device=self.device)
        zeros_b = torch.zeros(batch, device=self.device)
        return ObservedShopState(
            candidate=self.job_first_op.clone(),
            job_finished=torch.zeros((batch, jobs), dtype=torch.bool, device=self.device),
            op_scheduled=torch.zeros((batch, ops), dtype=torch.bool, device=self.device),
            observed_health=self.overlay.initial_health.clone(),
            observed_machine_status=torch.zeros(
                (batch, machines), dtype=torch.bool, device=self.device
            ),
            observed_job_ready_time=torch.zeros((batch, jobs), device=self.device),
            observed_machine_ready_time=zeros_bm.clone(),
            op_completion_time=torch.zeros((batch, ops), device=self.device),
            pm_count=torch.zeros((batch, machines), dtype=torch.long, device=self.device),
            cm_count=torch.zeros((batch, machines), dtype=torch.long, device=self.device),
            maintenance_decision_count=torch.zeros(
                batch, dtype=torch.long, device=self.device
            ),
            production_count=torch.zeros(batch, dtype=torch.long, device=self.device),
            pm_cost_total=zeros_b.clone(),
            cm_cost_total=zeros_b.clone(),
            maintenance_cost=zeros_b.clone(),
            failure_count=zeros_bm.clone(),
            productive_processing_time=zeros_bm.clone(),
            available_idle_time=zeros_bm.clone(),
            preventive_maintenance_time=zeros_bm.clone(),
            corrective_maintenance_time=zeros_bm.clone(),
            diagnosis_delay_time=zeros_bm.clone(),
            failed_waiting_time=zeros_bm.clone(),
            unplanned_downtime=zeros_bm.clone(),
            failed_since_time=torch.full((batch, machines), -1.0, device=self.device),
            failed_wait_accounted_until=torch.full(
                (batch, machines), -1.0, device=self.device
            ),
            current_makespan=zeros_b.clone(),
            decision_count=torch.zeros(batch, dtype=torch.long, device=self.device),
            terminated=torch.zeros(batch, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(batch, dtype=torch.bool, device=self.device),
            action_history=[[] for _ in range(batch)],
        )

    def reset(self) -> RAMPEnvState:
        """Start observed, state, and reward trajectories from one initial state."""

        self.reset_count += 1
        self.episode_id = self.reset_count
        self.transition_kernel.set_episode(self.episode_id)
        self._scenario_cpu_kernel.set_episode(self.episode_id)
        self.observed_state = self._new_observed_state()
        self.state_scenarios = ScenarioTrajectoryState.from_observed(
            self.observed_state,
            self.num_scenarios,
            episode_id=self.episode_id,
            noise_namespace="state",
        )
        self.reward_scenarios = ScenarioTrajectoryState.from_observed(
            self.observed_state,
            self.reward_num_scenarios,
            episode_id=self.episode_id,
            noise_namespace="reward",
        )
        if self.config.exogenous_processing_noise:
            self.reward_noise_bank.processing_noise_grid(
                episode_id=self.episode_id,
                scenario_count=self.reward_num_scenarios,
                operation_count=self.number_of_ops,
                machine_count=self.number_of_machines,
                cov=self.overlay.processing_cov,
                compatible=self.compatible,
                distribution=self.config.processing_distribution,
                device=self.device,
                dtype=self.nominal_processing_times.dtype,
            )
        self.state_noise_bank.prewarm_degradation_uniforms(
            episode_id=self.episode_id,
            scenario_count=self.num_scenarios,
            operation_count=self.number_of_ops,
            machine_count=self.number_of_machines,
            compatible=self.compatible,
        )
        self.reward_noise_bank.prewarm_degradation_uniforms(
            episode_id=self.episode_id,
            scenario_count=self.reward_num_scenarios,
            operation_count=self.number_of_ops,
            machine_count=self.number_of_machines,
            compatible=self.compatible,
        )
        self._update_scenario_costs(self.state_scenarios)
        self._update_scenario_costs(self.reward_scenarios)
        self.refresh_forecasts()
        return self.state

    @property
    def active(self) -> torch.Tensor:
        return ~(self.observed_state.terminated | self.observed_state.truncated)

    def _all_operation_forecast_scalar_reference(
        self,
        noise_bank: TrajectoryNoiseBank,
        scenario_count: int,
        trajectories: ScenarioTrajectoryState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, ops, machines = (
            self.batch_size,
            self.number_of_ops,
            self.number_of_machines,
        )
        shape = (batch, scenario_count, ops, machines)
        duration = torch.zeros(shape, device=self.device)
        expected = torch.zeros_like(duration)
        sampled = torch.zeros_like(duration)
        survival = torch.zeros(shape, dtype=torch.bool, device=self.device)
        forecast_root_health = torch.zeros_like(duration)
        bank = self.state_scenarios if trajectories is None else trajectories
        if bank.scenario_count != scenario_count:
            raise ValueError("trajectory and requested scenario counts differ")
        scenario_states = [bank.as_observed(s) for s in range(scenario_count)]
        scenario_invalid = self._trajectory_invalid_mask(bank)
        for b in range(batch):
            for scenario in range(scenario_count):
                if scenario_invalid[b, scenario]:
                    continue
                for operation in range(ops):
                    for machine in range(machines):
                        if not self.compatible[b, operation, machine]:
                            continue
                        candidate_state = scenario_states[scenario]
                        if candidate_state.observed_machine_status[b, machine]:
                            candidate_state = candidate_state.clone()
                            self.transition_kernel.apply_corrective_maintenance(
                                candidate_state,
                                batch_index=b,
                                machine=machine,
                                noise_bank=noise_bank,
                                scenario_id=scenario,
                            )
                        result = self.transition_kernel.prospective_production(
                            candidate_state,
                            batch_index=b,
                            scenario_id=scenario,
                            operation=operation,
                            machine=machine,
                            noise_bank=noise_bank,
                        )
                        duration[b, scenario, operation, machine] = result.duration
                        expected[b, scenario, operation, machine] = result.expected_delta
                        sampled[b, scenario, operation, machine] = result.sampled_delta
                        survival[b, scenario, operation, machine] = result.survival
                        forecast_root_health[b, scenario, operation, machine] = (
                            candidate_state.observed_health[b, machine]
                        )
        return duration, expected, sampled, survival, forecast_root_health

    def _all_operation_forecast_vectorized(
        self,
        noise_bank: TrajectoryNoiseBank,
        scenario_count: int,
        trajectories: ScenarioTrajectoryState | None = None,
        machine_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed forecast with the scalar implementation retained as authority.

        No physical formula changes here.  Host work is batched so candidate
        enumeration does not synchronize one CUDA scalar at a time.
        """

        bank = self.state_scenarios if trajectories is None else trajectories
        if bank.scenario_count != scenario_count:
            raise ValueError("trajectory and requested scenario counts differ")
        batch, operations, machines = (
            self.batch_size,
            self.number_of_ops,
            self.number_of_machines,
        )
        scenario_invalid = self._trajectory_invalid_mask(bank)
        active = (
            self.compatible[:, None, :, :]
            & ~scenario_invalid[:, :, None, None]
        )
        if machine_mask is not None:
            if machine_mask.shape != (batch, machines):
                raise ValueError("machine_mask must have shape [B,M]")
            active &= machine_mask[:, None, None, :]

        root_health = bank.health.clone()
        failed = bank.machine_status & ~scenario_invalid[:, :, None]
        if failed.any():
            # Recourse restoration depends on scenario/machine/count, not on
            # the prospective operation, so evaluate it once per failed root.
            health_cpu = root_health.detach().cpu()
            failed_cpu = failed.detach().cpu()
            count_cpu = bank.cm_count.detach().cpu()
            std_cpu = self.overlay.maintenance_noise_std.detach().cpu()
            rho_cpu = self.overlay.cm_rho.detach().cpu()
            failure_cpu = self.overlay.failure_level.detach().cpu()
            for row, scenario, machine in failed_cpu.nonzero(as_tuple=False).tolist():
                residual = noise_bank.maintenance_noise(
                    episode_id=self.episode_id,
                    scenario_id=scenario,
                    machine_id=machine,
                    maintenance_count=int(count_cpu[row, scenario, machine]),
                    maintenance_type="CM",
                    std=float(std_cpu[row, machine]),
                    device="cpu",
                    dtype=health_cpu.dtype,
                )
                health_cpu[row, scenario, machine] = restore_health(
                    health_cpu[row, scenario, machine],
                    rho_cpu[row, machine],
                    residual,
                    failure_cpu[row, machine],
                )
            root_health = health_cpu.to(self.device)

        nominal = self.nominal_processing_times[:, None]
        processing = (
            noise_bank.processing_noise_grid(
                episode_id=self.episode_id,
                scenario_count=scenario_count,
                operation_count=operations,
                machine_count=machines,
                cov=self.overlay.processing_cov,
                compatible=self.compatible,
                distribution=self.config.processing_distribution,
                device=self.device,
                dtype=self.nominal_processing_times.dtype,
            )
            if self.config.exogenous_processing_noise
            else torch.ones(
                (batch, scenario_count, operations, machines),
                device=self.device,
                dtype=self.nominal_processing_times.dtype,
            )
        )
        if self.config.health_dependent_processing_time:
            factor = health_time_factor(
                root_health[:, :, None, :],
                self.overlay.failure_level[:, None, None, :],
                self.overlay.eta[:, None, None, :],
                self.overlay.health_time_gamma[:, None, None, :],
            )
            duration = nominal * factor * processing
        else:
            duration = nominal * processing
        duration = duration.masked_fill(~active, 0)

        if self.config.action_conditioned_degradation:
            concentration = (
                self.overlay.alpha[:, None, None, :]
                * self.overlay.loads[:, None].clamp_min(0).pow(
                    self.overlay.load_sensitivity[:, None, None, :]
                )
                * duration.clamp_min(0)
            )
            expected = concentration * self.overlay.theta[:, None, None, :]
            sampled = noise_bank.degradation_noise_grid(
                episode_id=self.episode_id,
                concentration=concentration,
                scale=self.overlay.theta[:, None, None, :],
                active=active,
            )
        else:
            expected = torch.zeros_like(duration)
            sampled = torch.zeros_like(duration)
        expected = expected.masked_fill(~active, 0)
        sampled = sampled.masked_fill(~active, 0)
        survival = (
            root_health[:, :, None, :] + sampled
            < self.overlay.failure_level[:, None, None, :]
        ) & active
        forecast_root_health = root_health[:, :, None, :].expand(
            -1, -1, operations, -1
        ).clone()
        forecast_root_health.masked_fill_(~active, 0)
        return duration, expected, sampled, survival, forecast_root_health

    @profiled("env.state_scenario_production_forecast")
    def _all_operation_forecast(
        self,
        noise_bank: TrajectoryNoiseBank,
        scenario_count: int,
        trajectories: ScenarioTrajectoryState | None = None,
        machine_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.forecast_backend == "scalar_reference":
            if machine_mask is not None:
                raise ValueError("scalar reference does not support sparse refresh")
            return self._all_operation_forecast_scalar_reference(
                noise_bank, scenario_count, trajectories
            )
        return self._all_operation_forecast_vectorized(
            noise_bank, scenario_count, trajectories, machine_mask
        )

    @staticmethod
    def _trajectory_invalid_mask(
        trajectories: ScenarioTrajectoryState,
    ) -> torch.Tensor:
        """Return real invalid scenarios; high physical risk remains valid."""

        finite = (
            torch.isfinite(trajectories.health).all(dim=-1)
            & torch.isfinite(trajectories.job_ready_time).all(dim=-1)
            & torch.isfinite(trajectories.machine_ready_time).all(dim=-1)
            & torch.isfinite(trajectories.scenario_cost_components).all(dim=-1)
        )
        return ~finite

    def _gather_candidates(self, tensor: torch.Tensor) -> torch.Tensor:
        # [B,S,N,M] -> [B,S,J,M]
        index = self.observed_state.candidate[:, None, :, None].expand(
            -1, tensor.shape[1], -1, self.number_of_machines
        )
        return tensor.gather(2, index)

    def _refresh_action_masks(self) -> None:
        state = self.observed_state
        batch, scenarios, jobs, machines = (
            self.batch_size,
            self.num_scenarios,
            self.number_of_jobs,
            self.number_of_machines,
        )
        candidate_nominal = self._gather_candidates(
            self.nominal_processing_times[:, None].expand(-1, scenarios, -1, -1)
        )[:, 0]
        base_invalid = state.job_finished[:, :, None] | (candidate_nominal <= 0)
        observed_failed = state.observed_machine_status[:, None, :]
        self.scenario_invalid_mask = self._trajectory_invalid_mask(self.state_scenarios)
        scenario_safe = (
            (self._candidate_survival if self.config.scenario_safety_mask else torch.ones_like(self._candidate_survival))
            & ~base_invalid[:, None]
            & ~observed_failed[:, None]
            & ~self.scenario_invalid_mask[:, :, None, None]
        )
        self.health_pair_mask = ~scenario_safe
        valid_scenarios = (~self.scenario_invalid_mask).float()
        self.survival_probability = scenario_safe.float().sum(dim=1) / valid_scenarios.sum(
            dim=1, keepdim=True
        )[:, :, None].clamp_min(1)
        chance_invalid = (
            self.survival_probability < (1.0 - self.config.epsilon_use)
            if self.config.scenario_safety_mask
            else torch.zeros_like(self.survival_probability, dtype=torch.bool)
        )
        # The chance constraint is a soft safety filter.  Keep it separate
        # from physical incompatibility, completed jobs, and observed failure
        # so the production-only baseline can recover from an empty soft set
        # without ever unmasking a physically impossible action.
        production_hard_invalid = base_invalid | observed_failed
        self.production_mask = production_hard_invalid | chance_invalid

        remaining = self.compatible & ~state.op_scheduled[:, :, None]
        remaining_for_machine = remaining.any(dim=1)
        pm_invalid_observed = (
            state.observed_machine_status
            | ~remaining_for_machine
            | (state.pm_count >= self.config.max_pm_per_machine)
            | ~self.active[:, None]
        )
        cm_invalid_observed = ~state.observed_machine_status | ~self.active[:, None]
        # ``max_maintenance_decisions`` is an action budget, not an episode
        # termination condition.  Once it is exhausted, another maintenance
        # action is known in advance to violate the budget and therefore must
        # be masked before policy sampling.  Production remains available so
        # that a schedule which has used its maintenance allowance can still
        # finish normally.  If production is physically impossible as well,
        # the all-invalid check below records the genuine dead-end truncation.
        maintenance_budget_exhausted = (
            state.maintenance_decision_count
            >= self.config.max_maintenance_decisions
        )[:, None]
        pm_invalid_observed |= maintenance_budget_exhausted
        cm_invalid_observed |= maintenance_budget_exhausted
        if not self.config.scenario_recourse:
            # The no-recourse ablation permits only actions that are directly
            # compatible with every persistent state and reward trajectory.
            any_failed = self.state_scenarios.machine_status.any(dim=1) | self.reward_scenarios.machine_status.any(dim=1)
            any_healthy = (~self.state_scenarios.machine_status).any(dim=1) | (~self.reward_scenarios.machine_status).any(dim=1)
            self.production_mask |= any_failed[:, None, :]
            production_hard_invalid |= any_failed[:, None, :]
            pm_invalid_observed |= any_failed
            cm_invalid_observed |= any_healthy
        if not self.config.maintenance_actions:
            pm_invalid_observed = torch.ones_like(pm_invalid_observed)
            cm_invalid_observed = torch.ones_like(cm_invalid_observed)
        else:
            if not self.config.preventive_maintenance_actions:
                pm_invalid_observed = torch.ones_like(pm_invalid_observed)
            if not self.config.corrective_maintenance_actions:
                cm_invalid_observed = torch.ones_like(cm_invalid_observed)
        self.chance_constraint_backoff_mask = torch.zeros_like(
            self.production_mask
        )
        # Disabling proactive PM removes the policy's normal recovery action
        # when the soft chance constraint rejects every production pair.
        # In that ablation only, retain physical feasibility by exposing the
        # hard-legal production set.  This behavior is entailed by the single
        # PM switch; it does not alter health, risk, cost, or CM parameters.
        allow_empty_set_backoff = (
            self.config.chance_constraint_empty_set_backoff
            or not self.config.preventive_maintenance_actions
        )
        if allow_empty_set_backoff:
            no_ordinary_action = (
                self.active
                & self.production_mask.flatten(1).all(dim=1)
                & pm_invalid_observed.all(dim=1)
                & cm_invalid_observed.all(dim=1)
            )
            hard_legal = ~production_hard_invalid
            backoff_rows = no_ordinary_action & hard_legal.flatten(1).any(dim=1)
            if backoff_rows.any():
                row_mask = backoff_rows[:, None, None]
                self.chance_constraint_backoff_mask = (
                    row_mask & hard_legal & chance_invalid
                )
                self.production_mask = torch.where(
                    row_mask, production_hard_invalid, self.production_mask
                )
        self.pm_mask_observed = pm_invalid_observed
        self.cm_mask_observed = cm_invalid_observed
        self.pm_mask = pm_invalid_observed[:, None, :].expand(-1, scenarios, -1).clone()
        self.cm_mask = cm_invalid_observed[:, None, :].expand(-1, scenarios, -1).clone()
        self.action_mask = torch.cat(
            (
                self.production_mask.reshape(batch, jobs * machines),
                pm_invalid_observed,
                cm_invalid_observed,
            ),
            dim=1,
        )
        inactive = ~self.active
        if inactive.any():
            self.action_mask[inactive] = True
            self.action_mask[inactive, 0] = False
        without_action = self.active & self.action_mask.all(dim=1)
        if without_action.any():
            # A physical dead-end is an explicit truncation, never a nonterminal
            # all-masked state.
            state.truncated[without_action] = True
            self.action_mask[without_action] = True
            self.action_mask[without_action, 0] = False

    def _update_scenario_costs(self, trajectories: ScenarioTrajectoryState) -> None:
        """Update accumulated complete-trajectory objective components in place."""

        trajectories.scenario_cost_components.copy_(
            torch.stack(
                (
                    trajectories.current_makespan,
                    trajectories.pm_cost,
                    trajectories.cm_cost,
                    trajectories.unplanned_downtime.sum(dim=-1),
                    trajectories.failure_count.sum(dim=-1),
                ),
                dim=-1,
            )
        )

    def _risk_adjusted_total(
        self,
        scenario_costs: torch.Tensor | None = None,
        scenario_invalid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = self.scenario_total_cost(scenario_costs)
        invalid = scenario_invalid_mask
        if invalid is None and scenario_costs is None:
            invalid = self._trajectory_invalid_mask(self.reward_scenarios)
        weights = self.reward_scenario_weights
        if invalid is not None:
            weights = weights.masked_fill(invalid, 0.0)
        cvar_weights = weights
        if not self.reward_weighted_cvar_enabled:
            cvar_weights = (~invalid).to(total.dtype) if invalid is not None else torch.ones_like(total)
        return weighted_scenario_mean(total, weights) + self.config.objective.cvar_beta * weighted_upper_tail_cvar(
            total, cvar_weights, self.config.objective.cvar_alpha
        )

    def set_reward_scenario_weights(self, weights: torch.Tensor) -> None:
        """Install ATMSL representative masses; full fidelity stays uniform."""

        weights = torch.as_tensor(
            weights, dtype=self.nominal_processing_times.dtype, device=self.device
        )
        if weights.ndim == 1:
            weights = weights[None].expand(self.batch_size, -1)
        if weights.shape != (self.batch_size, self.reward_num_scenarios):
            raise ValueError("reward scenario weights must be [Sr] or [B,Sr]")
        if (
            not torch.isfinite(weights).all()
            or (weights < 0).any()
            or (weights.sum(dim=1) <= 0).any()
        ):
            raise ValueError("reward scenario weights must be finite nonnegative masses")
        self.reward_scenario_weights = weights / weights.sum(dim=1, keepdim=True)

    def configure_atmsl_scenario_support(
        self,
        *,
        state_scenario_ids: torch.Tensor | list[int],
        reward_scenario_ids: torch.Tensor | list[int],
        reward_weights: torch.Tensor,
        weighted_cvar_enabled: bool = True,
    ) -> RAMPEnvState:
        """Bind compact slots to paired full-fidelity paths and reset.

        The mapping affects the keyed event identity, not tensor slot indices.
        Consequently the tensor transition kernel stays dense/fast while a
        selected representative follows exactly the same noise path as its
        full-fidelity counterpart.
        """

        if len(state_scenario_ids) != self.num_scenarios:
            raise ValueError("state semantic support size must equal S")
        if len(reward_scenario_ids) != self.reward_num_scenarios:
            raise ValueError("reward semantic support size must equal Sr")
        self.state_noise_bank.set_semantic_scenario_ids(state_scenario_ids)
        self.reward_noise_bank.set_semantic_scenario_ids(reward_scenario_ids)
        self.set_reward_scenario_weights(reward_weights)
        self.reward_weighted_cvar_enabled = bool(weighted_cvar_enabled)
        return self.reset()

    def upper_tail_cvar(self, values: torch.Tensor) -> torch.Tensor:
        """Public empirical CVaR using the configured tail level."""

        return upper_tail_cvar(values, self.config.objective.cvar_alpha)

    def risk_potential(self) -> torch.Tensor:
        """Mean-plus-CVaR potential of persistent reward trajectories."""

        return self._risk_adjusted_total(self.reward_scenarios.scenario_cost_components)

    @profiled("env.refresh_forecasts_total")
    def refresh_forecasts(self, changed_machines: torch.Tensor | None = None) -> None:
        """Build pre-action candidate forecasts from persistent state trajectories."""

        machine_mask = None
        if changed_machines is not None:
            machine_mask = torch.zeros(
                (self.batch_size, self.number_of_machines),
                dtype=torch.bool,
                device=self.device,
            )
            machine_mask.scatter_(1, changed_machines[:, None], True)
        forecast = self._all_operation_forecast(
            self.state_noise_bank,
            self.num_scenarios,
            self.state_scenarios,
            machine_mask,
        )
        if machine_mask is None:
            (
                self._all_duration,
                self._all_expected_delta,
                self._all_sampled_delta,
                self._all_survival,
                self._all_forecast_root_health,
            ) = forecast
        else:
            update_mask = machine_mask[:, None, None, :]
            names = (
                "_all_duration",
                "_all_expected_delta",
                "_all_sampled_delta",
                "_all_survival",
                "_all_forecast_root_health",
            )
            for name, update in zip(names, forecast):
                current = getattr(self, name)
                setattr(self, name, torch.where(update_mask, update, current))
        self._candidate_effective_duration = self._gather_candidates(self._all_duration)
        self._candidate_expected_delta = self._gather_candidates(
            self._all_expected_delta
        )
        self._candidate_delta = self._gather_candidates(self._all_sampled_delta)
        self._candidate_survival = self._gather_candidates(self._all_survival)
        self._candidate_nominal = self._gather_candidates(
            self.nominal_processing_times[:, None].expand(
                -1, self.num_scenarios, -1, -1
            )
        )[:, 0]
        self._candidate_load = self._gather_candidates(
            self.overlay.loads[:, None].expand(-1, self.num_scenarios, -1, -1)
        )[:, 0]
        self._build_candidate_scenario_tensors(changed_machines)
        self._refresh_action_masks()
        self._update_scenario_costs(self.state_scenarios)
        self._update_scenario_costs(self.reward_scenarios)
        self.reward_scenario_cost_components = self.reward_scenarios.scenario_cost_components
        self.state = self._build_state()

    def _build_candidate_scenario_tensors_scalar_reference(self) -> None:
        """Materialize all production, PM, and CM consequences before sampling."""

        b, s, j, m = (
            self.batch_size,
            self.num_scenarios,
            self.number_of_jobs,
            self.number_of_machines,
        )
        tau = self.time_scale[:, None, None, None]
        health = self.state_scenarios.health
        status = self.state_scenarios.machine_status
        failure = self.overlay.failure_level[:, None, :]
        candidate_root_health = self._gather_candidates(self._all_forecast_root_health)
        post_health = candidate_root_health + self._candidate_delta
        crossing = post_health >= failure[:, :, None, :]
        cm_duration = self.overlay.cm_duration[:, None, None, :].expand(-1, s, j, -1)
        cm_cost = self.overlay.cm_cost[:, None, None, :].expand(-1, s, j, -1)
        recourse = status[:, :, None, :].expand(-1, -1, j, -1)
        if not self.config.scenario_recourse:
            recourse = torch.zeros_like(recourse)
        candidate_job_ready = self.state_scenarios.job_ready_time
        candidate_machine_ready = self.state_scenarios.machine_ready_time
        start = torch.maximum(
            candidate_job_ready[:, :, :, None],
            candidate_machine_ready[:, :, None, :]
            + recourse.float() * self.overlay.cm_duration[:, None, None, :],
        )
        post_ready = start + self._candidate_effective_duration
        post_ready = post_ready + crossing.float() * self.config.failure_diagnosis_delay
        self.production_candidate_scenarios = torch.stack(
            (
                self._candidate_effective_duration / tau,
                self._candidate_expected_delta / failure[:, :, None, :],
                self._candidate_delta / failure[:, :, None, :],
                post_health / failure[:, :, None, :],
                crossing.float(),
                self._candidate_survival.float(),
                recourse.float() * cm_duration / tau,
                recourse.float() * cm_cost / self.objective_scales[:, None, None, 2:3],
                post_ready / tau,
            ),
            dim=-1,
        )

        pm = torch.zeros((b, s, m, 7), device=self.device)
        cm = torch.zeros((b, s, m, 6), device=self.device)
        for row in range(b):
            for scenario in range(s):
                for machine in range(m):
                    current = health[row, scenario, machine]
                    pm_is_cm = bool(
                        self.config.scenario_recourse
                        and status[row, scenario, machine]
                    )
                    pm_type = "CM" if pm_is_cm else "PM"
                    pm_count = int(
                        self.state_scenarios.cm_count[row, scenario, machine]
                        if pm_is_cm
                        else self.state_scenarios.pm_count[row, scenario, machine]
                    )
                    pm_residual = self.state_noise_bank.maintenance_noise(
                        episode_id=self.episode_id,
                        scenario_id=scenario,
                        machine_id=machine,
                        maintenance_count=pm_count,
                        maintenance_type=pm_type,
                        std=float(self.overlay.maintenance_noise_std[row, machine]),
                        device=self.device,
                        dtype=current.dtype,
                    )
                    if pm_is_cm:
                        pm_duration = self.overlay.cm_duration[row, machine]
                        pm_cost = self.overlay.cm_cost[row, machine]
                        pm_rho = self.overlay.cm_rho[row, machine]
                    else:
                        pm_duration = self.overlay.pm_duration[row, machine]
                        pm_cost = self.overlay.pm_cost[row, machine]
                        pm_rho = self.overlay.pm_rho[row, machine]
                    pm_post = restore_health(
                        current, pm_rho, pm_residual, self.overlay.failure_level[row, machine]
                    )
                    avoided = (current - pm_post).clamp_min(0)
                    pm[row, scenario, machine] = torch.stack(
                        (
                            pm_duration / self.time_scale[row],
                            pm_cost / self.objective_scales[row, 1 if not pm_is_cm else 2],
                            pm_residual / self.overlay.failure_level[row, machine],
                            pm_post / self.overlay.failure_level[row, machine],
                            torch.ones((), device=self.device),
                            avoided / self.overlay.failure_level[row, machine],
                            (self.state_scenarios.machine_ready_time[row, scenario, machine] + pm_duration)
                            / self.time_scale[row],
                        )
                    )
                    cm_count = int(self.state_scenarios.cm_count[row, scenario, machine])
                    cm_residual = self.state_noise_bank.maintenance_noise(
                        episode_id=self.episode_id,
                        scenario_id=scenario,
                        machine_id=machine,
                        maintenance_count=cm_count,
                        maintenance_type="CM",
                        std=float(self.overlay.maintenance_noise_std[row, machine]),
                        device=self.device,
                        dtype=current.dtype,
                    )
                    cm_post = restore_health(
                        current,
                        self.overlay.cm_rho[row, machine],
                        cm_residual,
                        self.overlay.failure_level[row, machine],
                    )
                    cm[row, scenario, machine] = torch.stack(
                        (
                            self.overlay.cm_duration[row, machine] / self.time_scale[row],
                            self.overlay.cm_cost[row, machine] / self.objective_scales[row, 2],
                            cm_residual / self.overlay.failure_level[row, machine],
                            cm_post / self.overlay.failure_level[row, machine],
                            torch.ones((), device=self.device),
                            (self.state_scenarios.machine_ready_time[row, scenario, machine]
                             + self.overlay.cm_duration[row, machine]) / self.time_scale[row],
                        )
                    )
        self.pm_candidate_scenarios = pm
        self.cm_candidate_scenarios = cm

    def _build_candidate_scenario_tensors_vectorized(
        self, changed_machines: torch.Tensor | None = None
    ) -> None:
        """Materialize candidate tensors without per-scalar CUDA synchronization."""

        b, s, j, m = (
            self.batch_size,
            self.num_scenarios,
            self.number_of_jobs,
            self.number_of_machines,
        )
        tau = self.time_scale[:, None, None, None]
        health = self.state_scenarios.health
        status = self.state_scenarios.machine_status
        failure = self.overlay.failure_level[:, None, :]
        candidate_root_health = self._gather_candidates(self._all_forecast_root_health)
        post_health = candidate_root_health + self._candidate_delta
        crossing = post_health >= failure[:, :, None, :]
        cm_duration_pair = self.overlay.cm_duration[:, None, None, :].expand(-1, s, j, -1)
        cm_cost_pair = self.overlay.cm_cost[:, None, None, :].expand(-1, s, j, -1)
        recourse = status[:, :, None, :].expand(-1, -1, j, -1)
        if not self.config.scenario_recourse:
            recourse = torch.zeros_like(recourse)
        start = torch.maximum(
            self.state_scenarios.job_ready_time[:, :, :, None],
            self.state_scenarios.machine_ready_time[:, :, None, :]
            + recourse.float() * self.overlay.cm_duration[:, None, None, :],
        )
        post_ready = (
            start
            + self._candidate_effective_duration
            + crossing.float() * self.config.failure_diagnosis_delay
        )
        self.production_candidate_scenarios = torch.stack(
            (
                self._candidate_effective_duration / tau,
                self._candidate_expected_delta / failure[:, :, None, :],
                self._candidate_delta / failure[:, :, None, :],
                post_health / failure[:, :, None, :],
                crossing.float(),
                self._candidate_survival.float(),
                recourse.float() * cm_duration_pair / tau,
                recourse.float()
                * cm_cost_pair
                / self.objective_scales[:, None, None, 2:3],
                post_ready / tau,
            ),
            dim=-1,
        )

        if changed_machines is not None:
            machine = changed_machines[:, None].expand(-1, s)
            machine_index = machine[:, :, None]

            def gather_machine(value: torch.Tensor) -> torch.Tensor:
                return value.gather(2, machine_index).squeeze(2)

            def selected(value: torch.Tensor) -> torch.Tensor:
                return value.gather(1, changed_machines[:, None]).expand(-1, s)

            health_selected = gather_machine(health)
            status_selected = gather_machine(status)
            failure_selected = selected(self.overlay.failure_level)
            std_selected = selected(self.overlay.maintenance_noise_std)
            pm_is_cm_selected = status_selected & self.config.scenario_recourse
            pm_count_selected = torch.where(
                pm_is_cm_selected,
                gather_machine(self.state_scenarios.cm_count),
                gather_machine(self.state_scenarios.pm_count),
            )
            pm_residual_selected = self.state_noise_bank.maintenance_noise_selected(
                episode_id=self.episode_id,
                scenario_ids=self.state_scenarios.scenario_ids,
                machine_ids=machine,
                counts=pm_count_selected,
                maintenance_is_cm=pm_is_cm_selected,
                std=std_selected,
                active=torch.ones_like(pm_is_cm_selected),
                device=self.device,
                dtype=health.dtype,
            )
            pm_duration_selected = torch.where(
                pm_is_cm_selected,
                selected(self.overlay.cm_duration),
                selected(self.overlay.pm_duration),
            )
            pm_cost_selected = torch.where(
                pm_is_cm_selected,
                selected(self.overlay.cm_cost),
                selected(self.overlay.pm_cost),
            )
            pm_rho_selected = torch.where(
                pm_is_cm_selected,
                selected(self.overlay.cm_rho),
                selected(self.overlay.pm_rho),
            )
            pm_post_selected = restore_health(
                health_selected,
                pm_rho_selected,
                pm_residual_selected,
                failure_selected,
            )
            pm_scale_selected = torch.where(
                pm_is_cm_selected,
                self.objective_scales[:, 2:3].expand(-1, s),
                self.objective_scales[:, 1:2].expand(-1, s),
            )
            pm_update = torch.stack(
                (
                    pm_duration_selected / self.time_scale[:, None],
                    pm_cost_selected / pm_scale_selected,
                    pm_residual_selected / failure_selected,
                    pm_post_selected / failure_selected,
                    torch.ones_like(health_selected),
                    (health_selected - pm_post_selected).clamp_min(0)
                    / failure_selected,
                    (
                        gather_machine(self.state_scenarios.machine_ready_time)
                        + pm_duration_selected
                    )
                    / self.time_scale[:, None],
                ),
                dim=-1,
            )
            self.pm_candidate_scenarios.scatter_(
                2,
                machine_index[:, :, :, None].expand(-1, -1, -1, 7),
                pm_update[:, :, None, :],
            )

            cm_count_selected = gather_machine(self.state_scenarios.cm_count)
            cm_residual_selected = self.state_noise_bank.maintenance_noise_selected(
                episode_id=self.episode_id,
                scenario_ids=self.state_scenarios.scenario_ids,
                machine_ids=machine,
                counts=cm_count_selected,
                maintenance_is_cm=torch.ones_like(status_selected),
                std=std_selected,
                active=torch.ones_like(status_selected),
                device=self.device,
                dtype=health.dtype,
            )
            cm_duration_selected = selected(self.overlay.cm_duration)
            cm_post_selected = restore_health(
                health_selected,
                selected(self.overlay.cm_rho),
                cm_residual_selected,
                failure_selected,
            )
            cm_update = torch.stack(
                (
                    cm_duration_selected / self.time_scale[:, None],
                    selected(self.overlay.cm_cost)
                    / self.objective_scales[:, 2:3],
                    cm_residual_selected / failure_selected,
                    cm_post_selected / failure_selected,
                    torch.ones_like(health_selected),
                    (
                        gather_machine(self.state_scenarios.machine_ready_time)
                        + cm_duration_selected
                    )
                    / self.time_scale[:, None],
                ),
                dim=-1,
            )
            self.cm_candidate_scenarios.scatter_(
                2,
                machine_index[:, :, :, None].expand(-1, -1, -1, 6),
                cm_update[:, :, None, :],
            )
            return

        pm_is_cm = status & self.config.scenario_recourse
        pm_counts = torch.where(
            pm_is_cm, self.state_scenarios.cm_count, self.state_scenarios.pm_count
        )
        pm_residual = self.state_noise_bank.maintenance_noise_grid(
            episode_id=self.episode_id,
            counts=pm_counts,
            maintenance_type=pm_is_cm,
            std=self.overlay.maintenance_noise_std,
            device=self.device,
            dtype=health.dtype,
        )
        pm_duration = torch.where(
            pm_is_cm,
            self.overlay.cm_duration[:, None, :],
            self.overlay.pm_duration[:, None, :],
        )
        pm_cost = torch.where(
            pm_is_cm,
            self.overlay.cm_cost[:, None, :],
            self.overlay.pm_cost[:, None, :],
        )
        pm_rho = torch.where(
            pm_is_cm,
            self.overlay.cm_rho[:, None, :],
            self.overlay.pm_rho[:, None, :],
        )
        pm_post = restore_health(health, pm_rho, pm_residual, failure)
        pm_scale = torch.where(
            pm_is_cm,
            self.objective_scales[:, None, 2:3],
            self.objective_scales[:, None, 1:2],
        )
        self.pm_candidate_scenarios = torch.stack(
            (
                pm_duration / self.time_scale[:, None, None],
                pm_cost / pm_scale,
                pm_residual / failure,
                pm_post / failure,
                torch.ones((b, s, m), device=self.device),
                (health - pm_post).clamp_min(0) / failure,
                (self.state_scenarios.machine_ready_time + pm_duration)
                / self.time_scale[:, None, None],
            ),
            dim=-1,
        )

        cm_residual = self.state_noise_bank.maintenance_noise_grid(
            episode_id=self.episode_id,
            counts=self.state_scenarios.cm_count,
            maintenance_type="CM",
            std=self.overlay.maintenance_noise_std,
            device=self.device,
            dtype=health.dtype,
        )
        cm_post = restore_health(
            health,
            self.overlay.cm_rho[:, None, :],
            cm_residual,
            failure,
        )
        self.cm_candidate_scenarios = torch.stack(
            (
                self.overlay.cm_duration[:, None, :].expand(-1, s, -1)
                / self.time_scale[:, None, None],
                self.overlay.cm_cost[:, None, :].expand(-1, s, -1)
                / self.objective_scales[:, None, 2:3],
                cm_residual / failure,
                cm_post / failure,
                torch.ones((b, s, m), device=self.device),
                (
                    self.state_scenarios.machine_ready_time
                    + self.overlay.cm_duration[:, None, :]
                )
                / self.time_scale[:, None, None],
            ),
            dim=-1,
        )

    @profiled("env.production_pm_cm_candidate_construction")
    def _build_candidate_scenario_tensors(
        self, changed_machines: torch.Tensor | None = None
    ) -> None:
        if self.config.forecast_backend == "scalar_reference":
            self._build_candidate_scenario_tensors_scalar_reference()
        else:
            self._build_candidate_scenario_tensors_vectorized(changed_machines)

    def scenario_cost_components(
        self, scenario_costs: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (
            self.reward_scenario_cost_components
            if scenario_costs is None
            else scenario_costs
        )

    def scenario_total_cost(
        self, scenario_costs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dimensionless weighted scenario cost used by reward and critic."""

        components = self.scenario_cost_components(scenario_costs)
        weights = torch.tensor(
            (
                1.0,
                self.config.objective.lambda_pm,
                self.config.objective.lambda_cm,
                self.config.objective.lambda_downtime,
                self.config.objective.lambda_failure,
            ),
            device=self.device,
            dtype=components.dtype,
        )
        normalized = components / self.objective_scales[:, None, :]
        return (normalized * weights).sum(dim=-1)

    def scenario_raw_weighted_total_cost(
        self, scenario_costs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Raw-unit weighted total retained solely for transparent reporting."""

        components = self.scenario_cost_components(scenario_costs)
        weights = torch.tensor(
            (
                1.0,
                self.config.objective.lambda_pm,
                self.config.objective.lambda_cm,
                self.config.objective.lambda_downtime,
                self.config.objective.lambda_failure,
            ),
            device=self.device,
            dtype=components.dtype,
        )
        return (components * weights).sum(dim=-1)

    def _validate_actions(self, actions: torch.Tensor, active_before: torch.Tensor) -> None:
        if actions.shape != (self.batch_size,):
            raise ValueError(f"actions must have shape [{self.batch_size}]")
        invalid = self.action_mask.gather(1, actions[:, None]).squeeze(1) & active_before
        if invalid.any() and self.config.strict_invalid_actions:
            rows = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise AssertionError(
                f"invalid production/maintenance actions for active rows {rows}"
            )

    def _apply_action_forecast_device_scalar_reference(
        self,
        actions: torch.Tensor,
        *,
        noise_bank: TrajectoryNoiseBank,
        trajectories: ScenarioTrajectoryState,
    ) -> ForecastScenarioBatch:
        """Apply one observed-state action under every persistent future noise path.

        The selected common action mutates the same scenario identities that
        existed at the previous decision.  Scenario/observed legality mismatch
        is resolved by the fixed, fully costed recourse rule.
        """

        decoded = self.codec.decode(actions)
        scenario_count = trajectories.scenario_count
        batch = self.batch_size
        machines = self.number_of_machines
        jobs = self.number_of_jobs
        root_health = trajectories.health.clone()
        root_status = trajectories.machine_status.clone()
        post_health = torch.empty_like(root_health)
        post_status = torch.empty_like(root_status)
        post_job_ready = torch.empty(
            (batch, scenario_count, jobs), device=self.device
        )
        post_machine_ready = torch.empty(
            (batch, scenario_count, machines), device=self.device
        )
        duration = torch.zeros((batch, scenario_count), device=self.device)
        degradation = torch.zeros_like(duration)
        maintenance_cost = torch.zeros_like(duration)
        failure_crossing = torch.zeros(
            (batch, scenario_count), dtype=torch.bool, device=self.device
        )
        for scenario in range(scenario_count):
            future = trajectories.as_observed(scenario)
            for row in range(batch):
                if future.terminated[row] or future.truncated[row]:
                    continue
                result = self.transition_kernel.apply_primary_action_with_recourse(
                    future,
                    batch_index=row,
                    action=int(actions[row]),
                    action_codec=self.codec,
                    noise_bank=noise_bank,
                    scenario_id=scenario,
                )
                duration[row, scenario] = result.total_elapsed
                degradation[row, scenario] = result.degradation_increment
                maintenance_cost[row, scenario] = result.maintenance_cost_increment
                failure_crossing[row, scenario] = result.failure_crossing
            post_health[:, scenario] = future.observed_health
            post_status[:, scenario] = future.observed_machine_status
            post_job_ready[:, scenario] = future.observed_job_ready_time
            post_machine_ready[:, scenario] = future.observed_machine_ready_time
            trajectories.update_from_observed(scenario, future)

        self._update_scenario_costs(trajectories)

        return ForecastScenarioBatch(
            action=actions.clone(),
            root_health=root_health,
            root_machine_status=root_status,
            post_health=post_health,
            post_machine_status=post_status,
            post_job_ready_time=post_job_ready,
            post_machine_ready_time=post_machine_ready,
            duration=duration,
            degradation_increment=degradation,
            maintenance_cost_increment=maintenance_cost,
            failure_crossing=failure_crossing,
        )

    def _apply_action_forecast_cpu_scalar_kernel(
        self,
        actions: torch.Tensor,
        *,
        noise_bank: TrajectoryNoiseBank,
        trajectories: ScenarioTrajectoryState,
    ) -> ForecastScenarioBatch:
        """Run the identical scalar mutation kernel on host scenario state."""

        if self.device.type == "cuda":
            self.profiler.transfer(
                "gpu_to_cpu_bytes", tensor_bytes(trajectories) + tensor_bytes(actions)
            )
        actions_cpu = actions.detach().cpu()
        working = trajectories.clone(device="cpu")
        scenario_count = working.scenario_count
        batch = self.batch_size
        machines = self.number_of_machines
        jobs = self.number_of_jobs
        root_health = working.health.clone()
        root_status = working.machine_status.clone()
        post_health = torch.empty_like(root_health)
        post_status = torch.empty_like(root_status)
        post_job_ready = torch.empty((batch, scenario_count, jobs))
        post_machine_ready = torch.empty((batch, scenario_count, machines))
        duration = torch.zeros((batch, scenario_count))
        degradation = torch.zeros_like(duration)
        maintenance_cost = torch.zeros_like(duration)
        failure_crossing = torch.zeros((batch, scenario_count), dtype=torch.bool)
        for scenario in range(scenario_count):
            future = working.as_observed(scenario)
            for row in range(batch):
                if future.terminated[row] or future.truncated[row]:
                    continue
                result = self._scenario_cpu_kernel.apply_primary_action_with_recourse(
                    future,
                    batch_index=row,
                    action=int(actions_cpu[row]),
                    action_codec=self.codec,
                    noise_bank=noise_bank,
                    scenario_id=scenario,
                )
                duration[row, scenario] = result.total_elapsed
                degradation[row, scenario] = result.degradation_increment
                maintenance_cost[row, scenario] = result.maintenance_cost_increment
                failure_crossing[row, scenario] = result.failure_crossing
            post_health[:, scenario] = future.observed_health
            post_status[:, scenario] = future.observed_machine_status
            post_job_ready[:, scenario] = future.observed_job_ready_time
            post_machine_ready[:, scenario] = future.observed_machine_ready_time
            working.update_from_observed(scenario, future)
        self._update_scenario_costs(working)

        for field in fields(working):
            source = getattr(working, field.name)
            if isinstance(source, torch.Tensor):
                target = getattr(trajectories, field.name)
                target.copy_(source.to(target.device))
                if target.device.type == "cuda":
                    self.profiler.transfer("cpu_to_gpu_bytes", tensor_bytes(source))
            elif field.name == "action_history":
                trajectories.action_history = deepcopy(source)

        def on_device(value: torch.Tensor) -> torch.Tensor:
            if self.device.type == "cuda":
                self.profiler.transfer("cpu_to_gpu_bytes", tensor_bytes(value))
            return value.to(self.device)

        return ForecastScenarioBatch(
            action=actions.clone(),
            root_health=on_device(root_health),
            root_machine_status=on_device(root_status),
            post_health=on_device(post_health),
            post_machine_status=on_device(post_status),
            post_job_ready_time=on_device(post_job_ready),
            post_machine_ready_time=on_device(post_machine_ready),
            duration=on_device(duration),
            degradation_increment=on_device(degradation),
            maintenance_cost_increment=on_device(maintenance_cost),
            failure_crossing=on_device(failure_crossing),
        )

    @profiled("env.scenario_transition_one_bank")
    def _apply_action_forecast(
        self,
        actions: torch.Tensor,
        *,
        noise_bank: TrajectoryNoiseBank,
        trajectories: ScenarioTrajectoryState,
    ) -> ForecastScenarioBatch:
        if self.config.scenario_transition_backend == "device_scalar_reference":
            return self._apply_action_forecast_device_scalar_reference(
                actions, noise_bank=noise_bank, trajectories=trajectories
            )
        if self.config.scenario_transition_backend == "tensorized_selected_action":
            result = self.transition_kernel.apply_primary_actions_to_trajectories(
                trajectories,
                actions=actions,
                action_codec=self.codec,
                noise_bank=noise_bank,
            )
            self._update_scenario_costs(trajectories)
            return result
        return self._apply_action_forecast_cpu_scalar_kernel(
            actions, noise_bank=noise_bank, trajectories=trajectories
        )

    @profiled("env.step_total")
    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        *,
        return_tensors: bool = False,
    ) -> tuple[RAMPEnvState, Any, Any, Any, dict[str, Any]]:
        """Apply one common observed action and reconstruct both forecast sets."""

        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        active_before = self.active.clone()
        self._validate_actions(actions, active_before)
        old_risk = self.risk_potential().clone()
        state_action_forecast = self._apply_action_forecast(
            actions,
            noise_bank=self.state_noise_bank,
            trajectories=self.state_scenarios,
        )
        reward_action_forecast = self._apply_action_forecast(
            actions,
            noise_bank=self.reward_noise_bank,
            trajectories=self.reward_scenarios,
        )
        state = self.observed_state
        if self.config.scenario_transition_backend == "tensorized_selected_action":
            # Reuse the unique tensorized physical authority with a singleton
            # observed-noise trajectory.  This removes the per-row Python loop
            # and its device synchronizations without adding a second formula.
            observed_bank = ScenarioTrajectoryState.from_observed(
                state,
                1,
                episode_id=self.episode_id,
                noise_namespace="observed",
            )
            observed_bank.action_history = [
                [list(state.action_history[row])]
                for row in range(self.batch_size)
            ]
            # The observed namespace historically uses scenario_id=-1.  Keep
            # that stable key while the singleton tensor axis remains index 0.
            observed_bank.scenario_ids.fill_(-1)
            self.transition_kernel.apply_primary_actions_to_trajectories(
                observed_bank,
                actions=actions,
                action_codec=self.codec,
                noise_bank=self.observed_noise_bank,
            )
            self.observed_state = observed_bank.as_observed(0)
            state = self.observed_state
        else:
            for batch in range(self.batch_size):
                if not active_before[batch]:
                    continue
                self.transition_kernel.apply_primary_action_with_recourse(
                    state,
                    batch_index=batch,
                    action=int(actions[batch]),
                    action_codec=self.codec,
                )
        decoded = self.codec.decode(actions)
        self.refresh_forecasts(
            decoded.machine
            if self.config.forecast_backend == "vectorized"
            else None
        )
        new_risk = self.risk_potential()
        reward = old_risk - new_risk
        info: dict[str, Any] = {
            "active_before_step": active_before.clone(),
            "risk_adjusted_total_cost": new_risk.clone(),
            "scenario_total_cost": self.scenario_total_cost().clone(),
            "cost_components": self.scenario_cost_components().clone(),
            "survival_probability": self.survival_probability.clone(),
            "action_type": decoded.action_type.clone(),
            "state_action_forecast": state_action_forecast,
            "reward_action_forecast": reward_action_forecast,
        }
        if return_tensors:
            # The training loop is device-resident.  Deferring host conversion
            # avoids three CUDA synchronizations per scheduling decision while
            # preserving the public NumPy/Gym-compatible default below.
            return (
                self.state,
                reward.detach().clone(),
                state.terminated.detach().clone(),
                state.truncated.detach().clone(),
                info,
            )
        return (
            self.state,
            reward.detach().cpu().clone().numpy(),
            state.terminated.detach().cpu().clone().numpy(),
            state.truncated.detach().cpu().clone().numpy(),
            info,
        )

    @profiled("env.state_feature_and_serialization")
    def _build_state(self) -> RAMPEnvState:
        state = self.observed_state
        batch, scenarios, ops, machines, jobs = (
            self.batch_size,
            self.num_scenarios,
            self.number_of_ops,
            self.number_of_machines,
            self.number_of_jobs,
        )
        eps = 1e-8
        health = self.state_scenarios.health
        status = self.state_scenarios.machine_status
        failure = self.overlay.failure_level[:, None, :]
        normalized = health / failure
        remaining = ((failure - health) / failure).clamp_min(0)
        pm_after = self.overlay.pm_rho[:, None, :] * health
        recovery = (
            (health - pm_after).clamp_min(0)
            / (self.overlay.pm_duration[:, None, :] / self.time_scale[:, None, None] + eps)
            / failure
        )
        ready_machine = self.state_scenarios.machine_ready_time
        health_m = torch.stack(
            (
                remaining,
                normalized,
                recovery,
                status.float(),
                ready_machine / self.time_scale[:, None, None],
            ),
            dim=-1,
        )
        expected_by_machine = self._candidate_expected_delta.mean(dim=2)
        machine_risk = build_machine_risk(
            health, self.overlay.failure_level, expected_by_machine, recovery, status
        )

        remaining_pair = remaining[:, :, None, :].expand(-1, -1, jobs, -1)
        recovery_pair = recovery[:, :, None, :].expand(-1, -1, jobs, -1)
        budget = self._candidate_expected_delta / (
            (failure[:, :, None, :] - health[:, :, None, :]).clamp_min(eps)
        )
        duration_normalized = self._candidate_effective_duration / self.time_scale[
            :, None, None, None
        ]
        health_pair = torch.stack(
            (
                remaining_pair,
                budget,
                recovery_pair,
                duration_normalized,
                self._candidate_load[:, None].expand(-1, scenarios, -1, -1),
            ),
            dim=-1,
        )
        global_invalid = self.scenario_invalid_mask[:, :, None, None].expand_as(budget)
        pair_risk = build_pair_risk(
            budget, duration_normalized, self._candidate_survival, global_invalid
        ).masked_fill(self.health_pair_mask, 1e6)

        all_compatible = self.compatible[:, None]
        inf = torch.full_like(self._all_duration, float("inf"))
        op_min = torch.where(all_compatible, self._all_duration, inf).amin(dim=3)
        op_max = torch.where(
            all_compatible, self._all_duration, torch.zeros_like(self._all_duration)
        ).amax(dim=3)
        denom = all_compatible.sum(dim=3).clamp_min(1)
        op_mean = (self._all_duration * all_compatible).sum(dim=3) / denom
        op_job_ready = self.state_scenarios.job_ready_time.gather(
            2, self.job_of_op[:, None].expand(-1, scenarios, -1)
        )
        candidate_flag = torch.zeros((batch, ops), device=self.device)
        candidate_flag.scatter_(1, state.candidate, (~state.job_finished).float())
        job_lengths_for_op = self.job_lengths.gather(1, self.job_of_op)
        progress = self.position_in_job.float() / job_lengths_for_op.clamp_min(1)
        remaining_ops = (job_lengths_for_op - self.position_in_job).float()
        compatible_fraction = self.compatible.float().mean(dim=2)
        completion = self.state_scenarios.operation_completion_time
        scale = self.time_scale[:, None, None]
        fea_j = torch.stack(
            (
                self.state_scenarios.operation_scheduled.float(),
                candidate_flag[:, None].expand(-1, scenarios, -1),
                progress[:, None].expand(-1, scenarios, -1),
                op_min / scale,
                op_mean / scale,
                (op_max - op_min) / scale,
                op_job_ready / scale,
                remaining_ops[:, None].expand(-1, scenarios, -1) / self.number_of_ops,
                compatible_fraction[:, None].expand(-1, scenarios, -1),
                completion / scale,
            ),
            dim=-1,
        )

        current_valid = ~(
            state.job_finished[:, :, None] | (self._candidate_nominal <= 0)
        )
        available_jobs = current_valid.sum(dim=1).float()
        candidate_duration_mean = (
            self._candidate_effective_duration
            * current_valid[:, None].float()
        ).sum(dim=2) / current_valid.sum(dim=1)[:, None].clamp_min(1)
        remaining_work_count = (
            self.compatible & ~state.op_scheduled[:, :, None]
        ).sum(dim=1).float()
        fea_m = torch.stack(
            (
                available_jobs[:, None].expand(-1, scenarios, -1) / max(jobs, 1),
                remaining_work_count[:, None].expand(-1, scenarios, -1) / max(ops, 1),
                normalized,
                remaining,
                ready_machine / self.time_scale[:, None, None],
                candidate_duration_mean / self.time_scale[:, None, None],
                (~status).float(),
                recovery,
            ),
            dim=-1,
        )

        job_ready = self.state_scenarios.job_ready_time[:, :, :, None]
        machine_ready = self.state_scenarios.machine_ready_time[:, :, None, :]
        pair_start = torch.maximum(job_ready, machine_ready).expand(
            -1, scenarios, -1, -1
        )
        pair_wait = (pair_start - job_ready).clamp_min(0)
        fea_pairs = torch.stack(
            (
                self._candidate_nominal[:, None].expand(-1, scenarios, -1, -1)
                / self.time_scale[:, None, None, None],
                duration_normalized,
                self._candidate_load[:, None].expand(-1, scenarios, -1, -1),
                budget,
                self._candidate_survival.float(),
                pair_start / self.time_scale[:, None, None, None],
                pair_wait / self.time_scale[:, None, None, None],
                current_valid[:, None].expand(-1, scenarios, -1, -1).float(),
            ),
            dim=-1,
        )
        valid = current_valid
        comp_idx = (
            valid.transpose(1, 2)[:, :, None, :]
            & valid.transpose(1, 2)[:, None, :, :]
        )
        relation = comp_idx.any(dim=3)
        relation |= torch.eye(machines, dtype=torch.bool, device=self.device)[None]

        # Explicit scenario-free observed tensors.  These are built without
        # processing/degradation noise and therefore anchor both Actor and Critic.
        nominal = self.nominal_processing_times
        compatible = self.compatible
        nominal_inf = nominal.masked_fill(~compatible, float("inf"))
        nominal_min = nominal_inf.amin(dim=2)
        nominal_max = nominal.masked_fill(~compatible, 0).amax(dim=2)
        nominal_mean = nominal.sum(dim=2) / compatible.sum(dim=2).clamp_min(1)
        observed_operation = torch.stack(
            (
                state.op_scheduled.float(),
                candidate_flag,
                progress,
                nominal_min / self.time_scale[:, None],
                nominal_mean / self.time_scale[:, None],
                (nominal_max - nominal_min) / self.time_scale[:, None],
                state.observed_job_ready_time.gather(1, self.job_of_op) / self.time_scale[:, None],
                remaining_ops / self.number_of_ops,
                compatible_fraction,
                state.op_completion_time / self.time_scale[:, None],
            ),
            dim=-1,
        )
        observed_failure = self.overlay.failure_level
        observed_normalized = state.observed_health / observed_failure
        observed_remaining = ((observed_failure - state.observed_health) / observed_failure).clamp_min(0)
        observed_recovery = (
            (state.observed_health - self.overlay.pm_rho * state.observed_health).clamp_min(0)
            / (self.overlay.pm_duration / self.time_scale[:, None] + eps)
            / observed_failure
        )
        observed_candidate_nominal = self._candidate_nominal
        observed_candidate_mean = (
            observed_candidate_nominal * current_valid.float()
        ).sum(dim=1) / current_valid.sum(dim=1).clamp_min(1)
        observed_machine = torch.stack(
            (
                available_jobs / max(jobs, 1),
                remaining_work_count / max(ops, 1),
                observed_normalized,
                observed_remaining,
                state.observed_machine_ready_time / self.time_scale[:, None],
                observed_candidate_mean / self.time_scale[:, None],
                (~state.observed_machine_status).float(),
                observed_recovery,
            ),
            dim=-1,
        )
        candidate_health = state.observed_health[:, None, :].expand(-1, jobs, -1)
        candidate_failure = self.overlay.failure_level[:, None, :]
        candidate_health_factor = health_time_factor(
            candidate_health,
            candidate_failure,
            self.overlay.eta[:, None, :],
            self.overlay.health_time_gamma[:, None, :],
        )
        deterministic_duration = self._candidate_nominal * candidate_health_factor
        deterministic_expected = expected_degradation_increment(
            self.overlay.alpha[:, None, :],
            self._candidate_load,
            self.overlay.load_sensitivity[:, None, :],
            deterministic_duration,
            self.overlay.theta[:, None, :],
        )
        observed_start = torch.maximum(
            state.observed_job_ready_time[:, :, None],
            state.observed_machine_ready_time[:, None, :],
        )
        observed_wait = (observed_start - state.observed_job_ready_time[:, :, None]).clamp_min(0)
        observed_pair = torch.stack(
            (
                self._candidate_nominal / self.time_scale[:, None, None],
                deterministic_duration / self.time_scale[:, None, None],
                self._candidate_load,
                deterministic_expected / (candidate_failure - candidate_health).clamp_min(eps),
                (candidate_health + deterministic_expected < candidate_failure).float(),
                observed_start / self.time_scale[:, None, None],
                observed_wait / self.time_scale[:, None, None],
                current_valid.float(),
            ),
            dim=-1,
        )
        observed_global = torch.stack(
            (
                state.op_scheduled.float().mean(dim=1),
                observed_normalized.mean(dim=1),
                state.observed_machine_status.float().mean(dim=1),
                state.current_makespan / self.time_scale,
                state.maintenance_cost / (self.objective_scales[:, 1] + self.objective_scales[:, 2]),
            ),
            dim=-1,
        )
        return RAMPEnvState(
            fea_j_tensor=fea_j,
            op_mask_tensor=self.op_mask.clone(),
            fea_m_tensor=fea_m,
            mch_mask_tensor=~relation,
            dynamic_pair_mask_tensor=self.production_mask.clone(),
            comp_idx_tensor=comp_idx.float(),
            candidate_tensor=state.candidate.clone(),
            fea_pairs_tensor=fea_pairs,
            health_m_tensor=health_m,
            health_pair_tensor=health_pair,
            health_pair_mask_tensor=self.health_pair_mask.clone(),
            pm_mask_tensor=self.pm_mask.clone(),
            cm_mask_tensor=self.cm_mask.clone(),
            action_mask_tensor=self.action_mask.clone(),
            scenario_invalid_mask_tensor=self.scenario_invalid_mask.clone(),
            machine_risk_tensor=machine_risk,
            pair_risk_tensor=pair_risk,
            scenario_current_health_tensor=health.clone(),
            production_candidate_scenarios_tensor=self.production_candidate_scenarios.clone(),
            pm_candidate_scenarios_tensor=self.pm_candidate_scenarios.clone(),
            cm_candidate_scenarios_tensor=self.cm_candidate_scenarios.clone(),
            observed_operation_tensor=observed_operation,
            observed_machine_tensor=observed_machine,
            observed_pair_tensor=observed_pair,
            observed_global_tensor=observed_global,
            all_expected_delta_tensor=self._all_expected_delta.clone(),
            all_survival_tensor=self._all_survival.clone(),
            failure_level_tensor=self.overlay.failure_level.clone(),
            compatibility_tensor=self.compatible.clone(),
            terminated_tensor=state.terminated.clone(),
            truncated_tensor=state.truncated.clone(),
        )

    def metrics(self) -> dict[str, torch.Tensor]:
        """Return paper metrics and exact per-machine time accounting."""

        state = self.observed_state
        horizon = torch.maximum(
            state.current_makespan,
            state.observed_machine_ready_time.amax(dim=1),
        ).clamp_min(1e-8)
        self.transition_kernel.accrue_failed_waiting(state, horizon=horizon)
        occupied = (
            state.productive_processing_time
            + state.preventive_maintenance_time
            + state.corrective_maintenance_time
            + state.diagnosis_delay_time
            + state.failed_waiting_time
        )
        state.available_idle_time = (horizon[:, None] - occupied).clamp_min(0)
        conservation = occupied + state.available_idle_time
        if not torch.allclose(conservation, horizon[:, None].expand_as(conservation), atol=1e-5):
            raise RuntimeError("machine calendar-time conservation violated")
        availability = (
            state.productive_processing_time + state.available_idle_time
        ) / horizon[:, None]
        utilization = state.productive_processing_time / horizon[:, None]
        total_cost = self.scenario_total_cost()
        raw_total_cost = self.scenario_raw_weighted_total_cost()
        return {
            "expected_makespan": weighted_scenario_mean(
                self.scenario_cost_components()[..., 0], self.reward_scenario_weights
            ),
            "mean_total_cost": weighted_scenario_mean(
                total_cost, self.reward_scenario_weights
            ),
            "cvar_0_95_total_cost": weighted_upper_tail_cvar(
                total_cost,
                (
                    self.reward_scenario_weights
                    if self.reward_weighted_cvar_enabled
                    else torch.ones_like(self.reward_scenario_weights)
                ).masked_fill(self._trajectory_invalid_mask(self.reward_scenarios), 0.0),
                0.95,
            ),
            "raw_weighted_total_cost": weighted_scenario_mean(
                raw_total_cost, self.reward_scenario_weights
            ),
            "objective_scales": self.objective_scales.clone(),
            "pm_cost": state.pm_cost_total.clone(),
            "cm_cost": state.cm_cost_total.clone(),
            "failure_probability": (state.failure_count.sum(dim=1) > 0).float(),
            "failure_count": state.failure_count.sum(dim=1),
            "planned_downtime": state.preventive_maintenance_time.sum(dim=1),
            "unplanned_downtime": state.unplanned_downtime.sum(dim=1),
            "physical_availability": availability,
            "production_utilization": utilization,
            "calendar_horizon": horizon,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize trajectory authority and all three noise banks."""

        return {
            "format": "RAMP environment state v3",
            "reset_count": self.reset_count,
            "episode_id": self.episode_id,
            "observed_state": self.observed_state.state_dict(),
            "observed_noise_bank": self.observed_noise_bank.state_dict(),
            "state_noise_bank": self.state_noise_bank.state_dict(),
            "reward_noise_bank": self.reward_noise_bank.state_dict(),
            "state_scenarios": self.state_scenarios.state_dict(),
            "reward_scenarios": self.reward_scenarios.state_dict(),
            "reward_scenario_weights": self.reward_scenario_weights.detach().cpu(),
            "reward_weighted_cvar_enabled": self.reward_weighted_cvar_enabled,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore an episode then reconstruct forecasts without consuming noise."""

        if payload.get("format") not in {
            "RAMP environment state v1",
            "RAMP environment state v2",
            "RAMP environment state v3",
        }:
            raise ValueError("unsupported environment-state format")
        self.reset_count = int(payload["reset_count"])
        self.episode_id = int(payload["episode_id"])
        self.transition_kernel.set_episode(self.episode_id)
        self.observed_noise_bank.load_state_dict(payload["observed_noise_bank"])
        self.state_noise_bank.load_state_dict(payload["state_noise_bank"])
        self.reward_noise_bank.load_state_dict(payload["reward_noise_bank"])
        self.observed_state = ObservedShopState.from_state_dict(
            payload["observed_state"], self.device
        )
        if "state_scenarios" in payload:
            self.state_scenarios = ScenarioTrajectoryState.from_state_dict(
                payload["state_scenarios"], self.device
            )
            self.reward_scenarios = ScenarioTrajectoryState.from_state_dict(
                payload["reward_scenarios"], self.device
            )
        else:
            self.state_scenarios = ScenarioTrajectoryState.from_observed(
                self.observed_state, self.num_scenarios,
                episode_id=self.episode_id, noise_namespace="state"
            )
            self.reward_scenarios = ScenarioTrajectoryState.from_observed(
                self.observed_state, self.reward_num_scenarios,
                episode_id=self.episode_id, noise_namespace="reward"
            )
        weights = payload.get("reward_scenario_weights")
        if weights is not None:
            self.set_reward_scenario_weights(weights)
        self.reward_weighted_cvar_enabled = bool(
            payload.get("reward_weighted_cvar_enabled", True)
        )
        self.refresh_forecasts()


class _EnvironmentFacade:
    """Attribute forwarding only; it contains no physical transition logic."""

    def __init__(self, core: RAMPEnvironmentCore):
        object.__setattr__(self, "_core", core)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    @property
    def observed_state(self) -> ObservedShopState:
        return self._core.observed_state

    @property
    def transition_kernel(self) -> RAMPTransitionKernel:
        return self._core.transition_kernel

    def refresh_forecasts(self) -> None:
        self._core.refresh_forecasts()

    def metrics(self) -> dict[str, torch.Tensor]:
        return self._core.metrics()

    def state_dict(self) -> dict[str, Any]:
        return self._core.state_dict()

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._core.load_state_dict(payload)


class RAMPEnv(_EnvironmentFacade):
    """Feature environment composed around the shared transition core."""

    def __init__(
        self,
        job_lengths: torch.Tensor | np.ndarray | None = None,
        nominal_processing_times: torch.Tensor | np.ndarray | None = None,
        *,
        overlay: HealthOverlay | None = None,
        config: RAMPConfig | None = None,
        reward_num_scenarios: int | None = None,
        reward_seed: int | None = None,
        device: torch.device | str = "cpu",
        _core: RAMPEnvironmentCore | None = None,
    ):
        if _core is None:
            if job_lengths is None or nominal_processing_times is None:
                raise ValueError("job lengths and nominal times are required")
            cfg = config or RAMPConfig.from_paper_regime("H1")
            _core = RAMPEnvironmentCore(
                job_lengths,
                nominal_processing_times,
                overlay=overlay,
                config=cfg,
                reward_num_scenarios=reward_num_scenarios,
                reward_seed=reward_seed,
                device=device,
            )
        super().__init__(_core)

    def reset(self) -> RAMPEnvState:
        return self._core.reset()

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        *,
        return_tensors: bool = False,
    ) -> tuple[RAMPEnvState, Any, Any, Any, dict[str, Any]]:
        return self._core.step(actions, return_tensors=return_tensors)


class RAMPNoFeatureEnv(_EnvironmentFacade):
    """Compact view composed around the same kernel type, never inheritance."""

    def __init__(
        self,
        job_lengths: torch.Tensor | np.ndarray | None = None,
        nominal_processing_times: torch.Tensor | np.ndarray | None = None,
        *,
        overlay: HealthOverlay | None = None,
        config: RAMPConfig | None = None,
        reward_num_scenarios: int | None = None,
        reward_seed: int | None = None,
        device: torch.device | str = "cpu",
        _core: RAMPEnvironmentCore | None = None,
        _scenario_role: str = "state",
    ):
        if _core is None:
            if job_lengths is None or nominal_processing_times is None:
                raise ValueError("job lengths and nominal times are required")
            cfg = config or RAMPConfig.from_paper_regime("H1")
            _core = RAMPEnvironmentCore(
                job_lengths,
                nominal_processing_times,
                overlay=overlay,
                config=cfg,
                reward_num_scenarios=reward_num_scenarios,
                reward_seed=reward_seed,
                device=device,
            )
        super().__init__(_core)
        self._scenario_role = _scenario_role

    @property
    def num_scenarios(self) -> int:
        return (
            self._core.reward_num_scenarios
            if self._scenario_role == "reward"
            else self._core.num_scenarios
        )

    def _compact(self) -> dict[str, torch.Tensor]:
        state = self.observed_state
        return {
            "observed_health": state.observed_health.clone(),
            "machine_status": state.observed_machine_status.clone(),
            "job_ready_time": state.observed_job_ready_time.clone(),
            "machine_ready_time": state.observed_machine_ready_time.clone(),
            "operation_completion": state.op_completion_time.clone(),
            "maintenance_cost": state.maintenance_cost.clone(),
            "failure_count": state.failure_count.clone(),
            "unplanned_downtime": state.unplanned_downtime.clone(),
            "terminated": state.terminated.clone(),
            "truncated": state.truncated.clone(),
            "action_mask": self._core.action_mask.clone(),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        self._core.reset()
        return self._compact()

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        _, reward, terminated, truncated, info = self._core.step(actions)
        return self._compact(), reward, terminated, truncated, info


class RAMPScenarioEnv:
    """One observed environment with independent state/reward forecast banks."""

    def __init__(
        self,
        job_lengths: torch.Tensor | np.ndarray,
        nominal_processing_times: torch.Tensor | np.ndarray,
        *,
        state_config: RAMPConfig,
        reward_num_scenarios: int,
        state_overlay: HealthOverlay | None = None,
        reward_overlay: HealthOverlay | None = None,
        reward_seed: int | None = None,
        device: torch.device | str = "cpu",
        profiler: ThroughputProfiler | None = None,
    ):
        # A reward overlay may differ in scenario count in legacy callers, but
        # physical parameters and observed initial state come only from the
        # state overlay. Future reward variation comes from its independent bank.
        if reward_overlay is not None and state_overlay is not None:
            if not torch.equal(
                reward_overlay.failure_level.to(state_overlay.failure_level),
                state_overlay.failure_level,
            ):
                raise ValueError("state/reward overlays disagree on physical thresholds")
        core = RAMPEnvironmentCore(
            job_lengths,
            nominal_processing_times,
            overlay=state_overlay,
            config=state_config,
            reward_num_scenarios=reward_num_scenarios,
            reward_seed=reward_seed,
            device=device,
            profiler=profiler,
        )
        self.state_env = RAMPEnv(_core=core)
        self.reward_env = RAMPNoFeatureEnv(
            _core=core, _scenario_role="reward"
        )
        self.codec = core.codec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.state_env, name)

    @property
    def state(self) -> RAMPEnvState:
        return self.state_env.state

    @property
    def observed_state(self) -> ObservedShopState:
        return self.state_env.observed_state

    @property
    def reward_scenario_current_health(self) -> torch.Tensor:
        return self.reward_scenarios.health

    def reset(self) -> RAMPEnvState:
        return self.state_env.reset()

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        *,
        return_tensors: bool = False,
    ) -> tuple[RAMPEnvState, Any, Any, Any, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.state_env.step(
            actions, return_tensors=return_tensors
        )
        info = {
            **info,
            "state_scenarios": self.state_env.num_scenarios,
            "reward_scenarios": self.reward_env.num_scenarios,
        }
        return state, reward, terminated, truncated, info
