"""Production contract for adaptive tail-preserving multi-fidelity learning.

This module is deliberately independent of the neural architecture.  It owns
the training-fidelity schedule, deterministic representative-scenario
selection, probability masses, full/low correction diagnostics, and automatic
fallback.  Final paper evaluation never uses this approximation: it remains
fixed at the configured full fidelity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch


class ATMSLStage(str, Enum):
    PRODUCTION_WARM_START = "A_PRODUCTION_WARM_START"
    JOINT_LOW_FIDELITY = "B_JOINT_LOW_FIDELITY"
    FULL_FIDELITY_CORRECTION = "C_FULL_FIDELITY_CORRECTION"


@dataclass(frozen=True)
class ATMSLConfig:
    """Pre-registered three-stage ATMSL protocol."""

    warm_start_updates: int = 100
    joint_low_fidelity_until: int = 800
    final_full_fidelity_updates: int = 100
    correction_interval: int = 20
    warm_state_scenarios: int = 4
    warm_reward_scenarios: int = 8
    joint_state_scenarios: int = 8
    joint_reward_scenarios: int = 16
    full_state_scenarios: int = 32
    full_reward_scenarios: int = 128
    warm_ppo_epochs: int = 1
    joint_ppo_epochs: int = 4
    full_ppo_epochs: int = 4
    full_ppo_minibatch_size: int = 128
    correction_lambda: float = 1.0
    residual_relative_threshold: float = 0.15
    tail_coverage_threshold: float = 0.80
    degradation_patience: int = 2
    fallback_full_updates: int = 20
    residual_ewma_decay: float = 0.9
    production_warm_start_enabled: bool = True
    joint_low_fidelity_enabled: bool = True
    periodic_correction_enabled: bool = True
    final_full_fidelity_enabled: bool = True
    tail_preservation_enabled: bool = True
    extreme_event_anchors_enabled: bool = True
    probability_weights_enabled: bool = True
    weighted_cvar_enabled: bool = True
    paired_semantic_ids_enabled: bool = True
    control_variate_enabled: bool = True
    adaptive_fallback_enabled: bool = True
    fixed_fidelity_mode: str = "adaptive"

    def validate(self, total_updates: int) -> None:
        counts = (
            self.warm_state_scenarios,
            self.warm_reward_scenarios,
            self.joint_state_scenarios,
            self.joint_reward_scenarios,
            self.full_state_scenarios,
            self.full_reward_scenarios,
        )
        if total_updates < 1 or min(counts) < 1:
            raise ValueError("update and scenario counts must be positive")
        if self.warm_start_updates < 0 or self.joint_low_fidelity_until < self.warm_start_updates:
            raise ValueError("ATMSL stage boundaries are inconsistent")
        if not 0 <= self.final_full_fidelity_updates <= total_updates:
            raise ValueError("final full-fidelity window is outside the run")
        if self.correction_interval < 1 or min(
            self.warm_ppo_epochs, self.joint_ppo_epochs, self.full_ppo_epochs,
            self.full_ppo_minibatch_size,
        ) < 1:
            raise ValueError("correction interval and PPO epochs must be positive")
        if not 0 <= self.correction_lambda <= 1:
            raise ValueError("correction_lambda must be in [0,1]")
        if not 0 < self.tail_coverage_threshold <= 1:
            raise ValueError("tail_coverage_threshold must be in (0,1]")
        if self.degradation_patience < 1 or self.fallback_full_updates < 1:
            raise ValueError("fallback controls must be positive")
        if not 0 <= self.residual_ewma_decay < 1:
            raise ValueError("residual_ewma_decay must be in [0,1)")
        if self.fixed_fidelity_mode not in {"adaptive", "low", "full"}:
            raise ValueError("fixed_fidelity_mode must be adaptive, low, or full")


@dataclass(frozen=True)
class ATMSLPlan:
    update: int
    stage: ATMSLStage
    production_only: bool
    state_scenarios: int
    reward_scenarios: int
    ppo_epochs: int
    full_batch_ppo: bool
    paired_correction: bool
    forced_fallback: bool

    def uses_exact_full_fidelity(self, config: ATMSLConfig) -> bool:
        """Whether rollout/PPO must use the unmodified P2 noise authority."""

        return (
            not self.production_only
            and self.state_scenarios == config.full_state_scenarios
            and self.reward_scenarios == config.full_reward_scenarios
            and self.ppo_epochs == config.full_ppo_epochs
            and not self.full_batch_ppo
        )


def _validated_weights(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must be [B,S]")
    if weights.ndim == 1:
        weights = weights[None].expand(values.shape[0], -1)
    if weights.shape != values.shape:
        raise ValueError("weights must be [S] or [B,S]")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("scenario weights must be finite and nonnegative")
    mass = weights.sum(dim=1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("each row needs positive probability mass")
    return weights.to(device=values.device, dtype=values.dtype) / mass


def weighted_scenario_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Probability-weighted empirical mean."""

    normalized = _validated_weights(values, weights)
    return (values * normalized).sum(dim=1)


def weighted_upper_tail_cvar(
    values: torch.Tensor, weights: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Weighted upper-tail ES with exact fractional probability mass.

    Unlike top-k averaging, this integrates exactly ``1-alpha`` mass and splits
    the boundary representative when its cluster weight crosses the tail edge.
    """

    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    normalized = _validated_weights(values, weights)
    tail_mass = 1.0 - float(alpha)
    ordered_values, order = values.sort(dim=1, descending=True)
    ordered_weights = normalized.gather(1, order)
    cumulative_before = ordered_weights.cumsum(dim=1) - ordered_weights
    included = (tail_mass - cumulative_before).clamp_min(0).minimum(ordered_weights)
    return (ordered_values * included).sum(dim=1) / tail_mass


@dataclass(frozen=True)
class TailRepresentativeSet:
    scenario_ids: torch.Tensor
    weights: torch.Tensor
    assignment: torch.Tensor
    tail_coverage: float

    def state_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": self.scenario_ids.cpu(),
            "weights": self.weights.cpu(),
            "assignment": self.assignment.cpu(),
            "tail_coverage": float(self.tail_coverage),
        }


def identity_scenario_support(count: int) -> TailRepresentativeSet:
    """Return the canonical uniform support used by exact full fidelity."""

    if count < 1:
        raise ValueError("scenario support count must be positive")
    scenario_ids = torch.arange(count)
    return TailRepresentativeSet(
        scenario_ids=scenario_ids,
        weights=torch.full((count,), 1.0 / count),
        assignment=scenario_ids.clone(),
        tail_coverage=1.0,
    )


def select_tail_preserving_representatives(
    total_cost: torch.Tensor,
    cost_components: torch.Tensor,
    representative_count: int,
    *,
    alpha: float = 0.95,
    preserve_tail: bool = True,
    preserve_extreme_events: bool = True,
) -> TailRepresentativeSet:
    """Select deterministic tail/event anchors plus diverse representatives.

    Selection uses only completed full-fidelity correction trajectories.  The
    worst-cost, failure, CM and unplanned-downtime scenarios are mandatory
    anchors. Remaining representatives use deterministic farthest-point
    sampling. Every full scenario is assigned to its nearest representative;
    cluster cardinalities become probability weights.
    """

    if total_cost.ndim != 2 or cost_components.ndim != 3:
        raise ValueError("costs must be [B,S] and components [B,S,C]")
    if total_cost.shape[:2] != cost_components.shape[:2] or cost_components.shape[2] < 5:
        raise ValueError("component shape is incompatible")
    scenarios = total_cost.shape[1]
    if not 1 <= representative_count <= scenarios:
        raise ValueError("representative_count must lie in [1,S]")
    aggregate = torch.cat(
        (total_cost.mean(0, keepdim=False)[:, None], cost_components.mean(0)), dim=1
    ).float()
    scale = aggregate.std(dim=0, unbiased=False).clamp_min(1e-8)
    features = (aggregate - aggregate.mean(dim=0)) / scale
    # Preserve every full scenario carrying empirical CVaR tail mass whenever
    # K permits it, not merely the single worst scenario.
    tail_count = max(1, int(torch.ceil(torch.tensor((1 - alpha) * scenarios)).item()))
    anchors: list[int] = []
    if preserve_tail:
        anchors = total_cost.mean(0).topk(
            min(tail_count, representative_count)
        ).indices.tolist()
    # Add failure, CM and unplanned-downtime event anchors.
    if preserve_extreme_events:
        for column in (5, 3, 4):
            candidate = int(torch.argmax(aggregate[:, column]).item())
            if candidate not in anchors:
                anchors.append(candidate)
            if len(anchors) == representative_count:
                break
    if not anchors:
        anchors = [0]
    chosen = anchors
    while len(chosen) < representative_count:
        distances = torch.cdist(features, features[chosen]).amin(dim=1)
        distances[torch.tensor(chosen, device=distances.device)] = -1
        chosen.append(int(torch.argmax(distances).item()))
    ids = torch.tensor(chosen, dtype=torch.long, device=total_cost.device)
    assignment = torch.cdist(features, features[ids]).argmin(dim=1)
    # Identical feature rows can otherwise tie to the first representative and
    # leave a selected semantic path with zero probability mass.
    assignment[ids] = torch.arange(representative_count, device=assignment.device)
    weights = torch.bincount(assignment, minlength=representative_count).to(total_cost.dtype)
    weights = weights / weights.sum()
    tail_ids = total_cost.mean(0).topk(tail_count).indices
    covered = torch.isin(tail_ids, ids).float().mean().item()
    return TailRepresentativeSet(ids, weights, assignment, float(covered))


def corrected_rewards(
    low_reward: torch.Tensor, full_reward: torch.Tensor, correction_lambda: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Paired control-variate reward and residual diagnostics."""

    if low_reward.shape != full_reward.shape:
        raise ValueError("paired rewards must have identical shape")
    if not 0 <= correction_lambda <= 1:
        raise ValueError("correction_lambda must be in [0,1]")
    residual = full_reward - low_reward
    corrected = low_reward + correction_lambda * residual
    scale = full_reward.abs().mean().clamp_min(1e-8)
    mse = float(residual.square().mean().detach().cpu())
    mae = float(residual.abs().mean().detach().cpu())
    return corrected, {
        "correction_mse": mse,
        "correction_mae": mae,
        # Backward-compatible diagnostic aliases retained in existing logs.
        "correction_loss": mse,
        "correction_residual_mae": mae,
        "correction_residual_max": float(residual.abs().max().detach().cpu()),
        "correction_relative_residual": float((residual.abs().mean() / scale).detach().cpu()),
    }


class ATMSLScheduler:
    """Checkpointable adaptive scheduler with quality-triggered fallback."""

    FORMAT = "ATMSL scheduler state v1"

    def __init__(self, config: ATMSLConfig, total_updates: int):
        config.validate(total_updates)
        self.config = config
        self.total_updates = int(total_updates)
        self.completed_updates = 0
        self.stage = ATMSLStage.PRODUCTION_WARM_START
        self.forced_full_until = 0
        self.quality_violation_count = 0
        self.correction_count = 0
        self.residual_ewma = 0.0
        self.representatives: TailRepresentativeSet | None = None
        self.full_total_cost_archive: torch.Tensor | None = None
        self.full_cost_component_archive: torch.Tensor | None = None

    def representative_support(self, count: int) -> TailRepresentativeSet:
        """Return tail-preserving support for the requested compact fidelity."""

        if self.full_total_cost_archive is None or self.full_cost_component_archive is None:
            ids = torch.arange(count)
            return TailRepresentativeSet(
                ids, torch.full((count,), 1.0 / count), torch.arange(count), 0.0
            )
        support = select_tail_preserving_representatives(
            self.full_total_cost_archive,
            self.full_cost_component_archive,
            count,
            preserve_tail=self.config.tail_preservation_enabled,
            preserve_extreme_events=self.config.extreme_event_anchors_enabled,
        )
        if self.config.probability_weights_enabled:
            return support
        return TailRepresentativeSet(
            support.scenario_ids,
            torch.full_like(support.weights, 1.0 / count),
            support.assignment,
            support.tail_coverage,
        )

    def plan(self, update: int | None = None) -> ATMSLPlan:
        update = self.completed_updates + 1 if update is None else int(update)
        if not 1 <= update <= self.total_updates:
            raise ValueError("requested ATMSL update is outside the run")
        if self.config.fixed_fidelity_mode == "full":
            return ATMSLPlan(
                update, ATMSLStage.FULL_FIDELITY_CORRECTION, False,
                self.config.full_state_scenarios, self.config.full_reward_scenarios,
                self.config.full_ppo_epochs, False, True, False,
            )
        if self.config.fixed_fidelity_mode == "low":
            return ATMSLPlan(
                update, ATMSLStage.JOINT_LOW_FIDELITY, False,
                self.config.joint_state_scenarios, self.config.joint_reward_scenarios,
                self.config.joint_ppo_epochs, True, False, False,
            )
        final_start = self.total_updates - self.config.final_full_fidelity_updates + 1
        forced = update <= self.forced_full_until
        scheduled_correction = (
            self.config.periodic_correction_enabled
            and update > self.config.warm_start_updates
            and update <= self.config.joint_low_fidelity_until
            and update % self.config.correction_interval == 0
        )
        final = self.config.final_full_fidelity_enabled and update >= final_start
        correction_phase = update > self.config.joint_low_fidelity_until
        if forced or scheduled_correction or correction_phase or final:
            return ATMSLPlan(
                update, ATMSLStage.FULL_FIDELITY_CORRECTION, False,
                self.config.full_state_scenarios, self.config.full_reward_scenarios,
                self.config.full_ppo_epochs, False, True, forced,
            )
        if self.config.production_warm_start_enabled and update <= self.config.warm_start_updates:
            return ATMSLPlan(
                update, ATMSLStage.PRODUCTION_WARM_START, True,
                self.config.warm_state_scenarios, self.config.warm_reward_scenarios,
                self.config.warm_ppo_epochs, True, False, False,
            )
        if self.config.joint_low_fidelity_enabled:
            return ATMSLPlan(
                update, ATMSLStage.JOINT_LOW_FIDELITY, False,
                self.config.joint_state_scenarios, self.config.joint_reward_scenarios,
                self.config.joint_ppo_epochs, True, False, False,
            )
        return ATMSLPlan(
            update, ATMSLStage.FULL_FIDELITY_CORRECTION, False,
            self.config.full_state_scenarios, self.config.full_reward_scenarios,
            self.config.full_ppo_epochs, False, True, False,
        )

    def complete_update(self, plan: ATMSLPlan) -> None:
        if plan.update != self.completed_updates + 1:
            raise ValueError("ATMSL updates must complete sequentially")
        self.completed_updates = plan.update
        self.stage = plan.stage

    def observe_correction(
        self,
        *,
        relative_residual: float,
        tail_coverage: float,
        representatives: TailRepresentativeSet,
        full_total_cost: torch.Tensor | None = None,
        full_cost_components: torch.Tensor | None = None,
    ) -> bool:
        """Update fidelity evidence and return whether fallback was activated."""

        decay = self.config.residual_ewma_decay
        self.residual_ewma = (
            float(relative_residual) if self.correction_count == 0
            else decay * self.residual_ewma + (1 - decay) * float(relative_residual)
        )
        self.correction_count += 1
        self.representatives = representatives
        if full_total_cost is not None or full_cost_components is not None:
            if full_total_cost is None or full_cost_components is None:
                raise ValueError("full correction archive requires totals and components")
            self.full_total_cost_archive = full_total_cost.detach().cpu().clone()
            self.full_cost_component_archive = full_cost_components.detach().cpu().clone()
        degraded = self.config.adaptive_fallback_enabled and (
            self.residual_ewma > self.config.residual_relative_threshold
            or tail_coverage < self.config.tail_coverage_threshold
        )
        self.quality_violation_count = self.quality_violation_count + 1 if degraded else 0
        if self.quality_violation_count >= self.config.degradation_patience:
            self.forced_full_until = max(
                self.forced_full_until,
                self.completed_updates + self.config.fallback_full_updates,
            )
            self.quality_violation_count = 0
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "config": asdict(self.config),
            "total_updates": self.total_updates,
            "completed_updates": self.completed_updates,
            "stage": self.stage.value,
            "forced_full_until": self.forced_full_until,
            "quality_violation_count": self.quality_violation_count,
            "correction_count": self.correction_count,
            "residual_ewma": self.residual_ewma,
            "representatives": (
                None if self.representatives is None else self.representatives.state_dict()
            ),
            "full_total_cost_archive": self.full_total_cost_archive,
            "full_cost_component_archive": self.full_cost_component_archive,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "ATMSLScheduler":
        if payload.get("format") != cls.FORMAT:
            raise ValueError("unsupported ATMSL scheduler state")
        scheduler = cls(ATMSLConfig(**payload["config"]), int(payload["total_updates"]))
        scheduler.completed_updates = int(payload["completed_updates"])
        scheduler.stage = ATMSLStage(payload["stage"])
        scheduler.forced_full_until = int(payload["forced_full_until"])
        scheduler.quality_violation_count = int(payload["quality_violation_count"])
        scheduler.correction_count = int(payload["correction_count"])
        scheduler.residual_ewma = float(payload["residual_ewma"])
        representative = payload.get("representatives")
        if representative is not None:
            scheduler.representatives = TailRepresentativeSet(
                representative["scenario_ids"].long(),
                representative["weights"].float(),
                representative["assignment"].long(),
                float(representative["tail_coverage"]),
            )
        totals = payload.get("full_total_cost_archive")
        components = payload.get("full_cost_component_archive")
        scheduler.full_total_cost_archive = None if totals is None else totals.float()
        scheduler.full_cost_component_archive = None if components is None else components.float()
        return scheduler


V2_1_MIGRATABLE_CONFIG_FIELDS = frozenset({
    "joint_low_fidelity_until",
    "final_full_fidelity_updates",
})


def migrate_scheduler_config_for_suffix(
    scheduler: ATMSLScheduler, new_config: ATMSLConfig
) -> ATMSLScheduler:
    """Apply only the preregistered v2.0 -> v2.1 suffix migration.

    This is deliberately not a general configuration override.  It moves the
    exact full-fidelity suffix from update 801 to update 701, and rejects any
    checkpoint that has already crossed the new boundary.
    """

    new_config.validate(scheduler.total_updates)
    old_values = asdict(scheduler.config)
    new_values = asdict(new_config)
    changed = {key for key in old_values if old_values[key] != new_values[key]}
    if not changed:
        return scheduler
    if changed != V2_1_MIGRATABLE_CONFIG_FIELDS:
        raise ValueError(
            "ATMSL suffix migration only permits joint_low_fidelity_until and "
            f"final_full_fidelity_updates; changed={sorted(changed)}"
        )
    if (
        old_values["joint_low_fidelity_until"] != 800
        or old_values["final_full_fidelity_updates"] != 100
    ):
        raise ValueError("ATMSL suffix migration requires the frozen v2.0 schedule")
    if (
        new_values["joint_low_fidelity_until"] != 700
        or new_values["final_full_fidelity_updates"] != 300
    ):
        raise ValueError("ATMSL suffix migration requires the frozen v2.1 schedule")
    if scheduler.total_updates != 1000:
        raise ValueError("ATMSL v2.1 suffix migration requires a 1000-update budget")
    if scheduler.completed_updates > 700:
        raise ValueError(
            "ATMSL v2.1 cannot be reconstructed from a checkpoint after update 700"
        )
    scheduler.config = new_config
    return scheduler
