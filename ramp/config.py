"""Configuration contracts for the RAMP paper experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Public experiment regimes from the manuscript.
PAPER_REGIME_MATRIX: dict[str, dict[str, bool]] = {
    "H0": {
        "health_time": False,
        "degradation": False,
        "processing_noise": False,
        "maintenance": False,
    },
    "H1": {
        "health_time": True,
        "degradation": True,
        "processing_noise": True,
        "maintenance": True,
    },
}

# Fixed hardware admission for the first-paper three-seed protocol.  Each seed
# is pinned to one physical RTX 3090 across Gate100, Gate220, and update 1000.
FORMAL_PHYSICAL_GPUS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class HealthOverlayConfig:
    """Synthetic health-overlay parameters kept outside nominal FJSP data.

    ``failure_level`` is a physical/functional degradation boundary. It is not
    an exact remaining-useful-life quantity.
    """

    failure_level: float = 1.0
    initial_health_low: float = 0.05
    initial_health_high: float = 0.25
    degradation_alpha_low: float = 0.018
    degradation_alpha_high: float = 0.032
    degradation_theta_low: float = 0.025
    degradation_theta_high: float = 0.045
    load_transform_exponent: float = 1.0
    load_sensitivity_low: float = 1.0
    load_sensitivity_high: float = 1.5
    health_time_eta_low: float = 0.15
    health_time_eta_high: float = 0.45
    health_time_gamma_low: float = 1.2
    health_time_gamma_high: float = 2.2
    pm_rho_low: float = 0.10
    pm_rho_high: float = 0.35
    pm_duration_low: float = 4.0
    pm_duration_high: float = 8.0
    pm_cost_low: float = 2.0
    pm_cost_high: float = 5.0
    cm_rho: float = 0.0
    cm_duration_multiplier: float = 2.0
    cm_cost_multiplier: float = 3.0
    maintenance_noise_std: float = 0.005
    processing_cov: float = 0.20

    def validate(self) -> None:
        """Reject overlays that violate degradation or maintenance physics."""

        if self.failure_level <= 0:
            raise ValueError("failure_level must be positive")
        if not 0 <= self.initial_health_low <= self.initial_health_high < self.failure_level:
            raise ValueError("initial health range must lie below failure_level")
        if self.degradation_alpha_low <= 0 or self.degradation_theta_low <= 0:
            raise ValueError("Gamma degradation parameters must be positive")
        if not 0 <= self.pm_rho_low <= self.pm_rho_high < 1:
            raise ValueError("PM restoration factors must be in [0, 1)")
        if not 0 <= self.cm_rho < 1:
            raise ValueError("CM restoration factor must be in [0, 1)")
        if self.cm_duration_multiplier <= 1 or self.cm_cost_multiplier <= 1:
            raise ValueError("CM must be slower and more expensive than PM")
        if self.processing_cov < 0 or self.maintenance_noise_std < 0:
            raise ValueError("noise scales must be nonnegative")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Scalar scenario cost and upper-tail risk weights."""

    lambda_pm: float = 1.0
    lambda_cm: float = 1.0
    lambda_downtime: float = 2.0
    lambda_failure: float = 10.0
    cvar_beta: float = 0.5
    cvar_alpha: float = 0.95

    def validate(self) -> None:
        values = (
            self.lambda_pm,
            self.lambda_cm,
            self.lambda_downtime,
            self.lambda_failure,
            self.cvar_beta,
            self.cvar_alpha,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("objective parameters must be finite")
        if not 0.5 <= self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be in [0.5, 1)")
        if not 0 <= self.cvar_beta <= 10:
            raise ValueError("cvar_beta must be in [0, 10]")
        weights = (
            self.lambda_pm,
            self.lambda_cm,
            self.lambda_downtime,
            self.lambda_failure,
        )
        if min(weights) < 0 or max(weights) > 100:
            raise ValueError("dimensionless objective weights must be in [0, 100]")


@dataclass(frozen=True)
class RAMPConfig:
    """Fixed paper regime and online-environment contract."""

    problem_setting: str = "H1"
    num_scenarios: int = 32
    health_dependent_processing_time: bool = True
    action_conditioned_degradation: bool = True
    exogenous_processing_noise: bool = True
    maintenance_actions: bool = True
    preventive_maintenance_actions: bool = True
    corrective_maintenance_actions: bool = True
    scenario_safety_mask: bool = True
    scenario_recourse: bool = True
    chance_constraint_empty_set_backoff: bool = False
    processing_distribution: str = "lognormal"
    forecast_backend: str = "vectorized"
    scenario_transition_backend: str = "tensorized_selected_action"
    epsilon_use: float = 0.05
    max_pm_per_machine: int = 8
    max_maintenance_decisions: int = 64
    max_decisions: int | None = None
    failure_diagnosis_delay: float = 5.0
    scenario_seed: int = 400
    observed_seed: int | None = None
    degradation_rate_multiplier: float = 1.0
    gamma_shape_multiplier: float = 1.0
    gamma_scale_multiplier: float = 1.0
    initial_health_multiplier: float = 1.0
    cm_cost_ratio_multiplier: float = 1.0
    strict_invalid_actions: bool = True
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    def validate(self) -> None:
        """Validate switches, chance constraint, and explicit truncation limits."""

        self.objective.validate()
        if self.problem_setting not in {"H0", "H1"}:
            raise ValueError("problem_setting must be H0 or H1")
        if self.num_scenarios < 1:
            raise ValueError("num_scenarios must be positive")
        if not 0 <= self.epsilon_use < 1:
            raise ValueError("epsilon_use must be in [0, 1)")
        if self.max_pm_per_machine < 0 or self.max_maintenance_decisions < 0:
            raise ValueError("maintenance decision limits must be nonnegative")
        if self.max_decisions is not None and self.max_decisions < 1:
            raise ValueError("max_decisions must be positive when specified")
        if self.failure_diagnosis_delay < 0:
            raise ValueError("failure_diagnosis_delay must be nonnegative")
        if min(
            self.degradation_rate_multiplier,
            self.gamma_shape_multiplier,
            self.gamma_scale_multiplier,
            self.initial_health_multiplier,
            self.cm_cost_ratio_multiplier,
        ) <= 0:
            raise ValueError("health sensitivity multipliers must be positive")
        if self.processing_distribution not in {"lognormal", "beta", "mixture"}:
            raise ValueError("unsupported processing distribution")
        if self.forecast_backend not in {"vectorized", "scalar_reference"}:
            raise ValueError(
                "forecast_backend must be vectorized or scalar_reference"
            )
        if self.scenario_transition_backend not in {
            "cpu_scalar_kernel",
            "device_scalar_reference",
            "tensorized_selected_action",
        }:
            raise ValueError(
                "scenario_transition_backend must be cpu_scalar_kernel, "
                "device_scalar_reference, or tensorized_selected_action"
            )

    @classmethod
    def from_paper_regime(cls, regime: str, **kwargs: object) -> "RAMPConfig":
        """Construct the manuscript's healthy (H0) or stochastic (H1) regime."""

        name = regime.upper()
        try:
            physics = PAPER_REGIME_MATRIX[name]
        except KeyError as exc:
            raise ValueError("paper regime must be H0 or H1") from exc
        return cls(
            problem_setting=name,
            health_dependent_processing_time=physics["health_time"],
            action_conditioned_degradation=physics["degradation"],
            exogenous_processing_noise=physics["processing_noise"],
            maintenance_actions=physics["maintenance"],
            preventive_maintenance_actions=physics["maintenance"],
            corrective_maintenance_actions=physics["maintenance"],
            **kwargs,
        )
