"""Fixed baseline and ablation registry for unified paper experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .config import RAMPConfig, ObjectiveConfig
from model.ramp_core import RAMPModelConfig


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    category: str
    setting: str
    reporting_group: str = "H1_JOINT"
    policy_family: str = "neural"
    scenario_pooling: str = "dual"
    use_inducing_interaction: bool = True
    use_risk_query: bool = True
    use_structural_health: bool = True
    use_health_substitution_edge: bool = True
    use_observed_deterministic_encoder: bool = True
    action_conditioned_degradation: bool | None = None
    exogenous_processing_noise: bool | None = None
    health_dependent_processing_time: bool | None = None
    preventive_maintenance_actions: bool | None = None
    corrective_maintenance_actions: bool | None = None
    scenario_safety_mask: bool | None = None
    scenario_recourse: bool | None = None
    chance_constraint_empty_set_backoff: bool | None = None
    state_scenario_count: int | None = None
    reward_scenario_count: int | None = None
    cvar_beta: float | None = None
    survival_threshold: float = 0.95


BASELINES: tuple[MethodSpec, ...] = (
    MethodSpec(
        "dan",
        "DAN",
        "baseline",
        "H0",
        reporting_group="HISTORICAL_PRODUCTION_ONLY",
        scenario_pooling="mean",
        use_inducing_interaction=False,
        use_risk_query=False,
        use_structural_health=False,
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "stochastic_dan",
        "stochastic DAN",
        "baseline",
        "H1",
        reporting_group="HISTORICAL_PRODUCTION_ONLY",
        scenario_pooling="mean",
        use_inducing_interaction=False,
        use_risk_query=False,
        use_structural_health=False,
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "spm_dan",
        "SPM-DAN",
        "baseline",
        "H1",
        reporting_group="HISTORICAL_PRODUCTION_ONLY",
        scenario_pooling="mean",
        use_risk_query=False,
        use_structural_health=False,
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "dan_raw_health",
        "DAN with raw health stacking",
        "baseline",
        "H1",
        scenario_pooling="mean",
        use_inducing_interaction=False,
        use_risk_query=False,
        use_structural_health=False,
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "spm_dan_raw_health",
        "SPM-DAN with raw health stacking",
        "baseline",
        "H1",
        scenario_pooling="mean",
        use_risk_query=False,
        use_structural_health=False,
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "threshold_cbm",
        "threshold-based CBM",
        "baseline",
        "H1",
        policy_family="threshold_heuristic",
    ),
    MethodSpec(
        "production_only_auto_cm",
        "production-only RAMP with mandatory auto-CM",
        "baseline",
        "H1",
        preventive_maintenance_actions=False,
        corrective_maintenance_actions=True,
        chance_constraint_empty_set_backoff=True,
    ),
    MethodSpec("ramp_core", "RAMP core", "baseline", "H1"),
    MethodSpec(
        "ramp_core_exact",
        "full-fidelity RAMP with P2 exact backend",
        "baseline",
        "H1",
    ),
    MethodSpec("dan_joint", "DAN", "baseline", "H1",
               scenario_pooling="mean", use_inducing_interaction=False,
               use_risk_query=False, use_structural_health=False,
               use_health_substitution_edge=False),
    MethodSpec("scenario_mean_dan_joint", "SM-DAN", "baseline", "H1",
               scenario_pooling="mean", use_inducing_interaction=False,
               use_risk_query=False, use_structural_health=False,
               use_health_substitution_edge=False),
    MethodSpec("spm_dan_joint", "SPM-DAN", "baseline", "H1",
               scenario_pooling="mean", use_risk_query=False,
               use_structural_health=False, use_health_substitution_edge=False),
    MethodSpec("raw_health_dan_joint", "RH-DAN", "baseline", "H1",
               scenario_pooling="mean", use_inducing_interaction=False,
               use_risk_query=False, use_structural_health=False,
               use_health_substitution_edge=False),
    MethodSpec("raw_health_spm_dan_joint", "RH-SPM-DAN", "baseline", "H1",
               scenario_pooling="mean", use_risk_query=False,
               use_structural_health=False, use_health_substitution_edge=False),
)


PROPOSED_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "ramp",
        "RAMP",
        "proposed",
        "H1",
        survival_threshold=0.95,
    ),
)


ABLATIONS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "ramp_without_proactive_pm",
        "RAMP without proactive preventive maintenance",
        "ablation",
        "H1",
        preventive_maintenance_actions=False,
        survival_threshold=0.95,
    ),
    MethodSpec(
        "without_action_conditioned_degradation",
        "without action-conditioned degradation",
        "ablation",
        "H1",
        action_conditioned_degradation=False,
    ),
    MethodSpec(
        "without_external_processing_noise",
        "without external processing noise",
        "ablation",
        "H1",
        exogenous_processing_noise=False,
    ),
    MethodSpec(
        "without_health_dependent_processing_time",
        "without health-dependent processing time",
        "ablation",
        "H1",
        health_dependent_processing_time=False,
    ),
    MethodSpec(
        "without_preventive_maintenance",
        "without preventive maintenance",
        "ablation",
        "H1",
        preventive_maintenance_actions=False,
    ),
    MethodSpec(
        "without_corrective_maintenance",
        "without corrective maintenance",
        "ablation",
        "H1",
        corrective_maintenance_actions=False,
    ),
    MethodSpec(
        "without_observed_deterministic_encoder",
        "without observed deterministic encoder",
        "ablation",
        "H1",
        use_observed_deterministic_encoder=False,
    ),
    MethodSpec(
        "without_scenario_safety_mask",
        "without scenario safety mask",
        "ablation",
        "H1",
        scenario_safety_mask=False,
    ),
    MethodSpec(
        "without_scenario_recourse",
        "without scenario feasibility recourse",
        "ablation",
        "H1",
        scenario_recourse=False,
    ),
    MethodSpec(
        "without_risk_query",
        "without risk query",
        "ablation",
        "H1",
        use_risk_query=False,
    ),
    MethodSpec(
        "mean_pooling_replacing_dual_query",
        "mean pooling replacing central-risk dual query",
        "ablation",
        "H1",
        scenario_pooling="mean",
    ),
    MethodSpec(
        "without_inducing_interaction",
        "without inducing interaction",
        "ablation",
        "H1",
        use_inducing_interaction=False,
    ),
    MethodSpec(
        "raw_health_replacing_structural_quantities",
        "raw health replacing structural quantities",
        "ablation",
        "H1",
        use_structural_health=False,
    ),
    MethodSpec(
        "without_health_substitution_edge",
        "without health substitution edge",
        "ablation",
        "H1",
        use_health_substitution_edge=False,
    ),
    MethodSpec(
        "mean_replacing_mean_cvar",
        "mean replacing mean+CVaR",
        "ablation",
        "H1",
        cvar_beta=0.0,
    ),
)


ABLATION_CONSUMER_MAP: dict[str, str] = {
    "ramp_without_proactive_pm": "RAMPEnvironmentCore._refresh_action_masks",
    "without_action_conditioned_degradation": "RAMPTransitionKernel.prospective_production",
    "without_external_processing_noise": "RAMPTransitionKernel._duration",
    "without_health_dependent_processing_time": "RAMPTransitionKernel._duration",
    "without_preventive_maintenance": "RAMPEnvironmentCore._refresh_action_masks",
    "without_corrective_maintenance": "RAMPEnvironmentCore._refresh_action_masks",
    "without_observed_deterministic_encoder": "RAMPPolicyCore.encode_*_scenarios/forward",
    "without_scenario_safety_mask": "RAMPEnvironmentCore._refresh_action_masks",
    "without_scenario_recourse": "RAMPTransitionKernel.apply_primary_action_with_recourse",
    "without_risk_query": "ScenarioEncoder._masked_pool",
    "mean_pooling_replacing_dual_query": "ScenarioEncoder._masked_pool",
    "without_inducing_interaction": "ScenarioEncoder.forward",
    "raw_health_replacing_structural_quantities": "RAMPPolicyCore.encode_*_health_scenarios",
    "without_health_substitution_edge": "RAMPPolicyCore.health_substitution_edge",
    "mean_replacing_mean_cvar": "RAMPEnvironmentCore.risk_potential",
}


# This is the paper's conceptual contract, not a count of MethodSpec objects.
# Production-only is an independently registered baseline; scenario-count items
# are sensitivity families whose concrete variants are defined in the formal
# experiment configuration.
PAPER_ABLATION_CONCEPTS: tuple[dict[str, Any], ...] = (
    {"id": "without_action_conditioned_degradation", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_action_conditioned_degradation"]},
    {"id": "without_external_processing_noise", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_external_processing_noise"]},
    {"id": "without_health_dependent_processing_time", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_health_dependent_processing_time"]},
    {"id": "without_preventive_maintenance", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_preventive_maintenance"]},
    {"id": "without_corrective_maintenance", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_corrective_maintenance"]},
    {"id": "production_only", "registry": "BASELINES", "method": "production_only_auto_cm", "consumer": "ProductionOnlyAutoCMPolicy.forward"},
    {"id": "without_risk_query", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_risk_query"]},
    {"id": "mean_pooling_replacing_dual_query", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["mean_pooling_replacing_dual_query"]},
    {"id": "without_inducing_interaction", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_inducing_interaction"]},
    {"id": "raw_health_replacing_structural_quantities", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["raw_health_replacing_structural_quantities"]},
    {"id": "without_health_substitution_edge", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_health_substitution_edge"]},
    {"id": "without_observed_deterministic_encoder", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_observed_deterministic_encoder"]},
    {"id": "without_scenario_safety_mask", "registry": "ABLATIONS", "consumer": ABLATION_CONSUMER_MAP["without_scenario_safety_mask"]},
    {"id": "mean_objective_replacing_mean_cvar", "registry": "ABLATIONS", "method": "mean_replacing_mean_cvar", "consumer": ABLATION_CONSUMER_MAP["mean_replacing_mean_cvar"]},
    {"id": "without_recourse", "registry": "ABLATIONS", "method": "without_scenario_recourse", "consumer": ABLATION_CONSUMER_MAP["without_scenario_recourse"]},
    {"id": "state_scenario_count", "registry": "SENSITIVITY", "variants": (4, 8, 16, 32, 64), "consumer": "RAMPConfig.num_scenarios"},
    {"id": "reward_scenario_count", "registry": "SENSITIVITY", "variants": (16, 32, 64, 128, 256), "consumer": "RAMPScenarioEnv.reward_num_scenarios"},
)


# ATMSL is a training/scenario-computation protocol, not a neural method slug.
# Every entry below names its production ATMSLConfig/scheduler consumer so the
# efficiency table cannot be satisfied by documentation-only method labels.
ATMSL_ABLATION_CONCEPTS: tuple[dict[str, Any], ...] = (
    {"id": "without_production_warm_start", "override": {"production_warm_start_enabled": False}, "consumer": "ATMSLScheduler.plan"},
    {"id": "without_joint_low_fidelity_stage", "override": {"joint_low_fidelity_enabled": False}, "consumer": "ATMSLScheduler.plan"},
    {"id": "without_periodic_full_fidelity_correction", "override": {"periodic_correction_enabled": False}, "consumer": "ATMSLScheduler.plan"},
    {"id": "without_final_full_fidelity_window", "override": {"final_full_fidelity_enabled": False}, "consumer": "ATMSLScheduler.plan"},
    {"id": "uniform_scenario_reduction_replacing_tail_preservation", "override": {"tail_preservation_enabled": False}, "consumer": "ATMSLScheduler.representative_support"},
    {"id": "without_extreme_event_anchors", "override": {"extreme_event_anchors_enabled": False}, "consumer": "select_tail_preserving_representatives"},
    {"id": "without_probability_weights", "override": {"probability_weights_enabled": False}, "consumer": "ATMSLScheduler.representative_support"},
    {"id": "unweighted_cvar_replacing_weighted_cvar", "override": {"weighted_cvar_enabled": False}, "consumer": "RAMPEnvironmentCore._risk_adjusted_total"},
    {"id": "without_paired_semantic_scenario_ids", "override": {"paired_semantic_ids_enabled": False}, "consumer": "train_ramp.main/configure_atmsl_scenario_support"},
    {"id": "without_control_variate_correction", "override": {"control_variate_enabled": False}, "consumer": "corrected_rewards"},
    {"id": "without_adaptive_fallback", "override": {"adaptive_fallback_enabled": False}, "consumer": "ATMSLScheduler.observe_correction"},
    {"id": "fixed_low_fidelity_throughout", "override": {"fixed_fidelity_mode": "low"}, "consumer": "ATMSLScheduler.plan"},
    {"id": "fixed_full_fidelity_throughout", "override": {"fixed_fidelity_mode": "full"}, "consumer": "ATMSLScheduler.plan"},
    {"id": "correction_interval", "variants": (10, 20, 40), "field": "correction_interval", "consumer": "ATMSLScheduler.plan"},
    {"id": "representative_reward_scenario_count", "variants": (8, 16, 32), "field": "joint_reward_scenarios", "consumer": "ATMSLScheduler.representative_support"},
    {"id": "residual_threshold", "variants": (0.10, 0.15, 0.20), "field": "residual_relative_threshold", "consumer": "ATMSLScheduler.observe_correction"},
    {"id": "tail_coverage_threshold", "variants": (0.70, 0.80, 0.90), "field": "tail_coverage_threshold", "consumer": "ATMSLScheduler.observe_correction"},
)


METHODS: dict[str, MethodSpec] = {
    method.slug: method
    for method in (*BASELINES, *PROPOSED_METHODS, *ABLATIONS)
}


def get_method(slug: str) -> MethodSpec:
    try:
        return METHODS[slug]
    except KeyError as exc:
        raise ValueError(f"unknown experiment method: {slug}") from exc


def configure_environment(
    method: MethodSpec,
    *,
    num_scenarios: int,
    seed: int,
    epsilon_use: float,
    processing_distribution: str = "lognormal",
    degradation_rate_multiplier: float = 1.0,
    gamma_shape_multiplier: float = 1.0,
    gamma_scale_multiplier: float = 1.0,
    initial_health_multiplier: float = 1.0,
    cm_cost_ratio_multiplier: float = 1.0,
) -> RAMPConfig:
    """Construct an auditable environment configuration for one method."""

    config = RAMPConfig.from_paper_regime(
        method.setting,
        num_scenarios=(method.state_scenario_count or num_scenarios),
        scenario_seed=seed,
        epsilon_use=epsilon_use,
        processing_distribution=processing_distribution,
        degradation_rate_multiplier=degradation_rate_multiplier,
        gamma_shape_multiplier=gamma_shape_multiplier,
        gamma_scale_multiplier=gamma_scale_multiplier,
        initial_health_multiplier=initial_health_multiplier,
        cm_cost_ratio_multiplier=cm_cost_ratio_multiplier,
    )
    overrides = {
        name: getattr(method, name)
        for name in (
            "action_conditioned_degradation",
            "exogenous_processing_noise",
            "health_dependent_processing_time",
            "preventive_maintenance_actions",
            "corrective_maintenance_actions",
            "scenario_safety_mask",
            "scenario_recourse",
            "chance_constraint_empty_set_backoff",
        )
        if getattr(method, name) is not None
    }
    if method.cvar_beta is not None:
        overrides["objective"] = replace(
            config.objective, cvar_beta=method.cvar_beta
        )
    return replace(config, **overrides)


def configure_model(
    method: MethodSpec,
    *,
    smoke: bool,
) -> RAMPModelConfig:
    """Construct the neural architecture variant without relabeling methods."""

    independent_baselines = {
        "dan",
        "stochastic_dan",
        "spm_dan",
        "dan_raw_health",
        "spm_dan_raw_health",
        "dan_joint",
        "scenario_mean_dan_joint",
        "spm_dan_joint",
        "raw_health_dan_joint",
        "raw_health_spm_dan_joint",
    }
    if method.slug == "production_only_auto_cm":
        policy_backend = "production_only_auto_cm"
    elif method.slug in {
        "ramp",
        "ramp_without_proactive_pm",
    }:
        policy_backend = "ramp"
    elif method.slug in {"ramp_core", "ramp_core_exact"}:
        policy_backend = "ramp_core"
    elif method.slug in independent_baselines:
        policy_backend = method.slug
    else:
        policy_backend = "ramp_core"
    return RAMPModelConfig(
        policy_backend=policy_backend,
        scenario_dim=8 if smoke else 32,
        embedding_dim=8 if smoke else 32,
        scenario_heads=1 if smoke else 4,
        graph_heads=1 if smoke else 4,
        num_inducing_points=2 if smoke else 16,
        scenario_pooling=method.scenario_pooling,
        use_inducing_interaction=method.use_inducing_interaction,
        use_risk_query=method.use_risk_query,
        use_structural_health=method.use_structural_health,
        use_health_substitution_edge=method.use_health_substitution_edge,
        use_observed_deterministic_encoder=method.use_observed_deterministic_encoder,
        survival_threshold=method.survival_threshold,
    )
