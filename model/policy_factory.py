"""Policy construction boundary shared by training, evaluation, and resume."""

from __future__ import annotations

import torch.nn as nn

from model.baselines import (
    DANBaselinePolicy,
    DANJointPolicy,
    DANRawHealthPolicy,
    ScenarioMeanDANPolicy,
    ProductionOnlyAutoCMPolicy,
    SPMDANBaselinePolicy,
    SPMDANJointPolicy,
    SPMDANRawHealthPolicy,
)
from model.ramp_core import RAMPPolicyCore, RAMPModelConfig
from model.ramp_policy import RAMPPolicy


def build_policy(config: RAMPModelConfig) -> nn.Module:
    """Build the declared method family without relabeling RAMPPolicyCore toggles."""

    if config.policy_backend == "ramp_core":
        return RAMPPolicyCore(config)
    if config.policy_backend == "production_only_auto_cm":
        return ProductionOnlyAutoCMPolicy(config)
    if config.policy_backend == "ramp":
        return RAMPPolicy(config)
    if config.policy_backend == "dan_joint":
        return DANJointPolicy(config)
    if config.policy_backend == "scenario_mean_dan_joint":
        return ScenarioMeanDANPolicy(config)
    if config.policy_backend == "spm_dan_joint":
        return SPMDANJointPolicy(config)
    if config.policy_backend == "raw_health_dan_joint":
        return DANRawHealthPolicy(config)
    if config.policy_backend == "raw_health_spm_dan_joint":
        return SPMDANRawHealthPolicy(config)
    if config.policy_backend in {
        "spm_dan", "spm_dan_raw_health",
    }:
        return SPMDANBaselinePolicy(config)
    if config.policy_backend in {
        "dan",
        "stochastic_dan",
        "dan_raw_health",
    }:
        return DANBaselinePolicy(config)
    raise ValueError(f"unknown policy backend {config.policy_backend}")
