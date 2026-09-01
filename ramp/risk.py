"""Single production authority for transferable health scheduling risks."""

from __future__ import annotations

import torch


def build_operation_risk(
    expected_delta: torch.Tensor,
    remaining_health: torch.Tensor,
    compatible: torch.Tensor,
    safe: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    maximum_risk: float = 1e6,
) -> torch.Tensor:
    """Minimum compatible expected degradation-budget use per operation."""

    if expected_delta.ndim != 4:
        raise ValueError("expected_delta must be [B,S,N,M]")
    budget = expected_delta / remaining_health[:, :, None, :].clamp_min(epsilon)
    valid = compatible[:, None] & safe
    result = budget.masked_fill(~valid, float("inf")).amin(dim=-1)
    return torch.where(torch.isfinite(result), result, torch.full_like(result, maximum_risk))


def build_machine_risk(
    future_health: torch.Tensor,
    failure_level: torch.Tensor,
    expected_next_degradation: torch.Tensor,
    pm_recovery_value: torch.Tensor,
    scenario_failure_status: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Scenario-specific failure proximity and next-exposure risk."""

    failure = failure_level[:, None] if failure_level.ndim == 2 else failure_level
    normalized = future_health / failure.clamp_min(epsilon)
    remaining = ((failure - future_health) / failure.clamp_min(epsilon)).clamp_min(0)
    next_budget = expected_next_degradation / (failure - future_health).clamp_min(epsilon)
    return (
        normalized
        + 1.0 / (remaining + 0.05)
        + next_budget
        + 10.0 * scenario_failure_status.to(future_health.dtype)
        - pm_recovery_value
    )


def build_pair_risk(
    budget_consumption: torch.Tensor,
    effective_duration: torch.Tensor,
    scenario_survival: torch.Tensor,
    invalid_mask: torch.Tensor,
    *,
    maximum_risk: float = 1e6,
) -> torch.Tensor:
    """Candidate risk using empirical survival probability across scenarios."""

    if not (
        budget_consumption.shape
        == effective_duration.shape
        == scenario_survival.shape
        == invalid_mask.shape
    ):
        raise ValueError("pair risk inputs must have identical [B,S,J,M] shape")
    valid = ~invalid_mask
    count = valid.to(budget_consumption.dtype).sum(dim=1, keepdim=True).clamp_min(1)
    p_safe = (
        scenario_survival.to(budget_consumption.dtype).masked_fill(~valid, 0).sum(dim=1, keepdim=True)
        / count
    )
    risk = budget_consumption * (1.0 + effective_duration.clamp_min(0)) + 1.0 - p_safe
    return risk.masked_fill(invalid_mask, maximum_risk)
