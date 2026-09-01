"""Read-only accounting for RAMP routing decisions.

The accumulator consumes tensors already produced by the action forward pass.
It never calls the policy, samples an action, or mutates the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ROUTE_AUDIT_SCHEMA = "ramp_route_audit_v1"


@dataclass
class RouteAuditAccumulator:
    jobs: int
    machines: int
    active_steps: int = 0
    routing_eligible_steps: int = 0
    central_risk_steps: int = 0
    scenario_consensus_steps: int = 0
    forced_cm_steps: int = 0
    empty_safety_set_backoff_steps: int = 0
    pm_authorized_machines: int = 0
    pm_total_machines: int = 0
    pm_legal_machines: int = 0

    def observe(
        self,
        policy_output: Any,
        *,
        empty_safety_set_backoff_row: torch.Tensor,
    ) -> None:
        diagnostics = getattr(policy_output, "route_diagnostics", None)
        if diagnostics is None:
            raise ValueError(
                "route audit requires a RAMP routing policy output"
            )
        required = {
            "active_row",
            "forced_cm_row",
            "routing_eligible_row",
            "central_risk_row",
            "scenario_consensus_row",
            "pm_authorized_machine_count",
            "pm_legal_machine_count",
            "machine_count",
        }
        missing = required - set(diagnostics)
        if missing:
            raise ValueError(
                f"route diagnostics missing fields: {sorted(missing)}"
            )

        active = diagnostics["active_row"].bool()
        eligible = diagnostics["routing_eligible_row"].bool()
        central = diagnostics["central_risk_row"].bool()
        consensus = diagnostics["scenario_consensus_row"].bool()
        forced_cm = diagnostics["forced_cm_row"].bool()
        backoff = empty_safety_set_backoff_row.bool() & active

        if not torch.equal(central | consensus, eligible):
            raise AssertionError(
                "Central-Risk and Scenario-Consensus must partition eligible rows"
            )
        if bool((central & consensus).any()):
            raise AssertionError("route classifications overlap")
        if bool((forced_cm & eligible).any()):
            raise AssertionError("forced-CM rows cannot enter production routing")

        self.active_steps += int(active.sum().item())
        self.routing_eligible_steps += int(eligible.sum().item())
        self.central_risk_steps += int(central.sum().item())
        self.scenario_consensus_steps += int(consensus.sum().item())
        self.forced_cm_steps += int(forced_cm.sum().item())
        self.empty_safety_set_backoff_steps += int(backoff.sum().item())
        self.pm_authorized_machines += int(
            diagnostics["pm_authorized_machine_count"][eligible].sum().item()
        )
        self.pm_total_machines += int(
            diagnostics["machine_count"][eligible].sum().item()
        )
        self.pm_legal_machines += int(
            diagnostics["pm_legal_machine_count"][eligible].sum().item()
        )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        return (
            float(numerator / denominator)
            if denominator > 0
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_AUDIT_SCHEMA,
            "scale": f"{self.jobs}x{self.machines}",
            "jobs": self.jobs,
            "machines": self.machines,
            "counts": {
                "active_steps": self.active_steps,
                "routing_eligible_steps": self.routing_eligible_steps,
                "central_risk_steps": self.central_risk_steps,
                "scenario_consensus_steps": self.scenario_consensus_steps,
                "forced_cm_steps": self.forced_cm_steps,
                "empty_safety_set_backoff_steps": (
                    self.empty_safety_set_backoff_steps
                ),
                "pm_authorized_machines": self.pm_authorized_machines,
                "pm_total_machines": self.pm_total_machines,
                "pm_legal_machines": self.pm_legal_machines,
            },
            "rates": {
                "central_risk": self._rate(
                    self.central_risk_steps,
                    self.routing_eligible_steps,
                ),
                "scenario_consensus": self._rate(
                    self.scenario_consensus_steps,
                    self.routing_eligible_steps,
                ),
                "pm_authorized_machine": self._rate(
                    self.pm_authorized_machines,
                    self.pm_total_machines,
                ),
                "pm_authorized_among_legal": self._rate(
                    self.pm_authorized_machines,
                    self.pm_legal_machines,
                ),
                "forced_cm": self._rate(
                    self.forced_cm_steps,
                    self.active_steps,
                ),
                "empty_safety_set_backoff": self._rate(
                    self.empty_safety_set_backoff_steps,
                    self.active_steps,
                ),
            },
        }
