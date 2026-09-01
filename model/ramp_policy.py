"""RAMP risk-aware routing for stochastic flexible job shops.

Production sequencing and maintenance do not need the same uncertainty
representation. RAMP routes ordinary scenario consensus to the production
expert in low-risk states, while the central-risk representation retains
authority over the critic and every risk-triggered decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ramp.state import RAMPEnvState
from model.baselines import OfficialDANBackbone
from model.ramp_core import (
    RAMPPolicyCore,
    RAMPModelConfig,
    RAMPPolicyOutput,
    _masked_softmax,
)
from model.heads import Actor


class RoutineProductionExpert(nn.Module):
    """Scenario-mean DAN used only for production sequencing.

    Health magnitude, recovery value, survival and tail-risk fields are
    deliberately excluded here.  They remain available to the RAMP risk
    authority owned by :class:`RAMPPolicy`.
    """

    def __init__(self, config: RAMPModelConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        self.backbone = OfficialDANBackbone(config)
        self.operation_output_adapter = nn.Linear(8, d)
        self.machine_output_adapter = nn.Linear(8, d)
        self.pair_output_adapter = nn.Linear(config.pair_dim, d)
        self.production_actor = Actor(
            config.num_actor_layers, 5 * d, config.actor_hidden_dim, 1
        )

    @staticmethod
    def _scenario_mean(
        values: torch.Tensor, scenario_invalid: torch.Tensor
    ) -> torch.Tensor:
        valid = (~scenario_invalid).to(values.dtype)
        for _ in range(values.ndim - 2):
            valid = valid.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1.0)
        return (values * valid).sum(dim=1) / count

    def encode_production_features(self, state: RAMPEnvState) -> torch.Tensor:
        """Return the frozen per-pair representation used for production.

        Keeping feature construction behind this method gives post-training
        calibration heads a narrow insertion point.  The ordinary RAMP path
        still calls the same actor on the same tensor, so historical
        checkpoints and logits retain their exact semantics.
        """

        scenario_invalid = state.scenario_invalid_mask_tensor

        # Match the empirically strong Scenario-Mean DAN representation:
        # ordinary scheduling/time scenarios only, without structural health.
        scenario_machine = state.fea_m_tensor.clone()
        scenario_machine[..., [2, 3, 7]] = 0.0
        scenario_pair = state.fea_pairs_tensor.clone()
        scenario_pair[..., [2, 3, 4]] = 0.0

        operation_input = self._scenario_mean(
            state.fea_j_tensor, scenario_invalid
        )
        machine_input = self._scenario_mean(
            scenario_machine, scenario_invalid
        )
        pair_input = self._scenario_mean(
            scenario_pair, scenario_invalid
        )

        official_operation, official_machine, _, _ = self.backbone(
            operation_input,
            state.op_mask_tensor,
            state.candidate_tensor,
            machine_input,
            ~state.mch_mask_tensor,
            state.comp_idx_tensor,
        )
        operation = self.operation_output_adapter(official_operation)
        machine = self.machine_output_adapter(official_machine)
        pair = self.pair_output_adapter(pair_input)

        b, jobs, machines, d = pair.shape
        candidates = operation.gather(
            1, state.candidate_tensor.unsqueeze(-1).expand(-1, -1, d)
        )
        global_operation = operation.mean(dim=1)
        global_machine = machine.mean(dim=1)
        features = torch.cat(
            (
                candidates[:, :, None, :].expand(-1, -1, machines, -1),
                machine[:, None, :, :].expand(-1, jobs, -1, -1),
                global_operation[:, None, None, :].expand(
                    -1, jobs, machines, -1
                ),
                global_machine[:, None, None, :].expand(
                    -1, jobs, machines, -1
                ),
                pair,
            ),
            dim=-1,
        )
        return features

    def base_production_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the original production actor without any calibration."""

        return self.production_actor(features).squeeze(-1).flatten(1)

    def forward(
        self, state: RAMPEnvState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_production_features(state)
        logits = self.base_production_logits(features)
        invalid = state.dynamic_pair_mask_tensor.flatten(1)
        return logits, _masked_softmax(logits, invalid)


class RAMPPolicy(RAMPPolicyCore):
    """Decision-specific routing with safety-witnessed maintenance authority.

    Low risk
        Scenario consensus controls production and PM has exactly zero mass.
    Forecast risk
        Full RAMP production and PM conditionals become active.
    Observed failure
        CM is mandatory and deterministic, preserving the physical CM loop.

    The routing variable is a property of the state, never of the evaluation
    mode.  Greedy and sampling therefore execute the same policy.
    """

    representation_contract = (
        "scenario_consensus_production_plus_central_risk_maintenance_authority"
    )

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "ramp":
            raise ValueError("RAMPPolicy requires policy_backend='ramp'")
        super().__init__(config)
        threshold = float(config.survival_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("survival_threshold must be in (0, 1]")
        self.survival_threshold = threshold
        self.consensus_production = RoutineProductionExpert(config)

    def maintenance_authorization(
        self, state: RAMPEnvState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return authorized PM machines ``[B,M]`` and risk rows ``[B]``.

        A machine receives maintenance authority only when at least one
        currently relevant operation-machine candidate violates the same
        empirical survival threshold used by the Scenario Safety Mask.
        """

        consequences = state.production_candidate_scenarios_tensor
        scenario_valid = ~state.scenario_invalid_mask_tensor[:, :, None, None]
        scenario_valid = scenario_valid.expand(consequences.shape[:4])
        valid = scenario_valid.to(consequences.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        survival = (
            consequences[..., 5].masked_fill(~scenario_valid, 0.0).sum(dim=1)
            / count
        )
        relevant = state.observed_pair_tensor[..., 7] > 0
        unsafe_candidate = relevant & (
            survival < self.survival_threshold
        )
        risky_machine = unsafe_candidate.any(dim=1)

        pm_legal = ~state.pm_mask_tensor[:, 0]
        authorized = risky_machine & pm_legal

        # Feasibility fallback: if safety filtering removes every production
        # action, all legal PM machines must be available to break the dead end.
        production_unavailable = state.dynamic_pair_mask_tensor.flatten(1).all(
            dim=1
        )
        authorized = authorized | (production_unavailable[:, None] & pm_legal)
        return authorized, risky_machine.any(dim=1)

    @staticmethod
    def _renormalize(
        probabilities: torch.Tensor, enabled: torch.Tensor
    ) -> torch.Tensor:
        masked = probabilities.masked_fill(~enabled, 0.0)
        total = masked.sum(dim=1, keepdim=True)
        return torch.where(
            total > 0,
            masked / total.clamp_min(1e-12),
            torch.zeros_like(masked),
        )

    def _route_outputs(
        self,
        state: RAMPEnvState,
        risk_output: RAMPPolicyOutput,
        consensus_production: torch.Tensor,
    ) -> RAMPPolicyOutput:
        """Combine already-computed experts under the immutable RAMP rules."""

        b, _, machines = state.dynamic_pair_mask_tensor.shape
        device = risk_output.action_probs.device
        dtype = risk_output.action_probs.dtype
        inactive = state.terminated_tensor | state.truncated_tensor

        pm_authorized, forecast_risk_row = self.maintenance_authorization(state)
        cm_legal = ~state.cm_mask_tensor[:, 0]
        must_cm = cm_legal.any(dim=1) & ~inactive

        # Risk-free dispatch is insulated from tail-risk representation noise.
        production_probs = torch.where(
            forecast_risk_row[:, None],
            risk_output.production_probs,
            consensus_production,
        )
        production_probs = production_probs.masked_fill(
            state.dynamic_pair_mask_tensor.flatten(1), 0.0
        )
        production_probs = self._renormalize(
            production_probs,
            ~state.dynamic_pair_mask_tensor.flatten(1),
        )

        pm_probs = self._renormalize(
            risk_output.pm_probs, pm_authorized
        )
        has_authorized_pm = pm_authorized.any(dim=1)

        type_probs = torch.zeros((b, 3), device=device, dtype=dtype)
        active_nonfailed = ~inactive & ~must_cm
        joint_rows = active_nonfailed & has_authorized_pm
        production_only_rows = active_nonfailed & ~has_authorized_pm
        type_probs[production_only_rows, 0] = 1.0
        if joint_rows.any():
            production_pm = risk_output.type_probs[joint_rows, :2]
            production_pm = production_pm / production_pm.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            type_probs[joint_rows, :2] = production_pm

        # CM is an observed-failure recovery action, not an exploratory choice.
        priority = state.observed_machine_tensor[..., 1].masked_fill(
            ~cm_legal, float("-inf")
        )
        chosen_cm = priority.argmax(dim=1)
        cm_probs = torch.zeros((b, machines), device=device, dtype=dtype)
        if must_cm.any():
            cm_probs[must_cm, chosen_cm[must_cm]] = 1.0
            type_probs[must_cm, 2] = 1.0

        production_probs[must_cm | inactive] = 0.0
        pm_probs[must_cm | inactive] = 0.0
        action_probs = torch.cat(
            (
                type_probs[:, 0:1] * production_probs,
                type_probs[:, 1:2] * pm_probs,
                type_probs[:, 2:3] * cm_probs,
            ),
            dim=1,
        ).masked_fill(state.action_mask_tensor, 0.0)

        active = ~inactive
        total = action_probs.sum(dim=1, keepdim=True)
        if self.runtime_tensor_validation and (
            active & (total.squeeze(1) <= 0)
        ).any():
            raise RuntimeError("active RAMP row has no authorized action")
        action_probs = torch.where(
            active[:, None],
            action_probs / total.clamp_min(1e-12),
            torch.zeros_like(action_probs),
        )
        if inactive.any():
            action_probs[inactive, 0] = 1.0
            type_probs[inactive, 0] = 1.0

        action_log_probs = torch.where(
            action_probs > 0,
            action_probs.clamp_min(1e-45).log(),
            torch.full_like(action_probs, float("-inf")),
        )
        return RAMPPolicyOutput(
            action_probs=action_probs,
            action_log_probs=action_log_probs,
            value=risk_output.value,
            type_probs=type_probs,
            production_probs=production_probs,
            pm_probs=pm_probs,
            cm_probs=cm_probs,
            route_diagnostics={
                "active_row": active,
                "forced_cm_row": must_cm,
                "routing_eligible_row": active_nonfailed,
                "central_risk_row": active_nonfailed & forecast_risk_row,
                "scenario_consensus_row": (
                    active_nonfailed & ~forecast_risk_row
                ),
                "pm_authorized_machine_count": pm_authorized.sum(dim=1),
                "pm_legal_machine_count": (~state.pm_mask_tensor[:, 0]).sum(
                    dim=1
                ),
                "machine_count": torch.full(
                    (b,), machines, device=device, dtype=torch.long
                ),
            },
        )

    def forward(self, state: RAMPEnvState) -> RAMPPolicyOutput:
        risk_output = super().forward(state)
        _, consensus_production = self.consensus_production(state)
        return self._route_outputs(state, risk_output, consensus_production)
