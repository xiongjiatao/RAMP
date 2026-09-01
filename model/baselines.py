"""Independent DAN/SPM-DAN policy backends for paired paper baselines.

These classes deliberately do not subclass or call :class:`RAMPPolicyCore`.  They
retain the baseline method boundary: ordinary DAN aggregation or the original
inducing-point SPM followed by dual operation/machine attention.  Raw-health
variants append only measured degradation/status fields; they do not receive
RAMP structural budgets, survival estimates, recovery values, risk queries,
or substitution edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ramp.state import RAMPEnvState
from model.ramp_core import (
    RAMPPolicyCore,
    RAMPModelConfig,
    RAMPPolicyOutput,
    _masked_softmax,
)
from model.dan_backbone import DualAttentionNetwork, OfficialDANConfig
from model.scenario_encoder_layers import ScenarioProcessingModuleWithoutAggregation
from model.heads import Actor, Critic


class MaskedBaselineScenarioEncoder(nn.Module):
    """Baseline-specific scenario aggregation with explicit entity axes."""

    def __init__(
        self,
        input_dim: int,
        config: RAMPModelConfig,
        *,
        aggregation: str,
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        d = config.embedding_dim
        self.projection = nn.Linear(input_dim, d)
        self.spm = (
            ScenarioProcessingModuleWithoutAggregation(
                dim_in=d,
                dim_out=d,
                num_heads=config.scenario_heads,
                num_inds=config.num_inducing_points,
                ln=True,
            )
            if aggregation == "spm"
            else None
        )

    def forward(
        self, tokens: torch.Tensor, scenario_invalid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``[B,S,E,F]`` as ``[B,E,D]`` without node-axis mixing."""

        if tokens.ndim != 4:
            raise ValueError("baseline scenario tokens must be [B,S,E,F]")
        if scenario_invalid_mask.shape != tokens.shape[:2]:
            raise ValueError("baseline scenario mask must be [B,S]")
        if scenario_invalid_mask.all(dim=1).any():
            raise ValueError("baseline cannot aggregate an all-invalid scenario set")
        b, _, entities, _ = tokens.shape
        if self.aggregation == "central":
            return self.projection(tokens[:, 0])
        projected = self.projection(tokens)
        valid = (~scenario_invalid_mask)[:, :, None, None].to(projected.dtype)
        if self.aggregation == "mean":
            return (projected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        if self.spm is None:
            raise RuntimeError("SPM baseline encoder was not constructed")
        flattened = rearrange(projected, "b s e d -> (b e) s d")
        entity_invalid = rearrange(
            scenario_invalid_mask[:, :, None].expand(-1, -1, entities),
            "b s e -> (b e) s",
        )
        contextual = self.spm(flattened, entity_invalid)
        valid_flat = (~entity_invalid).to(contextual.dtype)
        pooled = (contextual * valid_flat[..., None]).sum(dim=1) / valid_flat.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return rearrange(pooled, "(b e) d -> b e d", b=b, e=entities)


class OfficialDANBackbone(DualAttentionNetwork):
    """Official DAN core with fixed published topology and mapped weights."""

    def __init__(self, config: RAMPModelConfig):
        if config.operation_dim != 10 or config.machine_dim != 8:
            raise ValueError("official DAN requires operation_dim=10 and machine_dim=8")
        super().__init__(OfficialDANConfig(dropout_prob=config.dropout))


class DANBaselinePolicy(nn.Module):
    """Official DAN graph encoder plus explicit H1 adapters.

    ``dan`` uses one deterministic scenario, ``stochastic_dan`` uses an
    ordinary scenario mean, and ``spm_dan`` uses inducing interaction followed
    by mean aggregation. Raw-health variants keep the corresponding DAN/SPM
    backbone and add a factorized production/maintenance decision head.
    """

    _BACKENDS = {
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

    def __init__(self, config: RAMPModelConfig) -> None:
        super().__init__()
        if config.policy_backend not in self._BACKENDS:
            raise ValueError(f"unsupported baseline backend {config.policy_backend}")
        self.config = config
        self.policy_backend = config.policy_backend
        self.raw_health = "raw_health" in self.policy_backend
        self.joint_maintenance = self.policy_backend.endswith("_joint") or self.raw_health
        suffix = "_joint"
        base_backend = (
            self.policy_backend[: -len(suffix)]
            if self.policy_backend.endswith(suffix)
            else self.policy_backend
        )
        if base_backend in {"dan", "dan_raw_health", "raw_health_dan"}:
            aggregation = "central"
        elif base_backend in {"stochastic_dan", "scenario_mean_dan"}:
            aggregation = "mean"
        else:
            aggregation = "spm"
        d = config.embedding_dim
        self.aggregation = aggregation
        self.backbone = (
            DualAttentionNetwork(
                OfficialDANConfig(
                    SAA_attention=True,
                    SAA_attention_dim=d,
                    dropout_prob=config.dropout,
                )
            )
            if aggregation == "spm"
            else OfficialDANBackbone(config)
        )
        self.operation_output_adapter = nn.Linear(8, d)
        self.machine_output_adapter = nn.Linear(8, d)
        self.observed_pair_adapter = nn.Linear(config.pair_dim, d)
        self.spm_pair_output_adapter = (
            nn.Linear(config.pair_dim + d, d) if aggregation == "spm" else None
        )
        scenario_aggregation = "spm" if aggregation == "spm" else "mean"
        self.operation_scenario_adapter = (
            MaskedBaselineScenarioEncoder(
                config.operation_dim, config, aggregation=scenario_aggregation
            )
            if aggregation != "central"
            else None
        )
        self.machine_scenario_adapter = (
            MaskedBaselineScenarioEncoder(
                config.machine_dim, config, aggregation=scenario_aggregation
            )
            if aggregation != "central"
            else None
        )
        self.pair_scenario_adapter = (
            MaskedBaselineScenarioEncoder(
                config.pair_dim, config, aggregation=scenario_aggregation
            )
            if aggregation != "central"
            else None
        )
        self.raw_machine_adapter = nn.Linear(2, d) if self.raw_health else None
        self.raw_pair_adapter = nn.Linear(1, d) if self.raw_health else None
        self.production_actor = Actor(
            config.num_actor_layers, 5 * d, config.actor_hidden_dim, 1
        )
        self.pm_actor = Actor(config.num_actor_layers, 3 * d, config.actor_hidden_dim, 1)
        self.cm_actor = Actor(config.num_actor_layers, 3 * d, config.actor_hidden_dim, 1)
        self.type_gate = Actor(config.num_actor_layers, 2 * d, config.actor_hidden_dim, 3)
        self.critic = Critic(
            config.num_critic_layers, 2 * d, config.critic_hidden_dim, 1
        )

    def _encode(
        self, state: RAMPEnvState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The official-DAN path receives scheduling/availability information
        # but no health magnitude or restoration value.  Health-aware methods
        # add those fields only through their explicitly named external adapter.
        observed_machine = state.observed_machine_tensor.clone()
        observed_machine[..., [2, 3, 7]] = 0.0
        observed_pair = state.observed_pair_tensor.clone()
        observed_pair[..., 1] = observed_pair[..., 0]
        observed_pair[..., [2, 3, 4]] = 0.0
        scenario_mask = state.scenario_invalid_mask_tensor
        b, _, jobs, machines, _ = state.fea_pairs_tensor.shape
        operation_input = state.observed_operation_tensor
        machine_input = observed_machine
        pair_input = observed_pair
        if self.operation_scenario_adapter is not None:
            # Generic scenario baselines receive ordinary stochastic
            # scheduling/time features.  Health budget, recovery, load-derived
            # risk and empirical survival are reserved for RAMP.
            scenario_machine = state.fea_m_tensor.clone()
            scenario_pair_input = state.fea_pairs_tensor.clone()
            if self.raw_health:
                # Raw-health means the measured normalized health and binary
                # availability only.  Remaining budget and PM recovery value
                # are structural quantities and must stay hidden.
                scenario_machine[..., [3, 7]] = 0.0
            else:
                scenario_machine[..., [2, 3, 7]] = 0.0
            scenario_pair_input[..., [2, 3, 4]] = 0.0
            if self.aggregation == "mean":
                valid = (~scenario_mask)[:, :, None, None].to(
                    state.fea_j_tensor.dtype
                )
                count = valid.sum(dim=1).clamp_min(1)
                operation_input = (state.fea_j_tensor * valid).sum(dim=1) / count
                machine_input = (scenario_machine * valid).sum(dim=1) / count
                pair_valid = valid[:, :, None]
                pair_input = (scenario_pair_input * pair_valid).sum(dim=1) / count[:, None]
            else:
                scenario_operation = self.operation_scenario_adapter(
                    state.fea_j_tensor, scenario_mask
                )
                scenario_machine_embedding = self.machine_scenario_adapter(
                    scenario_machine, scenario_mask
                )
                scenario_pair = self.pair_scenario_adapter(
                    rearrange(scenario_pair_input, "b s j m f -> b s (j m) f"),
                    scenario_mask,
                )
                scenario_pair = rearrange(
                    scenario_pair, "b (j m) d -> b j m d", j=jobs, m=machines
                )
                # Match official SPM-DAN's intervention point: concatenate
                # deterministic state and pooled SPM embeddings *before* DAN.
                operation_input = torch.cat(
                    (state.observed_operation_tensor, scenario_operation), dim=-1
                )
                machine_input = torch.cat(
                    (observed_machine, scenario_machine_embedding), dim=-1
                )
                pair_input = torch.cat((observed_pair, scenario_pair), dim=-1)

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
        pair = (
            self.spm_pair_output_adapter(pair_input)
            if self.aggregation == "spm"
            else self.observed_pair_adapter(pair_input)
        )
        if self.raw_health:
            raw_machine = state.observed_machine_tensor[..., [2, 6]]
            machine = machine + self.raw_machine_adapter(raw_machine)
            normalized = state.observed_machine_tensor[..., 2][:, None, :].expand(
                -1, jobs, -1
            )
            # Repeating raw machine health on compatible pairs is the only
            # pair-side health input.  Load and survival are deliberately not
            # included in this control baseline.
            raw_pair = normalized.unsqueeze(-1)
            pair = pair + self.raw_pair_adapter(raw_pair)
        return operation, machine, pair

    def forward(self, state: RAMPEnvState) -> RAMPPolicyOutput:
        operation, machine, pair = self._encode(state)
        global_operation = operation.mean(dim=1)
        global_machine = machine.mean(dim=1)
        b, jobs, machines, d = pair.shape
        candidates = operation.gather(
            1, state.candidate_tensor.unsqueeze(-1).expand(-1, -1, d)
        )
        production_features = torch.cat(
            (
                candidates[:, :, None].expand(-1, -1, machines, -1),
                machine[:, None].expand(-1, jobs, -1, -1),
                global_operation[:, None, None].expand(-1, jobs, machines, -1),
                global_machine[:, None, None].expand(-1, jobs, machines, -1),
                pair,
            ),
            dim=-1,
        )
        production_logits = self.production_actor(production_features).reshape(
            b, jobs * machines
        )
        production_invalid = state.dynamic_pair_mask_tensor.flatten(1)
        production_probs = _masked_softmax(production_logits, production_invalid)
        zeros_machine = torch.zeros(
            (b, machines), device=operation.device, dtype=operation.dtype
        )

        if self.joint_maintenance:
            maintenance_features = torch.cat(
                (
                    machine,
                    global_operation[:, None].expand(-1, machines, -1),
                    global_machine[:, None].expand(-1, machines, -1),
                ),
                dim=-1,
            )
            pm_probs = _masked_softmax(
                self.pm_actor(maintenance_features).squeeze(-1),
                state.pm_mask_tensor[:, 0],
            )
            cm_probs = _masked_softmax(
                self.cm_actor(maintenance_features).squeeze(-1),
                state.cm_mask_tensor[:, 0],
            )
            type_logits = self.type_gate(
                torch.cat((global_operation, global_machine), dim=-1)
            )
            branch_invalid = torch.stack(
                (
                    production_invalid.all(dim=1),
                    state.pm_mask_tensor[:, 0].all(dim=1),
                    state.cm_mask_tensor[:, 0].all(dim=1),
                ),
                dim=1,
            )
            type_probs = _masked_softmax(type_logits, branch_invalid)
        else:
            pm_probs = zeros_machine
            cm_probs = zeros_machine
            type_probs = torch.zeros((b, 3), device=operation.device, dtype=operation.dtype)
            type_probs[:, 0] = (~production_invalid.all(dim=1)).to(operation.dtype)

        action_probs = torch.cat(
            (
                type_probs[:, 0:1] * production_probs,
                type_probs[:, 1:2] * pm_probs,
                type_probs[:, 2:3] * cm_probs,
            ),
            dim=1,
        ).masked_fill(state.action_mask_tensor, 0.0)
        inactive = state.terminated_tensor | state.truncated_tensor
        active = ~inactive
        totals = action_probs.sum(dim=1, keepdim=True)
        if (active & (totals.squeeze(1) <= 0)).any():
            raise RuntimeError("active baseline row has no legal action probability")
        if active.any():
            action_probs[active] /= totals[active]
        if inactive.any():
            action_probs[inactive] = 0.0
            action_probs[inactive, 0] = 1.0
        action_log_probs = torch.where(
            action_probs > 0,
            action_probs.clamp_min(1e-45).log(),
            torch.full_like(action_probs, float("-inf")),
        )
        value = self.critic(
            torch.cat((global_operation, global_machine), dim=-1)
        ).squeeze(-1)
        return RAMPPolicyOutput(
            action_probs=action_probs,
            action_log_probs=action_log_probs,
            value=value,
            type_probs=type_probs,
            production_probs=production_probs,
            pm_probs=pm_probs,
            cm_probs=cm_probs,
        )


class SPMDANBaselinePolicy(DANBaselinePolicy):
    """Named independent SPM-DAN backend for source-level traceability."""

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend not in {
            "spm_dan", "spm_dan_raw_health", "spm_dan_joint",
            "raw_health_spm_dan_joint",
        }:
            raise ValueError("SPMDANBaselinePolicy requires an SPM backend")
        super().__init__(config)


class DANJointPolicy(DANBaselinePolicy):
    """DAN core plus the common H1 production/PM/CM adapter; no scenarios."""

    representation_contract = "official_dan_observed_only"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "dan_joint":
            raise ValueError("DANJointPolicy requires dan_joint")
        super().__init__(config)


class ScenarioMeanDANPolicy(DANBaselinePolicy):
    """Ordinary scenario features, arithmetic mean, DAN, common H1 adapter."""

    representation_contract = "ordinary_scenario_mean_then_dan"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "scenario_mean_dan_joint":
            raise ValueError("ScenarioMeanDANPolicy requires scenario-mean backend")
        super().__init__(config)


class SPMDANJointPolicy(SPMDANBaselinePolicy):
    """Official inducing-point SPM core plus DAN and the common H1 adapter."""

    representation_contract = "ordinary_scenarios_official_spm_then_dan"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "spm_dan_joint":
            raise ValueError("SPMDANJointPolicy requires spm_dan_joint")
        super().__init__(config)


class DANRawHealthPolicy(DANBaselinePolicy):
    """Observed DAN with only raw normalized health and availability appended."""

    representation_contract = "official_dan_plus_observed_raw_health"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "raw_health_dan_joint":
            raise ValueError("DANRawHealthPolicy requires raw_health_dan_joint")
        super().__init__(config)


class SPMDANRawHealthPolicy(SPMDANBaselinePolicy):
    """Generic SPM-DAN with raw scenario health but no structural risk fields."""

    representation_contract = "official_spm_dan_plus_scenario_raw_health"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "raw_health_spm_dan_joint":
            raise ValueError(
                "SPMDANRawHealthPolicy requires raw_health_spm_dan_joint"
            )
        super().__init__(config)


class ProductionOnlyAutoCMPolicy(RAMPPolicyCore):
    """Full RAMP encoder with learned production and mandatory auto-CM.

    PM is never exposed.  Whenever an observed failed machine has a legal CM
    action, the controller deterministically repairs the failed machine with
    the largest remaining compatible-work count.  Otherwise the learned actor
    is restricted to its production conditional distribution.  Consequently
    CM remains fully timed/costed by the common H1 transition authority but is
    not represented as a learned maintenance choice.
    """

    representation_contract = "ramp_production_actor_plus_mandatory_auto_cm"

    def __init__(self, config: RAMPModelConfig) -> None:
        if config.policy_backend != "production_only_auto_cm":
            raise ValueError(
                "ProductionOnlyAutoCMPolicy requires production_only_auto_cm"
            )
        # RAMPPolicyCore owns the representation; its constructor accepts the
        # backend tag because policy_factory is the architecture authority.
        super().__init__(config)

    def forward(self, state: RAMPEnvState) -> RAMPPolicyOutput:
        base = super().forward(state)
        b, jobs, machines = state.dynamic_pair_mask_tensor.shape
        device, dtype = base.action_probs.device, base.action_probs.dtype
        inactive = state.terminated_tensor | state.truncated_tensor
        cm_legal = ~state.cm_mask_tensor[:, 0]
        must_cm = cm_legal.any(dim=1) & ~inactive

        # observed_machine[..., 1] is the normalized count of unfinished
        # operations compatible with each machine.  It is deterministic and
        # contains no future scenario information.
        priority = state.observed_machine_tensor[..., 1].masked_fill(
            ~cm_legal, float("-inf")
        )
        chosen_cm = priority.argmax(dim=1)
        cm_probs = torch.zeros((b, machines), device=device, dtype=dtype)
        if must_cm.any():
            cm_probs[must_cm, chosen_cm[must_cm]] = 1.0

        production_invalid = state.dynamic_pair_mask_tensor.flatten(1)
        production_probs = base.production_probs.masked_fill(production_invalid, 0.0)
        production_total = production_probs.sum(dim=1, keepdim=True)
        production_rows = ~must_cm & ~inactive
        if (production_rows & (production_total.squeeze(1) <= 0)).any():
            raise RuntimeError(
                "production-only auto-CM row has neither mandatory CM nor production"
            )
        if production_rows.any():
            production_probs[production_rows] /= production_total[production_rows]
        production_probs[must_cm | inactive] = 0.0

        pm_probs = torch.zeros_like(cm_probs)
        type_probs = torch.zeros((b, 3), device=device, dtype=dtype)
        type_probs[production_rows, 0] = 1.0
        type_probs[must_cm, 2] = 1.0
        action_probs = torch.cat((production_probs, pm_probs, cm_probs), dim=1)
        action_probs = action_probs.masked_fill(state.action_mask_tensor, 0.0)
        if inactive.any():
            action_probs[inactive] = 0.0
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
            value=base.value,
            type_probs=type_probs,
            production_probs=production_probs,
            pm_probs=pm_probs,
            cm_probs=cm_probs,
        )
