"""RAMP policy core: scenario encoder, observed DAN, and factorized heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from ramp.state import RAMPEnvState
from ramp.risk import (
    build_operation_risk as _build_operation_risk,
    build_pair_risk as _build_pair_risk,
)
from model.attention import MultiHeadMchAttnBlock, MultiHeadOpAttnBlock
from model.heads import Actor, Critic


def flatten_entity_scenarios(tokens: torch.Tensor) -> torch.Tensor:
    """Move entity outside its scenario set: ``[B,S,E,F] -> [B*E,S,F]``."""

    if tokens.ndim != 4:
        raise ValueError("scenario tokens must have axes [B,S,E,F]")
    return rearrange(tokens, "b s e f -> (b e) s f")


@dataclass(frozen=True)
class RAMPModelConfig:
    policy_backend: str = "ramp"
    operation_dim: int = 10
    machine_dim: int = 8
    pair_dim: int = 8
    health_machine_dim: int = 5
    health_pair_dim: int = 5
    scenario_dim: int = 32
    embedding_dim: int = 32
    scenario_heads: int = 4
    graph_heads: int = 4
    num_inducing_points: int = 16
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    num_actor_layers: int = 3
    num_critic_layers: int = 3
    dropout: float = 0.0
    risk_logit_coefficient: float = 2.0
    substitution_degradation_weight: float = 1.0
    substitution_tail_weight: float = 1.0
    substitution_safety_weight: float = 1.0
    substitution_maintenance_weight: float = 1.0
    use_inducing_interaction: bool = True
    use_risk_query: bool = True
    use_structural_health: bool = True
    use_health_substitution_edge: bool = True
    use_observed_deterministic_encoder: bool = True
    scenario_pooling: str = "dual"
    survival_threshold: float = 0.95
    runtime_tensor_validation: bool = True


@dataclass
class RAMPPolicyOutput:
    action_probs: torch.Tensor
    action_log_probs: torch.Tensor
    value: torch.Tensor
    type_probs: torch.Tensor
    production_probs: torch.Tensor
    pm_probs: torch.Tensor
    cm_probs: torch.Tensor
    # Read-only tensors produced by policies that expose route provenance.
    # The field is intentionally optional so historical checkpoints and every
    # non-RAMP backend retain their exact public interface and behavior.
    route_diagnostics: dict[str, torch.Tensor] | None = None

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    def entropy(self) -> torch.Tensor:
        """Factorized entropy H(type)+E_type[H(entity|type)]."""

        def categorical_entropy(probabilities: torch.Tensor) -> torch.Tensor:
            safe_log = probabilities.clamp_min(1e-12).log()
            return -(probabilities * safe_log).sum(dim=1)

        type_entropy = categorical_entropy(self.type_probs)
        conditional = torch.stack(
            (
                categorical_entropy(self.production_probs),
                categorical_entropy(self.pm_probs),
                categorical_entropy(self.cm_probs),
            ),
            dim=1,
        )
        return type_entropy + (self.type_probs * conditional).sum(dim=1)


class ScenarioEncoder(nn.Module):
    """Masked inducing interaction with central and tail-risk queries.

    The linear input mapping is only dimensional alignment. The methodological
    structure is the inducing interaction plus two masked scenario queries.
    """

    def __init__(self, input_dim: int, config: RAMPModelConfig):
        super().__init__()
        d = config.scenario_dim
        if d % config.scenario_heads != 0:
            raise ValueError("scenario_dim must be divisible by scenario_heads")
        self.input_projection = nn.Linear(input_dim, d)
        self.inducing_points = nn.Parameter(
            torch.empty(1, config.num_inducing_points, d)
        )
        self.inducing_attention = nn.MultiheadAttention(
            d, config.scenario_heads, dropout=config.dropout, batch_first=True
        )
        self.scenario_attention = nn.MultiheadAttention(
            d, config.scenario_heads, dropout=config.dropout, batch_first=True
        )
        self.inducing_norm = nn.LayerNorm(d)
        self.scenario_norm = nn.LayerNorm(d)
        self.feed_forward = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        self.output_norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.central_query = nn.Parameter(torch.empty(d))
        self.risk_query = nn.Parameter(torch.empty(d))
        self.risk_logit_coefficient = float(config.risk_logit_coefficient)
        self.use_inducing_interaction = bool(config.use_inducing_interaction)
        self.use_risk_query = bool(config.use_risk_query)
        self.scenario_pooling = str(config.scenario_pooling)
        self.runtime_tensor_validation = bool(config.runtime_tensor_validation)
        if self.scenario_pooling not in {"dual", "mean"}:
            raise ValueError("scenario_pooling must be dual or mean")
        nn.init.normal_(self.central_query, std=d ** -0.5)
        nn.init.normal_(self.risk_query, std=d ** -0.5)
        nn.init.normal_(self.inducing_points, std=d ** -0.5)

    def _validate_inputs(
        self,
        tokens: torch.Tensor,
        risk: torch.Tensor,
        scenario_invalid_mask: torch.Tensor,
    ) -> None:
        if tokens.ndim != 4 or risk.shape != tokens.shape[:3]:
            raise ValueError("tokens must be [B,S,E,F] and risk [B,S,E]")
        if scenario_invalid_mask.shape != risk.shape:
            raise ValueError("scenario_invalid_mask must be [B,S,E]")
        if scenario_invalid_mask.dtype != torch.bool:
            raise TypeError("scenario_invalid_mask must be boolean with True=invalid")
        if self.runtime_tensor_validation and scenario_invalid_mask.all(dim=1).any():
            raise ValueError("all scenarios invalid for at least one scenario set")

    def _interact(
        self, tokens: torch.Tensor, invalid_mask: torch.Tensor
    ) -> torch.Tensor:
        invalid_values = invalid_mask.unsqueeze(-1)
        embedded = self.input_projection(tokens).masked_fill(invalid_values, 0.0)
        if not self.use_inducing_interaction:
            output = self.output_norm(embedded + self.feed_forward(embedded))
            return output.masked_fill(invalid_values, 0.0)
        inducing = self.inducing_points.expand(tokens.shape[0], -1, -1)
        induced, _ = self.inducing_attention(
            inducing,
            embedded,
            embedded,
            key_padding_mask=invalid_mask,
            need_weights=False,
        )
        induced = self.inducing_norm(inducing + induced)
        contextual, _ = self.scenario_attention(
            embedded, induced, induced, need_weights=False
        )
        # MultiheadAttention has a key-padding mask but no query-padding mask.
        # Explicitly zero the invalid scenario queries after inducing-to-scenario
        # attention so they cannot survive through residual/FFN paths.
        contextual = self.scenario_norm(embedded + contextual).masked_fill(
            invalid_values, 0.0
        )
        output = self.output_norm(contextual + self.feed_forward(contextual))
        return output.masked_fill(invalid_values, 0.0)

    def _weights(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        risk: torch.Tensor | None,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        keys = self.key(tokens)
        logits = torch.einsum("esd,d->es", keys, query) / math.sqrt(tokens.shape[-1])
        if risk is not None:
            valid = ~invalid_mask
            count = valid.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (risk.masked_fill(invalid_mask, 0.0).sum(dim=1, keepdim=True) / count)
            centered = (risk - mean).masked_fill(invalid_mask, 0.0)
            variance = (centered.square().sum(dim=1, keepdim=True) / count).clamp_min(1e-12)
            logits = logits + self.risk_logit_coefficient * centered / variance.sqrt()
        logits = logits.masked_fill(invalid_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = weights.masked_fill(invalid_mask, 0.0)
        if self.runtime_tensor_validation and not torch.isfinite(weights).all():
            raise FloatingPointError("masked scenario softmax produced non-finite weights")
        return weights

    def _pool(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        risk: torch.Tensor | None,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = self.value(tokens)
        weights = self._weights(tokens, query, risk, invalid_mask)
        return torch.einsum("es,esd->ed", weights, values)

    def scenario_weights(
        self,
        tokens: torch.Tensor,
        risk: torch.Tensor,
        scenario_invalid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return central/risk weights as ``[B,E,S]`` for diagnostics/tests."""

        self._validate_inputs(tokens, risk, scenario_invalid_mask)
        b, _, entities, _ = tokens.shape
        flattened = flatten_entity_scenarios(tokens)
        flat_risk = rearrange(risk, "b s e -> (b e) s")
        flat_invalid = rearrange(
            scenario_invalid_mask, "b s e -> (b e) s"
        )
        interacted = self._interact(flattened, flat_invalid)
        central = self._weights(
            interacted, self.central_query, None, flat_invalid
        )
        tail = self._weights(
            interacted, self.risk_query, flat_risk, flat_invalid
        )
        return (
            rearrange(central, "(b e) s -> b e s", b=b, e=entities),
            rearrange(tail, "(b e) s -> b e s", b=b, e=entities),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        risk: torch.Tensor,
        scenario_invalid_mask: torch.Tensor,
        entity_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate ``[B,S,E,F]`` to ``[B,E,2D]`` without axis mixing.

        A real entity with no valid scenario is a data error.  A padded entity
        may have every scenario invalid and deterministically returns a zero
        embedding; its entity mask remains responsible for downstream removal.
        """

        if entity_padding_mask is None:
            self._validate_inputs(tokens, risk, scenario_invalid_mask)
        else:
            if entity_padding_mask.shape != (tokens.shape[0], tokens.shape[2]):
                raise ValueError("entity_padding_mask must be [B,E]")
            if entity_padding_mask.dtype != torch.bool:
                raise TypeError("entity_padding_mask must be boolean")
            real_all_invalid = scenario_invalid_mask.all(dim=1) & ~entity_padding_mask
            if self.runtime_tensor_validation and real_all_invalid.any():
                raise ValueError("all scenarios invalid for at least one real scenario set")
            # Substitute one harmless valid token only to keep attention
            # numerically defined; padded outputs are zeroed below.
            scenario_invalid_mask = scenario_invalid_mask.clone()
            scenario_invalid_mask[:, 0] &= ~entity_padding_mask
            self._validate_inputs(tokens, risk, scenario_invalid_mask)
        b, _, entities, _ = tokens.shape
        scenario_tokens = flatten_entity_scenarios(tokens)
        scenario_risk = rearrange(risk, "b s e -> (b e) s")
        invalid = rearrange(scenario_invalid_mask, "b s e -> (b e) s")
        interacted = self._interact(scenario_tokens, invalid)
        if self.scenario_pooling == "mean":
            valid = (~invalid).to(interacted.dtype)
            mean = (interacted * valid[..., None]).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            mean = self.value(mean)
            result = rearrange(
                torch.cat((mean, mean), dim=-1),
                "(b e) d -> b e d",
                b=b,
                e=entities,
            )
            return result if entity_padding_mask is None else result.masked_fill(
                entity_padding_mask[..., None], 0.0
            )
        central = self._pool(interacted, self.central_query, None, invalid)
        tail = (
            self._pool(interacted, self.risk_query, scenario_risk, invalid)
            if self.use_risk_query
            else central
        )
        result = rearrange(
            torch.cat((central, tail), dim=-1),
            "(b e) d -> b e d",
            b=b,
            e=entities,
        )
        return result if entity_padding_mask is None else result.masked_fill(
            entity_padding_mask[..., None], 0.0
        )


class ProductionCandidateEncoder(ScenarioEncoder):
    """Central/risk processor for every production candidate before sampling."""


class PMCandidateEncoder(ScenarioEncoder):
    """Central/risk processor for every preventive-maintenance candidate."""


class CMCandidateEncoder(ScenarioEncoder):
    """Central/risk processor for every corrective-maintenance candidate."""


class ObservedOperationEncoder(nn.Module):
    """Encode scenario-free observed operation state."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.ELU())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class ObservedMachineEncoder(ObservedOperationEncoder):
    """Encode scenario-free observed machine health, status, and time."""


class ObservedPairEncoder(ObservedOperationEncoder):
    """Encode scenario-free compatibility, load, and pair timing."""


class ObservedGlobalEncoder(ObservedOperationEncoder):
    """Encode the scenario-free global scheduling anchor."""


def build_operation_risk(
    expected_delta: torch.Tensor,
    remaining_health: torch.Tensor,
    compatible: torch.Tensor,
    safe: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    maximum_risk: float = 1e6,
) -> torch.Tensor:
    """Minimum compatible expected budget consumption for each operation."""

    return _build_operation_risk(
        expected_delta, remaining_health, compatible, safe,
        epsilon=epsilon, maximum_risk=maximum_risk,
    )


def build_pair_risk(
    budget_consumption: torch.Tensor,
    effective_duration: torch.Tensor,
    scenario_survival: torch.Tensor,
    health_safe_invalid_mask: torch.Tensor,
    *,
    maximum_risk: float = 1e6,
) -> torch.Tensor:
    """Pair risk from budget use, duration, survival, and hard safety."""

    return _build_pair_risk(
        budget_consumption, effective_duration, scenario_survival,
        health_safe_invalid_mask, maximum_risk=maximum_risk,
    )


# Short aliases kept for callers that use the explicit ``compute_*`` wording.
compute_operation_specific_risk = build_operation_risk
compute_pair_risk = build_pair_risk


def _masked_softmax(logits: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != invalid_mask.shape:
        raise ValueError("logits and invalid mask must have identical shapes")
    has_valid = ~invalid_mask.all(dim=1)
    # Give an all-invalid row one temporary finite entry so softmax remains
    # defined without a device-to-host synchronization.  The original mask is
    # restored after softmax, hence such a branch still returns exact zeros.
    safe_invalid = invalid_mask.clone()
    safe_invalid[~has_valid, 0] = False
    probabilities = torch.softmax(
        logits.masked_fill(safe_invalid, float("-inf")), dim=1
    )
    return probabilities.masked_fill(invalid_mask, 0.0)


class RAMPPolicyCore(nn.Module):
    """RAMP with factorized production/PM/CM actor branches."""

    def __init__(self, config: RAMPModelConfig | object | None = None):
        super().__init__()
        if config is None:
            cfg = RAMPModelConfig()
        elif isinstance(config, RAMPModelConfig):
            cfg = config
        else:
            cfg = RAMPModelConfig(
                scenario_dim=int(getattr(config, "SAA_attention_dim", 32)),
                embedding_dim=int(getattr(config, "ramp_embedding_dim", 32)),
                actor_hidden_dim=int(getattr(config, "hidden_dim_actor", 128)),
                critic_hidden_dim=int(getattr(config, "hidden_dim_critic", 128)),
                num_actor_layers=int(getattr(config, "num_mlp_layers_actor", 3)),
                num_critic_layers=int(getattr(config, "num_mlp_layers_critic", 3)),
                dropout=float(getattr(config, "dropout_prob", 0.0)),
            )
        self.config = cfg
        self.use_structural_health = bool(cfg.use_structural_health)
        self.use_health_substitution_edge = bool(cfg.use_health_substitution_edge)
        self.use_observed_deterministic_encoder = bool(
            cfg.use_observed_deterministic_encoder
        )
        self.runtime_tensor_validation = bool(cfg.runtime_tensor_validation)
        self.joint_maintenance = True
        d = cfg.embedding_dim
        self.operation_scenario_processor = ScenarioEncoder(
            cfg.operation_dim, cfg
        )
        self.machine_health_scenario_processor = ScenarioEncoder(
            cfg.machine_dim + cfg.health_machine_dim, cfg
        )
        self.pair_health_scenario_processor = ScenarioEncoder(
            cfg.pair_dim + cfg.health_pair_dim, cfg
        )
        self.production_candidate_scenario_processor = ProductionCandidateEncoder(9, cfg)
        self.pm_candidate_scenario_processor = PMCandidateEncoder(7, cfg)
        self.cm_candidate_scenario_processor = CMCandidateEncoder(6, cfg)
        aggregated_dim = 2 * cfg.scenario_dim
        self.operation_projection = nn.Linear(aggregated_dim, d)
        self.machine_projection = nn.Linear(aggregated_dim, d)
        self.pair_projection = nn.Linear(aggregated_dim, d)
        self.production_candidate_projection = nn.Linear(aggregated_dim, d)
        self.pm_candidate_projection = nn.Linear(aggregated_dim, d)
        self.cm_candidate_projection = nn.Linear(aggregated_dim, d)
        self.observed_operation_encoder = ObservedOperationEncoder(cfg.operation_dim, d)
        self.observed_machine_encoder = ObservedMachineEncoder(cfg.machine_dim, d)
        self.observed_pair_encoder = ObservedPairEncoder(cfg.pair_dim, d)
        self.observed_global_encoder = ObservedGlobalEncoder(5, d)
        self.operation_fusion_gate = nn.Parameter(torch.zeros(d))
        self.machine_fusion_gate = nn.Parameter(torch.zeros(d))
        self.pair_fusion_gate = nn.Parameter(torch.zeros(d))
        self.operation_attention = MultiHeadOpAttnBlock(
            input_dim=d,
            output_dim=d,
            dropout_prob=cfg.dropout,
            num_heads=cfg.graph_heads,
            activation=nn.ELU(),
            concat=False,
        )
        self.machine_attention = MultiHeadMchAttnBlock(
            node_input_dim=d,
            edge_input_dim=d + 1,
            output_dim=d,
            dropout_prob=cfg.dropout,
            num_heads=cfg.graph_heads,
            activation=nn.ELU(),
            concat=False,
        )
        self.production_actor = Actor(
            cfg.num_actor_layers, 5 * d, cfg.actor_hidden_dim, 1
        )
        self.pm_actor = Actor(cfg.num_actor_layers, 4 * d, cfg.actor_hidden_dim, 1)
        self.cm_actor = Actor(cfg.num_actor_layers, 4 * d, cfg.actor_hidden_dim, 1)
        self.type_gate = Actor(cfg.num_actor_layers, 3 * d, cfg.actor_hidden_dim, 3)
        self.critic = Critic(
            cfg.num_critic_layers, 3 * d, cfg.critic_hidden_dim, 1
        )

    def encode_operation_scenarios(self, state: RAMPEnvState) -> torch.Tensor:
        b, s, n, _ = state.fea_j_tensor.shape
        invalid = state.scenario_invalid_mask_tensor[:, :, None].expand(b, s, n)
        operation_risk = build_operation_risk(
            state.all_expected_delta_tensor,
            (state.failure_level_tensor[:, None] - state.scenario_current_health_tensor).clamp_min(1e-8),
            state.compatibility_tensor,
            state.all_survival_tensor,
        )
        aggregated = self.operation_scenario_processor(
            state.fea_j_tensor, operation_risk, invalid
        )
        scenario = self.operation_projection(aggregated)
        observed = self.observed_operation_encoder(state.observed_operation_tensor)
        if not self.use_observed_deterministic_encoder:
            observed = torch.zeros_like(observed)
        return observed + torch.sigmoid(self.operation_fusion_gate) * scenario

    def encode_machine_health_scenarios(self, state: RAMPEnvState) -> torch.Tensor:
        health = state.health_m_tensor
        if not self.use_structural_health:
            raw = torch.zeros_like(health)
            raw[..., 0] = health[..., 1]
            raw[..., 1] = health[..., 3]
            health = raw
        tokens = torch.cat((state.fea_m_tensor, health), dim=-1)
        b, s, machines, _ = tokens.shape
        invalid = state.scenario_invalid_mask_tensor[:, :, None].expand(
            b, s, machines
        )
        scenario = self.machine_projection(
            self.machine_health_scenario_processor(
                tokens, state.machine_risk_tensor, invalid
            )
        )
        observed = self.observed_machine_encoder(state.observed_machine_tensor)
        if not self.use_observed_deterministic_encoder:
            observed = torch.zeros_like(observed)
        return observed + torch.sigmoid(self.machine_fusion_gate) * scenario

    def encode_pair_health_scenarios(self, state: RAMPEnvState) -> torch.Tensor:
        b, s, j, m, _ = state.fea_pairs_tensor.shape
        health = state.health_pair_tensor
        if not self.use_structural_health:
            raw = torch.zeros_like(health)
            raw[..., 0] = state.health_m_tensor[..., 1][:, :, None, :]
            raw[..., 1] = state.fea_pairs_tensor[..., 4]
            health = raw
        tokens = torch.cat((state.fea_pairs_tensor, health), dim=-1)
        risk = state.pair_risk_tensor
        flat_tokens = rearrange(tokens, "b s j m f -> b s (j m) f")
        flat_risk = rearrange(risk, "b s j m -> b s (j m)")
        invalid = state.health_pair_mask_tensor | state.scenario_invalid_mask_tensor[
            :, :, None, None
        ]
        flat_invalid = rearrange(invalid, "b s j m -> b s (j m)")
        # Pairs that are invalid in every scenario have no scenario set to
        # aggregate. They remain zero and are hard-masked by the decoder.
        entity_valid = (~flat_invalid).any(dim=1)
        flat_entity_valid = entity_valid.flatten()
        output = torch.zeros(
            (b * j * m, 2 * self.config.scenario_dim),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        # Trusted fast-path mode disables expensive assertions, not this
        # structural empty-set guard. MultiheadAttention cannot consume a
        # [0, S, F] batch, so an empty entity set keeps the exact zeros above.
        if flat_entity_valid.any():
            entity_tokens = rearrange(
                flat_tokens, "b s e f -> (b e) s f"
            )[flat_entity_valid]
            entity_risk = rearrange(flat_risk, "b s e -> (b e) s")[
                flat_entity_valid
            ]
            entity_invalid = rearrange(
                flat_invalid, "b s e -> (b e) s"
            )[flat_entity_valid]
            encoded_valid = self.pair_health_scenario_processor(
                entity_tokens[:, :, None, :],
                entity_risk[:, :, None],
                entity_invalid[:, :, None],
            )[:, 0]
            output[flat_entity_valid] = encoded_valid
        encoded = rearrange(output, "(b e) d -> b e d", b=b, e=j * m)
        scenario = rearrange(
            self.pair_projection(encoded),
            "b (j m) d -> b j m d",
            j=j,
            m=m,
        )
        # Candidate consequence embeddings are available before action sampling.
        candidate = self._encode_candidate_sets(
            state.production_candidate_scenarios_tensor,
            build_pair_risk(
                state.health_pair_tensor[..., 1],
                state.health_pair_tensor[..., 3],
                state.fea_pairs_tensor[..., 4] > 0.5,
                state.scenario_invalid_mask_tensor[:, :, None, None].expand_as(invalid),
            ),
            invalid,
            self.production_candidate_scenario_processor,
            self.production_candidate_projection,
            pair_shape=(j, m),
        )
        observed = self.observed_pair_encoder(state.observed_pair_tensor)
        if not self.use_observed_deterministic_encoder:
            observed = torch.zeros_like(observed)
        return observed + torch.sigmoid(self.pair_fusion_gate) * (scenario + candidate)

    def _encode_candidate_sets(
        self,
        tokens: torch.Tensor,
        risk: torch.Tensor,
        invalid: torch.Tensor,
        processor: ScenarioEncoder,
        projection: nn.Linear,
        *,
        pair_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Encode candidate scenario sets while retaining zero for absent entities."""

        b, s = tokens.shape[:2]
        entity_shape = tokens.shape[2:-1]
        entities = math.prod(entity_shape)
        flat_tokens = tokens.reshape(b, s, entities, tokens.shape[-1])
        flat_risk = risk.reshape(b, s, entities)
        flat_invalid = invalid.reshape(b, s, entities)
        valid_entity = (~flat_invalid).any(dim=1)
        result = torch.zeros((b, entities, self.config.embedding_dim), device=tokens.device, dtype=tokens.dtype)
        selected = valid_entity.flatten()
        # Candidate encoders require the same unconditional structural guard:
        # trusted fast-path mode must never send a zero-sized batch to
        # MultiheadAttention.
        if selected.any():
            selected_tokens = rearrange(flat_tokens, "b s e f -> (b e) s f")[selected]
            selected_risk = rearrange(flat_risk, "b s e -> (b e) s")[selected]
            selected_invalid = rearrange(flat_invalid, "b s e -> (b e) s")[selected]
            encoded = processor(
                selected_tokens[:, :, None],
                selected_risk[:, :, None],
                selected_invalid[:, :, None],
            )[:, 0]
            result.reshape(b * entities, -1)[selected] = projection(encoded)
        if pair_shape is not None:
            return result.reshape(b, pair_shape[0], pair_shape[1], -1)
        return result

    def encode_maintenance_candidate_scenarios(
        self, state: RAMPEnvState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return PM and CM candidate consequences consumed by their decoders."""

        b, s, machines, _ = state.pm_candidate_scenarios_tensor.shape
        invalid = state.scenario_invalid_mask_tensor[:, :, None].expand(-1, -1, machines)
        pm_tokens = state.pm_candidate_scenarios_tensor
        cm_tokens = state.cm_candidate_scenarios_tensor
        pm_risk = pm_tokens[..., 3] + pm_tokens[..., 0] + pm_tokens[..., 1] - pm_tokens[..., 5]
        cm_risk = cm_tokens[..., 3] + cm_tokens[..., 0] + cm_tokens[..., 1]
        pm = self._encode_candidate_sets(
            pm_tokens, pm_risk, invalid,
            self.pm_candidate_scenario_processor, self.pm_candidate_projection,
        )
        cm = self._encode_candidate_sets(
            cm_tokens, cm_risk, invalid,
            self.cm_candidate_scenario_processor, self.cm_candidate_projection,
        )
        return pm, cm

    def health_substitution_edge(self, state: RAMPEnvState) -> torch.Tensor:
        """Five-term directed substitution cost between shared machines.

        Terms are central duration, central degradation consumption, tail risk,
        unsafe probability, and maintenance contention. Edges exist only when
        both machines share a current candidate operation.
        """

        if not self.use_health_substitution_edge:
            machines = state.fea_m_tensor.shape[2]
            return torch.zeros(
                (state.fea_m_tensor.shape[0], machines, machines),
                device=state.fea_m_tensor.device,
                dtype=state.fea_m_tensor.dtype,
            )
        invalid = state.health_pair_mask_tensor
        valid = ~invalid
        count = valid.sum(dim=1).clamp_min(1)
        duration_values = state.health_pair_tensor[..., 3]
        budget_values = state.health_pair_tensor[..., 1]
        central_duration = duration_values.masked_fill(invalid, 0.0).sum(dim=1) / count
        central_budget = budget_values.masked_fill(invalid, 0.0).sum(dim=1) / count
        tail_risk = state.pair_risk_tensor.masked_fill(invalid, float("-inf")).amax(dim=1)
        tail_risk = torch.where(
            torch.isfinite(tail_risk), tail_risk, torch.full_like(tail_risk, 1e6)
        )
        unsafe_probability = invalid.float().mean(dim=1)
        compatible = state.fea_pairs_tensor[..., 7] > 0
        compatible = compatible.any(dim=1)
        machines = central_duration.shape[-1]
        shared = compatible[:, :, :, None] & compatible[:, :, None, :]

        pm_available = (~state.pm_mask_tensor[:, 0]).float()
        ready_time = state.health_m_tensor[..., 4].mean(dim=1)
        maintenance_contention = ready_time + (1.0 - pm_available)

        def directed_difference(values: torch.Tensor) -> torch.Tensor:
            target = values[:, :, None, :].expand(-1, -1, machines, -1)
            source = values[:, :, :, None].expand(-1, -1, -1, machines)
            return target - source

        contention_target = maintenance_contention[:, None, None, :].expand(
            -1, state.candidate_tensor.shape[1], machines, -1
        )
        contention_source = maintenance_contention[:, None, :, None].expand_as(
            contention_target
        )
        difference = (
            directed_difference(central_duration)
            + self.config.substitution_degradation_weight
            * directed_difference(central_budget)
            + self.config.substitution_tail_weight * directed_difference(tail_risk)
            + self.config.substitution_safety_weight
            * directed_difference(unsafe_probability)
            + self.config.substitution_maintenance_weight
            * (contention_target - contention_source)
        )
        return (difference * shared).sum(dim=1) / shared.sum(dim=1).clamp_min(1)

    def _dual_attention(
        self,
        state: RAMPEnvState,
        operation: torch.Tensor,
        machine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operation = self.operation_attention(operation, state.op_mask_tensor)
        b, _, d = operation.shape
        candidate_index = state.candidate_tensor.unsqueeze(-1).expand(-1, -1, d)
        candidate_operation = operation.gather(1, candidate_index)
        comp = state.comp_idx_tensor
        competition_sum = torch.einsum("bmkj,bjd->bmkd", comp, candidate_operation)
        competition = competition_sum / comp.sum(dim=3, keepdim=True).clamp_min(1)
        substitution = self.health_substitution_edge(state).unsqueeze(-1)
        edge_features = torch.cat((competition, substitution), dim=-1)
        machine = self.machine_attention(
            machine, ~state.mch_mask_tensor, edge_features
        )
        return operation, machine

    def decode_production_actions(
        self,
        state: RAMPEnvState,
        operation: torch.Tensor,
        machine: torch.Tensor,
        pair: torch.Tensor,
        global_operation: torch.Tensor,
        global_machine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, j, m, d = pair.shape
        candidate_index = state.candidate_tensor.unsqueeze(-1).expand(-1, -1, d)
        candidate_operation = operation.gather(1, candidate_index)
        op_serial = candidate_operation[:, :, None, :].expand(-1, -1, m, -1)
        machine_serial = machine[:, None, :, :].expand(-1, j, -1, -1)
        global_op = global_operation[:, None, None, :].expand(-1, j, m, -1)
        global_m = global_machine[:, None, None, :].expand(-1, j, m, -1)
        features = torch.cat(
            (op_serial, machine_serial, global_op, global_m, pair), dim=-1
        )
        logits = self.production_actor(features).squeeze(-1).reshape(b, j * m)
        invalid = state.dynamic_pair_mask_tensor.reshape(b, j * m)
        return logits, _masked_softmax(logits, invalid)

    def decode_maintenance_actions(
        self,
        state: RAMPEnvState,
        machine: torch.Tensor,
        global_operation: torch.Tensor,
        global_machine: torch.Tensor,
        pm_candidate: torch.Tensor,
        cm_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m = machine.shape[1]
        common = torch.cat(
            (
                machine,
                global_operation[:, None, :].expand(-1, m, -1),
                global_machine[:, None, :].expand(-1, m, -1),
            ),
            dim=-1,
        )
        pm_logits = self.pm_actor(torch.cat((common, pm_candidate), dim=-1)).squeeze(-1)
        cm_logits = self.cm_actor(torch.cat((common, cm_candidate), dim=-1)).squeeze(-1)
        pm_invalid = state.pm_mask_tensor[:, 0]
        cm_invalid = state.cm_mask_tensor[:, 0]
        return (
            pm_logits,
            _masked_softmax(pm_logits, pm_invalid),
            cm_logits,
            _masked_softmax(cm_logits, cm_invalid),
        )

    def forward(self, state: RAMPEnvState) -> RAMPPolicyOutput:
        operation = self.encode_operation_scenarios(state)
        machine = self.encode_machine_health_scenarios(state)
        pair = self.encode_pair_health_scenarios(state)
        pm_candidate, cm_candidate = self.encode_maintenance_candidate_scenarios(state)
        operation, machine = self._dual_attention(state, operation, machine)
        global_operation = operation.mean(dim=1)
        global_machine = machine.mean(dim=1)
        observed_global = self.observed_global_encoder(state.observed_global_tensor)
        if not self.use_observed_deterministic_encoder:
            observed_global = torch.zeros_like(observed_global)
        production_logits, production_conditional = self.decode_production_actions(
            state, operation, machine, pair, global_operation, global_machine
        )
        pm_logits, pm_conditional, cm_logits, cm_conditional = (
            self.decode_maintenance_actions(
                state, machine, global_operation, global_machine,
                pm_candidate, cm_candidate,
            )
        )
        type_logits = self.type_gate(
            torch.cat((global_operation, global_machine, observed_global), dim=-1)
        )
        branch_invalid = torch.stack(
            (
                state.dynamic_pair_mask_tensor.flatten(1).all(dim=1),
                state.pm_mask_tensor[:, 0].all(dim=1),
                state.cm_mask_tensor[:, 0].all(dim=1),
            ),
            dim=1,
        )
        type_probs = _masked_softmax(type_logits, branch_invalid)
        action_probs = torch.cat(
            (
                type_probs[:, 0:1] * production_conditional,
                type_probs[:, 1:2] * pm_conditional,
                type_probs[:, 2:3] * cm_conditional,
            ),
            dim=1,
        )
        action_probs = action_probs.masked_fill(state.action_mask_tensor, 0.0)
        inactive = state.terminated_tensor | state.truncated_tensor
        active = ~inactive
        normalizer = action_probs.sum(dim=1, keepdim=True)
        if self.runtime_tensor_validation and (
            active & (normalizer.squeeze(1) <= 0)
        ).any():
            raise RuntimeError("active policy row has no positive-probability legal action")
        normalized = action_probs / normalizer.clamp_min(1e-12)
        padding = torch.zeros_like(action_probs)
        padding[:, 0] = 1.0
        # Padding rows have no scientific action distribution. A dedicated
        # deterministic padding index keeps sampling numerically defined;
        # valid_transition_mask excludes it from every PPO statistic/loss.
        action_probs = torch.where(active[:, None], normalized, padding)
        action_log_probs = torch.where(
            action_probs > 0,
            action_probs.clamp_min(1e-45).log(),
            torch.full_like(action_probs, float("-inf")),
        )
        value = self.critic(
            torch.cat((global_operation, global_machine, observed_global), dim=-1)
        ).squeeze(-1)
        return RAMPPolicyOutput(
            action_probs=action_probs,
            action_log_probs=action_log_probs,
            value=value,
            type_probs=type_probs,
            production_probs=production_conditional,
            pm_probs=pm_conditional,
            cm_probs=cm_conditional,
        )
