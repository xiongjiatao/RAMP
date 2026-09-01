"""Vendored official DAN graph encoder with a one-to-one state-dict topology.

The implementation follows ``(official)SPM-DAN/FJSP-DRL/model/main_model.py``
``DualAttentionNetwork``.  Health/scenario/joint-action logic must be added by
callers after this module; changing this core would invalidate mapped-weight
forward parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.attention import MultiHeadMchAttnBlock, MultiHeadOpAttnBlock


@dataclass(frozen=True)
class OfficialDANConfig:
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    SAA_attention: bool = False
    SAA_attention_dim: int = 0
    layer_fea_output_dim: tuple[int, int] = (32, 8)
    num_heads_OAB: tuple[int, int] = (4, 4)
    num_heads_MAB: tuple[int, int] = (4, 4)
    dropout_prob: float = 0.0


def _nonzero_averaging(values: torch.Tensor) -> torch.Tensor:
    total = values.sum(dim=-2)
    nonzero = torch.count_nonzero(values, dim=-1)
    count = (nonzero != 0).sum(dim=-1, keepdim=True)
    inverse = 1 / count
    inverse[count == 0] = 0
    return inverse * total


class DualAttentionNetwork(nn.Module):
    """Official two-layer DAN core; default output dimension is eight."""

    def __init__(self, config: object):
        super().__init__()
        self.fea_j_input_dim = (
            config.fea_j_input_dim
            if not config.SAA_attention
            else config.fea_j_input_dim + config.SAA_attention_dim
        )
        self.fea_m_input_dim = (
            config.fea_m_input_dim
            if not config.SAA_attention
            else config.fea_m_input_dim + config.SAA_attention_dim
        )
        self.output_dim_per_layer = list(config.layer_fea_output_dim)
        self.num_heads_OAB = list(config.num_heads_OAB)
        self.num_heads_MAB = list(config.num_heads_MAB)
        self.last_layer_activate = nn.ELU()
        self.num_dan_layers = len(self.num_heads_OAB)
        if len(self.num_heads_MAB) != self.num_dan_layers:
            raise ValueError("official DAN operation/machine layer counts differ")
        if len(self.output_dim_per_layer) != self.num_dan_layers:
            raise ValueError("official DAN output dimensions do not match layer count")
        self.dropout_prob = float(config.dropout_prob)

        operation_heads = [1] + self.num_heads_OAB
        machine_heads = [1] + self.num_heads_MAB
        middle = self.output_dim_per_layer[:-1]
        operation_inputs = [self.fea_j_input_dim] + middle
        machine_inputs = [self.fea_m_input_dim] + middle
        self.op_attention_blocks = nn.ModuleList()
        self.mch_attention_blocks = nn.ModuleList()
        for layer in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=operation_heads[layer] * operation_inputs[layer],
                    num_heads=self.num_heads_OAB[layer],
                    output_dim=self.output_dim_per_layer[layer],
                    concat=layer < self.num_dan_layers - 1,
                    activation=nn.ELU(),
                    dropout_prob=self.dropout_prob,
                )
            )
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=machine_heads[layer] * machine_inputs[layer],
                    edge_input_dim=operation_heads[layer] * operation_inputs[layer],
                    num_heads=self.num_heads_MAB[layer],
                    output_dim=self.output_dim_per_layer[layer],
                    concat=layer < self.num_dan_layers - 1,
                    activation=nn.ELU(),
                    dropout_prob=self.dropout_prob,
                )
            )

    def forward(
        self,
        fea_j: torch.Tensor,
        op_mask: torch.Tensor,
        candidate: torch.Tensor,
        fea_m: torch.Tensor,
        mch_mask: torch.Tensor,
        comp_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, machines, _, jobs = comp_idx.size()
        comp_idx_for_mul = comp_idx.reshape(batch, -1, jobs)
        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1).repeat(
                1, 1, fea_j.shape[-1]
            ).to(torch.int64)
            candidate_features = torch.gather(fea_j, 1, candidate_idx).float()
            competition = torch.matmul(
                comp_idx_for_mul, candidate_features
            ).reshape(batch, machines, machines, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, competition)
        return fea_j, fea_m, _nonzero_averaging(fea_j), _nonzero_averaging(fea_m)
