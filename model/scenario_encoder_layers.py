"""
This code is inspired by the paper 'Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks'
The original code can be found at https://github.com/juho-lee/set_transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older supported torch builds
    SDPBackend = None
    sdpa_kernel = None


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim_Q, num_heads, batch_first=True)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        # CUDA's memory-efficient SDPA backward can emit NaN gradients for the
        # long, highly repeated real SPM-DAN scenario sets while the forward is
        # finite.  Historical inference remains the default; the factorial
        # trainable copy opts into the eager attention path explicitly.
        self.stable_training_attention = False

    def forward(
        self,
        Q,
        K,
        key_padding_mask=None,
        query_padding_mask=None,
    ):
        if self.stable_training_attention and Q.is_cuda and sdpa_kernel is not None:
            # Keep the no-weight MHA fast path, but forbid the CUDA
            # memory-efficient backward that produced NaN gradients on the
            # real 100-realization SPM-DAN scenario tensor.
            with sdpa_kernel(SDPBackend.MATH):
                attention = self.attention(
                    Q, K, K, key_padding_mask=key_padding_mask, need_weights=False
                )[0]
        else:
            # Older torch builds use eager matmul/softmax when weights are
            # requested, which is slower but numerically safe for training.
            attention = self.attention(
                Q, K, K, key_padding_mask=key_padding_mask,
                need_weights=bool(self.stable_training_attention)
            )[0]
        O = attention + Q
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        if query_padding_mask is not None:
            O = O.masked_fill(query_padding_mask[..., None], 0.0)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        if query_padding_mask is not None:
            O = O.masked_fill(query_padding_mask[..., None], 0.0)
        return O


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SelfAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ScenarioProcessingModuleWithoutAggregation(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True):
        super(ScenarioProcessingModuleWithoutAggregation, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiHeadAttentionBlock(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MultiHeadAttentionBlock(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, scenario_invalid_mask=None):
        if scenario_invalid_mask is not None:
            if scenario_invalid_mask.shape != X.shape[:2]:
                raise ValueError("scenario_invalid_mask must be [B,S]")
            if scenario_invalid_mask.dtype != torch.bool:
                raise TypeError("scenario_invalid_mask must be boolean")
            if scenario_invalid_mask.all(dim=1).any():
                raise ValueError("official SPM cannot process an all-invalid set")
        H = self.mab0(
            self.I.repeat(X.size(0), 1, 1), X,
            key_padding_mask=scenario_invalid_mask,
        )
        return self.mab1(X, H, query_padding_mask=scenario_invalid_mask)
