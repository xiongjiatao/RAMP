"""Risk-adjusted PPO with explicit transition validity and boundary semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramp.state import RAMPEnvState
from ramp.profiling import ThroughputProfiler, profiled, tensor_bytes


@dataclass(frozen=True)
class RAMPPPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.98
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 128
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma != 1.0:
            raise ValueError(
                "gamma must equal 1.0 for risk-adjusted total-cost potential differences"
            )
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("PPO update counts must be positive")


def masked_transition_mean(
    values: torch.Tensor, valid_transition_mask: torch.Tensor
) -> torch.Tensor:
    """Mean over active-before-step samples only.

    Indexing before reduction ensures NaN or Inf padding values cannot
    contaminate policy, value, entropy, KL, or clip-fraction statistics.
    """

    if values.shape != valid_transition_mask.shape:
        raise ValueError("values and valid_transition_mask must have identical shapes")
    valid = valid_transition_mask.bool()
    if not valid.any():
        raise ValueError("at least one valid transition is required")
    return values[valid].mean()


def compute_done_aware_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid_transition_mask: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for ``[T,B]`` with distinct terminal boundaries.

    Natural termination has bootstrap mask zero. Time truncation bootstraps
    the supplied next value but stops the recursive trace. Invalid padding
    transitions are exactly zero and reset the trace.
    """

    tensors = (
        values,
        terminated,
        truncated,
        valid_transition_mask,
        next_values,
    )
    if rewards.ndim != 2 or any(tensor.shape != rewards.shape for tensor in tensors):
        raise ValueError("all GAE tensors must share [T,B] shape")
    if gamma != 1.0:
        raise ValueError("gamma must equal 1.0 for this potential-difference reward")
    valid = valid_transition_mask.bool()
    terminated = terminated.bool()
    truncated = truncated.bool()
    if torch.any(terminated & truncated & valid):
        raise ValueError("a valid transition cannot be both terminated and truncated")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for step in reversed(range(rewards.shape[0])):
        bootstrap = (~terminated[step]).to(values.dtype)
        delta = (
            rewards[step]
            + gamma * bootstrap * next_values[step]
            - values[step]
        )
        delta = torch.where(valid[step], delta, torch.zeros_like(delta))
        continuation = (
            valid[step] & ~terminated[step] & ~truncated[step]
        ).to(values.dtype)
        running = delta + gamma * gae_lambda * continuation * running
        running = torch.where(valid[step], running, torch.zeros_like(running))
        advantages[step] = running
    returns = torch.where(valid, advantages + values, torch.zeros_like(values))
    return advantages, returns


class RAMPRolloutBuffer:
    """CPU rollout storage with explicit active-before-step validity."""

    FORMAT = "RAMP rollout buffer v1"

    def __init__(self) -> None:
        self.states: List[RAMPEnvState] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.next_values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.terminated: List[torch.Tensor] = []
        self.truncated: List[torch.Tensor] = []
        self.valid_transition_mask: List[torch.Tensor] = []

    def add(
        self,
        state: RAMPEnvState,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        valid_transition_mask: torch.Tensor,
        next_value: torch.Tensor,
    ) -> None:
        """Append one batched transition and reject valid invalid actions."""

        valid = valid_transition_mask.detach().bool()
        invalid_action = state.action_mask_tensor.gather(
            1, action.long()[:, None]
        ).squeeze(1)
        if (invalid_action & valid.to(invalid_action.device)).any():
            raise AssertionError("rollout contains an invalid active action")
        self.states.append(state.detach_clone(device="cpu"))
        self.actions.append(action.detach().cpu().clone())
        self.log_probs.append(log_prob.detach().cpu().clone())
        self.values.append(value.detach().cpu().clone())
        self.next_values.append(next_value.detach().cpu().clone())
        self.rewards.append(reward.detach().cpu().clone())
        self.terminated.append(terminated.detach().cpu().bool().clone())
        self.truncated.append(truncated.detach().cpu().bool().clone())
        self.valid_transition_mask.append(valid.cpu().clone())

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_values.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.valid_transition_mask.clear()

    def state_dict(self) -> dict[str, object]:
        """Serialize every pending transition for exact mid-rollout resume."""

        return {
            "format": self.FORMAT,
            "states": [state.detach_clone(device="cpu") for state in self.states],
            "actions": [value.detach().cpu().clone() for value in self.actions],
            "log_probs": [value.detach().cpu().clone() for value in self.log_probs],
            "values": [value.detach().cpu().clone() for value in self.values],
            "next_values": [value.detach().cpu().clone() for value in self.next_values],
            "rewards": [value.detach().cpu().clone() for value in self.rewards],
            "terminated": [value.detach().cpu().clone() for value in self.terminated],
            "truncated": [value.detach().cpu().clone() for value in self.truncated],
            "valid_transition_mask": [
                value.detach().cpu().clone() for value in self.valid_transition_mask
            ],
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        """Restore a pending rollout and reject partially serialized buffers."""

        if payload.get("format") != self.FORMAT:
            raise ValueError("unsupported rollout-buffer state")
        names = (
            "states",
            "actions",
            "log_probs",
            "values",
            "next_values",
            "rewards",
            "terminated",
            "truncated",
            "valid_transition_mask",
        )
        missing = [name for name in names if name not in payload]
        if missing:
            raise ValueError(f"rollout-buffer state lacks fields: {missing}")
        lengths = {len(payload[name]) for name in names}  # type: ignore[arg-type]
        if len(lengths) != 1:
            raise ValueError("rollout-buffer fields have inconsistent lengths")
        self.clear()
        self.states.extend(
            state.detach_clone(device="cpu") for state in payload["states"]  # type: ignore[union-attr]
        )
        for name in names[1:]:
            target = getattr(self, name)
            target.extend(
                value.detach().cpu().clone() for value in payload[name]  # type: ignore[union-attr]
            )

    def __len__(self) -> int:
        return len(self.states)


class RAMPPPO:
    """PPO learner for the scalar mean-plus-CVaR cost potential."""

    def __init__(
        self,
        policy: nn.Module,
        config: RAMPPPOConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ):
        self.config = config or RAMPPPOConfig()
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.policy_old = deepcopy(policy).to(self.device)
        self.policy_old.eval()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.profiler = ThroughputProfiler(device=self.device)
        self.sync_old_policy()

    def sync_old_policy(self) -> None:
        """Hard-synchronize behavior policy after each complete PPO update."""

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

    @torch.no_grad()
    def act_with_output(
        self, state: RAMPEnvState, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Select one action and expose the same forward output for auditing.

        This is deliberately a single-forward API: route instrumentation can
        inspect policy provenance without a second model call, an extra RNG
        draw, or any change to the sampled action.
        """

        state_on_device = state.to(self.device)
        output = self.policy_old(state_on_device)
        actions = (
            output.action_probs.argmax(dim=1)
            if deterministic
            else torch.distributions.Categorical(output.action_probs).sample()
        )
        invalid = state_on_device.action_mask_tensor.gather(
            1, actions[:, None]
        ).squeeze(1)
        active = ~(
            state_on_device.terminated_tensor | state_on_device.truncated_tensor
        )
        if (invalid & active).any():
            raise AssertionError("policy sampled an invalid active action")
        return actions, output.log_prob(actions), output.value, output

    @torch.no_grad()
    def act(
        self, state: RAMPEnvState, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, value, _ = self.act_with_output(
            state, deterministic=deterministic
        )
        return actions, log_prob, value

    @profiled("ppo.update_total")
    def update(self, buffer: RAMPRolloutBuffer) -> dict[str, float]:
        """Optimize exclusively over valid transitions and return diagnostics."""

        if len(buffer) == 0:
            raise ValueError("cannot update PPO from an empty buffer")
        rewards = torch.stack(buffer.rewards)
        values = torch.stack(buffer.values)
        next_values = torch.stack(buffer.next_values)
        terminated = torch.stack(buffer.terminated)
        truncated = torch.stack(buffer.truncated)
        valid = torch.stack(buffer.valid_transition_mask)
        advantages, returns = compute_done_aware_gae(
            rewards,
            values,
            terminated=terminated,
            truncated=truncated,
            valid_transition_mask=valid,
            next_values=next_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        valid_flat = valid.flatten()
        flat_advantages = advantages.flatten()[valid_flat]
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / flat_advantages.std(unbiased=False).clamp_min(1e-8)
        flat_returns = returns.flatten()[valid_flat]
        flat_values = values.flatten()[valid_flat]
        states_all = RAMPEnvState.cat(buffer.states)
        valid_indices = valid_flat.nonzero(as_tuple=False).flatten()
        states = states_all.index_select(valid_indices)
        actions = torch.cat(buffer.actions)[valid_flat]
        old_log_probs = torch.cat(buffer.log_probs)[valid_flat]
        total = actions.numel()
        if total == 0:
            raise ValueError("rollout has no valid transitions")

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        ratios: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        gradient_norms: list[float] = []
        # The rollout is reused for every PPO epoch. Transfer it once instead
        # of copying the same minibatch payload twenty times.
        with self.profiler.phase("ppo.rollout_cpu_to_gpu_once"):
            self.profiler.transfer("cpu_to_gpu_bytes", tensor_bytes(states))
            states_device = states.to(self.device)
            actions_device = actions.to(self.device)
            old_log_probs_device = old_log_probs.to(self.device)
            advantages_device = flat_advantages.to(self.device)
            returns_device = flat_returns.to(self.device)
        self.policy.train()
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(total)
            for start in range(0, total, self.config.minibatch_size):
                indices_cpu = permutation[start : start + self.config.minibatch_size]
                indices = indices_cpu.to(self.device)
                mini_state = states_device.index_select(indices)
                mini_actions = actions_device.index_select(0, indices)
                with self.profiler.phase("ppo.forward"):
                    output = self.policy(mini_state)
                new_log_prob = output.log_prob(mini_actions)
                if not torch.isfinite(new_log_prob).all():
                    raise FloatingPointError("non-finite selected action log probability")
                old_log = old_log_probs_device.index_select(0, indices)
                log_ratio = new_log_prob - old_log
                ratio = torch.exp(log_ratio)
                mini_advantage = advantages_device.index_select(0, indices)
                unclipped = ratio * mini_advantage
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * mini_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(
                    output.value,
                    returns_device.index_select(0, indices),
                )
                entropy = output.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                if not torch.isfinite(ratio).all() or not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO ratio or loss")
                with self.profiler.phase("ppo.backward_optimizer"):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("non-finite PPO gradient norm")
                    self.optimizer.step()
                losses.append(float(loss.detach()))
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))
                ratios.append(float(ratio.mean().detach()))
                approximate_kls.append(float((-log_ratio).mean().detach()))
                clip_fractions.append(
                    float(
                        ((ratio - 1.0).abs() > self.config.clip_ratio)
                        .float()
                        .mean()
                        .detach()
                    )
                )
                gradient_norms.append(float(grad_norm.detach()))
        self.sync_old_policy()
        prediction_error = ((flat_returns - flat_values) ** 2).mean()
        return_variance = flat_returns.var(unbiased=False)
        explained_variance = 1.0 - prediction_error / return_variance.clamp_min(1e-8)
        mean = lambda entries: float(sum(entries) / len(entries))
        return {
            "loss": mean(losses),
            "policy_loss": mean(policy_losses),
            "value_loss": mean(value_losses),
            "entropy": mean(entropies),
            "ratio_mean": mean(ratios),
            "approximate_kl": mean(approximate_kls),
            "clip_fraction": mean(clip_fractions),
            "explained_variance": float(explained_variance),
            "gradient_norm": mean(gradient_norms),
            "valid_transition_count": float(total),
        }
