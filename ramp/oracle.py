"""Exhaustive small-instance scenario oracle for RAMP validation.

The oracle is intentionally bounded to tiny instances. It enumerates every
common production/PM/CM action sequence, applies each action to every fixed
future scenario through the production transition kernel, and reports a
certificate only if the complete finite tree was exhausted. It is an exact
open-loop scenario oracle, not a scalable replacement for the learned policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product

import torch

from ramp.env import RAMPEnvironmentCore, upper_tail_cvar
from ramp.state import ActionType, ObservedShopState


@dataclass(frozen=True)
class ExactScenarioOracleResult:
    """Certified optimum and enumeration evidence for one tiny instance."""

    actions: tuple[int, ...]
    objective: float
    scenario_total_cost: torch.Tensor
    scenario_makespan: torch.Tensor
    explored_nodes: int
    certified_optimal: bool


class BoundedOpenLoopScenarioOracle:
    """Enumerate an open-loop common action sequence on fixed scenarios."""

    def __init__(
        self,
        environment: object,
        *,
        scenario_role: str = "reward",
        max_decisions: int | None = None,
        max_nodes: int = 250_000,
    ) -> None:
        core = getattr(environment, "_core", environment)
        if not isinstance(core, RAMPEnvironmentCore):
            raise TypeError("oracle requires an RAMP environment core or facade")
        if core.batch_size != 1:
            raise ValueError("exact scenario oracle supports batch size one")
        if core.number_of_ops > 8:
            raise ValueError("exact scenario oracle is restricted to at most 8 operations")
        if scenario_role not in {"state", "reward"}:
            raise ValueError("scenario_role must be state or reward")
        self.core = core
        self.kernel = core.transition_kernel
        self.bank = (
            core.state_noise_bank if scenario_role == "state" else core.reward_noise_bank
        )
        self.scenario_count = (
            core.num_scenarios
            if scenario_role == "state"
            else core.reward_num_scenarios
        )
        configured_limit = (
            core.number_of_ops + core.config.max_maintenance_decisions
            if core.config.max_decisions is None
            else core.config.max_decisions
        )
        self.max_decisions = int(
            configured_limit if max_decisions is None else max_decisions
        )
        if self.max_decisions < core.number_of_ops:
            raise ValueError("max_decisions cannot be smaller than operation count")
        self.max_nodes = int(max_nodes)
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.explored_nodes = 0
        self.best_actions: tuple[int, ...] | None = None
        self.best_objective = float("inf")
        self.best_costs: torch.Tensor | None = None
        self.best_makespan: torch.Tensor | None = None

    def _production_is_safe(
        self,
        states: list[ObservedShopState],
        job: int,
        machine: int,
    ) -> bool:
        survivals = []
        for scenario, state in enumerate(states):
            if state.job_finished[0, job]:
                return False
            operation = int(state.candidate[0, job])
            if not self.core.compatible[0, operation, machine]:
                return False
            probe = state.clone()
            if probe.observed_machine_status[0, machine]:
                self.kernel.apply_corrective_maintenance(
                    probe,
                    batch_index=0,
                    machine=machine,
                    noise_bank=self.bank,
                    scenario_id=scenario,
                )
            result = self.kernel.prospective_production(
                probe,
                batch_index=0,
                scenario_id=scenario,
                operation=operation,
                machine=machine,
                noise_bank=self.bank,
            )
            survivals.append(bool(result.survival))
        probability = sum(survivals) / len(survivals)
        return probability >= 1.0 - self.core.config.epsilon_use

    def _legal_actions(
        self, states: list[ObservedShopState]
    ) -> list[int]:
        actions: list[int] = []
        for job in range(self.core.number_of_jobs):
            for machine in range(self.core.number_of_machines):
                if self._production_is_safe(states, job, machine):
                    actions.append(self.core.codec.production(job, machine))
        if self.core.config.maintenance_actions:
            for machine in range(self.core.number_of_machines):
                pm_legal = self.core.config.preventive_maintenance_actions
                cm_legal = self.core.config.corrective_maintenance_actions
                for state in states:
                    pm_legal = pm_legal and (
                        not bool(state.observed_machine_status[0, machine])
                        and int(state.pm_count[0, machine])
                        < self.core.config.max_pm_per_machine
                        and self.kernel._has_remaining_compatible_operation(
                            state, 0, machine
                        )
                    )
                    cm_legal = cm_legal and bool(
                        state.observed_machine_status[0, machine]
                    )
                if pm_legal:
                    actions.append(self.core.codec.pm(machine))
                if cm_legal:
                    actions.append(self.core.codec.cm(machine))
        return actions

    def _apply(
        self,
        states: list[ObservedShopState],
        action: int,
    ) -> list[ObservedShopState]:
        next_states: list[ObservedShopState] = []
        for scenario, source in enumerate(states):
            state = source.clone()
            self.kernel.apply_primary_action_with_recourse(
                state,
                batch_index=0,
                action=action,
                action_codec=self.core.codec,
                noise_bank=self.bank,
                scenario_id=scenario,
            )
            next_states.append(state)
        return next_states

    def _terminal_cost(
        self, states: list[ObservedShopState]
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        components = []
        makespans = []
        for state in states:
            horizon = torch.maximum(
                state.current_makespan,
                state.observed_machine_ready_time.amax(dim=1),
            )
            self.kernel.accrue_failed_waiting(state, horizon=horizon)
            makespan = horizon[0]
            components.append(
                torch.stack(
                    (
                        makespan,
                        state.pm_cost_total[0],
                        state.cm_cost_total[0],
                        state.unplanned_downtime[0].sum(),
                        state.failure_count[0].sum(),
                    )
                )
            )
            makespans.append(makespan)
        component_tensor = torch.stack(components, dim=0)
        weights = torch.tensor(
            (
                1.0,
                self.core.config.objective.lambda_pm,
                self.core.config.objective.lambda_cm,
                self.core.config.objective.lambda_downtime,
                self.core.config.objective.lambda_failure,
            ),
            device=component_tensor.device,
            dtype=component_tensor.dtype,
        )
        costs = (
            component_tensor / self.core.objective_scales[0]
        ).mul(weights).sum(dim=1)
        objective_tensor = costs.mean() + self.core.config.objective.cvar_beta * (
            upper_tail_cvar(
                costs[None], self.core.config.objective.cvar_alpha
            )[0]
        )
        return costs, torch.stack(makespans), float(objective_tensor)

    def _search(
        self,
        states: list[ObservedShopState],
        actions: tuple[int, ...],
    ) -> None:
        self.explored_nodes += 1
        if self.explored_nodes > self.max_nodes:
            raise RuntimeError(
                "oracle node limit reached; no optimality certificate can be issued"
            )
        if all(bool(state.terminated[0]) for state in states):
            costs, makespan, objective = self._terminal_cost(states)
            if objective < self.best_objective:
                self.best_objective = objective
                self.best_actions = actions
                self.best_costs = costs.detach().cpu()
                self.best_makespan = makespan.detach().cpu()
            return
        if len(actions) >= self.max_decisions:
            return
        for action in self._legal_actions(states):
            self._search(self._apply(states, action), (*actions, action))

    def solve(self) -> ExactScenarioOracleResult:
        """Exhaust the bounded tree and return a truthful optimum certificate."""

        roots = [
            self.core.observed_state.clone() for _ in range(self.scenario_count)
        ]
        self._search(roots, ())
        if self.best_actions is None or self.best_costs is None or self.best_makespan is None:
            raise RuntimeError("no terminal common-action sequence exists within limits")
        return ExactScenarioOracleResult(
            actions=self.best_actions,
            objective=self.best_objective,
            scenario_total_cost=self.best_costs,
            scenario_makespan=self.best_makespan,
            explored_nodes=self.explored_nodes,
            certified_optimal=True,
        )


@dataclass(frozen=True)
class TinyScenarioTreeOracleResult:
    """Result of exhaustive non-anticipative finite policy-tree enumeration."""

    status: str
    objective: float
    scenario_total_cost: torch.Tensor
    explored_nodes: int
    certified_optimal: bool
    policy_tree: dict[str, int]
    decision_trace: tuple[dict[str, object], ...]


class TinyScenarioTreeOracle:
    """Exhaust a tiny non-anticipative policy tree over observed scenario branches.

    Scenarios that have the same fully observed scheduling state are constrained
    to the same action.  Once their observed health/status/timeline differs,
    they form distinct observation branches and may take different feedback
    actions.  A certificate is returned only when the finite tree is exhausted.
    """

    def __init__(self, environment: object, *, node_limit: int = 250_000):
        core = getattr(environment, "_core", environment)
        if not isinstance(core, RAMPEnvironmentCore):
            raise TypeError("scenario-tree oracle requires an RAMP environment")
        if core.batch_size != 1 or core.number_of_ops > 8 or core.number_of_machines > 3:
            raise ValueError("scenario-tree oracle supports B=1, operations<=8, machines<=3")
        self.core = core
        self.bank = core.reward_noise_bank
        self.node_limit = int(node_limit)
        if self.node_limit < 1:
            raise ValueError("node_limit must be positive")
        self.explored_nodes = 0
        self.limit_reached = False
        self.best_objective = float("inf")
        self.best_costs = torch.empty(0)
        self.best_policy: dict[str, int] = {}
        self.best_trace: tuple[dict[str, object], ...] = ()
        self._cost_authority = BoundedOpenLoopScenarioOracle(
            environment, scenario_role="reward", max_nodes=max(node_limit, 1)
        )

    @staticmethod
    def _signature(state: ObservedShopState) -> bytes:
        """Hash only information available in the complete observation history.

        Scenario identifiers and future noise keys are deliberately absent.
        All observed Markov fields, accumulated observed costs/counters, and
        primary-action history are included, so histories that are genuinely
        distinguishable may branch while future-only differences may not.
        """

        digest = hashlib.sha256()
        for name, value in state.state_dict().items():
            digest.update(name.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.digest()

    def _groups(
        self, states: list[ObservedShopState]
    ) -> list[list[int]]:
        groups: dict[bytes, list[int]] = {}
        for scenario, state in enumerate(states):
            if state.terminated[0] or state.truncated[0]:
                continue
            groups.setdefault(self._signature(state), []).append(scenario)
        return list(groups.values())

    def _legal_actions_for_group(
        self, states: list[ObservedShopState], group: list[int]
    ) -> list[int]:
        actions: list[int] = []
        for job in range(self.core.number_of_jobs):
            for machine in range(self.core.number_of_machines):
                survivals: list[bool] = []
                legal = True
                for scenario in group:
                    state = states[scenario]
                    if state.job_finished[0, job]:
                        legal = False
                        break
                    operation = int(state.candidate[0, job])
                    if not self.core.compatible[0, operation, machine]:
                        legal = False
                        break
                    probe = state.clone()
                    if probe.observed_machine_status[0, machine]:
                        self.core.transition_kernel.apply_corrective_maintenance(
                            probe, batch_index=0, machine=machine,
                            noise_bank=self.bank, scenario_id=scenario,
                        )
                    transition = self.core.transition_kernel.prospective_production(
                        probe, batch_index=0, scenario_id=scenario,
                        operation=operation, machine=machine, noise_bank=self.bank,
                    )
                    survivals.append(bool(transition.survival))
                if legal and sum(survivals) / len(survivals) >= 1 - self.core.config.epsilon_use:
                    actions.append(self.core.codec.production(job, machine))
        if self.core.config.maintenance_actions:
            representative = states[group[0]]
            for machine in range(self.core.number_of_machines):
                if (
                    self.core.config.preventive_maintenance_actions
                    and not representative.observed_machine_status[0, machine]
                    and representative.pm_count[0, machine] < self.core.config.max_pm_per_machine
                    and self.core.transition_kernel._has_remaining_compatible_operation(
                        representative, 0, machine
                    )
                ):
                    actions.append(self.core.codec.pm(machine))
                if (
                    self.core.config.corrective_maintenance_actions
                    and representative.observed_machine_status[0, machine]
                ):
                    actions.append(self.core.codec.cm(machine))
        return actions

    def _apply_one(
        self, state: ObservedShopState, scenario: int, action: int
    ) -> ObservedShopState:
        result = state.clone()
        self.core.transition_kernel.apply_primary_action_with_recourse(
            result,
            batch_index=0,
            action=action,
            action_codec=self.core.codec,
            noise_bank=self.bank,
            scenario_id=scenario,
        )
        return result

    def _search(
        self,
        states: list[ObservedShopState],
        policy: dict[str, int],
        trace: tuple[dict[str, object], ...],
    ) -> None:
        self.explored_nodes += 1
        if self.explored_nodes > self.node_limit:
            self.limit_reached = True
            return
        if all(bool(state.terminated[0]) for state in states):
            costs, _, objective = self._cost_authority._terminal_cost(states)
            if objective < self.best_objective:
                self.best_objective = objective
                self.best_costs = costs.detach().cpu()
                self.best_policy = dict(policy)
                self.best_trace = tuple(dict(row) for row in trace)
            return
        groups = self._groups(states)
        legal_sets = [self._legal_actions_for_group(states, group) for group in groups]
        if any(not actions for actions in legal_sets):
            return
        for assignments in product(*legal_sets):
            if self.limit_reached:
                return
            next_states = [state.clone() for state in states]
            next_policy = dict(policy)
            next_trace = list(trace)
            for group, action in zip(groups, assignments):
                depth = len(states[group[0]].action_history[0])
                signature = self._signature(states[group[0]]).hex()
                key = f"d{depth}:{signature}"
                next_policy[key] = int(action)
                next_trace.append(
                    {
                        "depth": depth,
                        "observed_history_signature": signature,
                        "scenario_members": tuple(group),
                        "action": int(action),
                    }
                )
                for scenario in group:
                    next_states[scenario] = self._apply_one(
                        states[scenario], scenario, int(action)
                    )
            self._search(next_states, next_policy, tuple(next_trace))

    def solve(self) -> TinyScenarioTreeOracleResult:
        roots = [
            self.core.reward_scenarios.as_observed(scenario)
            for scenario in range(self.core.reward_num_scenarios)
        ]
        self._search(roots, {}, ())
        certified = not self.limit_reached and bool(self.best_policy)
        status = "CERTIFIED_OPTIMAL" if certified else "UNCERTIFIED_NODE_LIMIT"
        return TinyScenarioTreeOracleResult(
            status=status,
            objective=self.best_objective,
            scenario_total_cost=self.best_costs,
            explored_nodes=self.explored_nodes,
            certified_optimal=certified,
            policy_tree=self.best_policy,
            decision_trace=self.best_trace,
        )


# Historical public name retained with the now-explicit scope.
ExactScenarioOracle = BoundedOpenLoopScenarioOracle
