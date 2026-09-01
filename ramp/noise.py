"""Persistent keyed future-noise banks and standalone reproducible samplers."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import torch

try:
    from scipy.special import betaincinv, gammaincinv
except ImportError:  # pragma: no cover - project requirements include SciPy
    betaincinv = None
    gammaincinv = None


def _generator(seed: int, device: torch.device | str) -> torch.Generator:
    device = torch.device(device)
    generator_device = device.type if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


class TrajectoryNoiseBank:
    """Episode-persistent common-random-number authority.

    Random primitives are keyed by semantic event identity, never by decision
    step. Consequently, inserting an unrelated action cannot change the future
    noise assigned to an unexecuted operation-machine pair. A namespace keeps
    state and reward scenario banks independent.
    """

    FORMAT = "RAMP trajectory noise bank v1"

    def __init__(
        self,
        *,
        seed: int,
        namespace: str,
        seed_source: torch.Tensor | None = None,
    ):
        if not namespace:
            raise ValueError("noise-bank namespace must be nonempty")
        self.seed = int(seed)
        self.namespace = str(namespace)
        self.seed_source = seed_source
        self._seed_source_values = (
            None
            if seed_source is None
            else seed_source.detach().cpu().long().tolist()
        )
        self._semantic_scenario_ids: list[int] | None = None
        self._uniform_cache: dict[str, float] = {}
        self._processing_grid_cache_key: tuple[Any, ...] | None = None
        self._processing_grid_cache_value: torch.Tensor | None = None
        self._degradation_uniform_cache_key: tuple[Any, ...] | None = None
        self._degradation_uniform_cache_value: np.ndarray | None = None
        self._degradation_uniform_grid_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._degradation_uniform_master_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.diagnostic_counts: dict[str, int] = {
            "uniform_calls": 0,
            "uniform_cache_hits": 0,
            "uniform_cache_misses": 0,
            "processing_noise_calls": 0,
            "degradation_noise_calls": 0,
            "maintenance_noise_calls": 0,
        }

    @staticmethod
    def _key(stream: str, values: tuple[Any, ...]) -> str:
        return "|".join((stream, *(str(value) for value in values)))

    def _uniform(self, stream: str, *values: Any) -> float:
        self.diagnostic_counts["uniform_calls"] += 1
        values = list(values)
        if self._semantic_scenario_ids is not None and len(values) >= 2:
            local = int(values[1])
            if local >= 0:
                if local >= len(self._semantic_scenario_ids):
                    raise IndexError("local scenario id exceeds semantic ATMSL map")
                values[1] = self._semantic_scenario_ids[local]
        values = tuple(values)
        key = self._key(stream, values)
        if key not in self._uniform_cache:
            self.diagnostic_counts["uniform_cache_misses"] += 1
            semantic_seed = self.seed
            if self._seed_source_values is not None and len(values) >= 2:
                scenario = max(int(values[1]), 0) % len(self._seed_source_values[0])
                semantic_seed = int(self._seed_source_values[0][scenario])
            digest = hashlib.sha256(
                f"{semantic_seed}|{self.namespace}|{key}".encode("utf-8")
            ).digest()
            integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
            # Map to the open interval so inverse CDFs remain finite.
            self._uniform_cache[key] = (integer + 0.5) / (2**64)
        else:
            self.diagnostic_counts["uniform_cache_hits"] += 1
        return self._uniform_cache[key]

    def set_semantic_scenario_ids(
        self, scenario_ids: torch.Tensor | list[int], *, paired_base_seed: bool = True
    ) -> None:
        """Map compact ATMSL slots to full-fidelity semantic scenario IDs.

        Paired mode intentionally uses the bank seed plus semantic event key,
        so a representative path is byte-identical in low and full banks.
        Must be called between episodes; all derived caches are invalidated.
        """

        ids = torch.as_tensor(scenario_ids, dtype=torch.long).flatten().tolist()
        if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
            raise ValueError("semantic scenario ids must be unique nonnegative integers")
        self._semantic_scenario_ids = [int(value) for value in ids]
        if paired_base_seed:
            self.seed_source = None
            self._seed_source_values = None
        self._uniform_cache.clear()
        self._processing_grid_cache_key = None
        self._processing_grid_cache_value = None
        self._degradation_uniform_cache_key = None
        self._degradation_uniform_cache_value = None
        self._degradation_uniform_grid_cache.clear()
        self._degradation_uniform_master_cache.clear()

    def rekey(self, *, seed: int, namespace: str | None = None) -> None:
        """Install a frozen semantic root before an evaluation episode.

        Public zero-shot evaluation derives this root from the benchmark hash,
        overlay id, replication id, and stream role.  Clearing every materialized
        primitive makes the result independent of method execution order while
        leaving the production training path unchanged.
        """

        if namespace is not None:
            if not namespace:
                raise ValueError("noise-bank namespace must be nonempty")
            self.namespace = str(namespace)
        self.seed = int(seed)
        self.seed_source = None
        self._seed_source_values = None
        self._semantic_scenario_ids = None
        self._uniform_cache.clear()
        self._processing_grid_cache_key = None
        self._processing_grid_cache_value = None
        self._degradation_uniform_cache_key = None
        self._degradation_uniform_cache_value = None
        self._degradation_uniform_grid_cache.clear()
        self._degradation_uniform_master_cache.clear()

    @staticmethod
    def _tensor(
        value: float,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(value, device=device, dtype=dtype)

    def processing_noise(
        self,
        *,
        episode_id: int,
        scenario_id: int,
        operation_id: int,
        machine_id: int,
        distribution: str,
        cov: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return unit-mean processing noise for one future operation pair."""

        self.diagnostic_counts["processing_noise_calls"] += 1
        if cov < 0:
            raise ValueError("processing coefficient of variation must be nonnegative")
        if cov == 0:
            return self._tensor(1.0, device=device, dtype=dtype)
        identity = (episode_id, scenario_id, operation_id, machine_id)
        if distribution == "lognormal":
            sigma2 = math.log1p(cov * cov)
            z = NormalDist().inv_cdf(self._uniform("processing_normal", *identity))
            value = math.exp(-0.5 * sigma2 + math.sqrt(sigma2) * z)
        elif distribution == "beta":
            if betaincinv is None:
                raise RuntimeError("SciPy is required for keyed beta processing noise")
            concentration = max(2.0, 0.5 / max(cov * cov, 1e-8) - 0.5)
            beta_draw = float(
                betaincinv(
                    concentration,
                    concentration,
                    self._uniform("processing_beta", *identity),
                )
            )
            amplitude = min(0.95, math.sqrt(12.0) * cov)
            value = (1.0 - amplitude) + 2.0 * amplitude * beta_draw
        elif distribution == "mixture":
            selector = self._uniform("processing_mixture", *identity)
            branch = "lognormal" if selector < 0.5 else "beta"
            value = float(
                self.processing_noise(
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    operation_id=operation_id,
                    machine_id=machine_id,
                    distribution=branch,
                    cov=cov,
                    device="cpu",
                    dtype=torch.float64,
                )
            )
        else:
            raise ValueError(f"unsupported processing-noise distribution: {distribution}")
        return self._tensor(value, device=device, dtype=dtype)

    def degradation_noise(
        self,
        *,
        episode_id: int,
        scenario_id: int,
        operation_id: int,
        machine_id: int,
        concentration: float | torch.Tensor,
        scale: float | torch.Tensor,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a Gamma increment from a keyed inverse-CDF primitive."""

        self.diagnostic_counts["degradation_noise_calls"] += 1
        shape = float(torch.as_tensor(concentration).detach().cpu())
        theta = float(torch.as_tensor(scale).detach().cpu())
        if shape < 0 or theta <= 0:
            raise ValueError("Gamma concentration must be nonnegative and scale positive")
        if shape == 0:
            return self._tensor(0.0, device=device, dtype=dtype)
        if gammaincinv is None:
            raise RuntimeError("SciPy is required for keyed Gamma degradation noise")
        uniform = self._uniform(
            "degradation_gamma",
            episode_id,
            scenario_id,
            operation_id,
            machine_id,
        )
        value = float(gammaincinv(shape, uniform)) * theta
        return self._tensor(value, device=device, dtype=dtype)

    def maintenance_noise(
        self,
        *,
        episode_id: int,
        scenario_id: int,
        machine_id: int,
        maintenance_count: int,
        maintenance_type: str,
        std: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a keyed nonnegative restoration residual for PM or CM."""

        self.diagnostic_counts["maintenance_noise_calls"] += 1
        maintenance_type = maintenance_type.upper()
        if maintenance_type not in {"PM", "CM"}:
            raise ValueError("maintenance_type must be PM or CM")
        if std < 0:
            raise ValueError("maintenance residual std must be nonnegative")
        if std == 0:
            return self._tensor(0.0, device=device, dtype=dtype)
        uniform = self._uniform(
            "maintenance_normal",
            episode_id,
            scenario_id,
            machine_id,
            maintenance_count,
            maintenance_type,
        )
        value = abs(NormalDist().inv_cdf(uniform)) * std
        return self._tensor(value, device=device, dtype=dtype)

    def processing_noise_grid(
        self,
        *,
        episode_id: int,
        scenario_count: int,
        operation_count: int,
        machine_count: int,
        cov: torch.Tensor,
        compatible: torch.Tensor,
        distribution: str,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize keyed processing factors without device-scalar round trips.

        Semantic keys and inverse-CDF formulae are identical to
        :meth:`processing_noise`.  The only optimization is to construct the
        scenario/operation/machine primitives once on the host and broadcast
        them over batch rows before one device transfer.
        """

        cov_np = cov.detach().cpu().double().numpy()
        compatible_np = compatible.detach().cpu().numpy().astype(bool)
        cache_key = (
            int(episode_id),
            int(scenario_count),
            int(operation_count),
            int(machine_count),
            str(distribution),
            cov_np.tobytes(),
            compatible_np.tobytes(),
            str(torch.device(device)),
            str(dtype),
        )
        if (
            cache_key == self._processing_grid_cache_key
            and self._processing_grid_cache_value is not None
        ):
            return self._processing_grid_cache_value
        batch = int(cov_np.shape[0])
        result = np.ones(
            (batch, scenario_count, operation_count, machine_count),
            dtype=np.float64,
        )
        logical = int(
            np.broadcast_to(compatible_np[:, None], result.shape).sum()
        )
        self.diagnostic_counts["processing_noise_calls"] += logical
        for scenario in range(scenario_count):
            for operation in range(operation_count):
                for machine in range(machine_count):
                    rows = compatible_np[:, operation, machine] & (cov_np[:, machine] > 0)
                    if not rows.any():
                        continue
                    identity = (episode_id, scenario, operation, machine)
                    if distribution == "lognormal":
                        z = NormalDist().inv_cdf(
                            self._uniform("processing_normal", *identity)
                        )
                        sigma2 = np.log1p(np.square(cov_np[rows, machine]))
                        result[rows, scenario, operation, machine] = np.exp(
                            -0.5 * sigma2 + np.sqrt(sigma2) * z
                        )
                    elif distribution == "beta":
                        uniform = self._uniform("processing_beta", *identity)
                        concentrations = np.maximum(
                            2.0,
                            0.5 / np.maximum(np.square(cov_np[rows, machine]), 1e-8) - 0.5,
                        )
                        draws = betaincinv(concentrations, concentrations, uniform)
                        amplitude = np.minimum(
                            0.95, np.sqrt(12.0) * cov_np[rows, machine]
                        )
                        result[rows, scenario, operation, machine] = (
                            1.0 - amplitude + 2.0 * amplitude * draws
                        )
                    elif distribution == "mixture":
                        selector = self._uniform("processing_mixture", *identity)
                        branch = "lognormal" if selector < 0.5 else "beta"
                        if branch == "lognormal":
                            z = NormalDist().inv_cdf(
                                self._uniform("processing_normal", *identity)
                            )
                            sigma2 = np.log1p(np.square(cov_np[rows, machine]))
                            result[rows, scenario, operation, machine] = np.exp(
                                -0.5 * sigma2 + np.sqrt(sigma2) * z
                            )
                        else:
                            uniform = self._uniform("processing_beta", *identity)
                            concentrations = np.maximum(
                                2.0,
                                0.5 / np.maximum(
                                    np.square(cov_np[rows, machine]), 1e-8
                                ) - 0.5,
                            )
                            draws = betaincinv(concentrations, concentrations, uniform)
                            amplitude = np.minimum(
                                0.95, np.sqrt(12.0) * cov_np[rows, machine]
                            )
                            result[rows, scenario, operation, machine] = (
                                1.0 - amplitude + 2.0 * amplitude * draws
                            )
                    else:
                        raise ValueError(
                            f"unsupported processing-noise distribution: {distribution}"
                        )
        tensor = torch.as_tensor(result, device=device, dtype=dtype)
        self._processing_grid_cache_key = cache_key
        self._processing_grid_cache_value = tensor
        return tensor

    def degradation_noise_grid(
        self,
        *,
        episode_id: int,
        concentration: torch.Tensor,
        scale: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized keyed Gamma inverse CDF for ``[B,S,N,M]`` candidates."""

        if gammaincinv is None:
            raise RuntimeError("SciPy is required for keyed Gamma degradation noise")
        concentration_np = concentration.detach().cpu().double().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        if concentration_np.ndim != 4 or active_np.shape != concentration_np.shape:
            raise ValueError("concentration and active must be matching [B,S,N,M]")
        if np.any(concentration_np[active_np] < 0):
            raise ValueError("Gamma concentration must be nonnegative")
        batch, scenarios, operations, machines = concentration_np.shape
        active_key = active_np.any(axis=0)
        uniform_key = (
            int(episode_id),
            int(scenarios),
            int(operations),
            int(machines),
            active_key.tobytes(),
        )
        master_key = (int(episode_id), scenarios, operations, machines)
        if master_key in self._degradation_uniform_master_cache:
            uniforms = self._degradation_uniform_master_cache[master_key]
        elif uniform_key in self._degradation_uniform_grid_cache:
            uniforms = self._degradation_uniform_grid_cache[uniform_key]
        else:
            uniforms = np.full(
                (scenarios, operations, machines), 0.5, dtype=np.float64
            )
            for scenario in range(scenarios):
                for operation in range(operations):
                    for machine in range(machines):
                        positive = active_np[:, scenario, operation, machine] & (
                            concentration_np[:, scenario, operation, machine] > 0
                        )
                        if positive.any():
                            uniforms[scenario, operation, machine] = self._uniform(
                                "degradation_gamma",
                                episode_id,
                                scenario,
                                operation,
                                machine,
                            )
            self._degradation_uniform_cache_key = uniform_key
            self._degradation_uniform_cache_value = uniforms
            self._degradation_uniform_grid_cache[uniform_key] = uniforms
            self._degradation_uniform_master_cache[master_key] = uniforms
        self.diagnostic_counts["degradation_noise_calls"] += int(active_np.sum())
        draws = np.zeros_like(concentration_np)
        positive_all = active_np & (concentration_np > 0)
        broadcast_uniforms = np.broadcast_to(uniforms[None], concentration_np.shape)
        draws[positive_all] = gammaincinv(
            concentration_np[positive_all], broadcast_uniforms[positive_all]
        )
        scale_np = scale.detach().cpu().double().numpy()
        draws *= np.broadcast_to(scale_np, draws.shape)
        return torch.as_tensor(
            draws, device=concentration.device, dtype=concentration.dtype
        )

    def prewarm_degradation_uniforms(
        self,
        *,
        episode_id: int,
        scenario_count: int,
        operation_count: int,
        machine_count: int,
        compatible: torch.Tensor,
    ) -> None:
        """Eagerly materialize the stable Gamma primitive grid for an episode."""

        master_key = (
            int(episode_id),
            int(scenario_count),
            int(operation_count),
            int(machine_count),
        )
        if master_key in self._degradation_uniform_master_cache:
            return
        compatible_np = compatible.detach().cpu().numpy().astype(bool).any(axis=0)
        uniforms = np.full(
            (scenario_count, operation_count, machine_count),
            0.5,
            dtype=np.float64,
        )
        for scenario in range(scenario_count):
            for operation in range(operation_count):
                for machine in range(machine_count):
                    if compatible_np[operation, machine]:
                        uniforms[scenario, operation, machine] = self._uniform(
                            "degradation_gamma",
                            int(episode_id),
                            scenario,
                            operation,
                            machine,
                        )
        self._degradation_uniform_master_cache[master_key] = uniforms

    def maintenance_noise_grid(
        self,
        *,
        episode_id: int,
        counts: torch.Tensor,
        maintenance_type: str | torch.Tensor,
        std: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize ``[B,S,M]`` restoration residuals in one transfer."""

        counts_np = counts.detach().cpu().numpy()
        std_np = std.detach().cpu().double().numpy()
        if counts_np.ndim != 3 or std_np.shape != (counts_np.shape[0], counts_np.shape[2]):
            raise ValueError("counts must be [B,S,M] and std must be [B,M]")
        if isinstance(maintenance_type, torch.Tensor):
            type_np = maintenance_type.detach().cpu().numpy().astype(bool)
            if type_np.shape != counts_np.shape:
                raise ValueError("maintenance type mask must match counts")
        else:
            name = str(maintenance_type).upper()
            if name not in {"PM", "CM"}:
                raise ValueError("maintenance_type must be PM or CM")
            type_np = np.full(counts_np.shape, name == "CM", dtype=bool)
        result = np.zeros(counts_np.shape, dtype=np.float64)
        batch, scenarios, machines = counts_np.shape
        self.diagnostic_counts["maintenance_noise_calls"] += int(result.size)
        for row in range(batch):
            for scenario in range(scenarios):
                for machine in range(machines):
                    sigma = float(std_np[row, machine])
                    if sigma == 0:
                        continue
                    name = "CM" if type_np[row, scenario, machine] else "PM"
                    uniform = self._uniform(
                        "maintenance_normal",
                        episode_id,
                        scenario,
                        machine,
                        int(counts_np[row, scenario, machine]),
                        name,
                    )
                    result[row, scenario, machine] = (
                        abs(NormalDist().inv_cdf(uniform)) * sigma
                    )
        return torch.as_tensor(result, device=device, dtype=dtype)

    def processing_noise_selected(
        self,
        *,
        episode_id: int,
        scenario_ids: torch.Tensor,
        operation_ids: torch.Tensor,
        machine_ids: torch.Tensor,
        cov: torch.Tensor,
        distribution: str,
        active: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Batched keyed processing noise for selected ``[B,S]`` transitions."""

        scenarios = scenario_ids.detach().cpu().long().numpy()
        operations = operation_ids.detach().cpu().long().numpy()
        machines = machine_ids.detach().cpu().long().numpy()
        cov_np = cov.detach().cpu().double().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        if not (
            scenarios.shape
            == operations.shape
            == machines.shape
            == cov_np.shape
            == active_np.shape
        ):
            raise ValueError("selected processing inputs must have matching [B,S]")
        result = np.ones(scenarios.shape, dtype=np.float64)
        self.diagnostic_counts["processing_noise_calls"] += int(result.size)
        for row, scenario in np.ndindex(scenarios.shape):
            if not active_np[row, scenario]:
                continue
            sigma = float(cov_np[row, scenario])
            if sigma == 0:
                continue
            identity = (
                int(episode_id),
                int(scenarios[row, scenario]),
                int(operations[row, scenario]),
                int(machines[row, scenario]),
            )
            branch = distribution
            if branch == "mixture":
                branch = (
                    "lognormal"
                    if self._uniform("processing_mixture", *identity) < 0.5
                    else "beta"
                )
            if branch == "lognormal":
                sigma2 = math.log1p(sigma * sigma)
                z = NormalDist().inv_cdf(
                    self._uniform("processing_normal", *identity)
                )
                result[row, scenario] = math.exp(
                    -0.5 * sigma2 + math.sqrt(sigma2) * z
                )
            elif branch == "beta":
                concentration = max(2.0, 0.5 / max(sigma * sigma, 1e-8) - 0.5)
                draw = float(
                    betaincinv(
                        concentration,
                        concentration,
                        self._uniform("processing_beta", *identity),
                    )
                )
                amplitude = min(0.95, math.sqrt(12.0) * sigma)
                result[row, scenario] = 1.0 - amplitude + 2.0 * amplitude * draw
            else:
                raise ValueError(
                    f"unsupported processing-noise distribution: {distribution}"
                )
        return torch.as_tensor(result, device=device, dtype=dtype)

    def degradation_noise_selected(
        self,
        *,
        episode_id: int,
        scenario_ids: torch.Tensor,
        operation_ids: torch.Tensor,
        machine_ids: torch.Tensor,
        concentration: torch.Tensor,
        scale: torch.Tensor,
        active: torch.Tensor,
        operation_count: int,
        machine_count: int,
        compatible: torch.Tensor,
    ) -> torch.Tensor:
        """One vectorized Gamma inverse-CDF call for selected ``[B,S]`` pairs."""

        shape_np = concentration.detach().cpu().double().numpy()
        scale_np = scale.detach().cpu().double().numpy()
        scenarios = scenario_ids.detach().cpu().long().numpy()
        operations = operation_ids.detach().cpu().long().numpy()
        machines = machine_ids.detach().cpu().long().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        if not (
            shape_np.shape
            == scale_np.shape
            == scenarios.shape
            == operations.shape
            == machines.shape
            == active_np.shape
        ):
            raise ValueError("selected degradation inputs must have matching [B,S]")
        if np.any(shape_np < 0) or np.any(scale_np <= 0):
            raise ValueError("Gamma concentration must be nonnegative and scale positive")
        if np.any(scenarios < 0):
            # Observed noise deliberately retains the historical scenario=-1
            # key.  It is not part of the nonnegative scenario grid.
            uniforms = np.zeros_like(shape_np)
            for index in np.ndindex(scenarios.shape):
                uniforms[index] = self._uniform(
                    "degradation_gamma",
                    int(episode_id),
                    int(scenarios[index]),
                    int(operations[index]),
                    int(machines[index]),
                )
        else:
            scenario_count = int(scenarios.max()) + 1
            self.prewarm_degradation_uniforms(
                episode_id=episode_id,
                scenario_count=scenario_count,
                operation_count=operation_count,
                machine_count=machine_count,
                compatible=compatible,
            )
            grid = self._degradation_uniform_master_cache[
                (int(episode_id), scenario_count, operation_count, machine_count)
            ]
            uniforms = grid[scenarios, operations, machines]
        positive = (shape_np > 0) & active_np
        self.diagnostic_counts["degradation_noise_calls"] += int(shape_np.size)
        result = np.zeros_like(shape_np)
        result[positive] = (
            gammaincinv(shape_np[positive], uniforms[positive]) * scale_np[positive]
        )
        return torch.as_tensor(
            result, device=concentration.device, dtype=concentration.dtype
        )

    def maintenance_noise_selected(
        self,
        *,
        episode_id: int,
        scenario_ids: torch.Tensor,
        machine_ids: torch.Tensor,
        counts: torch.Tensor,
        maintenance_is_cm: torch.Tensor,
        std: torch.Tensor,
        active: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Batched keyed PM/CM residuals for selected ``[B,S]`` machines."""

        scenarios = scenario_ids.detach().cpu().long().numpy()
        machines = machine_ids.detach().cpu().long().numpy()
        count_np = counts.detach().cpu().long().numpy()
        type_np = maintenance_is_cm.detach().cpu().numpy().astype(bool)
        std_np = std.detach().cpu().double().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        if not (
            scenarios.shape
            == machines.shape
            == count_np.shape
            == type_np.shape
            == std_np.shape
            == active_np.shape
        ):
            raise ValueError("selected maintenance inputs must have matching [B,S]")
        result = np.zeros(scenarios.shape, dtype=np.float64)
        self.diagnostic_counts["maintenance_noise_calls"] += int(result.size)
        for row, scenario in np.ndindex(scenarios.shape):
            if not active_np[row, scenario]:
                continue
            sigma = float(std_np[row, scenario])
            if sigma == 0:
                continue
            name = "CM" if type_np[row, scenario] else "PM"
            uniform = self._uniform(
                "maintenance_normal",
                int(episode_id),
                int(scenarios[row, scenario]),
                int(machines[row, scenario]),
                int(count_np[row, scenario]),
                name,
            )
            result[row, scenario] = abs(NormalDist().inv_cdf(uniform)) * sigma
        return torch.as_tensor(result, device=device, dtype=dtype)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the bank, including already materialized primitives."""

        return {
            "format": self.FORMAT,
            "seed": self.seed,
            "namespace": self.namespace,
            "uniform_cache": deepcopy(self._uniform_cache),
            "seed_source": (
                None
                if self.seed_source is None
                else self.seed_source.detach().cpu().tolist()
            ),
            "semantic_scenario_ids": self._semantic_scenario_ids,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore a bank without consuming a random draw."""

        if payload.get("format") != self.FORMAT:
            raise ValueError("unsupported trajectory-noise-bank format")
        if int(payload["seed"]) != self.seed or str(payload["namespace"]) != self.namespace:
            raise ValueError("noise-bank seed or namespace does not match environment")
        self._uniform_cache = {
            str(key): float(value) for key, value in payload["uniform_cache"].items()
        }
        semantic = payload.get("semantic_scenario_ids")
        self._semantic_scenario_ids = (
            None if semantic is None else [int(value) for value in semantic]
        )
        self._processing_grid_cache_key = None
        self._processing_grid_cache_value = None
        self._degradation_uniform_cache_key = None
        self._degradation_uniform_cache_value = None
        self._degradation_uniform_grid_cache.clear()
        self._degradation_uniform_master_cache.clear()


def sample_processing_noise(
    shape: Sequence[int],
    *,
    distribution: str = "lognormal",
    cov: float = 0.20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample positive, unit-mean multiplicative processing noise."""

    if cov < 0:
        raise ValueError("cov must be nonnegative")
    if cov == 0:
        return torch.ones(tuple(shape), device=device, dtype=dtype)
    generator = _generator(seed, device)
    if distribution == "lognormal":
        sigma2 = math.log1p(cov * cov)
        normal = torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype)
        return torch.exp((-0.5 * sigma2) + math.sqrt(sigma2) * normal)
    if distribution == "beta":
        concentration = max(2.0, 0.5 / max(cov * cov, 1e-8) - 0.5)
        alpha = torch.full(tuple(shape), concentration, device=device, dtype=dtype)
        x = torch._standard_gamma(alpha, generator=generator)
        y = torch._standard_gamma(alpha, generator=generator)
        unit_beta = x / (x + y).clamp_min(torch.finfo(dtype).eps)
        amplitude = min(0.95, math.sqrt(12.0) * cov)
        return (1.0 - amplitude) + 2.0 * amplitude * unit_beta
    if distribution == "mixture":
        log_noise = sample_processing_noise(
            shape,
            distribution="lognormal",
            cov=cov,
            seed=seed + 17,
            device=device,
            dtype=dtype,
        )
        beta_noise = sample_processing_noise(
            shape,
            distribution="beta",
            cov=cov,
            seed=seed + 31,
            device=device,
            dtype=dtype,
        )
        selector = torch.rand(tuple(shape), generator=generator, device=device) < 0.5
        return torch.where(selector, log_noise, beta_noise)
    raise ValueError(f"unsupported processing-noise distribution: {distribution}")


def sample_degradation_noise(
    concentration: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    seed: int,
) -> torch.Tensor:
    """Sample Gamma increments with explicit generator seed."""

    if torch.any(concentration < 0):
        raise ValueError("Gamma concentration must be nonnegative")
    generator = _generator(seed, concentration.device)
    draw = torch._standard_gamma(concentration.clamp_min(1e-8), generator=generator)
    result = draw * torch.as_tensor(
        scale, device=concentration.device, dtype=concentration.dtype
    )
    return torch.where(concentration > 0, result, torch.zeros_like(result))


def sample_maintenance_noise(
    shape: Sequence[int],
    *,
    std: float,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample nonnegative restoration residuals with an explicit seed."""

    if std < 0:
        raise ValueError("maintenance noise std must be nonnegative")
    if std == 0:
        return torch.zeros(tuple(shape), device=device, dtype=dtype)
    generator = _generator(seed, device)
    return torch.randn(
        tuple(shape), generator=generator, device=device, dtype=dtype
    ).mul(std).abs()
