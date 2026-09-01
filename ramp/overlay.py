"""Synthetic health overlays kept separate from the nominal SD1 data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import HealthOverlayConfig


@dataclass
class HealthOverlay:
    failure_level: torch.Tensor
    initial_health: torch.Tensor
    loads: torch.Tensor
    alpha: torch.Tensor
    theta: torch.Tensor
    load_sensitivity: torch.Tensor
    eta: torch.Tensor
    health_time_gamma: torch.Tensor
    pm_rho: torch.Tensor
    pm_duration: torch.Tensor
    pm_cost: torch.Tensor
    cm_rho: torch.Tensor
    cm_duration: torch.Tensor
    cm_cost: torch.Tensor
    maintenance_noise_std: torch.Tensor
    processing_cov: torch.Tensor
    scenario_seeds: torch.Tensor

    def clone(self) -> "HealthOverlay":
        """Deep-clone every tensor before applying an experiment overlay."""

        return HealthOverlay(
            **{name: value.detach().clone() for name, value in vars(self).items()}
        )

    def to(self, device: torch.device | str) -> "HealthOverlay":
        return HealthOverlay(
            **{name: value.to(device) for name, value in vars(self).items()}
        )

    @property
    def batch_size(self) -> int:
        return int(self.loads.shape[0])

    @property
    def number_of_machines(self) -> int:
        return int(self.loads.shape[-1])

    @property
    def num_scenarios(self) -> int:
        return int(self.scenario_seeds.shape[1])

    def validate(self, nominal_processing_times: torch.Tensor | None = None) -> None:
        if self.initial_health.ndim != 2:
            raise ValueError(
                "initial health must be scenario-free with shape [B,M]; "
                "use HealthOverlay.load for validated legacy migration"
            )
        b, m = self.initial_health.shape
        if self.loads.ndim != 3 or self.loads.shape[0] != b or self.loads.shape[2] != m:
            raise ValueError("loads must be [B,N,M] and match initial health")
        for name in (
            "failure_level", "alpha", "theta", "load_sensitivity", "eta",
            "health_time_gamma", "pm_rho", "pm_duration", "pm_cost", "cm_rho",
            "cm_duration", "cm_cost", "maintenance_noise_std", "processing_cov",
        ):
            if getattr(self, name).shape != (b, m):
                raise ValueError(f"{name} must have shape [B,M]")
        if self.scenario_seeds.ndim != 2 or self.scenario_seeds.shape[0] != b:
            raise ValueError("scenario_seeds must be [B,S]")
        if torch.any(self.failure_level <= 0):
            raise ValueError("failure levels must be positive")
        if torch.any(self.initial_health < 0) or torch.any(
            self.initial_health >= self.failure_level
        ):
            raise ValueError("initial health must lie in [0,L)")
        if torch.any(self.alpha <= 0) or torch.any(self.theta <= 0):
            raise ValueError("Gamma parameters must be positive")
        if torch.any(self.cm_duration <= self.pm_duration):
            raise ValueError("CM duration must exceed PM duration")
        if torch.any(self.cm_cost <= self.pm_cost):
            raise ValueError("CM cost must exceed PM cost")
        if nominal_processing_times is not None:
            nominal = torch.as_tensor(nominal_processing_times)
            if nominal.shape != self.loads.shape:
                raise ValueError("nominal processing matrix does not match overlay")
            if torch.any(self.loads[nominal == 0] != 0):
                raise ValueError("incompatible pairs must have zero load")

    def save(self, directory: str | Path, config: HealthOverlayConfig | None = None) -> None:
        """Write the five-paper overlay groups without touching nominal data."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "initial_health_states.npy", self.initial_health.cpu().numpy())
        np.save(directory / "operation_machine_loads.npy", self.loads.cpu().numpy())
        np.save(directory / "scenario_seeds.npy", self.scenario_seeds.cpu().numpy())
        degradation: Dict[str, list] = {
            name: getattr(self, name).cpu().tolist()
            for name in (
                "failure_level", "alpha", "theta", "load_sensitivity", "eta",
                "health_time_gamma", "processing_cov",
            )
        }
        maintenance: Dict[str, list] = {
            name: getattr(self, name).cpu().tolist()
            for name in (
                "pm_rho", "pm_duration", "pm_cost", "cm_rho", "cm_duration",
                "cm_cost", "maintenance_noise_std",
            )
        }
        (directory / "machine_degradation_parameters.json").write_text(
            json.dumps(degradation, indent=2), encoding="utf-8"
        )
        (directory / "maintenance_parameters.json").write_text(
            json.dumps(maintenance, indent=2), encoding="utf-8"
        )
        manifest = {
            "format": "RAMP health overlay v2",
            "initial_health_semantics": "one observed scenario-free state [B,M]",
            "nominal_data_modified": False,
            "batch_size": self.batch_size,
            "num_scenarios": self.num_scenarios,
            "config": asdict(config) if config is not None else None,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, device: torch.device | str = "cpu") -> "HealthOverlay":
        directory = Path(directory)
        degradation = json.loads(
            (directory / "machine_degradation_parameters.json").read_text(encoding="utf-8")
        )
        maintenance = json.loads(
            (directory / "maintenance_parameters.json").read_text(encoding="utf-8")
        )
        values = {**degradation, **maintenance}
        values["initial_health"] = np.load(directory / "initial_health_states.npy")
        values["loads"] = np.load(directory / "operation_machine_loads.npy")
        values["scenario_seeds"] = np.load(directory / "scenario_seeds.npy")
        tensors = {
            name: torch.as_tensor(value, device=device)
            for name, value in values.items()
        }
        tensors["scenario_seeds"] = tensors["scenario_seeds"].long()
        initial = tensors["initial_health"]
        if initial.ndim == 3:
            reference = initial[:, :1, :].expand_as(initial)
            if not torch.equal(initial, reference):
                raise ValueError(
                    "legacy overlay contains conflicting current health scenarios; "
                    "no scenario may be selected as observed authority"
                )
            tensors["initial_health"] = initial[:, 0, :].clone()
        overlay = cls(**tensors)
        overlay.validate()
        return overlay


def _uniform(
    rng: np.random.Generator, low: float, high: float, shape: tuple[int, ...]
) -> np.ndarray:
    if low == high:
        return np.full(shape, low, dtype=np.float32)
    return rng.uniform(low, high, size=shape).astype(np.float32)


def compute_operation_machine_loads(
    nominal_processing_times: torch.Tensor | np.ndarray,
    exponent: float = 1.0,
) -> torch.Tensor:
    """Compute dimensionless load relative to each machine's median task."""

    nominal = torch.as_tensor(nominal_processing_times, dtype=torch.float32)
    squeeze = nominal.ndim == 2
    if squeeze:
        nominal = nominal.unsqueeze(0)
    if nominal.ndim != 3:
        raise ValueError("nominal processing times must be [N,M] or [B,N,M]")
    loads = torch.zeros_like(nominal)
    for batch in range(nominal.shape[0]):
        for machine in range(nominal.shape[2]):
            values = nominal[batch, :, machine]
            feasible = values > 0
            if feasible.any():
                median = values[feasible].median().clamp_min(1e-8)
                loads[batch, feasible, machine] = (
                    values[feasible] / median
                ).pow(exponent)
    return loads.squeeze(0) if squeeze else loads


def build_health_overlay(
    nominal_processing_times: torch.Tensor | np.ndarray,
    num_scenarios: int,
    *,
    seed: int = 400,
    config: HealthOverlayConfig | None = None,
    device: torch.device | str = "cpu",
) -> HealthOverlay:
    config = config or HealthOverlayConfig()
    config.validate()
    nominal = torch.as_tensor(nominal_processing_times, dtype=torch.float32)
    if nominal.ndim == 2:
        nominal = nominal.unsqueeze(0)
    if nominal.ndim != 3:
        raise ValueError("nominal processing times must be [N,M] or [B,N,M]")
    b, _, m = nominal.shape
    rng = np.random.default_rng(seed)
    shape = (b, m)
    failure_level = np.full(shape, config.failure_level, dtype=np.float32)
    base_health = _uniform(
        rng, config.initial_health_low, config.initial_health_high, shape
    )
    # The current observation is scenario-free. Scenario uncertainty starts
    # only in keyed future transition noise.
    initial_health = base_health
    pm_duration = _uniform(rng, config.pm_duration_low, config.pm_duration_high, shape)
    pm_cost = _uniform(rng, config.pm_cost_low, config.pm_cost_high, shape)
    tensors = {
        "failure_level": failure_level,
        "initial_health": initial_health,
        "loads": compute_operation_machine_loads(
            nominal, config.load_transform_exponent
        ).cpu().numpy(),
        "alpha": _uniform(
            rng, config.degradation_alpha_low, config.degradation_alpha_high, shape
        ),
        "theta": _uniform(
            rng, config.degradation_theta_low, config.degradation_theta_high, shape
        ),
        "load_sensitivity": _uniform(
            rng, config.load_sensitivity_low, config.load_sensitivity_high, shape
        ),
        "eta": _uniform(rng, config.health_time_eta_low, config.health_time_eta_high, shape),
        "health_time_gamma": _uniform(
            rng, config.health_time_gamma_low, config.health_time_gamma_high, shape
        ),
        "pm_rho": _uniform(rng, config.pm_rho_low, config.pm_rho_high, shape),
        "pm_duration": pm_duration,
        "pm_cost": pm_cost,
        "cm_rho": np.full(shape, config.cm_rho, dtype=np.float32),
        "cm_duration": pm_duration * config.cm_duration_multiplier,
        "cm_cost": pm_cost * config.cm_cost_multiplier,
        "maintenance_noise_std": np.full(
            shape, config.maintenance_noise_std, dtype=np.float32
        ),
        "processing_cov": np.full(shape, config.processing_cov, dtype=np.float32),
        "scenario_seeds": np.arange(
            seed, seed + b * num_scenarios, dtype=np.int64
        ).reshape(b, num_scenarios),
    }
    overlay = HealthOverlay(
        **{
            name: torch.as_tensor(value, device=device)
            for name, value in tensors.items()
        }
    )
    overlay.scenario_seeds = overlay.scenario_seeds.long()
    overlay.validate(nominal.to(device))
    return overlay
