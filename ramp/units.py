"""Explicit, physics-consistent time-unit conversion for RAMP instances."""

from __future__ import annotations

from dataclasses import replace

import torch

from .config import RAMPConfig
from .overlay import HealthOverlay


def convert_time_units(
    nominal_processing_times: torch.Tensor,
    overlay: HealthOverlay,
    config: RAMPConfig,
    *,
    factor: float,
) -> tuple[torch.Tensor, HealthOverlay, RAMPConfig]:
    """Convert all calendar times by ``factor`` without changing physics.

    For example, minutes-to-seconds uses ``factor=60``. Processing,
    maintenance, and diagnosis durations scale by 60, while the Gamma shape
    rate per unit time (`alpha`) scales by 1/60. Dimensionless policy features,
    safety masks, scenario objective, and survival probabilities are invariant.
    """

    if not torch.isfinite(torch.tensor(factor)) or factor <= 0:
        raise ValueError("time-unit conversion factor must be finite and positive")
    converted = overlay.clone()
    converted.pm_duration.mul_(factor)
    converted.cm_duration.mul_(factor)
    converted.alpha.div_(factor)
    converted_config = replace(
        config,
        failure_diagnosis_delay=config.failure_diagnosis_delay * factor,
    )
    return nominal_processing_times * factor, converted, converted_config
