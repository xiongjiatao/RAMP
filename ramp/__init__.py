"""RAMP: action-conditioned health scenario scheduling.

The package reuses the nominal FJSP instance format; degradation and
maintenance data live in a reproducible health overlay.
"""

from .config import RAMPConfig, HealthOverlayConfig, ObjectiveConfig
from .env import (
    RAMPEnv,
    RAMPNoFeatureEnv,
    RAMPScenarioEnv,
)
from .overlay import HealthOverlay, build_health_overlay
from .noise import TrajectoryNoiseBank
from .oracle import (
    BoundedOpenLoopScenarioOracle,
    ExactScenarioOracle,
    ExactScenarioOracleResult,
    TinyScenarioTreeOracle,
    TinyScenarioTreeOracleResult,
)
from .state import (
    RAMPEnvState,
    ActionCodec,
    ActionType,
    ObservedShopState,
    ScenarioTrajectoryState,
)
from .transition import RAMPTransitionKernel
from .units import convert_time_units

__all__ = [
    "RAMPConfig",
    "RAMPEnvState",
    "ActionCodec",
    "RAMPEnv",
    "RAMPNoFeatureEnv",
    "ActionType",
    "ExactScenarioOracle",
    "ExactScenarioOracleResult",
    "BoundedOpenLoopScenarioOracle",
    "HealthOverlay",
    "HealthOverlayConfig",
    "RAMPTransitionKernel",
    "ObjectiveConfig",
    "ObservedShopState",
    "RAMPScenarioEnv",
    "TrajectoryNoiseBank",
    "ScenarioTrajectoryState",
    "TinyScenarioTreeOracle",
    "TinyScenarioTreeOracleResult",
    "build_health_overlay",
    "convert_time_units",
]
