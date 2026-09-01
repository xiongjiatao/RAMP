from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ramp import ActionCodec, RAMPConfig
from ramp.experiments import configure_environment, configure_model, get_method
from model.policy_factory import build_policy


ROOT = Path(__file__).resolve().parents[1]


def test_paper_contract_and_action_space() -> None:
    healthy = RAMPConfig.from_paper_regime("H0")
    active = configure_environment(
        get_method("ramp"), num_scenarios=32, seed=400, epsilon_use=0.05
    )
    assert not healthy.action_conditioned_degradation
    assert active.action_conditioned_degradation
    assert active.maintenance_actions
    assert ActionCodec(10, 5).total_actions == 60
    assert type(build_policy(configure_model(get_method("ramp"), smoke=True))).__name__ == "RAMPPolicy"


def test_smoke_entrypoint(tmp_path: Path) -> None:
    log_path = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            "train_ramp.py",
            "--smoke",
            "--updates",
            "1",
            "--validation-limit",
            "1",
            "--stochastic-eval-samples",
            "1",
            "--log",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "SMOKE_ONLY"' in result.stdout
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["status"] == "SMOKE_ONLY"
    assert payload["test"]["invalid_rows"] == 0
