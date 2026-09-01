"""Generate a health overlay next to, never inside, nominal SD1 raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from ramp.config import HealthOverlayConfig
from ramp.overlay import build_health_overlay
from data_utils import load_data_from_single_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAMP health overlay")
    parser.add_argument("instance", type=Path, help="nominal .fjs instance")
    parser.add_argument("output", type=Path, help="health_overlay output directory")
    parser.add_argument("--num-scenarios", type=int, default=32)
    parser.add_argument("--seed", type=int, default=400)
    args = parser.parse_args()
    job_lengths, nominal = load_data_from_single_file(str(args.instance))
    if len(job_lengths) == 0:
        raise FileNotFoundError(args.instance)
    config = HealthOverlayConfig()
    overlay = build_health_overlay(
        nominal,
        args.num_scenarios,
        seed=args.seed,
        config=config,
    )
    overlay.save(args.output, config=config)
    print(f"saved RAMP health overlay to {args.output}")


if __name__ == "__main__":
    main()
