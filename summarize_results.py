"""Aggregate RAMP raw-result files into the paper's mean +/- sample SD table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = {
    "Cost": "mean_total_cost",
    "Makespan": "expected_reward_scenario_makespan",
    "CVaR": "cvar_0_95_total_cost",
}


def summarize(paths: list[Path]) -> dict[str, object]:
    run_means: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} is not a raw-results list")
        per_file: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            if not row.get("evaluation_valid", False):
                continue
            mode = str(row["mode"])
            target = per_file.setdefault(mode, {name: [] for name in METRICS})
            for name, key in METRICS.items():
                target[name].append(float(row[key]))
            target.setdefault("Phi", []).append(
                float(row["mean_total_cost"])
                + 0.5 * float(row["cvar_0_95_total_cost"])
            )
        for mode, values in per_file.items():
            target = run_means.setdefault(mode, {name: [] for name in (*METRICS, "Phi")})
            for name, samples in values.items():
                if samples:
                    target[name].append(mean(samples))

    result: dict[str, object] = {}
    for mode, values in sorted(run_means.items()):
        result[mode] = {
            name: {
                "mean": mean(samples),
                "sample_sd": stdev(samples) if len(samples) > 1 else 0.0,
                "runs": len(samples),
            }
            for name, samples in values.items()
            if samples
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.raw_results)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
