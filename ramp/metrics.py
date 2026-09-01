"""Versioned two-level metric and result-admission contract for paper one."""

from __future__ import annotations

import math
from typing import Any, Mapping


PAPER_INSTANCE_SCALAR_METRICS: tuple[str, ...] = (
    "expected_reward_scenario_makespan",
    "mean_total_cost",
    "cvar_0_95_total_cost",
    "pm_cost",
    "cm_cost",
    "failure_probability",
    "failure_count",
    "planned_downtime",
    "unplanned_downtime",
    "physical_availability",
    "production_utilization",
    "inference_time",
)
PAPER_RUN_SCALAR_METRICS: tuple[str, ...] = (
    "training_time",
    "parameter_count",
)
PAPER_SCALAR_METRICS = PAPER_INSTANCE_SCALAR_METRICS + PAPER_RUN_SCALAR_METRICS

# Training-protocol diagnostics are intentionally separate from the fourteen
# paper outcome metrics. They establish ATMSL efficiency and quality controls;
# they never replace scheduling performance or risk outcomes.
ATMSL_DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "core_rollout_ppo_seconds",
    "full_update_seconds",
    "total_training_time",
    "average_seconds_per_update",
    "stage_A_updates",
    "stage_B_updates",
    "stage_C_updates",
    "fallback_update_count",
    "tail_coverage",
    "correction_mse",
    "correction_mae",
    "correction_relative_residual",
    "representative_scenario_count",
    "effective_scenario_mass",
    "training_speedup",
    "valid_completion_rate",
)

PAPER_STRUCTURED_METRICS: tuple[str, ...] = (
    "physical_availability_by_machine",
    "production_utilization_by_machine",
    "scenario_results",
)
BEHAVIOR_AUDIT_FIELDS: tuple[str, ...] = (
    "observed_pm_count",
    "observed_cm_count",
    "observed_failure_count",
    "observed_unplanned_downtime",
    "observed_pm_cost",
    "observed_cm_cost",
)
RESULT_ADMISSION_FIELDS: tuple[str, ...] = (
    "completed_schedule",
    "terminated",
    "truncated",
    "unfinished_operation_count",
    "completion_ratio",
    "evaluation_valid",
    "evaluation_status",
)
PAPER_METRIC_SCHEMA = "RAMP first-paper instance metrics v2"
PAPER_RUN_METRIC_SCHEMA = "RAMP first-paper run metrics v1"
VALID_EVALUATION_STATUS = "VALID_COMPLETED_SCHEDULE"
INVALID_EVALUATION_STATUS = "INVALID_INCOMPLETE_SCHEDULE"

# Reviewer-facing names are mapped explicitly to storage names.  Diagnostic
# quantities such as raw weighted cost and the observed calendar horizon are
# intentionally outside this 14-metric contract.
PAPER_METRIC_CONTRACT: tuple[tuple[str, str, str], ...] = (
    ("expected makespan", "instance", "expected_reward_scenario_makespan"),
    ("mean total cost", "instance", "mean_total_cost"),
    ("CVaR total cost", "instance", "cvar_0_95_total_cost"),
    ("PM cost", "instance", "pm_cost"),
    ("CM cost", "instance", "cm_cost"),
    ("failure probability", "instance", "failure_probability"),
    ("failure count", "instance", "failure_count"),
    ("planned downtime", "instance", "planned_downtime"),
    ("unplanned downtime", "instance", "unplanned_downtime"),
    ("physical availability", "instance", "physical_availability"),
    ("production utilization", "instance", "production_utilization"),
    ("inference time", "instance", "inference_time"),
    ("training time", "run", "training_time"),
    ("parameter count", "run", "parameter_count"),
)


def _require_finite(payload: Mapping[str, Any], names: tuple[str, ...], kind: str) -> None:
    missing = set(names) - payload.keys()
    if missing:
        raise ValueError(f"{kind} lacks metrics: {sorted(missing)}")
    nonfinite = [name for name in names if not math.isfinite(float(payload[name]))]
    if nonfinite:
        raise ValueError(f"{kind} has nonfinite metrics: {nonfinite}")


def validate_paper_run_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("paper_run_metric_schema") != PAPER_RUN_METRIC_SCHEMA:
        raise ValueError("paper run metric schema mismatch")
    _require_finite(summary, PAPER_RUN_SCALAR_METRICS, "paper-run summary")
    if float(summary["training_time"]) < 0 or int(summary["parameter_count"]) <= 0:
        raise ValueError("paper-run metrics must have nonnegative time and positive parameters")


def validate_paper_result_row(
    row: Mapping[str, Any], run_summary: Mapping[str, Any] | None = None
) -> None:
    """Reject incomplete schedules and rows lacking the complete paper contract."""

    if row.get("paper_metric_schema") != PAPER_METRIC_SCHEMA:
        raise ValueError("paper instance metric schema mismatch")
    missing_admission = set(RESULT_ADMISSION_FIELDS) - row.keys()
    if missing_admission:
        raise ValueError(f"paper-result row lacks admission fields: {sorted(missing_admission)}")
    valid = (
        bool(row["completed_schedule"])
        and bool(row["terminated"])
        and not bool(row["truncated"])
        and int(row["unfinished_operation_count"]) == 0
        and float(row["completion_ratio"]) == 1.0
        and bool(row["evaluation_valid"])
        and row["evaluation_status"] == VALID_EVALUATION_STATUS
    )
    if not valid:
        raise ValueError(f"inadmissible evaluation row: {row.get('evaluation_status')}")
    _require_finite(row, PAPER_INSTANCE_SCALAR_METRICS, "paper-result row")
    if int(row.get("total_operation_count", 0)) > 0 and float(row["production_utilization"]) <= 0:
        raise ValueError("a completed nonempty schedule must have positive production utilization")
    missing = set(PAPER_STRUCTURED_METRICS) - row.keys()
    if missing:
        raise ValueError(f"paper-result row lacks structured metrics: {sorted(missing)}")
    behavior_audit = row.get("behavior_audit")
    if behavior_audit is not None:
        if not isinstance(behavior_audit, Mapping):
            raise ValueError("behavior_audit must be a mapping")
        if set(behavior_audit) != set(BEHAVIOR_AUDIT_FIELDS):
            raise ValueError("behavior_audit schema mismatch")
        _require_finite(
            behavior_audit,
            BEHAVIOR_AUDIT_FIELDS,
            "behavior audit",
        )
        if any(float(behavior_audit[name]) < 0 for name in BEHAVIOR_AUDIT_FIELDS):
            raise ValueError("behavior audit values must be nonnegative")
    scenarios = row["scenario_results"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("paper-result row requires nonempty scenario_results")
    scenario_fields = {
        "scenario_id", "makespan", "pm_cost", "cm_cost",
        "unplanned_downtime", "failure_count", "total_cost",
    }
    for scenario in scenarios:
        if set(scenario) != scenario_fields:
            raise ValueError("scenario-result schema mismatch")
    if run_summary is not None:
        validate_paper_run_summary(run_summary)
