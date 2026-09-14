"""Metrics for masked normal-map evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diligent_validate import angular_error_map


def normal_error_metrics(estimated: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    errors = angular_error_map(estimated, ground_truth, mask)
    valid = errors[mask & np.isfinite(errors)]
    if valid.size == 0:
        raise ValueError("No finite angular errors exist inside the evaluation mask")
    metrics = {
        "mae_deg": float(np.mean(valid)),
        "median_deg": float(np.median(valid)),
        "p90_deg": float(np.percentile(valid, 90)),
        "p95_deg": float(np.percentile(valid, 95)),
        "valid_pixel_count": int(valid.size),
    }
    return metrics, errors


def attach_l2_comparison(records: list[dict[str, Any]]) -> None:
    references = {
        (str(record["object"]), int(record["lights"])): float(record["mae_deg"])
        for record in records
        if record.get("status") == "ok" and record.get("method") == "l2"
    }
    for record in records:
        if record.get("status") != "ok":
            # Failed and unsupported cases stay present in the summary, but must
            # not acquire metric-looking fields that downstream code could
            # accidentally coerce into an accuracy aggregate.
            record.pop("delta_mae_vs_l2_deg", None)
            record.pop("relative_improvement_vs_l2_pct", None)
            continue
        baseline = references.get((str(record["object"]), int(record["lights"])))
        if baseline is None:
            record["delta_mae_vs_l2_deg"] = None
            record["relative_improvement_vs_l2_pct"] = None
            continue
        mae = float(record["mae_deg"])
        record["delta_mae_vs_l2_deg"] = mae - baseline
        record["relative_improvement_vs_l2_pct"] = (
            100.0 * (baseline - mae) / baseline if baseline else None
        )
