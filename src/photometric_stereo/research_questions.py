"""Quantitative answers to RQ1-RQ3 from assembled benchmark records.

Every number here is derived mechanically from the summary records so the written
conclusions can be traced back to them. Aggregates follow the same coverage rule as
the figures: a method contributes at a light count only if it succeeded on every
evaluated object there, so no mean is taken over a different object set.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from .visualization import METHOD_LABELS, METHOD_ORDER, is_complete

TARGET_LIGHTS = 4
ALPHA = 0.05


def _by_object(records: Sequence[dict[str, Any]], method: str, lights: int, key: str) -> dict[str, float]:
    return {
        str(record["object"]): float(record[key])
        for record in records
        if record.get("status") == "ok" and record.get("method") == method
        and record.get("lights") is not None and int(record["lights"]) == lights
        and record.get(key) is not None
    }


def _methods(records: Sequence[dict[str, Any]]) -> list[str]:
    present = {record.get("method") for record in records}
    return [method for method in METHOD_ORDER if method in present]


def _light_counts(records: Sequence[dict[str, Any]]) -> list[int]:
    return sorted({int(record["lights"]) for record in records if record.get("lights") is not None})


def supported_light_counts(records: Sequence[dict[str, Any]], method: str) -> list[int]:
    return [count for count in _light_counts(records) if is_complete(list(records), method, count)]


def sign_test_p(differences: Sequence[float]) -> float | None:
    """Exact two-sided sign test on paired differences; ties are dropped."""
    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if n == 0:
        return None
    positive = sum(1 for value in nonzero if value > 0)
    extreme = min(positive, n - positive)
    tail = sum(math.comb(n, i) for i in range(extreme + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def wilcoxon_p(differences: Sequence[float]) -> float | None:
    """Exact two-sided Wilcoxon signed-rank test, or None when SciPy is unavailable."""
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        return float(wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue)
    except TypeError:  # SciPy < 1.9 names the argument `mode`
        return float(wilcoxon(nonzero, alternative="two-sided", mode="exact").pvalue)


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm-Bonferroni adjustment across the family of comparisons against L2."""
    valid = sorted(((p, name) for name, p in p_values.items() if p is not None))
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    m = len(valid)
    for rank, (p, name) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def _paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], list[float]]:
    objects = sorted(set(a) & set(b))
    return objects, [a[name] - b[name] for name in objects]


def rq1_accuracy_loss(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """RQ1: how much accuracy 4-light reconstruction loses against 32/96 lights."""
    result: dict[str, Any] = {}
    for method in _methods(records):
        supported = supported_light_counts(records, method)
        entry: dict[str, Any] = {"supported_light_counts": supported}
        if TARGET_LIGHTS not in supported:
            entry["note"] = f"{TARGET_LIGHTS}-light case lacks complete object coverage"
            result[method] = entry
            continue
        four = _by_object(records, method, TARGET_LIGHTS, "mae_deg")
        four_p95 = _by_object(records, method, TARGET_LIGHTS, "p95_deg")
        entry["mae_deg_at_4"] = statistics.mean(four.values())
        entry["p95_deg_at_4"] = statistics.mean(four_p95.values())
        references = sorted({count for count in (32, 96, max(supported)) if count in supported and count != TARGET_LIGHTS})
        for reference in references:
            other = _by_object(records, method, reference, "mae_deg")
            other_p95 = _by_object(records, method, reference, "p95_deg")
            objects, losses = _paired(four, other)
            reference_mean = statistics.mean(other[name] for name in objects)
            entry[f"vs_{reference}_lights"] = {
                "objects": len(objects),
                "mae_deg_reference": reference_mean,
                "mae_loss_deg": statistics.mean(losses),
                "mae_loss_median_deg": statistics.median(losses),
                "mae_loss_pct_of_reference": 100.0 * statistics.mean(losses) / reference_mean,
                "objects_worse_at_4": sum(1 for value in losses if value > 0),
                "p95_deg_reference": statistics.mean(other_p95[name] for name in objects),
                "p95_loss_deg": statistics.mean(four_p95[name] - other_p95[name] for name in objects),
                "is_max_supported_on_this_gpu": reference == max(supported),
            }
        unsupported = [count for count in _light_counts(records) if count not in supported]
        if unsupported:
            entry["unsupported_light_counts"] = unsupported
        result[method] = entry
    return result


def rq2_versus_l2_at_four(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """RQ2: whether robust or deep methods beat classical L2 when only 4 lights remain."""
    l2_mae = _by_object(records, "l2", TARGET_LIGHTS, "mae_deg")
    l2_p95 = _by_object(records, "l2", TARGET_LIGHTS, "p95_deg")
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float | None] = {}
    for method in _methods(records):
        if method == "l2" or not is_complete(list(records), method, TARGET_LIGHTS):
            continue
        mae = _by_object(records, method, TARGET_LIGHTS, "mae_deg")
        p95 = _by_object(records, method, TARGET_LIGHTS, "p95_deg")
        objects, deltas = _paired(mae, l2_mae)
        _, p95_deltas = _paired(p95, l2_p95)
        l2_mean = statistics.mean(l2_mae[name] for name in objects)
        method_mean = statistics.mean(mae[name] for name in objects)
        p_mae = wilcoxon_p(deltas)
        raw_p[method] = p_mae if p_mae is not None else sign_test_p(deltas)
        comparisons[method] = {
            "objects": len(objects),
            "mae_deg": method_mean,
            "l2_mae_deg": l2_mean,
            "delta_mae_deg": statistics.mean(deltas),
            "delta_mae_median_deg": statistics.median(deltas),
            "relative_improvement_pct": 100.0 * (l2_mean - method_mean) / l2_mean,
            "objects_better_than_l2": sum(1 for value in deltas if value < 0),
            "objects_worse_than_l2": sum(1 for value in deltas if value > 0),
            "wilcoxon_p": p_mae,
            "sign_test_p": sign_test_p(deltas),
            "delta_p95_deg": statistics.mean(p95_deltas),
            "objects_better_p95": sum(1 for value in p95_deltas if value < 0),
            "wilcoxon_p_p95": wilcoxon_p(p95_deltas),
            "per_object_delta_mae_deg": dict(zip(objects, deltas)),
        }
    adjusted = holm_adjust(raw_p)
    for method, entry in comparisons.items():
        p = adjusted[method]
        entry["p_holm"] = p
        if p is None:
            entry["verdict"] = "not testable"
        elif p >= ALPHA:
            entry["verdict"] = "no significant difference from L2"
        elif entry["delta_mae_deg"] < 0:
            entry["verdict"] = "significantly better than L2"
        else:
            entry["verdict"] = "significantly worse than L2"
    return {
        "lights": TARGET_LIGHTS,
        "test": "paired exact two-sided Wilcoxon signed-rank over objects (sign test if SciPy absent), "
                f"Holm-adjusted across methods, alpha={ALPHA}",
        "comparisons": comparisons,
    }


IMPLEMENTATION = {
    "l2": {
        "runs_on": "CPU", "inputs": "images + calibrated light directions",
        "dependencies": "NumPy only; closed-form least squares",
        "complexity": "lowest; no model, no training, deterministic",
        "licence": "benchmarked through GPL RobustPhotometricStereo; least squares is a few lines of NumPy, so an independent reimplementation carries no licence constraint",
    },
    "l1": {
        "runs_on": "CPU", "inputs": "images + calibrated light directions",
        "dependencies": "NumPy/SciPy; per-pixel sparse regression (official RobustPhotometricStereo)",
        "complexity": "low to implement, but slow per pixel",
        "licence": "GPL (RobustPhotometricStereo); commercial use needs author licence or an independent implementation",
    },
    "ps_fcn": {
        "runs_on": "GPU or CPU (PyTorch)", "inputs": "images + calibrated light directions",
        "dependencies": "PyTorch, 2.2M-parameter pretrained checkpoint (author HuggingFace mirror)",
        "complexity": "moderate; needs the [0,1] input-scale convention, deterministic",
        "licence": "MIT (PS-FCN repository)",
    },
    "sdm_unips": {
        "runs_on": "CUDA only on this machine (CPU path crashes natively)", "inputs": "images only; lights not used",
        "dependencies": "PyTorch transformer, pretrained checkpoint, process isolation",
        "complexity": "highest; VRAM-bound, non-commercial licence blocks product deployment",
        "licence": "MIT with a non-commercial clause (SDM-UniPS repository); research use only",
    },
}


def rq3_tradeoff(records: Sequence[dict[str, Any]], licences: dict[str, str] | None = None) -> dict[str, Any]:
    """RQ3: accuracy / runtime / memory / lights / complexity trade-off at the 4-light target."""
    rows: dict[str, Any] = {}
    for method in _methods(records):
        supported = supported_light_counts(records, method)
        at_four = [
            record for record in records
            if record.get("method") == method and record.get("status") == "ok"
            and record.get("lights") is not None and int(record["lights"]) == TARGET_LIGHTS
        ]
        timed = [record for record in at_four if record.get("runtime_sec") is not None]
        spill = sorted({
            int(record["lights"]) for record in records
            if record.get("method") == method and record.get("gpu_memory_exceeds_physical")
        })
        peaks = [float(record["peak_gpu_memory_mb"]) for record in at_four if record.get("peak_gpu_memory_mb") is not None]
        rows[method] = {
            "label": METHOD_LABELS.get(method, method),
            "complete_at_4": is_complete(list(records), method, TARGET_LIGHTS),
            "mae_deg_at_4": statistics.mean(float(r["mae_deg"]) for r in at_four) if at_four else None,
            "p95_deg_at_4": statistics.mean(float(r["p95_deg"]) for r in at_four) if at_four else None,
            "runtime_sec_at_4_median": statistics.median(float(r["runtime_sec"]) for r in timed) if timed else None,
            "runtime_objects_at_4": sorted(str(r["object"]) for r in timed),
            "runtime_sec_at_4_median_excluding_caveats": (
                statistics.median(float(r["runtime_sec"]) for r in timed if not r.get("runtime_caveat"))
                if any(not r.get("runtime_caveat") for r in timed) else None
            ),
            "runtime_caveats_at_4": sorted(str(r["object"]) for r in timed if r.get("runtime_caveat")),
            "peak_gpu_memory_mb_at_4_max": max(peaks) if peaks else None,
            "supported_light_counts_on_this_gpu": supported,
            "max_supported_lights": max(supported) if supported else None,
            "light_counts_spilling_past_physical_vram": spill,
            **IMPLEMENTATION.get(method, {}),
            "licence": (licences or {}).get(method, IMPLEMENTATION.get(method, {}).get("licence")),
        }

    candidates = {
        method: row for method, row in rows.items()
        if row["complete_at_4"] and row["runtime_sec_at_4_median"] is not None
    }

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        keys = ("mae_deg_at_4", "runtime_sec_at_4_median", "peak_gpu_memory_mb_at_4_max")
        pairs = [(float(a[k] or 0.0), float(b[k] or 0.0)) for k in keys]
        return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)

    pareto = [
        method for method, row in candidates.items()
        if not any(dominates(other, row) for name, other in candidates.items() if name != method)
    ]
    return {
        "lights": TARGET_LIGHTS,
        "methods": rows,
        "pareto_front_mae_runtime_memory": pareto,
        "pareto_note": "Pareto over mean MAE, median runtime and peak GPU memory at 4 lights; "
                       "methods without repeated-run runtime are excluded rather than guessed.",
    }


def answer_research_questions(
    records: Sequence[dict[str, Any]], licences: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "rq1_accuracy_loss_4_vs_many": rq1_accuracy_loss(records),
        "rq2_robust_deep_vs_l2_at_4": rq2_versus_l2_at_four(records),
        "rq3_tradeoff_at_4": rq3_tradeoff(records, licences),
    }
