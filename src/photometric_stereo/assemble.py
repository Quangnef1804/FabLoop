"""Assemble final benchmark tables and figures from the two locked passes.

Pass 1 (accuracy) runs every object x method x light count once; its metrics and
artifacts are authoritative, its single-sample runtime is not. Pass 2 (runtime)
re-runs a representative subset with warm-up and repeated measurement; it is the
only source of cited runtime.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .benchmark import write_benchmark_tables
from .research_questions import answer_research_questions
from .visualization import METHOD_ORDER, save_aggregate_plots, save_method_comparisons

SUMMARY_LEADING_FIELDS = (
    "object", "method", "lights", "status",
    "mae_deg", "median_deg", "p90_deg", "p95_deg",
    "runtime_sec", "runtime_iqr_sec", "runtime_std_sec", "runtime_measured_runs", "runtime_source", "runtime_caveat",
    "peak_gpu_memory_mb", "gpu_memory_exceeds_physical", "delta_mae_vs_l2_deg", "relative_improvement_vs_l2_pct",
    "valid_pixel_count", "device", "gpu_name", "gpu_total_memory_mb", "error",
)
RUNTIME_LEADING_FIELDS = (
    "object", "method", "lights", "status",
    "runtime_sec", "runtime_iqr_sec", "runtime_std_sec", "runtime_min_sec", "runtime_max_sec",
    "runtime_measured_runs", "runtime_warmup_runs", "end_to_end_sec", "peak_gpu_memory_mb", "gpu_memory_exceeds_physical",
    "runtime_caveat", "runtime_caveat_evidence",
    "repeat_outputs_identical", "mae_deg", "device", "gpu_name", "gpu_total_memory_mb",
    "runtime_protocol", "error",
)
# Fields that pass 1 only measured once; they must not be mistaken for the cited runtime.
_SINGLE_SAMPLE_TIMING_PREFIXES = ("runtime_", "end_to_end_")
_SINGLE_SAMPLE_TIMING_KEYS = ("repeat_outputs_identical", "mae_deg_median_over_runs", "mae_deg_std_over_runs")
_REPEATED_TIMING_KEYS = ("repeat_outputs_identical", "mae_deg_std_over_runs")
MAE_AGREEMENT_TOLERANCE_DEG = 1.0e-6


def exceeds_physical_memory(record: dict[str, Any]) -> bool:
    """True when the Windows driver spilled CUDA allocations into shared system memory.

    Such a case still computes the same normals, so its accuracy is valid, but its
    runtime measures PCIe traffic rather than the GPU and must not be cited as one.
    """
    peak = record.get("peak_gpu_memory_mb")
    total = record.get("gpu_total_memory_mb")
    return bool(peak is not None and total and float(peak) > float(total))


def load_records(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / "benchmark_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing pass output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_caveats(directory: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Measurement-environment annotations recorded next to the runtime pass.

    A caveat never removes a row or changes a number; it travels with the case into
    every table and figure so a reader can see which runtimes were disturbed.
    """
    if directory is None:
        return {}
    path = Path(directory) / "runtime_caveats.json"
    if not path.is_file():
        return {}
    return {case_key(entry): entry for entry in json.loads(path.read_text(encoding="utf-8"))}


def case_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["object"]), str(record["method"]), int(record["lights"])


def _sort_key(record: dict[str, Any]) -> tuple[str, int, int]:
    method = str(record.get("method"))
    order = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return str(record.get("object")), order, int(record.get("lights") or 0)


def merge_passes(
    accuracy_records: Sequence[dict[str, Any]],
    runtime_records: Sequence[dict[str, Any]],
    caveats: dict[tuple[str, str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine authoritative accuracy with repeated-run runtime, never mixing their roles."""
    runtime_by_case = {
        case_key(record): record for record in runtime_records
        if record.get("object") is not None and record.get("lights") is not None
    }
    merged: list[dict[str, Any]] = []
    for source in accuracy_records:
        if source.get("object") is None or source.get("lights") is None:
            continue
        record = dict(source)
        single_sample = record.get("runtime_sec")
        for key in list(record):
            if key.startswith(_SINGLE_SAMPLE_TIMING_PREFIXES) or key in _SINGLE_SAMPLE_TIMING_KEYS:
                record.pop(key)
        record["accuracy_pass_single_run_sec"] = single_sample
        record["runtime_sec"] = None
        record["runtime_source"] = "not_measured"

        timing = runtime_by_case.get(case_key(record))
        if timing is not None and record.get("status") == "ok":
            if timing.get("status") == "ok":
                for key, value in timing.items():
                    if key.startswith(_SINGLE_SAMPLE_TIMING_PREFIXES) or key in _REPEATED_TIMING_KEYS:
                        record[key] = value
                record["runtime_source"] = "runtime_pass"
                caveat = (caveats or {}).get(case_key(record))
                if caveat is not None:
                    record["runtime_caveat"] = caveat["caveat"]
                    record["runtime_caveat_evidence"] = caveat.get("evidence")
                # Both passes use the same seed and input, so a deterministic method
                # must reproduce the same MAE; a mismatch invalidates the pairing.
                record["runtime_pass_mae_deg"] = timing.get("mae_deg")
                record["runtime_pass_mae_matches"] = (
                    timing.get("mae_deg") is not None
                    and abs(float(timing["mae_deg"]) - float(record["mae_deg"])) <= MAE_AGREEMENT_TOLERANCE_DEG
                )
            else:
                record["runtime_source"] = f"runtime_pass_{timing.get('status')}"
        record["gpu_memory_exceeds_physical"] = exceeds_physical_memory(record)
        merged.append(record)
    merged.sort(key=_sort_key)
    return merged


def write_runtime_table(
    output_dir: Path,
    runtime_records: Sequence[dict[str, Any]],
    caveats: dict[tuple[str, str, int], dict[str, Any]] | None = None,
) -> Path:
    """Resource table for the repeated-run subset, including cases that did not fit the GPU."""
    rows = []
    for record in runtime_records:
        if record.get("object") is None:
            continue
        row = {**record, "gpu_memory_exceeds_physical": exceeds_physical_memory(record)}
        caveat = (caveats or {}).get(case_key(record))
        if caveat is not None:
            row["runtime_caveat"] = caveat["caveat"]
            row["runtime_caveat_evidence"] = caveat.get("evidence")
        rows.append(row)
    rows.sort(key=_sort_key)
    fields = [field for field in RUNTIME_LEADING_FIELDS if any(field in row for row in rows)]
    path = Path(output_dir) / "runtime_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def assemble(
    accuracy_dirs: Sequence[Path],
    runtime_dir: Path | None,
    output_dir: Path,
    error_max_deg: float = 90.0,
) -> dict[str, Any]:
    accuracy_records = [record for directory in accuracy_dirs for record in load_records(directory)]
    runtime_records = load_records(runtime_dir) if runtime_dir is not None else []
    caveats = load_runtime_caveats(runtime_dir)
    duplicates = {
        key for key in (case_key(r) for r in accuracy_records if r.get("object") is not None)
        if sum(1 for r in accuracy_records if r.get("object") is not None and case_key(r) == key) > 1
    }
    if duplicates:
        raise ValueError(f"accuracy passes contain duplicate cases: {sorted(duplicates)[:5]}")

    records = merge_passes(accuracy_records, runtime_records, caveats)
    output_dir = Path(output_dir)
    write_benchmark_tables(output_dir, records, leading_fields=SUMMARY_LEADING_FIELDS)
    runtime_table = write_runtime_table(output_dir, runtime_records, caveats) if runtime_records else None
    figures = save_aggregate_plots(records, output_dir / "figures")
    figures += save_method_comparisons(records, output_dir, error_max_deg)
    (output_dir / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    research = answer_research_questions(records)
    (output_dir / "rq_summary.json").write_text(json.dumps(research, indent=2), encoding="utf-8")

    mismatches = [case_key(r) for r in records if r.get("runtime_pass_mae_matches") is False]
    manifest = {
        "accuracy_dirs": [str(Path(d)) for d in accuracy_dirs],
        "runtime_dir": str(runtime_dir) if runtime_dir is not None else None,
        "cases": len(records),
        "status_counts": {
            status: sum(1 for r in records if r.get("status") == status)
            for status in sorted({str(r.get("status")) for r in records})
        },
        "runtime_measured_cases": sum(1 for r in records if r.get("runtime_source") == "runtime_pass"),
        "runtime_pass_mae_mismatches": [list(key) for key in mismatches],
        "gpu_memory_spill_cases": [list(case_key(r)) for r in records if r.get("gpu_memory_exceeds_physical")],
        "runtime_caveat_cases": [list(key) + [entry["caveat"]] for key, entry in sorted(caveats.items())],
        "runtime_table": str(runtime_table) if runtime_table else None,
        "figures": len(figures),
    }
    (output_dir / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
