"""End-to-end DiLiGenT benchmark orchestration."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .datasets import discover_diligent_objects, load_benchmark_input
from .diligent_validate import attach_baseline, default_baseline_path, load_baselines
from .lighting import describe_subset, select_light_subsets
from .metrics import attach_l2_comparison, normal_error_metrics
from .visualization import save_aggregate_plots, save_case_artifacts, save_method_comparisons
from .wrappers import PSFCNEstimator, RobustPSEstimator, SDMUniPSEstimator
from .wrappers.base import NormalEstimator


@dataclass(frozen=True)
class BenchmarkPaths:
    diligent_root: Path
    output_dir: Path
    rps_root: Path
    ps_fcn_repo: Path
    ps_fcn_checkpoint: Path
    sdm_repo: Path
    sdm_checkpoint_root: Path
    baseline_csv: Path


def build_estimators(methods: Sequence[str], paths: BenchmarkPaths, device: str, seed: int = 42) -> dict[str, NormalEstimator]:
    result: dict[str, NormalEstimator] = {}
    for method in methods:
        if method in {"l2", "l1", "l1-multicore"}:
            result[method] = RobustPSEstimator(method, paths.rps_root)
        elif method == "ps_fcn":
            result[method] = PSFCNEstimator(paths.ps_fcn_repo, paths.ps_fcn_checkpoint, device)
        elif method == "sdm_unips":
            result[method] = SDMUniPSEstimator(paths.sdm_repo, paths.sdm_checkpoint_root, device, seed)
        else:
            raise ValueError(f"Unknown benchmark method: {method}")
    return result


def run_preflight(paths: BenchmarkPaths, methods: Sequence[str], device: str = "auto", seed: int = 42) -> dict[str, dict[str, Any]]:
    estimators = build_estimators(methods, paths, device, seed)
    result = {name: asdict(estimator.preflight()) for name, estimator in estimators.items()}
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def gpu_context() -> dict[str, Any]:
    """Report the accelerator the measurement actually ran on, for reproducibility."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_name": None, "gpu_total_memory_mb": None}
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "gpu_name": properties.name,
            "gpu_total_memory_mb": float(properties.total_memory) / (1024.0**2),
        }
    except Exception:  # torch missing or driver unreachable; absence is not a benchmark failure
        return {"gpu_name": None, "gpu_total_memory_mb": None}


def is_out_of_memory(exc: BaseException) -> bool:
    """Recognise CUDA OOM raised in-process or surfaced through the SDM-UniPS subprocess."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemoryerror" in text


def runtime_statistics(values: Sequence[float], prefix: str) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        f"{prefix}_sec": float(statistics.median(ordered)),
        f"{prefix}_median_sec": float(statistics.median(ordered)),
        f"{prefix}_min_sec": ordered[0],
        f"{prefix}_max_sec": ordered[-1],
        f"{prefix}_std_sec": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
        f"{prefix}_iqr_sec": float(np.percentile(ordered, 75) - np.percentile(ordered, 25)),
        f"{prefix}_runs_sec": [float(value) for value in values],
    }


def measure_case(
    estimator: NormalEstimator,
    images: np.ndarray,
    light_directions: np.ndarray,
    mask: np.ndarray,
    normal_gt: np.ndarray,
    *,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, dict[str, Any]]:
    """Warm up, then repeat the measured run so runtime is a median rather than one sample.

    Reported metrics and artifacts always come from the first measured run; the
    remaining runs quantify runtime spread and confirm seeded determinism.
    """
    for _ in range(max(0, warmup_runs)):
        estimator.estimate_normals(images, light_directions, mask)

    forward: list[float] = []
    end_to_end: list[float] = []
    peaks: list[float] = []
    maes: list[float] = []
    reference: np.ndarray | None = None
    reference_metrics: dict[str, float] | None = None
    reference_errors: np.ndarray | None = None
    deterministic = True

    for _ in range(max(1, measured_runs)):
        estimated = estimator.estimate_normals(images, light_directions, mask)
        metrics, errors = normal_error_metrics(estimated, normal_gt, mask)
        if reference is None:
            reference, reference_metrics, reference_errors = estimated, metrics, errors
        elif not np.array_equal(estimated, reference):
            deterministic = False
        forward.append(float(estimator.runtime_sec or 0.0))
        end_to_end.append(float(estimator.end_to_end_sec or estimator.runtime_sec or 0.0))
        if estimator.peak_gpu_memory_mb is not None:
            peaks.append(float(estimator.peak_gpu_memory_mb))
        maes.append(float(metrics["mae_deg"]))

    assert reference is not None and reference_metrics is not None and reference_errors is not None
    timing: dict[str, Any] = {
        **runtime_statistics(forward, "runtime"),
        **runtime_statistics(end_to_end, "end_to_end"),
        "runtime_warmup_runs": max(0, warmup_runs),
        "runtime_measured_runs": len(forward),
        "peak_gpu_memory_mb": max(peaks) if peaks else None,
        "repeat_outputs_identical": deterministic,
        "mae_deg_median_over_runs": float(statistics.median(maes)),
        "mae_deg_std_over_runs": float(statistics.stdev(maes)) if len(maes) > 1 else 0.0,
    }
    return reference, reference_metrics, reference_errors, timing


def _flat_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in record.items()}


def write_benchmark_tables(
    output_dir: Path, records: list[dict[str, Any]], leading_fields: Sequence[str] = ()
) -> None:
    attach_l2_comparison(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields: list[str] = [field for field in leading_fields if any(field in record for record in records)]
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    for name, selected in (
        ("benchmark_summary.csv", records),
        ("four_light_summary.csv", [record for record in records if record.get("lights") == 4]),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_flat_record(record) for record in selected)


def run_benchmark(
    paths: BenchmarkPaths,
    *,
    objects: Sequence[str] | None,
    methods: Sequence[str],
    light_counts: Sequence[int],
    device: str = "auto",
    allow_missing_methods: bool = False,
    continue_on_error: bool = False,
    error_max_deg: float = 90.0,
    seed: int = 42,
    runtime_warmup_runs: int = 1,
    runtime_measured_runs: int = 3,
) -> list[dict[str, Any]]:
    estimators = build_estimators(methods, paths, device, seed)
    preflight = {name: asdict(estimator.preflight()) for name, estimator in estimators.items()}
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    unavailable = {name: status["reason"] for name, status in preflight.items() if not status["available"]}
    if unavailable and not allow_missing_methods:
        detail = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        raise RuntimeError(f"Benchmark preflight failed before expensive execution: {detail}")
    active = {name: estimator for name, estimator in estimators.items() if preflight[name]["available"]}

    accelerator = gpu_context()
    samples = discover_diligent_objects(paths.diligent_root, objects)
    baselines = load_baselines(paths.baseline_csv)
    records: list[dict[str, Any]] = []
    for sample in samples:
        subsets = select_light_subsets(sample.light_directions, light_counts)
        for count in sorted(subsets):
            indices = subsets[count]
            benchmark_input = load_benchmark_input(sample, indices)
            geometry = describe_subset(sample.light_directions, indices)
            for method, estimator in active.items():
                case_dir = paths.output_dir / sample.object_name / f"lights_{count:03d}" / method
                base: dict[str, Any] = {
                    "object": sample.object_name,
                    "method": method,
                    "lights": count,
                    "image_indices_zero_based": list(indices),
                    "image_names": list(benchmark_input.image_names),
                    "light_geometry": geometry,
                    "intensity_normalization": "per_channel_calibrated",
                    "method_input_preprocessing": (
                        "raw RGB [0,1], upstream per-image normalization; calibrated directions ignored"
                        if method == "sdm_unips"
                        else (
                            "per-channel light-intensity correction, upstream [0,1] PS-FCN input scale"
                            if method == "ps_fcn"
                            else "per-channel light-intensity correction, RGB mean"
                        )
                    ),
                    "source_mask_pixel_count": sample.source_mask_pixel_count,
                    "invalid_gt_pixels_excluded": sample.invalid_gt_pixel_count,
                    "runtime_scope": "solver/model forward only",
                    "inference_seed": seed,
                    "runtime_protocol": (
                        f"{max(0, runtime_warmup_runs)} warm-up run(s) then "
                        f"{max(1, runtime_measured_runs)} measured run(s); median reported"
                    ),
                    **accelerator,
                }
                try:
                    method_images = benchmark_input.raw_images if method == "sdm_unips" else benchmark_input.images
                    estimated, metrics, errors, timing = measure_case(
                        estimator,
                        method_images,
                        benchmark_input.light_directions,
                        benchmark_input.mask,
                        benchmark_input.normal_gt,
                        warmup_runs=runtime_warmup_runs,
                        measured_runs=runtime_measured_runs,
                    )
                    artifacts = save_case_artifacts(
                        case_dir, estimated, benchmark_input.normal_gt, errors, benchmark_input.mask,
                        error_max_deg=error_max_deg,
                    )
                    record = {
                        **base,
                        "status": "ok",
                        **metrics,
                        **timing,
                        "device": estimator.device,
                        "output_dir": str(case_dir),
                        **artifacts,
                    }
                    if method in {"l2", "l1", "l1-multicore"}:
                        attach_baseline(record, baselines, sample.object_name, method, count)
                    records.append(record)
                    print(
                        f"{sample.object_name} {method} {count}: MAE={record['mae_deg']:.3f} deg, "
                        f"median={record['median_deg']:.3f}, P95={record['p95_deg']:.3f}, "
                        f"runtime={record['runtime_sec']:.3f}s"
                    )
                except Exception as exc:
                    out_of_memory = is_out_of_memory(exc)
                    failure = {
                        **base,
                        "status": "unsupported_oom" if out_of_memory else "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "device": estimator.device,
                    }
                    if out_of_memory:
                        # The input light set is part of the benchmark protocol, so an
                        # OOM case is reported as unsupported on this accelerator rather
                        # than silently re-run on a split subset of the lights.
                        failure["oom_note"] = (
                            "Case does not fit this accelerator; lights were not split, "
                            "because splitting would change the benchmarked problem."
                        )
                    records.append(failure)
                    write_benchmark_tables(paths.output_dir, records)
                    if out_of_memory:
                        print(
                            f"UNSUPPORTED_OOM {sample.object_name} {method} {count} on "
                            f"{accelerator.get('gpu_name')}"
                        )
                    elif not continue_on_error:
                        raise
                    else:
                        print(f"ERROR {sample.object_name} {method} {count}: {exc}")
                write_benchmark_tables(paths.output_dir, records)

    for method, reason in unavailable.items():
        records.append({"object": None, "method": method, "lights": None, "status": "unavailable", "error": reason})
    write_benchmark_tables(paths.output_dir, records)
    figures = save_aggregate_plots(records, paths.output_dir / "figures")
    figures += save_method_comparisons(records, paths.output_dir, error_max_deg)
    (paths.output_dir / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    return records


def gate_result(records: list[dict[str, Any]]) -> dict[str, Any]:
    matching = [record for record in records if record.get("object") == "ball" and record.get("method") == "l2" and record.get("lights") == 96 and record.get("status") == "ok"]
    if len(matching) != 1:
        raise RuntimeError("Gate requires exactly one successful Ball + L2 + 96 record")
    record = matching[0]
    if record.get("baseline_status") != "reference_available":
        raise RuntimeError("Gate cannot be assessed because the official baseline is missing")
    return {
        "status": "reproduced_for_review",
        "measured_mae_deg": record["mae_deg"],
        "baseline_mae_deg": record["baseline_mae_deg"],
        "delta_deg": record["baseline_delta_deg"],
        "relative_delta_pct": 100.0 * record["baseline_delta_deg"] / record["baseline_mae_deg"],
        "note": "No unpublished pass/fail tolerance was invented; review the measured delta before the full matrix.",
    }


def default_paths(project_root: Path) -> BenchmarkPaths:
    return BenchmarkPaths(
        diligent_root=project_root / "data" / "diligent" / "DiLiGenT" / "pmsData",
        output_dir=project_root / "outputs" / "photometric_stereo",
        rps_root=project_root / "third-party" / "RobustPhotometricStereo",
        ps_fcn_repo=project_root / "third-party" / "PS-FCN",
        ps_fcn_checkpoint=project_root / "third-party" / "PS-FCN" / "data" / "models" / "PS-FCN_B_S_32.pth.tar",
        sdm_repo=project_root / "third-party" / "SDM-UniPS",
        sdm_checkpoint_root=project_root / "third-party" / "SDM-UniPS" / "checkpoint",
        baseline_csv=default_baseline_path(),
    )
