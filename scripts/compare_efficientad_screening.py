"""Compare existing EfficientAD baseline and Slim-0.5 evaluation artifacts.

This script does not evaluate predictions or define a metric. It reads the
image-level AUROC already written by ``src/evaluator.py`` and applies the
architecture-screening thresholds documented for Slim-0.5.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
BASELINE_ARCHITECTURE = "anomalib-baseline"
SLIM_ARCHITECTURE = "efficientad-s-slim-0.5"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _metrics_path(root: Path, category: str, seed: int) -> Path:
    return root / category / "predictions" / f"seed_{seed}" / "metrics.json"


def _samples_path(root: Path, category: str, seed: int) -> Path:
    return root / category / "predictions" / f"seed_{seed}" / "samples.jsonl"


def _architecture_id(record: dict[str, Any]) -> str:
    architecture = record.get("architecture")
    if not isinstance(architecture, dict) or not isinstance(architecture.get("id"), str):
        raise ValueError("metrics.json has no architecture.id provenance")
    return architecture["id"]


def _sample_identity(path: Path) -> list[tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required sample ledger is missing: {path}")
    result: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                result.append((str(record["sample_id"]), int(record["label"])))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid sample identity at {path}:{line_number}") from error
    if not result:
        raise ValueError(f"Sample ledger is empty: {path}")
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    profile = _read_json(args.profile)
    reductions = profile["reductions"]["student_only"]
    params_reduction = float(reductions["parameters"]["reduction_fraction"])
    flops_reduction = float(reductions["supported_flops"]["reduction_fraction"])

    rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        baseline_path = _metrics_path(args.baseline_root, category, args.seed)
        slim_path = _metrics_path(args.slim_root, category, args.seed)
        baseline = _read_json(baseline_path)
        slim = _read_json(slim_path)

        for label, record, expected_architecture in (
            ("baseline", baseline, BASELINE_ARCHITECTURE),
            ("slim", slim, SLIM_ARCHITECTURE),
        ):
            if record.get("category") != category or int(record.get("seed", -1)) != args.seed:
                raise ValueError(f"{label} provenance mismatch in {category}")
            if _architecture_id(record) != expected_architecture:
                raise ValueError(f"Unexpected {label} architecture in {category}")
            if abs(float(record.get("target_fpr", -1.0)) - args.target_fpr) > 1e-12:
                raise ValueError(f"{label} target_fpr mismatch in {category}")
            if int(record.get("test_passes", -1)) != 1:
                raise ValueError(f"{label} evaluation was not single-pass in {category}")

        baseline_samples = _sample_identity(_samples_path(args.baseline_root, category, args.seed))
        slim_samples = _sample_identity(_samples_path(args.slim_root, category, args.seed))
        if baseline_samples != slim_samples:
            raise ValueError(f"Baseline and Slim test sample ledgers differ in {category}")

        auroc_old = float(baseline["image_level_auroc"])
        auroc_slim = float(slim["image_level_auroc"])
        delta_pp = (auroc_slim - auroc_old) * 100.0
        degradation_pp = max(0.0, -delta_pp)
        rows.append(
            {
                "category": category,
                "auroc_old": auroc_old,
                "auroc_slim": auroc_slim,
                "delta_auroc_pp": delta_pp,
                "degradation_pp": degradation_pp,
                "screening_quality_passed": degradation_pp <= args.max_degradation_pp + 1e-12,
                "test_sample_count": len(baseline_samples),
                "baseline_metrics": str(baseline_path),
                "slim_metrics": str(slim_path),
            }
        )

    mean_old = sum(row["auroc_old"] for row in rows) / len(rows)
    mean_slim = sum(row["auroc_slim"] for row in rows) / len(rows)
    mean_delta_pp = (mean_slim - mean_old) * 100.0
    params_passed = params_reduction >= args.min_params_reduction
    flops_passed = flops_reduction >= args.min_flops_reduction
    quality_passed = all(row["screening_quality_passed"] for row in rows)

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "VisA architecture screening",
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "delta_definition": "delta_auroc_pp = (AUROC_slim - AUROC_old) * 100",
        "screening_gate": {
            "student_parameters_reduction_min_fraction": args.min_params_reduction,
            "student_supported_flops_reduction_min_fraction": args.min_flops_reduction,
            "max_auroc_degradation_pp_per_category": args.max_degradation_pp,
            "student_parameters_reduction_fraction": params_reduction,
            "student_supported_flops_reduction_fraction": flops_reduction,
            "parameters_passed": params_passed,
            "flops_passed": flops_passed,
            "quality_passed": quality_passed,
            "passed": params_passed and flops_passed and quality_passed,
        },
        "final_pcba_jetson_target": {
            "max_auroc_degradation_pp": 1.0,
            "status": "NOT_EVALUATED_BY_VISA_SCREENING",
        },
        "categories": rows,
        "mean": {
            "auroc_old": mean_old,
            "auroc_slim": mean_slim,
            "delta_auroc_pp": mean_delta_pp,
        },
        "provenance": {
            "baseline_root": str(args.baseline_root),
            "slim_root": str(args.slim_root),
            "compression_profile": str(args.profile),
            "comparison_scope": "Existing image_level_auroc from src/evaluator.py; no new metric",
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    gate = report["screening_gate"]
    lines = [
        "# EfficientAD-S vs Slim-0.5 — VisA screening seed 42",
        "",
        "`ΔAUROC = AUROC_slim - AUROC_old`, reported in percentage points (pp).",
        "",
        "| Category | AUROC_old | AUROC_slim | ΔAUROC (pp) | Gate ≤2 pp |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in report["categories"]:
        lines.append(
            f"| {row['category']} | {row['auroc_old']:.4f} | {row['auroc_slim']:.4f} | "
            f"{row['delta_auroc_pp']:+.2f} | {'PASS' if row['screening_quality_passed'] else 'FAIL'} |"
        )
    mean = report["mean"]
    lines.extend(
        [
            f"| **Mean** | **{mean['auroc_old']:.4f}** | **{mean['auroc_slim']:.4f}** | "
            f"**{mean['delta_auroc_pp']:+.2f}** | — |",
            "",
            "## Screening gate",
            "",
            f"- Student parameters: {gate['student_parameters_reduction_fraction'] * 100:.2f}% reduction "
            f"(target ≥40%): **{'PASS' if gate['parameters_passed'] else 'FAIL'}**",
            f"- Student fvcore-supported FLOPs: {gate['student_supported_flops_reduction_fraction'] * 100:.2f}% "
            f"reduction (target ≥30%): **{'PASS' if gate['flops_passed'] else 'FAIL'}**",
            f"- AUROC degradation: ≤2 pp in every VisA category: "
            f"**{'PASS' if gate['quality_passed'] else 'FAIL'}**",
            f"- Overall architecture screening: **{'PASS' if gate['passed'] else 'FAIL'}**",
            "",
            "The ≤1 pp objective belongs to the later real-PCBA/Jetson validation and is not evaluated or claimed here.",
            "",
            "All AUROC values come directly from the existing `src/evaluator.py` output at `target_fpr: 0.10`; "
            "this comparison does not add or recompute a model metric.",
            "",
            "## Implementation and literature roles",
            "",
            "- [Batzner et al., EfficientAD](https://arxiv.org/abs/2303.14535): architecture and training source.",
            "- [Anomalib EfficientAD](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/efficient_ad): "
            "FabLoop implementation source and pretrained Teacher loader.",
            "- [Lee & Kim, arXiv:2407.17909](https://arxiv.org/abs/2407.17909): reference for EfficientAD "
            "feature distances and logical anomalies; it is not a channel-compression source.",
            "- [rximg/EfficientAD](https://github.com/rximg/EfficientAD) and "
            "[DistillationAD](https://github.com/SimonThomine/DistillationAD): reading references only.",
            "- [DepGraph/Torch-Pruning](https://github.com/VainF/Torch-Pruning): Plan B for structural pruning "
            "only if manual width reduction fails the quality gate.",
            "- [CDA](https://github.com/WHUer-cloud/CDA): lightweight anomaly-distillation literature; "
            "it does not replace EfficientAD in this branch.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/efficientad_baseline_screening"))
    parser.add_argument("--slim-root", type=Path, default=Path("outputs/efficientad_slim/slim_0_5"))
    parser.add_argument("--profile", type=Path, default=Path("docs/preflight/efficientad_slim_05_profile.json"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/results/efficientad_slim_05_vs_original_seed42.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/results/efficientad_slim_05_vs_original_seed42.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-params-reduction", type=float, default=0.40)
    parser.add_argument("--min-flops-reduction", type=float, default=0.30)
    parser.add_argument("--max-degradation-pp", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)
    print(f"screening_gate={'PASS' if report['screening_gate']['passed'] else 'FAIL'}")
    raise SystemExit(0 if report["screening_gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
