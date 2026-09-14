"""CLI for the independent FabLoop photometric-stereo benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photometric_stereo.benchmark import BenchmarkPaths, default_paths, gate_result, run_benchmark, run_preflight


def parse_args() -> argparse.Namespace:
    defaults = default_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Benchmark L2/L1/PS-FCN/SDM-UniPS on DiLiGenT.")
    parser.add_argument("--diligent-root", type=Path, default=defaults.diligent_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--rps-root", type=Path, default=defaults.rps_root)
    parser.add_argument("--ps-fcn-repo", type=Path, default=defaults.ps_fcn_repo)
    parser.add_argument("--ps-fcn-checkpoint", type=Path, default=defaults.ps_fcn_checkpoint)
    parser.add_argument("--sdm-repo", type=Path, default=defaults.sdm_repo)
    parser.add_argument("--sdm-checkpoint-root", type=Path, default=defaults.sdm_checkpoint_root)
    parser.add_argument("--baseline-csv", type=Path, default=defaults.baseline_csv)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=["l2", "l1", "l1-multicore", "ps_fcn", "sdm_unips"], default=["l2", "l1", "ps_fcn", "sdm_unips"])
    parser.add_argument("--light-counts", nargs="+", type=int, default=[4, 8, 16, 32, 64, 96])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Seed for stochastic pretrained inference, especially SDM-UniPS.")
    parser.add_argument("--error-max-deg", type=float, default=90.0)
    parser.add_argument(
        "--runtime-warmup-runs", type=int, default=1,
        help="Unmeasured warm-up runs per case, to exclude lazy init and CUDA context setup.",
    )
    parser.add_argument(
        "--runtime-measured-runs", type=int, default=3,
        help="Measured runs per case; the reported runtime is their median.",
    )
    parser.add_argument("--allow-missing-methods", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--gate", action="store_true", help="Run only Ball + L2 + 96 and write gate_result.json.")
    parser.add_argument("--preflight-only", action="store_true", help="Check source/checkpoints/device, write preflight.json, and stop.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = BenchmarkPaths(
        diligent_root=args.diligent_root,
        output_dir=args.output_dir,
        rps_root=args.rps_root,
        ps_fcn_repo=args.ps_fcn_repo,
        ps_fcn_checkpoint=args.ps_fcn_checkpoint,
        sdm_repo=args.sdm_repo,
        sdm_checkpoint_root=args.sdm_checkpoint_root,
        baseline_csv=args.baseline_csv,
    )
    if args.preflight_only:
        result = run_preflight(paths, args.methods, args.device, args.seed)
        print(json.dumps(result, indent=2))
        return 0 if all(status["available"] for status in result.values()) else 2
    records = run_benchmark(
        paths,
        objects=["ball"] if args.gate else args.objects,
        methods=["l2"] if args.gate else args.methods,
        light_counts=[96] if args.gate else args.light_counts,
        device=args.device,
        allow_missing_methods=args.allow_missing_methods,
        continue_on_error=args.continue_on_error,
        error_max_deg=args.error_max_deg,
        seed=args.seed,
        runtime_warmup_runs=args.runtime_warmup_runs,
        runtime_measured_runs=args.runtime_measured_runs,
    )
    if args.gate:
        result = gate_result(records)
        path = args.output_dir / "gate_result.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
