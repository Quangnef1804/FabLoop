"""CLI: assemble the final photometric-stereo tables and figures from the two locked passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photometric_stereo.assemble import assemble

DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "photometric_stereo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy-dirs", nargs="+", type=Path,
        default=[DEFAULT_ROOT / folder / method for folder in ("matrix_accuracy", "matrix_accuracy_lights064") for method in ("l2", "l1", "ps_fcn", "sdm_unips")],
        help="Pass-1 output directories, one per method; their metrics and artifacts are authoritative.",
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=DEFAULT_ROOT / "matrix_runtime",
        help="Pass-2 output directory with warm-up and repeated runs; the only source of cited runtime.",
    )
    parser.add_argument("--no-runtime", action="store_true", help="Assemble accuracy only, before pass 2 exists.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--error-max-deg", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = assemble(
        args.accuracy_dirs,
        None if args.no_runtime else args.runtime_dir,
        args.output_dir,
        args.error_max_deg,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
