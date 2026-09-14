"""Adapters for the unmodified yasumat/RobustPhotometricStereo source."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..diligent_validate import import_rps_class, solve_with_rps
from .base import NormalEstimator, PreflightResult


class RobustPSEstimator(NormalEstimator):
    def __init__(self, solver: str, rps_root: Path) -> None:
        if solver not in {"l2", "l1", "l1-multicore"}:
            raise ValueError(f"Unsupported RobustPS solver: {solver}")
        self.name = solver
        self.solver = solver
        self.rps_root = Path(rps_root)
        self.runtime_sec = None
        self.end_to_end_sec = None
        self.peak_gpu_memory_mb = 0.0
        self.device = "cpu"

    def preflight(self) -> PreflightResult:
        required = ("rps.py", "psutil.py", "rpsnumerics.py")
        missing = [name for name in required if not (self.rps_root / name).is_file()]
        if missing:
            return PreflightResult(False, f"missing upstream files: {', '.join(missing)}", "cpu")
        return PreflightResult(True, "official RobustPhotometricStereo source found", "cpu")

    def estimate_normals(self, images: np.ndarray, light_directions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.float64)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must have shape N,H,W,3")
        if images.shape[1:3] != mask.shape:
            raise ValueError("images and mask have different spatial shapes")
        gray = np.mean(images, axis=-1)
        measurements = np.moveaxis(gray, 0, -1).reshape(mask.size, images.shape[0])
        RPS = import_rps_class(self.rps_root)
        normal, elapsed = solve_with_rps(RPS, measurements, light_directions, mask, self.solver)
        self.runtime_sec = elapsed
        self.end_to_end_sec = elapsed
        return normal.astype(np.float32)
