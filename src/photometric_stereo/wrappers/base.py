"""Shared estimator contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreflightResult:
    available: bool
    reason: str
    device: str


class NormalEstimator(ABC):
    name: str
    runtime_sec: float | None = None
    peak_gpu_memory_mb: float | None = None
    end_to_end_sec: float | None = None
    device: str = "cpu"

    @abstractmethod
    def preflight(self) -> PreflightResult:
        raise NotImplementedError

    @abstractmethod
    def estimate_normals(
        self, images: np.ndarray, light_directions: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Return H,W,3 unit normals in the input coordinate convention."""
        raise NotImplementedError
