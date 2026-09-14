"""Process-isolated adapter for official SDM-UniPS pretrained inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..diligent_validate import normalize_normals
from .base import NormalEstimator, PreflightResult

# 0xC0000005; Windows reports a native segfault through the process exit code.
WINDOWS_ACCESS_VIOLATION = 3221225477


def excerpt_worker_output(text: str, head: int = 2500, tail: int = 2500) -> str:
    """Keep both ends of worker stderr.

    faulthandler prints the fatal-exception type first and the extension-module list
    last, so a tail-only excerpt drops exactly the line that classifies a native crash.
    """
    if len(text) <= head + tail:
        return text
    return text[:head] + f"\n... [{len(text) - head - tail} characters omitted] ...\n" + text[-tail:]


class SDMUniPSEstimator(NormalEstimator):
    name = "sdm_unips"

    def __init__(self, repo: Path, checkpoint_root: Path, device: str = "auto", seed: int = 42) -> None:
        self.repo = Path(repo).resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.requested_device = device
        self.seed = int(seed)
        self.runtime_sec = None
        self.end_to_end_sec = None
        self.peak_gpu_memory_mb = None
        self.device = "unresolved"
        self.checkpoint_key_match: dict | None = None

    def preflight(self) -> PreflightResult:
        source = self.repo / "sdm_unips" / "main.py"
        checkpoint = self.checkpoint_root / "normal" / "nml.pytmodel"
        if not source.is_file():
            return PreflightResult(False, f"missing official source: {source}", "unavailable")
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            return PreflightResult(False, f"missing official checkpoint: {checkpoint}", "unavailable")
        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError as exc:
            return PreflightResult(False, str(exc), "unavailable")
        if self.requested_device.startswith("cuda") and not cuda:
            return PreflightResult(False, "CUDA was requested for SDM-UniPS but is unavailable", "unavailable")
        self.device = "cuda" if (self.requested_device == "auto" and cuda) else self.requested_device
        if self.device == "auto":
            self.device = "cpu"
        match = self.verify_checkpoint()
        if not match.get("exact_match"):
            return PreflightResult(
                False,
                "SDM-UniPS checkpoint does not match the model state dict exactly. Upstream "
                "loads it with strict=False, so inference would silently keep randomly "
                f"initialised weights: {match}",
                self.device,
            )
        return PreflightResult(
            True,
            "official SDM-UniPS source and checkpoint found; checkpoint keys match the model "
            "state dict exactly; forward runtime is instrumented separately",
            self.device,
        )

    def _worker_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        # A native crash in the worker would otherwise surface only as a bare
        # Windows exit code, with no indication of which upstream op failed.
        environment["PYTHONFAULTHANDLER"] = "1"
        if self.device == "cpu":
            # Upstream wraps the network in torch.nn.DataParallel unconditionally.
            # With a visible CUDA device that wrapper scatters inputs to cuda:0
            # while the module stays on CPU, which fails on the first indexing
            # op. Hiding CUDA from the worker makes DataParallel a pass-through,
            # so the official CPU path runs without patching third-party code.
            # "-1" rather than "": on Windows an empty value leaves torch in a
            # half-initialised state where is_available() is True but the device
            # count is 0, and DataParallel then fails on device_ids[0].
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        return environment

    def verify_checkpoint(self) -> dict:
        """Confirm the checkpoint matches the model state dict exactly.

        Upstream loads with strict=False, so a drifted checkpoint would leave
        randomly initialised weights in place and still produce a plausible-looking
        normal map. Model construction is cheap and device-independent, so this runs
        on CPU and never competes for the GPU.
        """
        if self.checkpoint_key_match is not None:
            return self.checkpoint_key_match
        with tempfile.TemporaryDirectory(prefix="fabloop_sdm_verify_") as temporary:
            root = Path(temporary)
            metadata_path = root / "verify.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("sdm_worker.py")),
                "--repo", str(self.repo),
                "--checkpoint", str(self.checkpoint_root),
                "--test-dir", str(root),
                "--session-dir", str(root / "session"),
                "--image-count", "1",
                "--metadata", str(metadata_path),
                "--device", "cpu",
                "--seed", str(self.seed),
                "--verify-only",
            ]
            environment = dict(os.environ)
            environment["PYTHONFAULTHANDLER"] = "1"
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False, env=environment
            )
            if completed.returncode != 0 or not metadata_path.is_file():
                tail = (completed.stderr or completed.stdout)[-2000:]
                self.checkpoint_key_match = {
                    "exact_match": False,
                    "verification_failed": True,
                    "reason": f"checkpoint verification subprocess failed ({completed.returncode})",
                    "detail": tail,
                }
                return self.checkpoint_key_match
            self.checkpoint_key_match = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )["checkpoint_key_match"]
        return self.checkpoint_key_match

    def estimate_normals(self, images: np.ndarray, light_directions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del light_directions  # UniPS intentionally estimates without calibrated light directions.
        images = np.asarray(images, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[1:3] != mask.shape:
            raise ValueError("SDM-UniPS expects images N,H,W,3 and mask H,W")
        status = self.preflight()
        if not status.available:
            raise RuntimeError(status.reason)

        with tempfile.TemporaryDirectory(prefix="fabloop_sdm_unips_") as temporary:
            root = Path(temporary)
            input_dir = root / "input" / "object.data"
            session_dir = root / "session"
            input_dir.mkdir(parents=True)
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("opencv-python is required for SDM-UniPS input adaptation") from exc
            for index, image in enumerate(images):
                encoded = np.clip(image * 65535.0, 0.0, 65535.0).astype(np.uint16)
                cv2.imwrite(str(input_dir / f"L{index:03d}.png"), encoded[:, :, ::-1])
            cv2.imwrite(str(input_dir / "mask.png"), mask.astype(np.uint8) * 255)

            metadata_path = root / "runtime.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("sdm_worker.py")),
                "--repo", str(self.repo),
                "--checkpoint", str(self.checkpoint_root),
                "--test-dir", str(root / "input"),
                "--session-dir", str(session_dir),
                "--image-count", str(len(images)),
                "--metadata", str(metadata_path),
                "--device", self.device,
                "--seed", str(self.seed),
            ]
            environment = self._worker_environment()
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False, env=environment
            )
            if completed.returncode != 0:
                tail = excerpt_worker_output(completed.stderr or completed.stdout)
                detail = ""
                if completed.returncode == WINDOWS_ACCESS_VIOLATION and self.device == "cpu":
                    detail = (
                        " The worker died with a Windows access violation inside the upstream"
                        " attention feed-forward GEMM. Measured on this machine, SDM-UniPS CPU"
                        " inference is not viable at the official pixel_samples=10000; run this"
                        " method on CUDA. Lowering pixel_samples would change the benchmarked"
                        " protocol, so it is not done automatically."
                    )
                raise RuntimeError(
                    f"SDM-UniPS subprocess failed ({completed.returncode}).{detail}\n{tail}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.runtime_sec = float(metadata["forward_runtime_sec"])
            self.end_to_end_sec = float(metadata["end_to_end_runtime_sec"])
            self.peak_gpu_memory_mb = float(metadata["peak_gpu_memory_mb"])
            self.device = str(metadata["device"])
            output_path = session_dir / "results" / "object.data" / "normal_raw.npy"
            if not output_path.is_file():
                raise RuntimeError(f"SDM-UniPS adapter did not capture {output_path}")
            normal = np.load(output_path).astype(np.float32)
            if normal.shape != (*mask.shape, 3):
                raise RuntimeError(f"Unexpected SDM-UniPS normal shape {normal.shape}")
            normal[~mask] = 0.0
            return normalize_normals(normal).astype(np.float32)
