"""Adapter for the official, unmodified PS-FCN pretrained implementation."""

from __future__ import annotations

import importlib.util
import sys
import time
import types
import uuid
from pathlib import Path

import numpy as np

from ..diligent_validate import normalize_normals
from .base import NormalEstimator, PreflightResult


class PSFCNEstimator(NormalEstimator):
    name = "ps_fcn"

    def __init__(
        self, repo: Path, checkpoint: Path, device: str = "auto", source_integer_max: float = 255.0
    ) -> None:
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.requested_device = device
        self.source_integer_max = float(source_integer_max)
        self.runtime_sec = None
        self.end_to_end_sec = None
        self.peak_gpu_memory_mb = None
        self.device = "unresolved"
        self._model = None

    def _resolve_device(self):
        import torch

        if self.requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PS-FCN but is unavailable")
        return device

    def preflight(self) -> PreflightResult:
        source = self.repo / "models" / "PS_FCN_run.py"
        if not source.is_file():
            return PreflightResult(False, f"missing official source: {source}", "unavailable")
        if not self.checkpoint.is_file() or self.checkpoint.stat().st_size == 0:
            return PreflightResult(False, f"missing official checkpoint: {self.checkpoint}", "unavailable")
        try:
            device = str(self._resolve_device())
        except (ImportError, RuntimeError) as exc:
            return PreflightResult(False, str(exc), "unavailable")
        return PreflightResult(True, "official PS-FCN source and checkpoint found", device)

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch

        package_name = f"_fabloop_psfcn_{uuid.uuid4().hex}"
        package = types.ModuleType(package_name)
        package.__path__ = [str(self.repo / "models")]
        sys.modules[package_name] = package
        try:
            for module_name in ("model_utils", "PS_FCN_run"):
                qualified = f"{package_name}.{module_name}"
                spec = importlib.util.spec_from_file_location(
                    qualified, self.repo / "models" / f"{module_name}.py"
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load official PS-FCN module {module_name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
            model_class = sys.modules[f"{package_name}.PS_FCN_run"].PS_FCN
            model = model_class(fuse_type="max", batchNorm=False, c_in=6)
        finally:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    sys.modules.pop(key, None)

        device = self._resolve_device()
        # This is an official research checkpoint, intentionally loaded as a full
        # PyTorch checkpoint rather than claiming it is a weights-only artifact.
        checkpoint = torch.load(self.checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
        model.to(device).eval()
        self.device = str(device)
        self._model = model
        return model

    def estimate_normals(self, images: np.ndarray, light_directions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        images = np.asarray(images, dtype=np.float32)
        lights = np.asarray(light_directions, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] != len(lights):
            raise ValueError("PS-FCN expects images N,H,W,3 and matching lights N,3")
        if images.shape[1:3] != mask.shape or lights.shape != (len(images), 3):
            raise ValueError("PS-FCN input shapes are inconsistent")

        height, width = mask.shape
        image_chw = images.transpose(0, 3, 1, 2).reshape(1, -1, height, width)
        light_chw = np.broadcast_to(lights[:, :, None, None], (len(lights), 3, height, width))
        light_chw = light_chw.reshape(1, -1, height, width).copy()
        # The official DiLiGenT adapter reads each PNG with imageio and divides by
        # 255. DiLiGenT PNGs declare 16-bit depth, but imageio down-converts them to
        # uint8, so upstream actually feeds the network [0,1] radiance. Our loader
        # reads the true 16-bit samples and already stores [0,1], so the upstream
        # input scale is reproduced with a factor of one. Using 65535 here instead
        # would feed the network a 257x over-bright image and collapse PS-FCN on
        # DiLiGenT Ball from ~2.4 deg MAE to ~46 deg.
        image_tensor = torch.from_numpy(image_chw) * (self.source_integer_max / 255.0)
        light_tensor = torch.from_numpy(light_chw)
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]

        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if pad_h or pad_w:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
            light_tensor = F.pad(light_tensor, (0, pad_w, 0, pad_h))
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h))

        model = self._load_model()
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device) * mask_tensor.to(device)
        light_tensor = light_tensor.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model([image_tensor, light_tensor])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        else:
            self.peak_gpu_memory_mb = 0.0
        self.runtime_sec = time.perf_counter() - start
        self.end_to_end_sec = self.runtime_sec
        normal = output[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
        normal[~mask] = 0.0
        return normalize_normals(normal).astype(np.float32)
