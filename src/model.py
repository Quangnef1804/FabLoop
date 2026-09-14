"""Architecture-aware wrapper around Anomalib's EfficientAD lifecycle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from anomalib.models import EfficientAd

if __package__:
    from .models.efficientad_slim import SlimEfficientAdModel
    from .models.efficientad_slim.shape_check import check_feature_shapes
else:
    from models.efficientad_slim import SlimEfficientAdModel
    from models.efficientad_slim.shape_check import check_feature_shapes


def _anomalib_model_size(value: str) -> str:
    aliases = {"small": "small", "s": "small", "medium": "medium", "m": "medium"}
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise ValueError("model.size must be one of: small, s, medium, m") from error


def _architecture_id(model_config: dict[str, Any]) -> str:
    value = str(model_config.get("architecture", "anomalib-baseline")).lower()
    aliases = {
        "baseline": "anomalib-baseline",
        "anomalib": "anomalib-baseline",
        "anomalib-baseline": "anomalib-baseline",
        "slim-0.5": "efficientad-s-slim-0.5",
        "slim_0_5": "efficientad-s-slim-0.5",
        "efficientad-s-slim-0.5": "efficientad-s-slim-0.5",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(
            "model.architecture must be anomalib-baseline or efficientad-s-slim-0.5"
        ) from error


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_checksum(module: torch.nn.Module) -> str:
    """Create a deterministic SHA-256 over a module state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class EfficientAdWrapper:
    """Own model lifecycle checks while leaving architecture, maps and losses to Anomalib."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        model_config = config["model"]
        train_config = config["training"]
        model_size = _anomalib_model_size(str(model_config["size"]))
        architecture_id = _architecture_id(model_config)
        teacher_out_channels = int(model_config["teacher_out_channels"])
        self.lightning_model = EfficientAd(
            imagenet_dir=Path(config["imagenette"]["root"]),
            teacher_out_channels=teacher_out_channels,
            model_size=model_size,
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            padding=bool(model_config["padding"]),
            pad_maps=bool(model_config["pad_maps"]),
            pre_processor=False,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        if architecture_id == "efficientad-s-slim-0.5":
            if model_size != "small":
                raise ValueError("Slim-0.5 is defined only for EfficientAD-S (model.size: small)")
            self.lightning_model.model = SlimEfficientAdModel(
                teacher_out_channels=teacher_out_channels,
                padding=bool(model_config["padding"]),
                pad_maps=bool(model_config["pad_maps"]),
            )
        self.lightning_model.to(device)
        self.core = self.lightning_model.model
        if self.lightning_model.model is not self.core:
            raise RuntimeError("Lightning helper and runtime core must reference the same model")
        self.architecture_id = architecture_id
        self.architecture = self._architecture_metadata(model_size)
        self.device = device
        self.teacher_checksum: str | None = None

    def _architecture_metadata(self, model_size: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "id": self.architecture_id,
            "model_size": model_size,
            "core_class": type(self.core).__name__,
            "teacher_out_channels": int(self.core.teacher_out_channels),
            "student_output_channels": int(self.core.teacher_out_channels) * 2,
            "candidate": self.architecture_id == "efficientad-s-slim-0.5",
        }
        if metadata["candidate"]:
            source_root = Path(__file__).resolve().parent / "models" / "efficientad_slim"
            metadata.update(
                {
                    "width_multiplier": float(self.core.width_multiplier),
                    "student_channels": list(self.core.student_channels),
                    "autoencoder_encoder_channels": list(self.core.encoder_channels),
                    "autoencoder_decoder_channels": list(self.core.decoder_channels),
                    "source_sha256": {
                        "slim_model.py": _file_checksum(source_root / "slim_model.py"),
                        "torch_model.py": _file_checksum(source_root / "torch_model.py"),
                        "SOURCE.json": _file_checksum(source_root / "SOURCE.json"),
                    },
                }
            )
        return metadata

    def load_pretrained_teacher(self) -> str:
        """Use Anomalib's downloader/loader, then freeze the teacher completely."""

        self.lightning_model.to(self.device)
        self.lightning_model.prepare_pretrained_model()
        self.freeze_teacher()
        self.teacher_checksum = module_checksum(self.core.teacher)
        return self.teacher_checksum

    def freeze_teacher(self) -> None:
        self.core.teacher.eval()
        self.core.teacher.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.core.teacher.parameters()):
            raise RuntimeError("EfficientAD teacher is not fully frozen")

    def verify_feature_shapes(self, image_size: tuple[int, int]) -> dict[str, Any]:
        """Require matching Teacher, both Student heads and AE before training."""
        return check_feature_shapes(self.core, image_size=image_size)

    def verify_student_output_channels(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Compatibility entrypoint; now checks the complete feature contract."""
        self.verify_feature_shapes(image_size)
        channels = int(self.core.teacher_out_channels)
        return channels, channels

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.core.student.parameters()) + list(self.core.ae.parameters())

    def enforce_teacher_frozen(self) -> None:
        """Keep teacher frozen/eval after recursive ``train()`` calls."""

        self.core.teacher.eval()
        if any(parameter.requires_grad for parameter in self.core.teacher.parameters()):
            raise RuntimeError("Teacher unexpectedly became trainable")

    def load_core_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.core.load_state_dict(state_dict, strict=True)
        self.freeze_teacher()
        self.teacher_checksum = module_checksum(self.core.teacher)
