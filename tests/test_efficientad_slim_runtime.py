"""Runtime selection and checkpoint provenance for the Slim-0.5 candidate."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import EfficientAdWrapper
from models.efficientad_slim import SlimEfficientAdModel


def _config(architecture: str | None = None) -> dict:
    model = {
        "size": "small",
        "teacher_out_channels": 384,
        "padding": False,
        "pad_maps": True,
    }
    if architecture is not None:
        model["architecture"] = architecture
    return {
        "model": model,
        "training": {"learning_rate": 1.0e-4, "weight_decay": 1.0e-5},
        "imagenette": {"root": "data/imagenette"},
    }


class SlimRuntimeTests(unittest.TestCase):
    def test_explicit_slim_config_replaces_only_the_core(self) -> None:
        wrapper = EfficientAdWrapper(_config("efficientad-s-slim-0.5"), torch.device("cpu"))
        self.assertIsInstance(wrapper.core, SlimEfficientAdModel)
        self.assertIs(wrapper.lightning_model.model, wrapper.core)
        self.assertEqual(wrapper.architecture_id, "efficientad-s-slim-0.5")
        self.assertEqual(wrapper.architecture["student_channels"], [3, 64, 128, 128, 768])
        self.assertEqual(wrapper.architecture["teacher_out_channels"], 384)
        self.assertEqual(wrapper.architecture["student_output_channels"], 768)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.core.teacher.parameters()))
        self.assertEqual(
            {id(parameter) for parameter in wrapper.trainable_parameters()},
            {id(parameter) for parameter in wrapper.core.student.parameters()}
            | {id(parameter) for parameter in wrapper.core.ae.parameters()},
        )

    def test_missing_architecture_keeps_anomalib_baseline(self) -> None:
        wrapper = EfficientAdWrapper(_config(), torch.device("cpu"))
        self.assertNotIsInstance(wrapper.core, SlimEfficientAdModel)
        self.assertEqual(wrapper.architecture_id, "anomalib-baseline")

    def test_slim_rejects_non_small_model(self) -> None:
        config = _config("slim-0.5")
        config["model"]["size"] = "medium"
        with self.assertRaisesRegex(ValueError, "only for EfficientAD-S"):
            EfficientAdWrapper(config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
