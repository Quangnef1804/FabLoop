from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photometric_stereo.benchmark import is_out_of_memory, measure_case, runtime_statistics
from photometric_stereo.lighting import select_cardinal_four, select_light_subsets
from photometric_stereo.metrics import attach_l2_comparison, normal_error_metrics
from photometric_stereo.visualization import save_case_artifacts
from photometric_stereo.wrappers.base import PreflightResult
from photometric_stereo.wrappers.ps_fcn import PSFCNEstimator
from photometric_stereo.wrappers.l2_l1 import RobustPSEstimator
from photometric_stereo.wrappers.sdm_unips import WINDOWS_ACCESS_VIOLATION, SDMUniPSEstimator
from photometric_stereo.wrappers.sdm_worker import decode_upstream_float_bgr


class LightingSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        azimuth = np.deg2rad(np.arange(-180, 180, 15, dtype=np.float64))
        elevation = np.deg2rad(45.0)
        self.lights = np.column_stack(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.full_like(azimuth, np.sin(elevation))]
        )

    def test_cardinal_four_are_unique_and_near_targets(self) -> None:
        selected = select_cardinal_four(self.lights)
        self.assertEqual(len(set(selected)), 4)
        azimuth = np.degrees(np.arctan2(self.lights[selected, 1], self.lights[selected, 0]))
        expected = np.array([0.0, 90.0, -180.0, -90.0])
        np.testing.assert_allclose(azimuth, expected, atol=1.0e-7)

    def test_subsets_are_deterministic_nested_and_full_at_total_count(self) -> None:
        first = select_light_subsets(self.lights, [4, 8, 16, 24])
        second = select_light_subsets(self.lights, [4, 8, 16, 24])
        self.assertEqual(first, second)
        self.assertEqual(first[4], first[8][:4])
        self.assertEqual(first[8], first[16][:8])
        self.assertEqual(set(first[24]), set(range(24)))


class MetricTests(unittest.TestCase):
    def test_masked_metrics_and_percentiles(self) -> None:
        gt = np.zeros((2, 2, 3), dtype=np.float32); gt[..., 2] = 1
        est = gt.copy(); est[0, 1] = [1, 0, 0]; est[1, 0] = [0, 1, 0]
        mask = np.array([[True, True], [False, False]])
        metrics, errors = normal_error_metrics(est, gt, mask)
        self.assertAlmostEqual(metrics["mae_deg"], 45.0)
        self.assertAlmostEqual(metrics["median_deg"], 45.0)
        self.assertAlmostEqual(metrics["p90_deg"], 81.0)
        self.assertTrue(np.isnan(errors[1, 0]))

    def test_l2_delta_and_relative_improvement(self) -> None:
        rows = [
            {"object": "ball", "method": "l2", "lights": 4, "mae_deg": 20.0, "status": "ok"},
            {"object": "ball", "method": "l1", "lights": 4, "mae_deg": 15.0, "status": "ok"},
        ]
        attach_l2_comparison(rows)
        self.assertEqual(rows[1]["delta_mae_vs_l2_deg"], -5.0)
        self.assertEqual(rows[1]["relative_improvement_vs_l2_pct"], 25.0)

    def test_oom_record_does_not_acquire_metric_fields(self) -> None:
        rows = [
            {"object": "ball", "method": "l2", "lights": 96, "mae_deg": 4.0, "status": "ok"},
            {
                "object": "ball",
                "method": "sdm_unips",
                "lights": 96,
                "status": "unsupported_oom",
                "gpu_name": "test GPU",
                "gpu_total_memory_mb": 4096,
                "oom_note": "unsupported at this light count",
            },
        ]
        attach_l2_comparison(rows)
        oom = rows[1]
        self.assertNotIn("mae_deg", oom)
        self.assertNotIn("delta_mae_vs_l2_deg", oom)
        self.assertNotIn("relative_improvement_vs_l2_pct", oom)


class ArtifactAndPreflightTests(unittest.TestCase):
    def test_sdm_float_normal_capture_avoids_png_quantization(self) -> None:
        expected = np.array([[[0.12345, -0.23456, 0.96412]]], dtype=np.float32)
        upstream_bgr = 255.0 * (0.5 * (1.0 + expected[:, :, ::-1]))
        actual = decode_upstream_float_bgr(upstream_bgr)
        np.testing.assert_allclose(actual, expected, atol=1.0e-6)

    def test_case_artifacts_include_portable_vtk(self) -> None:
        rows, cols = 8, 9
        normal = np.zeros((rows, cols, 3), dtype=np.float32); normal[..., 2] = 1
        mask = np.ones((rows, cols), dtype=bool)
        errors = np.zeros((rows, cols), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = save_case_artifacts(Path(temporary), normal, normal, errors, mask)
            for path in outputs.values():
                self.assertTrue(Path(path).is_file(), path)
            self.assertIn("DATASET POLYDATA", Path(outputs["mesh_vtk"]).read_text(encoding="ascii"))

    def test_ps_fcn_missing_checkpoint_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "models" / "PS_FCN_run.py").write_text("", encoding="utf-8")
            status = PSFCNEstimator(root, root / "missing.pth").preflight()
            self.assertFalse(status.available)
            self.assertIn("checkpoint", status.reason)

    def test_classical_wrapper_common_interface_recovers_normals(self) -> None:
        vendor = Path(__file__).resolve().parents[1] / "third-party" / "RobustPhotometricStereo"
        if not (vendor / "rps.py").is_file():
            self.skipTest("Optional official RobustPhotometricStereo clone is unavailable")
        lights = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=np.float32)
        lights /= np.linalg.norm(lights, axis=1, keepdims=True)
        expected = np.array([[[0.1, 0.2, 1.0], [-0.2, 0.1, 1.0]]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=2, keepdims=True)
        intensity = np.einsum("hwc,nc->nhw", expected, lights)
        images = np.repeat(intensity[:, :, :, None], 3, axis=3)
        estimator = RobustPSEstimator("l2", vendor)
        actual = estimator.estimate_normals(images, lights, np.ones((1, 2), dtype=bool))
        np.testing.assert_allclose(actual, expected, atol=1.0e-6)


class _ScriptedEstimator:
    """Stand-in estimator with prescribed per-call runtimes and outputs."""

    def __init__(self, runtimes, outputs=None, error=None):
        self.runtimes = list(runtimes)
        self.outputs = outputs
        self.error = error
        self.calls = 0
        self.runtime_sec = None
        self.end_to_end_sec = None
        self.peak_gpu_memory_mb = None
        self.device = "cpu"

    def estimate_normals(self, images, light_directions, mask):
        del images, light_directions
        if self.error is not None:
            raise self.error
        index = self.calls
        self.calls += 1
        self.runtime_sec = self.runtimes[index]
        self.end_to_end_sec = self.runtimes[index] + 1.0
        self.peak_gpu_memory_mb = 100.0 * (index + 1)
        if self.outputs is not None:
            return self.outputs[index]
        normal = np.zeros((*mask.shape, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        return normal


class RuntimeProtocolTests(unittest.TestCase):
    def test_runtime_statistics_report_median_and_spread_not_a_single_sample(self):
        stats = runtime_statistics([5.0, 1.0, 3.0], "runtime")
        self.assertEqual(stats["runtime_sec"], 3.0)
        self.assertEqual(stats["runtime_median_sec"], 3.0)
        self.assertEqual(stats["runtime_min_sec"], 1.0)
        self.assertEqual(stats["runtime_max_sec"], 5.0)
        self.assertAlmostEqual(stats["runtime_std_sec"], 2.0)
        self.assertAlmostEqual(stats["runtime_iqr_sec"], 2.0)
        self.assertEqual(stats["runtime_runs_sec"], [5.0, 1.0, 3.0])

    def test_single_measured_run_has_zero_spread_rather_than_failing(self):
        stats = runtime_statistics([7.5], "end_to_end")
        self.assertEqual(stats["end_to_end_sec"], 7.5)
        self.assertEqual(stats["end_to_end_std_sec"], 0.0)
        self.assertEqual(stats["end_to_end_iqr_sec"], 0.0)

    def test_measure_case_discards_warmup_and_reports_median_of_measured_runs(self):
        mask = np.ones((2, 2), dtype=bool)
        normal_gt = np.zeros((2, 2, 3), dtype=np.float32)
        normal_gt[..., 2] = 1.0
        estimator = _ScriptedEstimator([99.0, 5.0, 1.0, 3.0])
        _, metrics, _, timing = measure_case(
            estimator, np.zeros((1, 2, 2, 3), np.float32), np.array([[0.0, 0.0, 1.0]]),
            mask, normal_gt, warmup_runs=1, measured_runs=3,
        )
        self.assertEqual(estimator.calls, 4)
        self.assertEqual(timing["runtime_measured_runs"], 3)
        self.assertEqual(timing["runtime_warmup_runs"], 1)
        self.assertNotIn(99.0, timing["runtime_runs_sec"])
        self.assertEqual(timing["runtime_sec"], 3.0)
        self.assertEqual(timing["peak_gpu_memory_mb"], 400.0)
        self.assertTrue(timing["repeat_outputs_identical"])
        self.assertAlmostEqual(metrics["mae_deg"], 0.0)

    def test_nondeterministic_repeats_are_flagged_and_metrics_come_from_first_run(self):
        mask = np.ones((1, 1), dtype=bool)
        normal_gt = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        first = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        second = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        estimator = _ScriptedEstimator([1.0, 2.0], outputs=[first, second])
        normal, metrics, _, timing = measure_case(
            estimator, np.zeros((1, 1, 1, 3), np.float32), np.array([[0.0, 0.0, 1.0]]),
            mask, normal_gt, warmup_runs=0, measured_runs=2,
        )
        np.testing.assert_array_equal(normal, first)
        self.assertAlmostEqual(metrics["mae_deg"], 0.0)
        self.assertFalse(timing["repeat_outputs_identical"])
        self.assertGreater(timing["mae_deg_std_over_runs"], 0.0)


class PSFCNInputScaleTests(unittest.TestCase):
    def test_default_input_scale_matches_upstream_zero_to_one_radiance(self):
        """Upstream reads DiLiGenT PNGs through imageio, which down-converts the
        declared 16-bit data to uint8 before dividing by 255. Feeding 65535-scaled
        data instead is a 257x over-brightening that collapses PS-FCN accuracy."""
        estimator = PSFCNEstimator(Path("repo"), Path("ckpt.pth.tar"), "cpu")
        self.assertEqual(estimator.source_integer_max, 255.0)


class SDMNativeCrashReportingTests(unittest.TestCase):
    def test_cpu_access_violation_is_explained_rather_than_left_as_an_exit_code(self):
        """A bare 3221225477 gives a reader nothing to act on. CPU inference dies in
        the upstream attention GEMM, and the actionable answer is to use CUDA."""
        estimator = SDMUniPSEstimator(Path("repo"), Path("checkpoint"), "cpu")
        estimator.device = "cpu"
        completed = SimpleNamespace(
            returncode=WINDOWS_ACCESS_VIOLATION, stderr="native crash", stdout=""
        )
        with mock.patch("photometric_stereo.wrappers.sdm_unips.subprocess.run", return_value=completed),              mock.patch.object(SDMUniPSEstimator, "preflight",
                               return_value=PreflightResult(True, "stub", "cpu")):
            images = np.zeros((2, 4, 4, 3), dtype=np.float32)
            with self.assertRaises(RuntimeError) as caught:
                estimator.estimate_normals(images, np.zeros((2, 3)), np.ones((4, 4), dtype=bool))
        message = str(caught.exception)
        self.assertIn("access violation", message)
        self.assertIn("run this method on CUDA", message)
        self.assertIn("not done automatically", message)


class SDMCheckpointGateTests(unittest.TestCase):
    """Upstream loads with strict=False, so a drifted checkpoint would leave randomly
    initialised weights in place and still emit a plausible normal map. Any key
    mismatch must block inference rather than produce an unverified benchmark number."""

    def _estimator(self):
        estimator = SDMUniPSEstimator(Path("repo"), Path("checkpoint"), "cpu")
        estimator.repo = Path("repo")
        return estimator

    def test_key_mismatch_fails_preflight(self):
        estimator = self._estimator()
        estimator.checkpoint_key_match = {
            "checkpoint_keys": 373, "model_keys": 374,
            "missing_in_checkpoint": ["regressor.bias"], "unexpected_in_checkpoint": [],
            "exact_match": False,
        }
        with mock.patch.object(Path, "is_file", return_value=True),              mock.patch.object(Path, "stat") as stat:
            stat.return_value = SimpleNamespace(st_size=1)
            result = estimator.preflight()
        self.assertFalse(result.available)
        self.assertIn("strict=False", result.reason)
        self.assertIn("regressor.bias", result.reason)

    def test_failed_verification_is_not_silently_treated_as_a_match(self):
        estimator = self._estimator()
        completed = SimpleNamespace(returncode=1, stderr="import blew up", stdout="")
        with mock.patch("photometric_stereo.wrappers.sdm_unips.subprocess.run", return_value=completed):
            match = estimator.verify_checkpoint()
        self.assertFalse(match["exact_match"])
        self.assertTrue(match["verification_failed"])

    def test_verification_result_is_cached_so_the_matrix_pays_for_it_once(self):
        estimator = self._estimator()
        completed = SimpleNamespace(returncode=1, stderr="", stdout="")
        with mock.patch("photometric_stereo.wrappers.sdm_unips.subprocess.run",
                        return_value=completed) as run:
            estimator.verify_checkpoint()
            estimator.verify_checkpoint()
        self.assertEqual(run.call_count, 1)


class WorkerOutputExcerptTests(unittest.TestCase):
    def test_excerpt_keeps_the_fatal_exception_header_and_the_tail(self):
        from photometric_stereo.wrappers.sdm_unips import excerpt_worker_output
        text = 'Windows fatal exception: code 0xc0000409' + 'x' * 20000 + 'Extension modules: torch'
        excerpt = excerpt_worker_output(text)
        self.assertIn('Windows fatal exception', excerpt)
        self.assertIn('Extension modules', excerpt)
        self.assertIn('characters omitted', excerpt)
        self.assertLess(len(excerpt), 6000)

    def test_short_output_is_returned_unchanged(self):
        from photometric_stereo.wrappers.sdm_unips import excerpt_worker_output
        self.assertEqual(excerpt_worker_output('short'), 'short')


class OutOfMemoryClassificationTests(unittest.TestCase):
    def test_subprocess_surfaced_cuda_oom_is_recognised(self):
        exc = RuntimeError(
            "SDM-UniPS subprocess failed (1): "
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 2.00 GiB"
        )
        self.assertTrue(is_out_of_memory(exc))

    def test_unrelated_failures_are_not_misreported_as_oom(self):
        self.assertFalse(is_out_of_memory(ValueError("images and mask have different spatial shapes")))
        self.assertFalse(is_out_of_memory(FileNotFoundError("missing checkpoint")))


class LightLevelStabilityTests(unittest.TestCase):
    def test_adding_a_light_level_does_not_change_existing_subsets(self):
        """64 lights were added after 4/8/16/32/96 had been benchmarked, so the nested
        selection must leave every already-measured subset untouched."""
        index = np.arange(96)
        z = 0.2 + 0.8 * (index + 0.5) / 96
        radius = np.sqrt(1.0 - z**2)
        theta = index * np.pi * (3.0 - np.sqrt(5.0))
        lights = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
        before = select_light_subsets(lights, [4, 8, 16, 32, 96])
        after = select_light_subsets(lights, [4, 8, 16, 32, 64, 96])
        for count in (4, 8, 16, 32, 96):
            self.assertEqual(list(before[count]), list(after[count]))
        self.assertTrue(set(after[32]) <= set(after[64]) <= set(after[96]))
        self.assertEqual(len(set(after[64])), 64)

if __name__ == "__main__":
    unittest.main()
