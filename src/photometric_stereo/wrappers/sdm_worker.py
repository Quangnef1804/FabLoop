"""Internal isolated runner; imports official SDM-UniPS without modifying it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def decode_upstream_float_bgr(image: np.ndarray) -> np.ndarray:
    """Invert upstream's float BGR visualization transform before PNG quantization."""
    return 2.0 * (np.asarray(image, dtype=np.float32)[:, :, ::-1] / 255.0) - 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--image-count", type=int, required=True)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Build the model, compare checkpoint keys, write metadata, and exit before inference.",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    source_root = args.repo.resolve() / "sdm_unips"
    sys.path.insert(0, str(source_root))
    import torch
    import cv2

    device = torch.device(args.device)

    # Upstream's model_utils.loadmodel calls torch.load without map_location, so a
    # CUDA-saved checkpoint cannot be deserialised on CPU. Redirect the default in
    # this worker process only, exactly as the cv2.imwrite capture below does,
    # rather than editing the third-party source.
    original_torch_load = torch.load

    def torch_load_on_target_device(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", device)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_on_target_device

    from modules.builder import builder
    from modules.io import dataio
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    upstream_args = SimpleNamespace(
        target="normal",
        checkpoint=str(args.checkpoint.resolve()),
        pixel_samples=10000,
        scalable=True,
        canonical_resolution=256,
        test_dir=str(args.test_dir.resolve()),
        test_ext=".data",
        test_prefix="L*",
        mask_margin=0,
        max_image_num=args.image_count,
        session_name=str(args.session_dir.resolve()),
    )
    model = builder.builder(upstream_args, device)

    # Upstream loads the checkpoint with strict=False, which would silently leave a
    # partially random network if the names ever drifted. Record the actual overlap
    # so a benchmark number can never rest on an unverified load.
    checkpoint_path = "".join(
        str(path) for path in sorted((args.checkpoint.resolve() / "normal").glob("*.pytmodel"))
    )
    checkpoint_keys = set(original_torch_load(checkpoint_path, map_location="cpu").keys())
    model_keys = set(model.net_nml.state_dict().keys())
    checkpoint_key_match = {
        "checkpoint_keys": len(checkpoint_keys),
        "model_keys": len(model_keys),
        "missing_in_checkpoint": sorted(model_keys - checkpoint_keys)[:10],
        "unexpected_in_checkpoint": sorted(checkpoint_keys - model_keys)[:10],
        "exact_match": checkpoint_keys == model_keys,
    }

    if args.verify_only:
        args.metadata.write_text(
            json.dumps({"checkpoint_key_match": checkpoint_key_match, "device": str(device)}),
            encoding="utf-8",
        )
        return 0

    if not checkpoint_key_match["exact_match"]:
        raise RuntimeError(
            "Refusing to run SDM-UniPS inference: the checkpoint does not match the model "
            "state dict exactly, and upstream loads it with strict=False, so the network "
            "would silently keep randomly initialised weights. "
            f"{checkpoint_key_match}"
        )

    dataset = dataio.dataio("Test", upstream_args)

    forward_runtime_sec = 0.0
    forward_started = 0.0

    def before_forward(_module, _inputs):
        nonlocal forward_started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started = time.perf_counter()

    def after_forward(_module, _inputs, _output):
        nonlocal forward_runtime_sec
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_runtime_sec += time.perf_counter() - forward_started

    pre_handle = model.net_nml.register_forward_pre_hook(before_forward)
    post_handle = model.net_nml.register_forward_hook(after_forward)

    original_imwrite = cv2.imwrite

    def capture_float_normal(filename, image, *write_args):
        if Path(filename).name.lower() == "normal.png":
            # Upstream hands cv2 a float BGR visualization before uint8 encoding.
            # Recover the full-precision RGB normal here so benchmark metrics do
            # not include PNG quantization error.
            normal_rgb = decode_upstream_float_bgr(image)
            np.save(Path(filename).with_name("normal_raw.npy"), normal_rgb)
        return original_imwrite(filename, image, *write_args)

    cv2.imwrite = capture_float_normal
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        model.run(testdata=dataset, max_image_resolution=512, canonical_resolution=256)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        else:
            peak = 0.0
    finally:
        cv2.imwrite = original_imwrite
        pre_handle.remove()
        post_handle.remove()
    end_to_end_runtime_sec = time.perf_counter() - start
    args.metadata.write_text(
        json.dumps({
            "forward_runtime_sec": forward_runtime_sec,
            "end_to_end_runtime_sec": end_to_end_runtime_sec,
            "peak_gpu_memory_mb": peak,
            "device": str(device),
            "seed": args.seed,
            "normal_precision": "float32 captured before upstream PNG encoding",
            "checkpoint_key_match": checkpoint_key_match,
        }),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
