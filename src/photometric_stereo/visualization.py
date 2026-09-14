"""Reproducible benchmark artifacts and comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .diligent_validate import (
    normal_to_rgb,
    normals_to_height_frankot_chellappa,
    save_error_heatmap,
    save_grayscale_png,
    save_rgb_png,
)


METHOD_ORDER = ("l2", "l1", "ps_fcn", "sdm_unips")
METHOD_LABELS = {"l2": "L2", "l1": "L1", "ps_fcn": "PS-FCN", "sdm_unips": "SDM-UniPS"}


def _write_mesh_vtk(path: Path, height: np.ndarray, mask: np.ndarray) -> None:
    """Write a portable legacy VTK surface without requiring PyVista."""
    rows, cols = height.shape
    index_map = np.full((rows, cols), -1, dtype=np.int64)
    coordinates = np.argwhere(mask & np.isfinite(height))
    index_map[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates))
    faces: list[tuple[int, int, int, int]] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            ids = (
                index_map[row, col], index_map[row, col + 1],
                index_map[row + 1, col + 1], index_map[row + 1, col],
            )
            if min(ids) >= 0:
                faces.append(tuple(int(value) for value in ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\nFabLoop relative height\nASCII\nDATASET POLYDATA\n")
        handle.write(f"POINTS {len(coordinates)} float\n")
        for row, col in coordinates:
            handle.write(f"{col} {row} {float(height[row, col]):.9g}\n")
        handle.write(f"POLYGONS {len(faces)} {len(faces) * 5}\n")
        for face in faces:
            handle.write("4 " + " ".join(str(value) for value in face) + "\n")


def _save_mesh_png(path: Path, height: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = height.shape
    step = max(1, int(max(rows, cols) / 220))
    y, x = np.mgrid[0:rows:step, 0:cols:step]
    z = np.where(mask, height, np.nan)[::step, ::step]
    figure = plt.figure(figsize=(8, 6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=False)
    axis.set_title("Relative height (Frankot-Chellappa)")
    axis.set_xlabel("x [pixel]")
    axis.set_ylabel("y [pixel]")
    axis.set_zlabel("relative height")
    axis.view_init(elev=35, azim=-60)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_case_artifacts(
    output_dir: Path,
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
    *,
    error_max_deg: float = 90.0,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    height = normals_to_height_frankot_chellappa(estimated, mask, normal_y_axis="up")
    paths = {
        "normal_est_npy": output_dir / "normal_est.npy",
        "normal_est_rgb_png": output_dir / "normal_est_rgb.png",
        "normal_gt_rgb_png": output_dir / "normal_gt_rgb.png",
        "angular_error_npy": output_dir / "angular_error.npy",
        "angular_error_heatmap_png": output_dir / "angular_error_heatmap.png",
        "height_map_npy": output_dir / "height_map.npy",
        "height_map_png": output_dir / "height_map.png",
        "mesh_vtk": output_dir / "mesh.vtk",
        "mesh_png": output_dir / "mesh.png",
    }
    np.save(paths["normal_est_npy"], estimated.astype(np.float32))
    np.save(paths["angular_error_npy"], errors.astype(np.float32))
    np.save(paths["height_map_npy"], height.astype(np.float32))
    save_rgb_png(paths["normal_est_rgb_png"], normal_to_rgb(estimated, mask))
    save_rgb_png(paths["normal_gt_rgb_png"], normal_to_rgb(ground_truth, mask))
    save_error_heatmap(paths["angular_error_heatmap_png"], errors, mask, error_max_deg)
    save_grayscale_png(paths["height_map_png"], height, mask)
    _write_mesh_vtk(paths["mesh_vtk"], height, mask)
    _save_mesh_png(paths["mesh_png"], height, mask)
    return {key: str(value) for key, value in paths.items()}


def _successful(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("status") == "ok"]


def _light_counts(records: Iterable[dict[str, Any]]) -> list[int]:
    return sorted({int(record["lights"]) for record in records if record.get("lights") is not None})


def method_coverage(records: list[dict[str, Any]], method: str, lights: int) -> tuple[set[str], set[str]]:
    """Objects a method succeeded on at one light count, and the objects the matrix evaluated there."""
    expected = {
        str(record["object"]) for record in records
        if record.get("object") is not None and record.get("lights") is not None and int(record["lights"]) == lights
    }
    succeeded = {
        str(record["object"]) for record in _successful(records)
        if record["method"] == method and int(record["lights"]) == lights
    }
    return succeeded, expected


def is_complete(records: list[dict[str, Any]], method: str, lights: int) -> bool:
    """An aggregate is comparable across methods only if it covers every evaluated object."""
    succeeded, expected = method_coverage(records, method, lights)
    return bool(expected) and succeeded == expected


def coverage_note(records: list[dict[str, Any]], method: str, lights: int) -> str:
    succeeded, expected = method_coverage(records, method, lights)
    statuses: dict[str, int] = {}
    for record in records:
        if record.get("method") == method and record.get("lights") is not None and int(record["lights"]) == lights:
            if record.get("status") != "ok":
                status = str(record.get("status"))
                statuses[status] = statuses.get(status, 0) + 1
    detail = ", ".join(f"{status}x{count}" for status, count in sorted(statuses.items()))
    label = METHOD_LABELS.get(method, method)
    return f"{label} @ {lights}: {len(succeeded)}/{len(expected)} ok" + (f" ({detail})" if detail else "")


def _methods_present(records: list[dict[str, Any]]) -> list[str]:
    present = {record.get("method") for record in records}
    return [method for method in METHOD_ORDER if method in present]


def _rows_at(records: list[dict[str, Any]], method: str, lights: int) -> list[dict[str, Any]]:
    return [
        record for record in records
        if record.get("method") == method and record.get("lights") is not None and int(record["lights"]) == lights
    ]


def _mean_mae(records: list[dict[str, Any]], method: str, lights: int) -> float:
    return float(np.mean([float(row["mae_deg"]) for row in _successful(_rows_at(records, method, lights))]))


def _masked_errors(record: dict[str, Any]) -> np.ndarray:
    values = np.load(Path(str(record["angular_error_npy"])))
    return values[np.isfinite(values)]


def save_aggregate_plots(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = _successful(records)
    if not valid:
        return []
    paths: list[str] = []
    counts = _light_counts(records)
    methods = _methods_present(records)

    # Plot 1: a point is drawn only when the method covers every evaluated object,
    # otherwise its mean would be over a different object set than the other methods.
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    omitted: list[str] = []
    for method in methods:
        xs, ys = [], []
        for count in counts:
            if is_complete(records, method, count):
                xs.append(count)
                ys.append(_mean_mae(records, method, count))
            elif _rows_at(records, method, count):
                omitted.append(coverage_note(records, method, count))
        if xs:
            ax.plot(xs, ys, marker="o", label=METHOD_LABELS[method])
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(count) for count in counts])
    ax.set(xlabel="Number of lights", ylabel="Mean angular error over objects [deg]", title="MAE vs number of lights")
    ax.grid(alpha=0.25)
    if omitted:
        # Listed with the legend, outside the axes, so the note can never hide a curve.
        ax.plot([], [], " ", label="Omitted (incomplete coverage):")
        for note in omitted:
            ax.plot([], [], " ", label=note)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    path = output_dir / "plot_1_mae_vs_lights.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Plot 2: only runtimes that came from repeated measurement; one point per
    # method and light count, median runtime and mean MAE over the same objects.
    timed = [row for row in valid if row.get("runtime_sec") is not None]
    if timed:
        fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
        for method in methods:
            points = []
            for count in counts:
                rows = [row for row in timed if row["method"] == method and int(row["lights"]) == count]
                if rows:
                    points.append((
                        count,
                        float(np.median([float(row["runtime_sec"]) for row in rows])),
                        float(np.mean([float(row["mae_deg"]) for row in rows])),
                        any(row.get("runtime_caveat") for row in rows),
                    ))
            if points:
                ax.plot([p[1] for p in points], [p[2] for p in points], marker="o", label=METHOD_LABELS[method])
                for count, runtime, mae, caveated in points:
                    label = f"{count}*" if caveated else str(count)
                    ax.annotate(label, (runtime, mae), textcoords="offset points", xytext=(4, 4), fontsize=7)
        protocols = sorted({str(row.get("runtime_protocol")) for row in timed})
        objects = sorted({str(row["object"]) for row in timed})
        caveated_cases = sorted({
            f"{row['object']}/{row['method']}/{row['lights']}: {row['runtime_caveat']}"
            for row in timed if row.get("runtime_caveat")
        })
        if caveated_cases:
            ax.text(
                0.01, 0.99, "* median includes runtimes with a measurement caveat:\n" + "\n".join(caveated_cases),
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left", color="crimson",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        ax.set_xscale("log")
        ax.set(
            xlabel="Runtime, median over measured runs [s]", ylabel="Mean angular error [deg]",
            title="Accuracy vs runtime (point labels = number of lights)",
        )
        ax.text(
            0.99, 0.01, f"objects: {', '.join(objects)}\n" + "\n".join(protocols),
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
        )
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="lower left")
        path = output_dir / "plot_2_accuracy_vs_runtime.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    # Plot 3: 4 vs 96 lights; a bar is drawn only on complete coverage, otherwise
    # a hatched outline states how many objects succeeded and why the rest did not.
    low, high = (4, 96) if {4, 96} <= set(counts) else (counts[0], counts[-1])
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    width = 0.38
    for index, method in enumerate(methods):
        for offset, count, colour in ((-width / 2, low, "C0"), (width / 2, high, "C1")):
            x = index + offset
            if is_complete(records, method, count):
                value = _mean_mae(records, method, count)
                ax.bar(x, value, width, color=colour)
                ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
            else:
                succeeded, expected = method_coverage(records, method, count)
                statuses = sorted({
                    str(row.get("status")) for row in _rows_at(records, method, count) if row.get("status") != "ok"
                })
                ax.bar(x, 0.0, width, color="none", edgecolor=colour, hatch="//")
                ax.text(
                    x, 0.2, f"{len(succeeded)}/{len(expected)} " + " ".join(statuses),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
    ax.set_ylabel("Mean angular error over objects [deg]")
    ax.set_title(f"{low}-light vs {high}-light")
    # Explicit patches: empty bar containers do not carry their colour into the legend.
    ax.legend(handles=[
        Patch(facecolor="C0", label=f"{low} lights"),
        Patch(facecolor="C1", label=f"{high} lights"),
        Patch(facecolor="none", edgecolor="0.4", hatch="//", label="incomplete coverage (count/total)"),
    ])
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "plot_3_four_vs_96.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Plot 4: per-pixel error distribution at a fixed light count. Pooling across
    # counts would penalise a method that is missing its easy many-light cases.
    panels = [count for count in (low, high) if any(is_complete(records, method, count) for method in methods)]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.4), constrained_layout=True, squeeze=False)
        for axis, count in zip(axes[0], panels):
            distributions, labels, missing = [], [], []
            for method in methods:
                if is_complete(records, method, count):
                    rows = _successful(_rows_at(records, method, count))
                    distributions.append(np.concatenate([_masked_errors(row) for row in rows]))
                    labels.append(METHOD_LABELS[method])
                else:
                    missing.append(coverage_note(records, method, count))
            axis.boxplot(distributions, tick_labels=labels, whis=(5, 95), showfliers=False, showmeans=True)
            for position, values in enumerate(distributions, start=1):
                axis.plot(position, float(np.percentile(values, 99)), marker="x", color="crimson")
            axis.set_ylabel("Angular error [deg]")
            axis.set_title(f"{count} lights: box = IQR, whiskers = P5-P95, x = P99, triangle = mean", fontsize=9)
            axis.grid(axis="y", alpha=0.25)
            if missing:
                axis.text(
                    0.99, 0.99, "Not shown:\n" + "\n".join(missing),
                    transform=axis.transAxes, fontsize=7, va="top", ha="right",
                )
        path = output_dir / "plot_4_angular_error_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_method_comparisons(records: list[dict[str, Any]], output_dir: Path, error_max_deg: float) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    cases = sorted({(str(row["object"]), int(row["lights"])) for row in _successful(records)})
    for object_name, lights in cases:
        by_method = {
            row["method"]: row for row in records
            if str(row.get("object")) == object_name and row.get("lights") is not None and int(row["lights"]) == lights
        }
        methods = [method for method in METHOD_ORDER if method in by_method]
        ok = [method for method in methods if by_method[method].get("status") == "ok"]
        if not ok:
            continue
        case_dir = output_dir / "comparisons" / object_name / f"lights_{lights:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        def placeholder(axis, method):
            # A method that could not run stays visible, so the figure never
            # implies that it was simply not evaluated.
            axis.text(0.5, 0.5, str(by_method[method].get("status")), ha="center", va="center", fontsize=11, color="crimson")
            axis.set_title(METHOD_LABELS[method])
            axis.set_xticks([])
            axis.set_yticks([])

        gt = plt.imread(by_method[ok[0]]["normal_gt_rgb_png"])
        fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 4), constrained_layout=True)
        axes[0].imshow(gt)
        axes[0].set_title("GT")
        axes[0].axis("off")
        for axis, method in zip(axes[1:], methods):
            if method in ok:
                axis.imshow(plt.imread(by_method[method]["normal_est_rgb_png"]))
                axis.set_title(METHOD_LABELS[method])
                axis.axis("off")
            else:
                placeholder(axis, method)
        path = case_dir / "plot_5_normal_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), constrained_layout=True, squeeze=False)
        image = None
        for axis, method in zip(axes[0], methods):
            if method in ok:
                errors = np.load(by_method[method]["angular_error_npy"])
                image = axis.imshow(errors, cmap="turbo", vmin=0, vmax=error_max_deg)
                axis.set_title(METHOD_LABELS[method])
                axis.axis("off")
            else:
                placeholder(axis, method)
        fig.colorbar(image, ax=axes.ravel().tolist(), label=f"Angular error [deg], common scale 0-{error_max_deg:g}")
        path = case_dir / "plot_6_error_heatmaps_common_scale.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), constrained_layout=True, squeeze=False)
        for axis, method in zip(axes[0], methods):
            if method in ok:
                axis.imshow(np.load(by_method[method]["height_map_npy"]), cmap="viridis")
                axis.set_title(METHOD_LABELS[method])
                axis.axis("off")
            else:
                placeholder(axis, method)
        fig.suptitle("Relative height (Frankot-Chellappa), per-panel scale, illustration only", fontsize=9)
        path = case_dir / "plot_7_height_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths
