#!/usr/bin/env python3
"""Plot calorie decision trade-offs and validation NLL using plotnine."""

import hashlib
import json
import math
import os
import tempfile
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parent
PALETTE = {
    "ResNet-152": "#0072B2",
    "Swin V2": "#D55E00",
    "DINOv3": "#009E73",
    "Training mean": "#444444",
    "Luna/low": "#AA4499",
}
app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class Population:
    path: Path
    sha256: str
    ids: tuple[str, ...]
    labels: np.ndarray
    missing: tuple[str, ...]


def load_population(path: Path) -> Population:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    ids = tuple(sample["id"] for sample in manifest["samples"])
    labels = np.array(
        [s["target"]["calories_kcal"] for s in manifest["samples"]], dtype=float
    )
    if len(set(ids)) != len(ids) or len(ids) != manifest["n"] or not len(ids):
        raise ValueError("Invalid population IDs/count")
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Invalid calorie labels")
    return Population(
        path,
        hashlib.sha256(raw).hexdigest(),
        ids,
        labels,
        tuple(manifest["missing_overhead_images"]),
    )


def task_metrics(estimates, labels) -> dict[str, Any]:
    x, y = np.asarray(estimates, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not len(x):
        raise ValueError(
            "Need nonempty, aligned one-dimensional predictions and labels"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Predictions and labels must be finite")
    errors = x - y
    over = errors >= 0  # Equality belongs to X >= Y, as specified.
    n_over = int(over.sum())
    return {
        "p_under": float((errors < 0).mean()),
        "mean_over_kcal": float(errors[over].mean()) if n_over else None,
        "n_over": n_over,
        "n_under": int((errors < 0).sum()),
        "n": len(x),
    }


def quantile_sweep(mean, std, labels, quantiles, *, clip_zero: bool = True):
    mu, sigma, y = (np.asarray(a, dtype=float) for a in (mean, std, labels))
    qs = np.asarray(quantiles, dtype=float)
    if mu.ndim != 1 or mu.shape != sigma.shape or mu.shape != y.shape or not len(mu):
        raise ValueError("Gaussian parameters and labels must align")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("Gaussian means must be finite and stds finite and positive")
    if (
        qs.ndim != 1
        or not len(qs)
        or not np.isfinite(qs).all()
        or ((qs <= 0) | (qs >= 1)).any()
    ):
        raise ValueError("Quantiles must lie strictly between zero and one")
    rows = []
    for q in qs:
        raw = mu + sigma * norm.ppf(q)
        deployed = np.maximum(raw, 0) if clip_zero else raw
        rows.append(
            {
                "q": float(q),
                **task_metrics(deployed, y),
                "raw_p_under": float((raw < y).mean()),
                "raw_quantile_coverage": float((y <= raw).mean()),
                "n_floored": int((raw < 0).sum()) if clip_zero else 0,
            }
        )
    return pd.DataFrame(rows)


def nondominated(frame: pd.DataFrame) -> np.ndarray:
    """Minimize both coordinates; duplicate points remain nondominated."""
    x = frame["p_under"].to_numpy(dtype=float)
    y = frame["mean_over_kcal"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    valid = np.flatnonzero(finite)
    result = np.zeros(len(frame), dtype=bool)
    xx, yy = x[valid], y[valid]
    weak = (xx[:, None] <= xx[None, :]) & (yy[:, None] <= yy[None, :])
    strict = (xx[:, None] < xx[None, :]) | (yy[:, None] < yy[None, :])
    result[valid] = ~(weak & strict).any(axis=0)
    return result


def read_predictions(directory: Path, population: Population):
    config = json.loads((directory / "config.json").read_text())
    raw = (directory / "predictions.json").read_bytes()
    records = json.loads(raw)
    if config["manifest_sha256"] != population.sha256:
        raise ValueError(f"Population fingerprint mismatch: {directory}")
    if len(records) != len(population.ids) or any(r["status"] != "ok" for r in records):
        raise ValueError(
            f"Need successful predictions for the ENTIRE population: {directory}"
        )
    indexed = {r["id"]: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(population.ids):
        raise ValueError(f"Prediction IDs do not match the population: {directory}")
    ordered = [indexed[key] for key in population.ids]
    labels = np.array([r["target"]["calories_kcal"] for r in ordered], dtype=float)
    if not np.array_equal(labels, population.labels):
        raise ValueError(
            f"Prediction labels differ from population labels: {directory}"
        )
    return ordered, {
        "directory": str(directory),
        "predictions_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": config["predictor"].get("checkpoint_sha256"),
    }


def collect_task_data(spec, population, quantiles, clip_zero):
    curves, points, sources = [], [], []
    for run in spec["gaussian"]:
        records, source = read_predictions(PROJECT / run["test_run"], population)
        try:
            mean = [
                r["backend_metadata"]["gaussian_mean"]["calories_kcal"] for r in records
            ]
            std = [
                r["backend_metadata"]["gaussian_std"]["calories_kcal"] for r in records
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Gaussian curves require actual raw means/stds, not legacy sigma half-widths"
            ) from error
        frame = quantile_sweep(
            mean, std, population.labels, quantiles, clip_zero=clip_zero
        )
        frame["model"], frame["series"] = run["name"], "Gaussian " + run["name"]
        curves.append(frame)
        sources.append({"name": run["name"], "kind": "Gaussian", **source})
    for run in spec.get("point", []):
        records, source = read_predictions(PROJECT / run["test_run"], population)
        estimates = [r["prediction"]["mean"]["calories_kcal"] for r in records]
        points.append(
            {
                "model": run["name"],
                "series": "Point " + run["name"],
                "estimate": run.get("estimate", "Point head"),
                **task_metrics(estimates, population.labels),
            }
        )
        sources.append({"name": run["name"], "kind": "Point", **source})
    curve = pd.concat(curves, ignore_index=True)
    point = pd.DataFrame(points)
    all_points = pd.concat([curve, point], ignore_index=True)
    all_points["pooled_nondominated"] = nondominated(all_points)
    curve["pooled_nondominated"] = all_points.iloc[: len(curve)][
        "pooled_nondominated"
    ].to_numpy()
    if len(point):
        point["pooled_nondominated"] = all_points.iloc[len(curve) :][
            "pooled_nondominated"
        ].to_numpy()
    return curve, point, all_points, sources


def validation_data(gaussian_runs):
    rows, markers, summary_rows = [], [], []
    reference_ids, reference_manifest = None, None
    for run in gaussian_runs:
        root = PROJECT / run["training_run"]
        cfg = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        ids = tuple(cfg["validation_ids"])
        if reference_ids is None:
            reference_ids, reference_manifest = ids, cfg["manifest_sha256"]
        if ids != reference_ids or cfg["manifest_sha256"] != reference_manifest:
            raise ValueError(
                "Validation runs must use the same held-out validation population"
            )
        history = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text().splitlines()
        ]
        for record in history:
            val = record["validation"]
            physical = val["nll"]["calories_kcal"]
            expected = val["nll_standardized"]["calories_kcal"] + math.log(
                cfg["target_std"][0]
            )
            if not math.isclose(physical, expected, abs_tol=1e-5):
                raise ValueError("Calorie NLL units/standardization mismatch")
            rows.append(
                {
                    "model": run["name"],
                    "epoch": record["epoch"],
                    "calorie_nll": physical,
                    "five_target_loss": val["loss"],
                }
            )
        selected = next(r for r in history if r["epoch"] == summary["best_epoch"])
        minimum = min(history, key=lambda r: r["validation"]["nll"]["calories_kcal"])
        for record, label in (
            (selected, "Saved checkpoint"),
            (minimum, "Minimum calorie NLL"),
        ):
            markers.append(
                {
                    "model": run["name"],
                    "epoch": record["epoch"],
                    "calorie_nll": record["validation"]["nll"]["calories_kcal"],
                    "marker": label,
                }
            )
        summary_rows.append(
            {
                "model": run["name"],
                "validation_n": len(ids),
                "saved_epoch": selected["epoch"],
                "saved_calorie_nll": selected["validation"]["nll"]["calories_kcal"],
                "minimum_calorie_nll_epoch": minimum["epoch"],
                "minimum_calorie_nll": minimum["validation"]["nll"]["calories_kcal"],
                "checkpoint_selection": cfg["selection"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(markers), pd.DataFrame(summary_rows)


def make_plots(curve, point, n, subset=False, *, clip_zero=True):
    from plotnine import (
        aes,
        coord_cartesian,
        element_line,
        element_text,
        geom_abline,
        geom_path,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_shape_manual,
        scale_x_continuous,
        scale_y_continuous,
        theme,
        theme_minimal,
    )

    style = theme_minimal(base_size=11, base_family="DejaVu Sans") + theme(
        figure_size=(9, 6),
        legend_position="bottom",
        legend_box="vertical",
        plot_title=element_text(size=16, weight="bold", color="#172B4D"),
        plot_subtitle=element_text(size=11, color="#4A5568"),
        plot_caption=element_text(size=8, ha="left", color="#596579"),
        panel_grid_minor=element_line(alpha=0),
    )
    population_label = f"{n} test images" + (" · saved subset" if subset else "")
    finite = curve[curve["mean_over_kcal"].notna()].copy()
    anchors = finite[np.isclose(finite["q"], 0.95)].copy()
    anchors["estimate"] = "Gaussian q=0.95"
    visible_points = point[point["mean_over_kcal"].notna()] if len(point) else point
    markers = pd.concat([anchors, visible_points], ignore_index=True)
    front = finite[finite["pooled_nondominated"]]
    policy = (
        "Deployment estimates are floored at 0 kcal"
        if clip_zero
        else "Raw Gaussian quantiles are used without clipping"
    )
    pareto_base = (
        ggplot(finite, aes("p_under", "mean_over_kcal", color="model", group="series"))
        + geom_path(size=0.85, alpha=0.85)
        + geom_point(data=front, size=1.6, alpha=0.9, show_legend=False)
        + geom_point(data=markers, mapping=aes(shape="estimate"), size=3.1)
        + scale_color_manual(values=PALETTE, name="Model")
        + scale_shape_manual(
            values={
                "Gaussian q=0.95": "*",
                "Point head": "s",
                "Training mean": "X",
                "Saved Luna/low": "D",
            },
            name="Operating point",
        )
        + labs(
            title="Calorie overestimation versus underestimation",
            subtitle=f"{population_label} · lower is better on both axes",
            x="Probability of underestimating, P(X < Y)",
            y="Mean excess given X ≥ Y (kcal)",
            caption="Lines: quantile sweeps; small circles: pooled non-dominated sampled operating points.\n"
            + policy
            + "; equality counts in the conditional excess mean.",
        )
        + style
    )
    pareto = (
        pareto_base
        + scale_x_continuous(breaks=np.linspace(0, 1, 6).tolist())
        + coord_cartesian(xlim=(0, 1), ylim=(0, None))
    )
    focus = (
        pareto_base
        + scale_x_continuous(breaks=[0, 0.05, 0.1, 0.15, 0.2])
        + coord_cartesian(xlim=(0, 0.2), ylim=(0, None))
        + labs(
            title="Low underestimation risk: calorie trade-off",
        )
    )
    calibration = (
        ggplot(curve, aes("q", "raw_quantile_coverage", color="model", group="series"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#8993A4")
        + geom_path(size=0.9)
        + geom_point(data=curve[np.isclose(curve["q"], 0.95)], size=2.5)
        + scale_color_manual(values=PALETTE, name="Model")
        + scale_x_continuous(breaks=np.linspace(0, 1, 6).tolist(), limits=(0, 1))
        + scale_y_continuous(breaks=np.linspace(0, 1, 6).tolist(), limits=(0, 1))
        + labs(
            title="Gaussian calorie quantile calibration",
            subtitle=f"{population_label} · dashed line: nominal coverage",
            x="Model quantile q",
            y="Empirical P(Y ≤ Q(q))",
            caption="Uses raw Gaussian quantiles; dots mark q=0.95.\nThe 95th percentile is one-sided: μ + 1.645σ. Model quantiles do not guarantee the same empirical coverage.",
        )
        + style
        + coord_cartesian(xlim=(0, 1), ylim=(0, 1), expand=False)
    )
    return {
        "task_tradeoff": pareto,
        "task_tradeoff_low_risk": focus,
        "quantile_calibration": calibration,
    }, style


def save_figures(figures, output):
    for name, figure in figures.items():
        for extension in ("png", "pdf", "svg"):
            figure.save(
                output / f"{name}.{extension}",
                width=9,
                height=6,
                dpi=200,
                verbose=False,
            )


@app.command()
def build(
    spec: Path = Path("configs/task-plots.toml"),
    output: Path = Path("runs/babel/task-plots"),
):
    """Build reproducible plotnine figures from complete saved prediction sets."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "macs-plotnine")
    )
    cache = Path(tempfile.gettempdir()) / "macs-plotnine-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    from plotnine import (
        aes,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_shape_manual,
    )

    raw_spec = spec.read_bytes()
    config = tomllib.loads(raw_spec.decode())
    output.mkdir(parents=True, exist_ok=True)
    qs = np.unique(
        np.round(
            np.r_[
                np.linspace(0.01, 0.99, 197),
                0.5,
                0.75,
                0.9,
                0.95,
                0.975,
                0.995,
                0.999,
                0.9999,
            ],
            8,
        )
    )
    metadata: dict[str, Any] = {
        "spec_sha256": hashlib.sha256(raw_spec).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quantiles": qs.tolist(),
        "clip_zero": config.get("clip_zero", True),
        "versions": {
            name: version(name)
            for name in ("plotnine", "matplotlib", "pandas", "numpy", "scipy")
        },
        "task_metrics": {
            "p_under": "count(X < Y)/N",
            "mean_over_kcal": "sum((X-Y)[X>=Y])/count(X>=Y); undefined if denominator is zero",
        },
        "model_metric": "calorie-only full Gaussian validation NLL, physical kcal units, nats/example",
        "checkpoint_selection": "existing five-target standardized NLL; no reselection or retraining",
    }
    for key, section in (
        ("full_test", config),
        ("saved_200", config.get("comparison")),
    ):
        if section is None:
            continue
        population = load_population(PROJECT / section["population_manifest"])
        curve, point, all_points, sources = collect_task_data(
            section, population, qs, metadata["clip_zero"]
        )
        folder = output if key == "full_test" else output / key
        folder.mkdir(exist_ok=True)
        curve.to_csv(folder / "quantile_sweep.csv", index=False)
        point.to_csv(folder / "point_baselines.csv", index=False)
        all_points[all_points["pooled_nondominated"]].to_csv(
            folder / "pareto_frontier.csv", index=False
        )
        operating = curve[curve["q"].isin([0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999])]
        operating.to_csv(folder / "operating_points.csv", index=False)
        figures, style = make_plots(
            curve,
            point,
            len(population.ids),
            key != "full_test",
            clip_zero=metadata["clip_zero"],
        )
        save_figures(figures, folder)
        metadata[key] = {
            "n": len(population.ids),
            "population_manifest": str(population.path),
            "manifest_sha256": population.sha256,
            "excluded_missing_overhead_ids": list(population.missing),
            "sources": sources,
            "undefined_conditional_means": int(curve["mean_over_kcal"].isna().sum()),
        }
        if key == "full_test":
            nll, markers, nll_summary = validation_data(section["gaussian"])
            nll.to_csv(output / "validation_calorie_nll.csv", index=False)
            nll_summary.to_csv(
                output / "validation_checkpoint_summary.csv", index=False
            )
            nll_figure = (
                ggplot(nll, aes("epoch", "calorie_nll", color="model", group="model"))
                + geom_line(size=0.9)
                + geom_point(data=markers, mapping=aes(shape="marker"), size=3)
                + scale_color_manual(values=PALETTE, name="Model")
                + scale_shape_manual(
                    values={"Saved checkpoint": "o", "Minimum calorie NLL": "^"},
                    name="Selection",
                )
                + labs(
                    title="Validation calorie negative log-likelihood",
                    subtitle=f"{int(nll_summary.iloc[0]['validation_n'])} validation images · full Gaussian density in kcal units",
                    x="Epoch",
                    y="Calorie NLL (nats per example)",
                    caption="Saved checkpoints were selected by the mean NLL of five targets, not calorie-only NLL.\nTriangles show calorie-only minima for diagnosis; existing test checkpoints were not reselected.",
                )
                + style
            )
            save_figures({"validation_calorie_nll": nll_figure}, output)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "test_n": metadata["full_test"]["n"],
                "figures": 4,
                "saved_subset_figures": 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
