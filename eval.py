#!/usr/bin/env python3
"""Small, backend-independent Nutrition5k image-to-nutrition evaluation.

Run with uv. See README.md for usage and backend contracts.
"""

import csv
import hashlib
import io
import random
import time
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Annotated
from urllib.error import HTTPError
from urllib.request import urlopen

import polars as pl
import typer

from predictors import (
    CodexPredictor,
    MeanPredictor,
    Predictor,
)
from run import EvaluationRun, RunConfig, write_json
from schemas import (
    FIELDS,
    CalorieAccuracy,
    EvaluationRecord,
    EvaluationSummary,
    ImageInput,
    Manifest,
    Metric,
    Nutrition,
    Sample,
)

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
SOURCE = "https://github.com/google-research-datasets/Nutrition5k"
app = typer.Typer(no_args_is_help=True, help=__doc__)


class Backend(StrEnum):
    codex = "codex"
    mean = "mean"


def download(relative, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE + relative, timeout=60) as response:
        content = response.read()
    if relative.endswith(".png") and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG received: " + relative)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)


def read_metadata(paths: list[Path]) -> dict[str, Nutrition]:
    labels = {}
    for path in paths:
        for row in csv.reader(io.StringIO(path.read_text())):
            if not row or not row[0].startswith("dish_"):
                continue
            # Nutrition5k: dish_id, calories, mass, fat, carb, protein, ...
            labels[row[0]] = Nutrition(
                calories_kcal=float(row[1]),
                protein_g=float(row[5]),
                carbs_g=float(row[4]),
                fat_g=float(row[3]),
            )
    return labels


@app.command()
def prepare(
    data_dir: Annotated[Path, typer.Option(help="Dataset cache directory")] = Path(
        "data/nutrition5k"
    ),
    n: Annotated[int, typer.Option(min=1)] = 20,
    seed: Annotated[int, typer.Option()] = 42,
):
    """Download a reproducible small test subset."""
    root = data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise ValueError(
            "Manifest already exists. Reuse it, or choose a new --data-dir."
        )
    metadata_paths = []
    for cafe in (1, 2):
        relative = f"metadata/dish_metadata_cafe{cafe}.csv"
        path = root / relative
        download(relative, path)
        metadata_paths.append(path)
    labels = read_metadata(metadata_paths)
    splits = {}
    for split in ("train", "test"):
        relative = f"dish_ids/splits/rgb_{split}_ids.txt"
        path = root / relative
        download(relative, path)
        splits[split] = set(path.read_text().split())
    if splits["train"] & splits["test"]:
        raise ValueError("Training and test splits overlap")
    train = [labels[key] for key in sorted(splits["train"]) if key in labels]
    if not train:
        raise ValueError("No training labels found")
    means = Nutrition(
        **pl.DataFrame(
            {key: [getattr(label, key) for label in train] for key in FIELDS}
        )
        .select(pl.all().mean())
        .row(0, named=True)
    )
    candidates = sorted(splits["test"] & labels.keys())
    random.Random(seed).shuffle(candidates)
    samples, missing = [], []
    for dish in candidates:
        relative = f"imagery/realsense_overhead/{dish}/rgb.png"
        image = root / "images" / (dish + ".png")
        try:
            download(relative, image)
        except HTTPError as error:
            if error.code != 404:
                raise
            missing.append(dish)
            continue
        samples.append(
            Sample(
                id=dish,
                image=str(image.relative_to(root)),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                target=labels[dish],
            )
        )
        print(f"Downloaded {len(samples)}/{n}: {dish}", flush=True)
        if len(samples) == n:
            break
    if len(samples) != n:
        raise ValueError(f"Only {len(samples)} test images available")
    manifest = Manifest(
        dataset="Nutrition5k",
        source=SOURCE,
        license="CC BY 4.0",
        split="rgb_test",
        seed=seed,
        n=n,
        missing_overhead_images=missing,
        training_mean=means,
        training_label_count=len(train),
        samples=samples,
    )
    write_json(manifest_path, manifest)
    print("Prepared " + str(manifest_path))


def score(rows: list[EvaluationRecord]) -> EvaluationSummary:
    good = [
        (row.prediction, row.target)
        for row in rows
        if row.status == "ok" and row.prediction is not None
    ]
    metrics = {key: Metric() for key in FIELDS}
    within, eligible = 0, 0
    if good:
        frame = pl.DataFrame(
            {
                **{
                    key: [getattr(prediction, key) for prediction, _ in good]
                    for key in FIELDS
                },
                **{
                    key + "_target": [getattr(target, key) for _, target in good]
                    for key in FIELDS
                },
            }
        )
        means = frame.select(
            [
                (pl.col(key) - pl.col(key + "_target")).abs().mean().alias(key)
                for key in FIELDS
            ]
        ).row(0, named=True)
        metrics = {key: Metric(mae=means[key]) for key in FIELDS}
        calories = frame.filter(pl.col("calories_kcal_target") > 0)
        eligible = calories.height
        within = calories.filter(
            (pl.col("calories_kcal") - pl.col("calories_kcal_target")).abs()
            <= 0.2 * pl.col("calories_kcal_target")
        ).height
    return EvaluationSummary(
        attempted=len(rows),
        succeeded=len(good),
        failed=len(rows) - len(good),
        metrics=metrics,
        calories_within_20pct=CalorieAccuracy(
            count=within,
            eligible_successes=eligible,
            percent=100 * within / eligible if eligible else None,
        ),
        mean_latency_seconds=fmean(row.latency_seconds for row in rows)
        if rows
        else None,
    )


@app.command()
def run(
    data_dir: Annotated[Path, typer.Option(help="Dataset cache directory")] = Path(
        "data/nutrition5k"
    ),
    predictor: Annotated[Backend, typer.Option()] = Backend.codex,
    model: Annotated[str | None, typer.Option(help="Codex model")] = None,
    timeout: Annotated[int, typer.Option(min=1)] = 300,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Run only the first N samples")
    ] = None,
    prompt_file: Annotated[
        Path | None, typer.Option(help="Override the Codex prompt")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="New directory for this run")
    ] = None,
):
    """Run any predictor and score its answers."""
    manifest_path = (data_dir / "manifest.json").resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(raw_manifest)
    samples = manifest.samples
    if limit:
        samples = samples[:limit]
    # Validate everything before launching any potentially paid inference.
    for sample in samples:
        image = manifest_path.parent / sample.image
        if hashlib.sha256(image.read_bytes()).hexdigest() != sample.image_sha256:
            raise ValueError("Image checksum mismatch: " + sample.id)
    backend: Predictor
    if predictor == "codex":
        backend = (
            CodexPredictor(model, timeout, prompt=prompt_file.read_text())
            if prompt_file
            else CodexPredictor(model, timeout)
        )
    else:
        backend = MeanPredictor(manifest.training_mean)
    recording = EvaluationRun(
        RunConfig.capture(
            predictor=backend.config,
            manifest=raw_manifest,
            sample_ids=[sample.id for sample in samples],
            timeout_seconds=timeout,
        ),
        summarize=score,
        output=output,
    )
    for index, sample in enumerate(samples, 1):
        row = EvaluationRecord(id=sample.id, target=sample.target)
        backend.last_metadata = None
        started = time.monotonic()
        try:
            # Backends receive only the photo, never the target.
            row.prediction = backend.predict(
                ImageInput(
                    data=(manifest_path.parent / sample.image).read_bytes(),
                    media_type="image/png",
                ),
            )
            row.status = "ok"
        except Exception as error:
            row.error = f"{type(error).__name__}: {error}"
        row.latency_seconds = time.monotonic() - started
        row.backend_metadata = backend.last_metadata
        recording.record(row)
        print(
            f"[{index}/{len(samples)}] {sample.id}: {row.status}",
            flush=True,
        )
    print(recording.summary.model_dump_json(indent=2))
    print("Results: " + str(recording.output.resolve()))
    raise typer.Exit(1 if recording.summary.failed else 0)


def main():
    try:
        app()
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
