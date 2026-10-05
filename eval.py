#!/usr/bin/env python3
"""Small, backend-independent Nutrition5k image-to-nutrition evaluation.

Run with uv. See README.md for usage and backend contracts.
"""

import csv
import hashlib
import io
import random
import time
from email.message import Message
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Annotated
from urllib.error import HTTPError
from urllib.request import urlopen

import polars as pl
import typer
from openai_codex.generated.v2_all import ReasoningEffort
from tqdm import tqdm

from classical import ClassicalPredictor
from determinism import MAX_SEED, configure_determinism
from predictors import (
    CodexPredictor,
    MeanPredictor,
    Predictor,
    SiftPredictor,
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
    sift = "sift"
    classical = "classical"


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
        "data/nutrition5k-200"
    ),
    source_dir: Annotated[
        Path | None,
        typer.Option(
            help="Existing raw Nutrition5k directory; manifests are written to data-dir"
        ),
    ] = None,
    n: Annotated[int, typer.Option(min=1)] = 200,
    reference_n: Annotated[
        int,
        typer.Option(
            min=0,
            help="Number of labeled RGB training images to cache for SIFT retrieval",
        ),
    ] = 0,
    seed: Annotated[int, typer.Option(min=0, max=MAX_SEED)] = 42,
    all_samples: Annotated[
        bool, typer.Option("--all", help="Prepare every available overhead test image")
    ] = False,
):
    """Download a seeded, shuffled test subset (default: 200 images)."""
    configure_determinism(seed)
    root = data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_root = source_dir.resolve() if source_dir else root
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise ValueError(
            "Manifest already exists. Reuse it, or choose a new --data-dir."
        )
    # The official Nutrition5k download uses the raw layout
    # ``imagery/realsense_overhead/<dish>/rgb.png``.  In that layout many
    # split IDs legitimately have no overhead image, so use existing files
    # directly instead of trying to materialize a separate cache.
    local_raw = (source_root / "dish_ids" / "splits").is_dir() and (
        source_root / "imagery" / "realsense_overhead"
    ).is_dir()
    metadata_paths = []
    for cafe in (1, 2):
        relative = f"metadata/dish_metadata_cafe{cafe}.csv"
        path = source_root / relative if local_raw else root / relative
        if local_raw:
            if not path.exists():
                raise ValueError("Missing local dataset file: " + str(path))
        else:
            download(relative, path)
        metadata_paths.append(path)
    labels = read_metadata(metadata_paths)
    splits = {}
    for split in ("train", "test"):
        relative = f"dish_ids/splits/rgb_{split}_ids.txt"
        path = source_root / relative if local_raw else root / relative
        if local_raw:
            if not path.exists():
                raise ValueError("Missing local dataset file: " + str(path))
        else:
            download(relative, path)
        splits[split] = set(path.read_text().split())
    if splits["train"] & splits["test"]:
        raise ValueError("Training and test splits overlap")
    train = [labels[key] for key in sorted(splits["train"]) if key in labels]
    if not train:
        raise ValueError("No training labels found")
    means = Nutrition(
        **{key: fmean(getattr(label, key) for label in train) for key in FIELDS}
    )
    candidates = sorted(splits["test"] & labels.keys())
    random.Random(seed).shuffle(candidates)
    samples, missing = [], []
    for dish in tqdm(candidates, desc="Preparing test images", unit="image"):
        relative = f"imagery/realsense_overhead/{dish}/rgb.png"
        image = (
            source_root / relative if local_raw else root / "images" / (dish + ".png")
        )
        try:
            if local_raw:
                if not image.exists():
                    raise HTTPError(
                        relative, 404, "missing local image", Message(), None
                    )
            else:
                download(relative, image)
        except HTTPError as error:
            if error.code != 404:
                raise
            missing.append(dish)
            continue
        samples.append(
            Sample(
                id=dish,
                image=str(image) if local_raw else str(image.relative_to(root)),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                target=labels[dish],
            )
        )
        requested = "all" if all_samples else str(n)
        print(f"Downloaded {len(samples)}/{requested}: {dish}", flush=True)
        if not all_samples and len(samples) == n:
            break
    if not samples or (not all_samples and len(samples) != n):
        raise ValueError(f"Only {len(samples)} test images available")
    reference_candidates = sorted(splits["train"] & labels.keys())
    random.Random(seed + 1).shuffle(reference_candidates)
    references, missing_references = [], []
    for dish in tqdm(
        reference_candidates, desc="Preparing train references", unit="image"
    ):
        if len(references) == reference_n:
            break
        relative = f"imagery/realsense_overhead/{dish}/rgb.png"
        image = (
            source_root / relative
            if local_raw
            else root / "references" / (dish + ".png")
        )
        try:
            if local_raw:
                if not image.exists():
                    raise HTTPError(
                        relative, 404, "missing local image", Message(), None
                    )
            else:
                download(relative, image)
        except HTTPError as error:
            if error.code != 404:
                raise
            missing_references.append(dish)
            continue
        references.append(
            Sample(
                id=dish,
                image=str(image) if local_raw else str(image.relative_to(root)),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                target=labels[dish],
            )
        )
    if len(references) != reference_n:
        raise ValueError(f"Only {len(references)} training reference images available")
    manifest = Manifest(
        dataset="Nutrition5k",
        source=SOURCE,
        license="CC BY 4.0",
        split="rgb_test",
        seed=seed,
        n=len(samples),
        missing_overhead_images=missing,
        training_mean=means,
        training_label_count=len(train),
        samples=samples,
        references=references,
        missing_reference_images=missing_references,
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
                    key: [getattr(prediction.mean, key) for prediction, _ in good]
                    for key in FIELDS
                },
                **{
                    key + "_target": [getattr(target, key) for _, target in good]
                    for key in FIELDS
                },
            }
        )
        # Vector arithmetic preserves row order; fmean avoids parallel reductions
        # whose floating-point addition order can vary with worker scheduling.
        errors = frame.select(
            [(pl.col(key) - pl.col(key + "_target")).abs().alias(key) for key in FIELDS]
        )
        metrics = {key: Metric(mae=fmean(errors[key].to_list())) for key in FIELDS}
        interval_rows = [
            (row.prediction_interval, row.target)
            for row in rows
            if row.status == "ok"
            and row.prediction is not None
            and row.prediction_interval is not None
        ]
        uncertain = [
            (row.prediction.mean, row.prediction.sigma, row.target)
            for row in rows
            if row.status == "ok"
            and row.prediction is not None
            and row.prediction.sigma is not None
            and row.prediction_interval is None
        ]
        for key in FIELDS:
            interval_count = len(uncertain) + len(interval_rows)
            if interval_count:
                covered = sum(
                    abs(getattr(mean, key) - getattr(target, key))
                    <= getattr(sigma, key)
                    for mean, sigma, target in uncertain
                ) + sum(
                    getattr(interval.lower, key)
                    <= getattr(target, key)
                    <= getattr(interval.upper, key)
                    for interval, target in interval_rows
                )
                metrics[key].interval_count = interval_count
                metrics[key].interval_coverage_pct = 100 * covered / interval_count
                metrics[key].mean_interval_width = fmean(
                    [2 * getattr(sigma, key) for _, sigma, _ in uncertain]
                    + [
                        getattr(interval.upper, key) - getattr(interval.lower, key)
                        for interval, _ in interval_rows
                    ]
                )
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
        "data/nutrition5k-200"
    ),
    predictor: Annotated[Backend, typer.Option()] = Backend.codex,
    model: Annotated[str, typer.Option(help="Codex model ID")] = "gpt-6-luna",
    reasoning_effort: Annotated[
        ReasoningEffort, typer.Option(help="Codex reasoning effort")
    ] = ReasoningEffort.low,
    timeout: Annotated[int, typer.Option(min=1)] = 300,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Run only the first N samples")
    ] = None,
    prompt_file: Annotated[
        Path | None, typer.Option(help="Override the Codex prompt")
    ] = None,
    sift_neighbors: Annotated[
        int, typer.Option(min=1, help="Number of SIFT retrieval neighbors")
    ] = 5,
    sift_ratio_threshold: Annotated[
        float,
        typer.Option(min=0.01, max=0.99, help="Lowe ratio threshold for SIFT matches"),
    ] = 0.75,
    sift_max_features: Annotated[
        int, typer.Option(min=1, help="Maximum SIFT descriptors per image")
    ] = 500,
    output: Annotated[
        Path | None, typer.Option(help="New directory for this run")
    ] = None,
):
    """Run any predictor and score its answers."""
    manifest_path = (data_dir / "manifest.json").resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(raw_manifest)
    determinism = configure_determinism(manifest.seed)
    samples = manifest.samples
    if limit:
        samples = samples[:limit]
    # Validate everything before launching any potentially paid inference.
    for sample in tqdm(
        [*samples, *manifest.references], desc="Checking image files", unit="image"
    ):
        image = manifest_path.parent / sample.image
        if hashlib.sha256(image.read_bytes()).hexdigest() != sample.image_sha256:
            raise ValueError("Image checksum mismatch: " + sample.id)
    backend: Predictor
    if predictor == "codex":
        backend = (
            CodexPredictor(
                model,
                timeout,
                prompt=prompt_file.read_text(),
                reasoning_effort=reasoning_effort,
            )
            if prompt_file
            else CodexPredictor(model, timeout, reasoning_effort=reasoning_effort)
        )
    elif predictor == "classical":
        backend = ClassicalPredictor(manifest.training_mean)
    elif predictor == "mean":
        backend = MeanPredictor(manifest.training_mean)
    else:
        if not manifest.references:
            raise ValueError(
                "SIFT predictor needs training references; run prepare with --reference-n"
            )

        def load_references():
            for reference in tqdm(
                manifest.references, desc="Loading train references", unit="image"
            ):
                yield (
                    reference.id,
                    ImageInput(
                        data=(manifest_path.parent / reference.image).read_bytes(),
                        media_type="image/png",
                    ),
                    reference.target,
                )

        backend = SiftPredictor(
            load_references(),
            manifest.training_mean,
            neighbors=sift_neighbors,
            ratio_threshold=sift_ratio_threshold,
            max_features=sift_max_features,
            feature_cache=manifest_path.parent / "sift-features",
        )
    if isinstance(backend, SiftPredictor):
        for sample in tqdm(samples, desc="Caching test SIFT features", unit="image"):
            backend.cache_features(
                ImageInput(
                    data=(manifest_path.parent / sample.image).read_bytes(),
                    media_type="image/png",
                )
            )
    recording = EvaluationRun(
        RunConfig.capture(
            predictor=backend.config,
            determinism=determinism,
            manifest=raw_manifest,
            sample_ids=[sample.id for sample in samples],
            timeout_seconds=timeout,
        ),
        summarize=score,
        output=output,
    )
    for index, sample in enumerate(
        tqdm(samples, desc=f"Evaluating {backend.config.backend}", unit="image"), 1
    ):
        row = EvaluationRecord(id=sample.id, target=sample.target)
        backend.last_metadata = None
        backend.last_interval = None
        started = time.monotonic()
        try:
            # Backends receive only the photo, never the target.
            row.prediction = backend.predict(
                ImageInput(
                    data=(manifest_path.parent / sample.image).read_bytes(),
                    media_type="image/png",
                ),
            )
            row.prediction_interval = backend.last_interval
            row.status = "ok"
        except Exception as error:
            row.error = f"{type(error).__name__}: {error}"
        row.latency_seconds = time.monotonic() - started
        row.backend_metadata = backend.last_metadata
        recording.record(row)
        print(
            f"[{index}/{len(samples)}] {sample.id}: {row.status} "
            f"({row.latency_seconds:.1f}s)",
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
