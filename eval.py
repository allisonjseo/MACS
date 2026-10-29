#!/usr/bin/env python3
"""Dataset- and backend-independent image-to-nutrition evaluation.

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

import typer
from openai_codex.generated.v2_all import ReasoningEffort
from tqdm import tqdm

from classical import ClassicalPredictor
from determinism import MAX_SEED, configure_determinism
from eval_data import image_media_type, load_image, prepare_csv
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
    vision = "vision"


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
        training_split="rgb_train",
        samples=samples,
        references=references,
        missing_reference_images=missing_references,
    )
    write_json(manifest_path, manifest)
    print("Prepared " + str(manifest_path))


@app.command("import-csv")
def import_csv(
    labels: Annotated[Path, typer.Option(help="Normalized CSV; see docs/datasets.md")],
    data_dir: Annotated[Path, typer.Option(help="New manifest directory")],
    dataset: Annotated[str, typer.Option(help="Dataset name and version")],
    source: Annotated[str, typer.Option(help="Dataset source URL or DOI")],
    license: Annotated[str, typer.Option(help="License for the source data")],
    split: Annotated[str, typer.Option(help="Evaluation split or protocol name")],
    image_root: Annotated[
        Path | None,
        typer.Option(help="Relative image paths resolve here (default: CSV parent)"),
    ] = None,
    training_labels: Annotated[
        Path | None, typer.Option(help="Separate training CSV for baselines/retrieval")
    ] = None,
    training_image_root: Annotated[
        Path | None,
        typer.Option(help="Training image root (default: training CSV parent)"),
    ] = None,
    training_split: Annotated[
        str | None,
        typer.Option(help="Required with training-labels; recorded as provenance"),
    ] = None,
    reference_n: Annotated[int, typer.Option(min=0)] = 0,
    n: Annotated[
        int | None, typer.Option(min=1, help="Seeded subset; default: all rows")
    ] = None,
    seed: Annotated[int, typer.Option(min=0, max=MAX_SEED)] = 42,
):
    """Import any local dataset into the shared evaluation manifest."""
    manifest = prepare_csv(
        labels=labels,
        data_dir=data_dir,
        dataset=dataset,
        source=source,
        license=license,
        split=split,
        image_root=image_root,
        training_labels=training_labels,
        training_image_root=training_image_root,
        training_split=training_split,
        reference_n=reference_n,
        n=n,
        seed=seed,
    )
    print(f"Prepared {data_dir.resolve() / 'manifest.json'} ({manifest.n} samples)")


def score(rows: list[EvaluationRecord]) -> EvaluationSummary:
    """Score only observed labels; prediction and uncertainty contracts stay complete."""
    good = [row for row in rows if row.status == "ok" and row.prediction is not None]
    metrics = {}
    for key in FIELDS:
        labeled = [row for row in good if getattr(row.target, key) is not None]
        errors, widths, covered = [], [], 0
        for row in labeled:
            assert row.prediction is not None
            target = getattr(row.target, key)
            mean = getattr(row.prediction.mean, key)
            errors.append(abs(mean - target))
            if row.prediction_interval is not None:
                lower = getattr(row.prediction_interval.lower, key)
                upper = getattr(row.prediction_interval.upper, key)
                widths.append(upper - lower)
                covered += lower <= target <= upper
            elif row.prediction.sigma is not None:
                half_width = getattr(row.prediction.sigma, key)
                widths.append(2 * half_width)
                covered += abs(mean - target) <= half_width
            else:
                continue
        metrics[key] = Metric(
            count=len(labeled),
            mae=fmean(errors) if errors else None,
            interval_count=len(widths),
            interval_coverage_pct=100 * covered / len(widths) if widths else None,
            mean_interval_width=fmean(widths) if widths else None,
        )
    calories = [
        row
        for row in good
        if row.target.calories_kcal is not None and row.target.calories_kcal > 0
    ]
    within = 0
    for row in calories:
        assert row.prediction is not None and row.target.calories_kcal is not None
        within += abs(row.prediction.mean.calories_kcal - row.target.calories_kcal) <= (
            0.2 * row.target.calories_kcal
        )
    return EvaluationSummary(
        attempted=len(rows),
        succeeded=len(good),
        failed=len(rows) - len(good),
        metrics=metrics,
        calories_within_20pct=CalorieAccuracy(
            count=within,
            eligible_successes=len(calories),
            percent=100 * within / len(calories) if calories else None,
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
    checkpoint: Annotated[
        Path | None, typer.Option(help="Trained vision checkpoint (.pt)")
    ] = None,
    device: Annotated[str, typer.Option(help="Vision inference device")] = "cpu",
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
        image_media_type(sample)
    if (
        predictor in (Backend.mean, Backend.classical, Backend.sift)
        and manifest.training_mean is None
    ):
        raise ValueError(
            f"{predictor} requires complete training-only statistics; "
            "supply --training-labels to import-csv. Evaluation labels are never used as a fallback."
        )
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
        assert manifest.training_mean is not None
        backend = ClassicalPredictor(manifest.training_mean)
    elif predictor == "mean":
        assert manifest.training_mean is not None
        backend = MeanPredictor(manifest.training_mean)
    elif predictor == "vision":
        from vision import VisionPredictor

        if checkpoint is None:
            raise ValueError("Vision predictor requires --checkpoint")
        backend = VisionPredictor(checkpoint, device=device)
    else:
        assert manifest.training_mean is not None
        if not manifest.references:
            raise ValueError(
                "SIFT predictor needs training references; use prepare or import-csv with --reference-n"
            )

        def load_references():
            for reference in tqdm(
                manifest.references, desc="Loading train references", unit="image"
            ):
                yield (
                    reference.id,
                    load_image(reference, manifest_path.parent),
                    Nutrition.model_validate(reference.target.model_dump()),
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
            backend.cache_features(load_image(sample, manifest_path.parent))
    recording = EvaluationRun(
        RunConfig.capture(
            predictor=backend.config,
            determinism=determinism,
            manifest=raw_manifest,
            sample_ids=[sample.id for sample in samples],
            timeout_seconds=timeout,
            dataset=manifest.dataset,
            split=manifest.split,
        ),
        summarize=score,
        output=output,
    )
    # Preserve the exact input, including labels and preparation provenance.
    (recording.output / "manifest.json").write_bytes(raw_manifest)
    print(
        "Run configuration: " + recording.config.model_dump_json(indent=2), flush=True
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
                load_image(sample, manifest_path.parent),
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
