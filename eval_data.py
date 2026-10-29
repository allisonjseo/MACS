"""Dataset boundary: normalize local labels once, then evaluate any manifest.

Adapters should produce Manifest/Sample objects with portion totals in kcal/g.
This CSV adapter deliberately does no nutrient lookup, unit guessing, splitting,
or dataset-specific label aggregation; those choices belong to the protocol.
"""

import csv
import hashlib
import random
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, ConfigDict

from determinism import MAX_SEED
from run import write_json
from schemas import (
    FIELDS,
    ImageInput,
    ImageMediaType,
    Manifest,
    Nutrition,
    NutritionTarget,
    Sample,
)

MEDIA_TYPES: dict[str, ImageMediaType] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_media_type(sample: Sample) -> ImageMediaType:
    """Legacy PNG manifests need no migration; new adapters may declare MIME."""
    if sample.media_type is not None:
        return sample.media_type
    suffix = Path(sample.image).suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError(f"Unsupported image extension for {sample.id}: {suffix}")
    return MEDIA_TYPES[suffix]


def load_image(sample: Sample, root: Path) -> ImageInput:
    return ImageInput(
        data=(root / sample.image).read_bytes(), media_type=image_media_type(sample)
    )


class LabelRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image: str | None
    target: NutritionTarget
    group_id: str | None = None


def read_label_csv(path: Path) -> list[LabelRow]:
    """Read canonical labels, rejecting ambiguous columns and invalid numbers."""
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        allowed = {"id", "image", "group_id", *FIELDS}
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate CSV columns in {path}")
        if not {"id", "image"} <= set(columns) or not set(columns) & set(FIELDS):
            raise ValueError(f"{path} requires id, image, and at least one of {FIELDS}")
        if set(columns) - allowed:
            raise ValueError(f"Unknown CSV columns in {path}: {set(columns) - allowed}")
        for index, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed CSV row {index} in {path}")
            values = {key: value.strip() for key, value in raw.items()}
            if not values["id"]:
                raise ValueError(f"Empty ID at row {index} in {path}")
            try:
                target = NutritionTarget.model_validate(
                    {key: float(values[key]) for key in FIELDS if values.get(key)}
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid nutrition at row {index} in {path}: {error}"
                ) from error
            rows.append(
                LabelRow(
                    id=values["id"],
                    image=values["image"] or None,
                    group_id=values.get("group_id") or None,
                    target=target,
                )
            )
    if not rows:
        raise ValueError(f"No labeled rows in {path}")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return rows


def _sample(row: LabelRow, root: Path) -> Sample:
    if row.image is None:
        raise ValueError(f"Missing image path for {row.id}")
    image = (root / row.image).resolve()
    sample = Sample(
        id=row.id,
        image=str(image),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        target=row.target,
        group_id=row.group_id,
    )
    sample.media_type = image_media_type(sample)
    return sample


def prepare_csv(
    *,
    labels: Path,
    data_dir: Path,
    dataset: str,
    source: str,
    license: str,
    split: str,
    image_root: Path | None = None,
    training_labels: Path | None = None,
    training_image_root: Path | None = None,
    training_split: str | None = None,
    reference_n: int = 0,
    n: int | None = None,
    seed: int = 42,
) -> Manifest:
    """Import a fixed evaluation split, with optional independent training data."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError(
            "Manifest already exists. Reuse it, or choose a new --data-dir."
        )
    if not 0 <= seed <= MAX_SEED or reference_n < 0 or (n is not None and n < 1):
        raise ValueError("Invalid seed or sample count")
    if not all(value.strip() for value in (dataset, source, license, split)):
        raise ValueError("Dataset, source, license, and split must be nonempty")
    if training_labels is None and (
        reference_n or training_split or training_image_root
    ):
        raise ValueError("Training options require --training-labels")
    if training_labels is not None and (
        not training_split or not training_split.strip() or training_split == split
    ):
        raise ValueError("Supply a distinct --training-split with --training-labels")
    candidates = read_label_csv(labels)
    train = read_label_csv(training_labels) if training_labels else []
    if {row.id for row in candidates} & {row.id for row in train}:
        raise ValueError("Evaluation and training IDs overlap")
    if {row.group_id for row in candidates if row.group_id} & {
        row.group_id for row in train if row.group_id
    }:
        raise ValueError("Evaluation and training groups overlap")
    # Validate the entire supplied training table, including rows not retrieved.
    full_train = [Nutrition.model_validate(row.target.model_dump()) for row in train]
    means = (
        Nutrition(
            **{
                key: fmean(getattr(label, key) for label in full_train)
                for key in FIELDS
            }
        )
        if full_train
        else None
    )
    if n is not None and n > len(candidates):
        raise ValueError(f"Only {len(candidates)} evaluation rows available")
    if reference_n > len(train):
        raise ValueError(f"Only {len(train)} training references available")
    candidates.sort(key=lambda row: row.id)
    random.Random(seed).shuffle(candidates)
    train.sort(key=lambda row: row.id)
    random.Random(seed + 1).shuffle(train)
    samples = [_sample(row, image_root or labels.parent) for row in candidates[:n]]
    references = (
        [
            _sample(row, training_image_root or training_labels.parent)
            for row in train[:reference_n]
        ]
        if training_labels
        else []
    )
    if {s.image_sha256 for s in samples} & {s.image_sha256 for s in references}:
        raise ValueError("Evaluation and training references share image content")
    manifest = Manifest(
        dataset=dataset,
        source=source,
        license=license,
        split=split,
        seed=seed,
        n=len(samples),
        samples=samples,
        references=references,
        training_mean=means,
        training_label_count=len(train),
        training_split=training_split,
        preparation={
            "adapter": "csv-v1",
            "labels": str(labels.resolve()),
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "training_labels": str(training_labels.resolve())
            if training_labels
            else None,
            "training_labels_sha256": hashlib.sha256(
                training_labels.read_bytes()
            ).hexdigest()
            if training_labels
            else None,
            "candidate_count": len(candidates),
            "requested_n": n,
            "reference_n": reference_n,
            "image_root": str((image_root or labels.parent).resolve()),
            "training_image_root": str(
                (training_image_root or training_labels.parent).resolve()
            )
            if training_labels
            else None,
            "ordering": "sorted IDs shuffled with seed; references use seed+1",
            "units": "portion totals: calories_kcal in kcal; protein_g/carbs_g/fat_g in grams",
        },
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest
