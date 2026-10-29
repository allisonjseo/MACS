"""Portable, checksummed supervised train/validation data, never test labels."""

import csv
import hashlib
import io
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import Dataset
from tqdm import tqdm

from schemas import Nutrition
from vision import TARGETS


class Quantities(Nutrition):
    mass_g: float = Field(ge=0, allow_inf_nan=False)


class TrainingSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    image: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: Quantities


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["Nutrition5k"] = "Nutrition5k"
    split: Literal["rgb_train"] = "rgb_train"
    seed: int = Field(ge=0, le=2**31 - 1)
    validation_fraction: float = Field(gt=0, lt=1)
    split_method: Literal["seeded_dish_id"] = "seeded_dish_id"
    train: list[TrainingSample] = Field(min_length=1)
    validation: list[TrainingSample] = Field(min_length=1)
    missing_overhead_images: list[str]
    missing_labels: list[str]
    official_test_ids: list[str]

    @model_validator(mode="after")
    def disjoint(self):
        train = [sample.id for sample in self.train]
        val = [sample.id for sample in self.validation]
        all_ids = train + val
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Training/validation IDs must be unique and disjoint")
        if set(all_ids) & set(self.official_test_ids):
            raise ValueError("Training/validation must exclude official test IDs")
        return self


def prepare_training(
    source: Path,
    output: Path,
    seed: int,
    validation_fraction: float,
    download_workers: int = 1,
):
    # Reuse the established official download URL and atomic download helper.
    from eval import download
    from run import write_json

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if output.exists():
        raise ValueError("Training manifest already exists; choose another output")
    source = source.resolve()
    output = output.resolve()
    labels: dict[str, Quantities] = {}
    for cafe in (1, 2):
        relative = f"metadata/dish_metadata_cafe{cafe}.csv"
        path = source / relative
        download(relative, path)
        for row in csv.reader(io.StringIO(path.read_text())):
            if row and row[0].startswith("dish_"):
                labels[row[0]] = Quantities(
                    calories_kcal=float(row[1]),
                    mass_g=float(row[2]),
                    fat_g=float(row[3]),
                    carbs_g=float(row[4]),
                    protein_g=float(row[5]),
                )
    splits = {}
    for split in ("train", "test"):
        relative = f"dish_ids/splits/rgb_{split}_ids.txt"
        path = source / relative
        download(relative, path)
        splits[split] = set(path.read_text().split())
    if splits["train"] & splits["test"]:
        raise ValueError("Official training/test IDs overlap")
    if not 1 <= download_workers <= 64:
        raise ValueError("download_workers must be between 1 and 64")

    def prepare_photo(dish):
        relative = f"imagery/realsense_overhead/{dish}/rgb.png"
        path = source / relative
        # Also reuse photos downloaded by the existing evaluation prepare command.
        for cached in (
            source / "references" / f"{dish}.png",
            source / "images" / f"{dish}.png",
        ):
            if not path.exists() and cached.exists():
                path = cached
        try:
            if not path.exists():
                download(relative, path)
        except HTTPError as error:
            if error.code != 404:
                raise
            return dish, None
        return dish, TrainingSample(
            id=dish,
            image=os.path.relpath(path, output.parent),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            target=labels[dish],
        )

    samples, missing = [], []
    candidates = sorted(splits["train"] & labels.keys())
    # Preserve sorted order regardless of download completion order.
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        for dish, prepared in tqdm(
            pool.map(prepare_photo, candidates),
            total=len(candidates),
            desc="Training photos",
        ):
            if prepared is None:
                missing.append(dish)
            else:
                samples.append(prepared)
    if len(samples) < 2:
        raise ValueError("Need at least two available training photos")
    random.Random(seed).shuffle(samples)
    count = min(len(samples) - 1, max(1, math.ceil(len(samples) * validation_fraction)))
    manifest = TrainingManifest(
        seed=seed,
        validation_fraction=validation_fraction,
        train=samples[count:],
        validation=samples[:count],
        missing_overhead_images=missing,
        missing_labels=sorted(splits["train"] - labels.keys()),
        official_test_ids=sorted(splits["test"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    return manifest


class NutritionDataset(Dataset):
    def __init__(self, samples: list[TrainingSample], root: Path, transform: Any):
        self.samples, self.root, self.transform = samples, root, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        with Image.open(self.root / sample.image) as image:
            tensor = self.transform(image.convert("RGB"))
        target = torch.tensor([getattr(sample.target, key) for key in TARGETS])
        return tensor, target


def verify_images(manifest: TrainingManifest, root: Path):
    for sample in tqdm([*manifest.train, *manifest.validation], desc="Checking images"):
        with (root / sample.image).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != sample.image_sha256:
            raise ValueError("Image checksum mismatch: " + sample.id)
