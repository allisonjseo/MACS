"""Exclude test dishes whose exact ingredient-ID set occurs in official training.

Writes a new evaluator-compatible manifest and audit files; source data stays intact.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from schemas import Manifest

ROOT = Path(__file__).resolve().parent


def read_ingredient_sets(paths: list[Path]) -> dict[str, frozenset[str]]:
    signatures = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if not row or not row[0].startswith("dish_"):
                    continue
                if len(row) < 13 or (len(row) - 6) % 7:
                    raise ValueError(
                        f"Invalid ingredient metadata for {row[0]} in {path}"
                    )
                ids = row[6::7]
                if any(not ingredient.startswith("ingr_") for ingredient in ids):
                    raise ValueError(f"Invalid ingredient ID for {row[0]} in {path}")
                signature = frozenset(ids)
                if row[0] in signatures and signatures[row[0]] != signature:
                    raise ValueError(f"Conflicting ingredient sets for {row[0]}")
                signatures[row[0]] = signature
    return signatures


def exclude_exact_matches(
    manifest: Manifest,
    signatures: dict[str, frozenset[str]],
    train_ids: set[str],
    test_ids: set[str],
) -> tuple[Manifest, list[dict]]:
    if manifest.dataset != "Nutrition5k" or manifest.split != "rgb_test":
        raise ValueError("Input must be an official Nutrition5k rgb_test manifest")
    if train_ids & test_ids:
        raise ValueError("Official training and test IDs overlap")
    if not train_ids:
        raise ValueError("Official training split is empty")
    sample_ids = {sample.id for sample in manifest.samples}
    if sample_ids - test_ids:
        raise ValueError("Input samples contain IDs outside the official test split")
    if {sample.id for sample in manifest.references} - train_ids:
        raise ValueError(
            "Input references contain IDs outside the official train split"
        )
    required = train_ids | sample_ids
    missing = sorted(dish for dish in required if not signatures.get(dish))
    if missing:
        raise ValueError(
            f"Missing ingredient metadata for {len(missing)} dishes: {', '.join(missing[:5])}"
        )
    recipes = defaultdict(list)
    for dish in sorted(train_ids):
        recipes[signatures[dish]].append(dish)
    kept, removed = [], []
    for sample in manifest.samples:
        matches = recipes.get(signatures[sample.id])
        if matches:
            removed.append(
                {
                    "id": sample.id,
                    "ingredient_ids": sorted(signatures[sample.id]),
                    "matching_train_ids": matches,
                }
            )
        else:
            kept.append(sample)
    if not kept:
        raise ValueError("No test samples remain after exact ingredient-set filtering")
    filtered = Manifest.model_validate(
        {
            **manifest.model_dump(),
            "split": "rgb_test_exact_recipe_disjoint",
            "n": len(kept),
            "samples": [sample.model_dump() for sample in kept],
        }
    )
    return filtered, removed


def rebase_images(manifest: Manifest, input_dir: Path, output_dir: Path) -> Manifest:
    """Preserve image identity when moving a manifest containing relative paths."""

    def rebase(sample):
        if Path(sample.image).is_absolute():
            return sample
        original = (input_dir / sample.image).resolve()
        return sample.model_copy(
            update={"image": os.path.relpath(original, output_dir)}
        )

    return manifest.model_copy(
        update={
            "samples": [rebase(sample) for sample in manifest.samples],
            "references": [rebase(sample) for sample in manifest.references],
        }
    )


def filter_test_manifest(
    manifest_path: Path, source_dir: Path, output_dir: Path
) -> dict:
    manifest_path = manifest_path.resolve()
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    raw = manifest_path.read_bytes()
    original = Manifest.model_validate_json(raw)
    metadata = [source_dir / "metadata" / f"dish_metadata_cafe{i}.csv" for i in (1, 2)]
    split_paths = {
        split: source_dir / "dish_ids/splits" / f"rgb_{split}_ids.txt"
        for split in ("train", "test")
    }
    signatures = read_ingredient_sets(metadata)
    splits = {
        split: set(path.read_text().split()) for split, path in split_paths.items()
    }
    filtered, removed = exclude_exact_matches(
        original, signatures, splits["train"], splits["test"]
    )
    filtered = rebase_images(filtered, manifest_path.parent, output_dir)
    report = {
        "rule": "Exclude exact ingredient-ID set matches against the entire official RGB training split.",
        "definition": "Order, repeated IDs, ingredient weights, and ingredient names do not affect set equality. All ingredient IDs, including seasonings, are included.",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*metadata, *split_paths.values()]
        },
        "comparison_train_dishes": len(splits["train"]),
        "comparison_train_ingredient_sets": len(
            {signatures[dish] for dish in splits["train"]}
        ),
        "input_test_dishes": original.n,
        "retained_test_dishes": filtered.n,
        "removed_test_dishes": len(removed),
        "retained_test_ids": [sample.id for sample in filtered.samples],
        "removed": removed,
    }
    # Complete validation before creating output. Refuse to replace existing results.
    files = {
        "manifest.json": filtered.model_dump_json(indent=2) + "\n",
        "filter_report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "test_ids.txt": "".join(sample.id + "\n" for sample in filtered.samples),
        "removed_test_ids.txt": "".join(row["id"] + "\n" for row in removed),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output_dir / name).write_text(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/nutrition5k-sift-full/manifest.json",
        help="Existing evaluation manifest; its sample order and training references are preserved",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/nutrition5k",
        help="Raw Nutrition5k directory with metadata and official RGB split lists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for filtered data; must not exist",
    )
    args = parser.parse_args()
    try:
        report = filter_test_manifest(args.manifest, args.source_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Kept {report['retained_test_dishes']} / {report['input_test_dishes']} test dishes; "
        f"removed {report['removed_test_dishes']} exact ingredient-set matches "
        f"against {report['comparison_train_dishes']} official training dishes."
    )
    print("Filtered manifest:", args.output_dir.resolve() / "manifest.json")
    print("Audit report:", args.output_dir.resolve() / "filter_report.json")


if __name__ == "__main__":
    main()
