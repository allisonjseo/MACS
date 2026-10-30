"""Exact recipe filtering, auditability, and evaluator path compatibility."""

import csv
import hashlib
import json

import pytest

from filter_test_recipes import (
    exclude_exact_matches,
    filter_test_manifest,
    read_ingredient_sets,
    rebase_images,
)
from schemas import Manifest

TARGET = {"calories_kcal": 10.0, "protein_g": 1.0, "carbs_g": 2.0, "fat_g": 1.0}


def manifest(ids, reference_ids=()):
    def sample(dish):
        return {
            "id": dish,
            "image": f"images/{dish}.png",
            "image_sha256": "0" * 64,
            "target": TARGET,
        }

    return Manifest(
        dataset="Nutrition5k",
        source="test",
        license="CC BY 4.0",
        split="rgb_test",
        seed=42,
        n=len(ids),
        missing_overhead_images=["missing_photo"],
        training_mean=TARGET,
        training_label_count=1,
        samples=[sample(dish) for dish in ids],
        references=[sample(dish) for dish in reference_ids],
    )


def test_removes_only_exact_sets_and_preserves_order_and_training():
    original = manifest(["test_partial", "test_same", "test_new"], ["train"])
    signatures = {
        "train": frozenset(["a", "b"]),
        "test_partial": frozenset(["a"]),
        "test_same": frozenset(["b", "a"]),
        "test_new": frozenset(["a", "b", "c"]),
    }
    filtered, removed = exclude_exact_matches(
        original, signatures, {"train"}, set(signatures) - {"train"}
    )
    assert [sample.id for sample in filtered.samples] == ["test_partial", "test_new"]
    assert filtered.n == 2
    assert filtered.references == original.references
    assert filtered.training_mean == original.training_mean
    assert filtered.missing_overhead_images == original.missing_overhead_images
    assert original.n == 3
    assert removed == [
        {
            "id": "test_same",
            "ingredient_ids": ["a", "b"],
            "matching_train_ids": ["train"],
        }
    ]


def test_csv_signature_ignores_order_duplicates_names_and_weights(tmp_path):
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dish_train",
                10,
                20,
                1,
                2,
                3,
                "ingr_a",
                "rice",
                100,
                0,
                0,
                0,
                0,
                "ingr_b",
                "beans",
                50,
                0,
                0,
                0,
                0,
            ]
        )
        writer.writerow(
            [
                "dish_test",
                10,
                20,
                1,
                2,
                3,
                "ingr_b",
                "other name",
                200,
                0,
                0,
                0,
                0,
                "ingr_a",
                "rice",
                25,
                0,
                0,
                0,
                0,
                "ingr_a",
                "rice",
                75,
                0,
                0,
                0,
                0,
            ]
        )
    signatures = read_ingredient_sets([path])
    assert signatures["dish_train"] == signatures["dish_test"] == {"ingr_a", "ingr_b"}


@pytest.mark.parametrize("missing_id", ["train", "test"])
def test_missing_metadata_fails_closed(missing_id):
    signatures = {"train": frozenset(["a"]), "test": frozenset(["b"])}
    signatures.pop(missing_id)
    with pytest.raises(ValueError, match="Missing ingredient metadata"):
        exclude_exact_matches(manifest(["test"]), signatures, {"train"}, {"test"})


def test_empty_result_and_invalid_split_are_rejected():
    signatures = {"train": frozenset(["a"]), "test": frozenset(["a"])}
    with pytest.raises(ValueError, match="No test samples remain"):
        exclude_exact_matches(manifest(["test"]), signatures, {"train"}, {"test"})
    with pytest.raises(ValueError, match="outside the official test"):
        exclude_exact_matches(manifest(["wrong"]), signatures, {"train"}, {"test"})
    with pytest.raises(ValueError, match="overlap"):
        exclude_exact_matches(
            manifest(["test"]), signatures, {"train", "test"}, {"test"}
        )


def test_rebases_both_sample_and_reference_images(tmp_path):
    original = manifest(["test"], ["train"])
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    filtered = rebase_images(original, input_dir, output_dir)
    for old, new in zip(
        [*original.samples, *original.references],
        [*filtered.samples, *filtered.references],
        strict=True,
    ):
        assert (output_dir / new.image).resolve() == (input_dir / old.image).resolve()
        assert new.image_sha256 == old.image_sha256
        assert new.target == old.target
    absolute = original.model_copy(
        update={
            "samples": [
                original.samples[0].model_copy(update={"image": "/absolute/image.png"})
            ]
        }
    )
    assert (
        rebase_images(absolute, input_dir, output_dir).samples[0].image
        == "/absolute/image.png"
    )


def test_writes_auditable_manifest_without_changing_source(tmp_path):
    source = tmp_path / "raw"
    metadata = source / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "dish_metadata_cafe1.csv").write_text(
        "dish_train,10,20,1,2,3,ingr_a,rice,20,10,1,2,3\n"
        "dish_same,10,20,1,2,3,ingr_a,rice,80,10,1,2,3\n"
        "dish_new,10,20,1,2,3,ingr_b,beans,20,10,1,2,3\n"
    )
    (metadata / "dish_metadata_cafe2.csv").write_text("")
    splits = source / "dish_ids/splits"
    splits.mkdir(parents=True)
    (splits / "rgb_train_ids.txt").write_text("dish_train\n")
    (splits / "rgb_test_ids.txt").write_text("dish_same\ndish_new\n")
    input_path = tmp_path / "manifest.json"
    input_path.write_text(
        manifest(["dish_same", "dish_new"], ["dish_train"]).model_dump_json()
    )
    before = input_path.read_bytes()
    output = tmp_path / "filtered"
    report = filter_test_manifest(input_path, source, output)
    assert input_path.read_bytes() == before
    filtered = Manifest.model_validate_json((output / "manifest.json").read_bytes())
    assert filtered.n == 1 and filtered.samples[0].id == "dish_new"
    assert report["source_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["removed_test_dishes"] == 1
    assert report["removed"][0]["matching_train_ids"] == ["dish_train"]
    assert json.loads((output / "filter_report.json").read_text()) == report
    assert (output / "test_ids.txt").read_text() == "dish_new\n"
    assert (output / "removed_test_ids.txt").read_text() == "dish_same\n"
    with pytest.raises(ValueError, match="already exists"):
        filter_test_manifest(input_path, source, output)


def test_invalid_metadata_is_not_silently_treated_as_no_match(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("dish_train,10,20,1,2,3\n")
    with pytest.raises(ValueError, match="Invalid ingredient metadata"):
        read_ingredient_sets([path])
