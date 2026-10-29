"""Offline checks for importing datasets and scoring partially observed targets."""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

import eval as evaluation
from eval_data import load_image, prepare_csv, read_label_csv
from predictors import MeanPredictor
from schemas import (
    EvaluationRecord,
    Manifest,
    Nutrition,
    NutritionInterval,
    NutritionTarget,
    Prediction,
)


def photo(path: Path, value: int = 80):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, np.full((12, 12, 3), value, np.uint8))
    assert ok
    path.write_bytes(encoded.tobytes())


def labels(path: Path, rows: list[dict]):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def import_data(tmp_path: Path, **kwargs):
    return prepare_csv(
        labels=tmp_path / "test.csv",
        data_dir=tmp_path / "cache",
        dataset="external-v1",
        source="https://example.org/data",
        license="CC BY 4.0",
        split="held-out",
        **kwargs,
    )


def test_calorie_only_import_needs_no_train_data_and_is_immutable(tmp_path):
    photo(tmp_path / "meal.JPG")
    labels(
        tmp_path / "test.csv",
        [{"id": "meal", "image": "meal.JPG", "calories_kcal": 250}],
    )
    manifest = import_data(tmp_path)
    assert manifest.training_mean is None
    assert manifest.training_label_count == 0
    assert manifest.samples[0].target.protein_g is None
    assert manifest.samples[0].media_type == "image/jpeg"
    assert (
        manifest.samples[0].image_sha256
        == hashlib.sha256((tmp_path / "meal.JPG").read_bytes()).hexdigest()
    )
    assert (
        manifest.preparation["labels_sha256"]
        == hashlib.sha256((tmp_path / "test.csv").read_bytes()).hexdigest()
    )
    assert load_image(manifest.samples[0], tmp_path).media_type == "image/jpeg"
    assert (
        Manifest.model_validate_json((tmp_path / "cache/manifest.json").read_bytes())
        == manifest
    )
    with pytest.raises(ValueError, match="already exists"):
        import_data(tmp_path)


def test_subset_and_references_are_reproducible_and_mean_is_training_only(tmp_path):
    for name, value in [
        ("a.png", 50),
        ("b.jpg", 80),
        ("c.png", 100),
        ("train.png", 200),
    ]:
        photo(tmp_path / name, value)
    rows = [
        {"id": name, "image": image, "calories_kcal": 9000}
        for name, image in [("a", "a.png"), ("b", "b.jpg"), ("c", "c.png")]
    ]
    labels(tmp_path / "test.csv", rows)
    training = labels(
        tmp_path / "train.csv",
        [
            {
                "id": "train",
                "image": "train.png",
                "calories_kcal": 100,
                "protein_g": 10,
                "carbs_g": 20,
                "fat_g": 5,
            }
        ],
    )
    first = import_data(
        tmp_path,
        training_labels=training,
        training_split="train",
        reference_n=1,
        n=2,
        seed=7,
    )
    labels(tmp_path / "test.csv", list(reversed(rows)))
    second = prepare_csv(
        labels=tmp_path / "test.csv",
        data_dir=tmp_path / "second",
        dataset="other",
        source="url",
        license="license",
        split="test",
        n=2,
        seed=7,
    )
    assert [s.id for s in first.samples] == [s.id for s in second.samples]
    assert first.training_mean.calories_kcal == 100
    assert first.references[0].id == "train"
    assert first.training_label_count == 1
    assert first.preparation["candidate_count"] == 3
    assert first.training_split == "train"


@pytest.mark.parametrize(
    "field,value",
    [
        ("calories_kcal", "nan"),
        ("fat_g", "-1"),
        ("protein_g", "inf"),
        ("carbs_g", "unknown"),
    ],
)
def test_csv_rejects_invalid_labels(tmp_path, field, value):
    path = labels(tmp_path / "test.csv", [{"id": "a", "image": "a.jpg", field: value}])
    with pytest.raises(ValueError, match="Invalid nutrition"):
        read_label_csv(path)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("id,image,calories_kcal\na,a.jpg,\n", "At least one"),
        ("id,image,calories_kcal\na,a.jpg,2\na,b.jpg,3\n", "Duplicate sample"),
        ("id,image,calories_kcal,calories_kcal\na,a.jpg,2,3\n", "Duplicate CSV"),
        ("id,image,calories_kcal\na,a.jpg,2,3\n", "Malformed"),
        ("id,image,calories_kcal\na,a.jpg\n", "Malformed"),
        ("id,image,calories_kcal,per100g\na,a.jpg,2,true\n", "Unknown CSV"),
    ],
)
def test_csv_rejects_ambiguous_tables(tmp_path, contents, match):
    path = tmp_path / "test.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=match):
        read_label_csv(path)


@pytest.mark.parametrize("overlap", ["ids", "groups", "content"])
def test_import_rejects_training_leakage(tmp_path, overlap):
    photo(tmp_path / "test.jpg", 80)
    photo(tmp_path / "train.jpg", 80 if overlap == "content" else 160)
    labels(
        tmp_path / "test.csv",
        [
            {
                "id": "test",
                "image": "test.jpg",
                "group_id": "person1",
                "calories_kcal": 200,
            }
        ],
    )
    training = labels(
        tmp_path / "train.csv",
        [
            {
                "id": "test" if overlap == "ids" else "train",
                "image": "train.jpg",
                "group_id": "person1" if overlap == "groups" else "person2",
                "calories_kcal": 100,
                "protein_g": 10,
                "carbs_g": 20,
                "fat_g": 5,
            }
        ],
    )
    with pytest.raises(ValueError, match="overlap|share image"):
        import_data(
            tmp_path, training_labels=training, training_split="train", reference_n=1
        )
    assert not (tmp_path / "cache/manifest.json").exists()


def test_baseline_statistics_can_use_training_labels_without_images(tmp_path):
    photo(tmp_path / "test.jpg")
    labels(
        tmp_path / "test.csv",
        [{"id": "test", "image": "test.jpg", "calories_kcal": 200}],
    )
    training = labels(
        tmp_path / "train.csv",
        [
            {
                "id": "train",
                "image": "",
                "calories_kcal": 100,
                "protein_g": 10,
                "carbs_g": 20,
                "fat_g": 5,
            }
        ],
    )
    manifest = import_data(tmp_path, training_labels=training, training_split="train")
    assert manifest.training_mean.calories_kcal == 100
    assert not manifest.references
    with pytest.raises(ValueError, match="Missing image"):
        prepare_csv(
            labels=tmp_path / "test.csv",
            data_dir=tmp_path / "bad",
            dataset="d",
            source="url",
            license="lic",
            split="test",
            training_labels=training,
            training_split="train",
            reference_n=1,
        )


def test_scoring_uses_per_field_observations_and_uncertainty_denominators():
    mean = Nutrition(calories_kcal=120, protein_g=10, carbs_g=20, fat_g=5)
    sigma = Nutrition(calories_kcal=20, protein_g=2, carbs_g=3, fat_g=1)
    rows = [
        EvaluationRecord(
            id="calorie-only",
            target=NutritionTarget(calories_kcal=100),
            status="ok",
            prediction=Prediction(mean=mean, sigma=sigma),
        ),
        EvaluationRecord(
            id="protein-only",
            target=NutritionTarget(protein_g=14),
            status="ok",
            prediction=Prediction(mean=mean, sigma=sigma),
        ),
        EvaluationRecord(
            id="zero",
            target=NutritionTarget(calories_kcal=0),
            status="ok",
            prediction=Prediction(mean=mean),
            prediction_interval=NutritionInterval(lower=mean, upper=mean),
        ),
        EvaluationRecord(id="failed", target=NutritionTarget(fat_g=20), error="failed"),
    ]
    summary = evaluation.score(rows)
    assert (summary.succeeded, summary.failed) == (3, 1)
    assert summary.metrics["calories_kcal"].count == 2
    assert summary.metrics["calories_kcal"].mae == 70
    assert summary.metrics["calories_kcal"].interval_count == 2
    assert summary.metrics["calories_kcal"].interval_coverage_pct == 50
    assert summary.metrics["calories_kcal"].mean_interval_width == 20
    assert summary.metrics["protein_g"].count == 1
    assert summary.metrics["protein_g"].mae == 4
    assert summary.metrics["protein_g"].interval_count == 1
    assert summary.metrics["protein_g"].interval_coverage_pct == 0
    assert summary.metrics["fat_g"].count == 0
    assert summary.metrics["fat_g"].mae is None
    assert summary.metrics["fat_g"].interval_count == 0
    assert summary.metrics["carbs_g"].mae is None
    assert summary.calories_within_20pct.eligible_successes == 1
    assert summary.calories_within_20pct.percent == 100


def test_csv_cli_and_generic_checkpoint_evaluation(tmp_path, monkeypatch):
    photo(tmp_path / "meal.jpg")
    table = labels(
        tmp_path / "test.csv",
        [{"id": "meal", "image": "meal.jpg", "calories_kcal": 200}],
    )
    cache = tmp_path / "cache"
    runner = CliRunner()
    imported = runner.invoke(
        evaluation.app,
        [
            "import-csv",
            "--labels",
            str(table),
            "--data-dir",
            str(cache),
            "--dataset",
            "external-v1",
            "--source",
            "https://example.org",
            "--license",
            "CC BY 4.0",
            "--split",
            "test",
        ],
    )
    assert imported.exit_code == 0, imported.output
    mean = Nutrition(calories_kcal=100, protein_g=10, carbs_g=20, fat_g=5)
    calls = []

    def predict(self, image):
        calls.append(image)
        assert image.media_type == "image/jpeg"
        assert image.data == (tmp_path / "meal.jpg").read_bytes()
        return Prediction(mean=mean)

    # Exercise the actual shared loop without a checkpoint or external inference.
    monkeypatch.setattr(
        "vision.VisionPredictor", lambda *args, **kwargs: MeanPredictor(mean)
    )
    monkeypatch.setattr(MeanPredictor, "predict", predict)
    output = tmp_path / "results"
    result = runner.invoke(
        evaluation.app,
        [
            "run",
            "--data-dir",
            str(cache),
            "--predictor",
            "vision",
            "--checkpoint",
            "mock.pt",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (output / "manifest.json").read_bytes() == (
        cache / "manifest.json"
    ).read_bytes()
    config = json.loads((output / "config.json").read_text())
    assert (config["dataset"], config["split"]) == ("external-v1", "test")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metrics"]["calories_kcal"]["mae"] == 100
    assert summary["metrics"]["protein_g"]["mae"] is None
    assert summary["metrics"]["protein_g"]["count"] == 0
    for backend in ("mean", "classical", "sift"):
        failed = runner.invoke(
            evaluation.app, ["run", "--data-dir", str(cache), "--predictor", backend]
        )
        assert failed.exit_code != 0
        assert "training-only statistics" in str(failed.exception)
    assert len(calls) == 1


def test_checksums_fail_before_backend_initialization(tmp_path, monkeypatch):
    photo(tmp_path / "meal.jpg")
    labels(
        tmp_path / "test.csv",
        [{"id": "meal", "image": "meal.jpg", "calories_kcal": 200}],
    )
    import_data(tmp_path)
    (tmp_path / "meal.jpg").write_bytes(b"modified")

    def should_not_initialize(*args, **kwargs):
        pytest.fail("Backend initialized before validating the dataset")

    monkeypatch.setattr(evaluation, "CodexPredictor", should_not_initialize)
    result = CliRunner().invoke(
        evaluation.app, ["run", "--data-dir", str(tmp_path / "cache")]
    )
    assert result.exit_code != 0
    assert "checksum mismatch" in str(result.exception)


@pytest.mark.parametrize(
    "kwargs", [{"reference_n": 1}, {"training_split": "train"}, {"n": 10}, {"seed": -1}]
)
def test_import_rejects_invalid_selection(tmp_path, kwargs):
    photo(tmp_path / "meal.jpg")
    labels(
        tmp_path / "test.csv",
        [{"id": "meal", "image": "meal.jpg", "calories_kcal": 200}],
    )
    with pytest.raises(ValueError):
        import_data(tmp_path, **kwargs)
