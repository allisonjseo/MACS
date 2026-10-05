"""Offline checks for labels, scoring, and the external predictor boundary."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from openai_codex import ApprovalMode, LocalImageInput, Sandbox, TextInput, TurnResult
from openai_codex.generated.v2_all import ThreadItem, TurnStatus
from typer.testing import CliRunner

import eval as evaluation
from predictors import CodexPredictor, SiftPredictor
from schemas import (
    FIELDS,
    SCHEMA,
    EvaluationRecord,
    ImageInput,
    Manifest,
    Nutrition,
    Sample,
)


def nutrients(calories: float, protein: float = 10, carbs: float = 20, fat: float = 5):
    return Nutrition(
        calories_kcal=calories, protein_g=protein, carbs_g=carbs, fat_g=fat
    )


def feature_image(seed: int) -> ImageInput:
    """Return a deterministic, textured PNG that yields SIFT descriptors."""
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, size=(180, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return ImageInput(data=encoded.tobytes(), media_type="image/png")


def test_metadata_column_order_with_variable_ingredients(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    path.write_text("dish_123,400,250,12,45,30,1,ingr_1,soup,250,400,12,45,30\n")
    assert evaluation.read_metadata([path])["dish_123"] == nutrients(400, 30, 45, 12)


@pytest.mark.parametrize("bad", [True, "100", -1, float("nan"), float("inf"), None])
def test_validation_rejects_non_numeric_or_invalid_answers(bad):
    with pytest.raises(ValueError):
        Nutrition.model_validate({**nutrients(100).model_dump(), "calories_kcal": bad})


def test_validation_rejects_missing_and_extra_fields():
    with pytest.raises(ValueError):
        Nutrition.model_validate({"calories_kcal": 10})
    with pytest.raises(ValueError):
        Nutrition.model_validate(dict(nutrients(10).model_dump(), explanation="guess"))


def test_manifest_references_must_be_disjoint_from_test_samples():
    sample = {
        "id": "dish_1",
        "image": "food.png",
        "image_sha256": "0" * 64,
        "target": nutrients(100),
    }
    with pytest.raises(ValueError, match="must not overlap"):
        Manifest.model_validate(
            {
                "dataset": "Nutrition5k",
                "source": "test",
                "license": "CC BY 4.0",
                "split": "rgb_test",
                "seed": 42,
                "n": 1,
                "missing_overhead_images": [],
                "training_mean": nutrients(100),
                "training_label_count": 1,
                "samples": [sample],
                "references": [sample],
            }
        )


def test_sift_retrieves_identical_training_image_and_records_neighbor():
    target = nutrients(600, protein=30, carbs=60, fat=20)
    predictor = SiftPredictor(
        [
            ("dish_match", feature_image(1), target),
            ("dish_other", feature_image(2), nutrients(100)),
        ],
        fallback=nutrients(300),
        neighbors=1,
    )
    assert predictor.predict(feature_image(1)) == target
    assert predictor.last_metadata["neighbors"][0]["id"] == "dish_match"


def test_sift_uses_training_mean_when_photo_has_no_features():
    blank = np.zeros((100, 100), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    fallback = nutrients(321)
    predictor = SiftPredictor(
        [("dish_reference", feature_image(1), nutrients(500))], fallback
    )
    assert (
        predictor.predict(ImageInput(data=encoded.tobytes(), media_type="image/png"))
        == fallback
    )
    assert predictor.last_metadata["neighbors"] == []


def test_cli_runs_sift_with_training_reference(tmp_path: Path):
    test_image = feature_image(1).data
    reference_image = feature_image(1).data
    (tmp_path / "test.png").write_bytes(test_image)
    (tmp_path / "reference.png").write_bytes(reference_image)
    reference_target = nutrients(600, protein=30, carbs=60, fat=20)
    manifest = Manifest(
        dataset="Nutrition5k",
        source="test",
        license="CC BY 4.0",
        split="rgb_test",
        seed=42,
        n=1,
        missing_overhead_images=[],
        training_mean=nutrients(100),
        training_label_count=1,
        samples=[
            Sample(
                id="test_dish",
                image="test.png",
                image_sha256=hashlib.sha256(test_image).hexdigest(),
                target=nutrients(200),
            )
        ],
        references=[
            Sample(
                id="training_dish",
                image="reference.png",
                image_sha256=hashlib.sha256(reference_image).hexdigest(),
                target=reference_target,
            )
        ],
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    output = tmp_path / "sift-results"
    result = CliRunner().invoke(
        evaluation.app,
        [
            "run",
            "--data-dir",
            str(tmp_path),
            "--predictor",
            "sift",
            "--sift-neighbors",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    config = json.loads((output / "config.json").read_text())
    assert config["predictor"]["backend"] == "sift"
    records = json.loads((output / "predictions.json").read_text())
    assert records[0]["prediction"] == reference_target.model_dump(mode="json")
    assert records[0]["backend_metadata"]["neighbors"][0]["id"] == "training_dish"


def test_metrics_and_failures():
    rows = [
        EvaluationRecord(
            id="1",
            status="ok",
            prediction=nutrients(120),
            target=nutrients(100),
            latency_seconds=1,
        ),
        EvaluationRecord(
            id="2",
            status="ok",
            prediction=nutrients(100),
            target=nutrients(200),
            latency_seconds=3,
        ),
        EvaluationRecord(
            id="3", target=nutrients(200), latency_seconds=2, error="failed"
        ),
    ]
    summary = evaluation.score(rows)
    assert summary.metrics["calories_kcal"].mae == 60
    assert summary.calories_within_20pct.percent == 50
    assert summary.failed == 1
    assert summary.mean_latency_seconds == 2
    assert evaluation.score([rows[-1]]).metrics["calories_kcal"].mae is None


def test_zero_calories_excluded_from_percentage_only():
    summary = evaluation.score(
        [
            EvaluationRecord(
                id="1", status="ok", prediction=nutrients(50), target=nutrients(0)
            ),
        ]
    )
    assert summary.metrics["calories_kcal"].mae == 50
    assert summary.calories_within_20pct.percent is None


@pytest.fixture
def codex_sdk(monkeypatch: pytest.MonkeyPatch):
    result = TurnResult(
        id="turn-test",
        status=TurnStatus.completed,
        error=None,
        started_at=None,
        completed_at=None,
        duration_ms=None,
        final_response=nutrients(100).model_dump_json(),
        items=[
            ThreadItem.model_validate(
                {
                    "id": "message-test",
                    "type": "agentMessage",
                    "text": nutrients(100).model_dump_json(),
                }
            )
        ],
        usage=None,
    )
    thread = SimpleNamespace(run=AsyncMock(return_value=result))
    client = MagicMock()
    client.thread_start = AsyncMock(return_value=thread)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("predictors.AsyncCodex", factory)
    return SimpleNamespace(result=result, thread=thread, client=client, factory=factory)


def test_codex_rejects_tool_use(tmp_path: Path, codex_sdk):
    image = ImageInput(data=b"image", media_type="image/png")
    backend = CodexPredictor(None, 5)
    codex_sdk.result.items = [
        ThreadItem.model_validate(
            {
                "id": "tool-test",
                "type": "webSearch",
                "query": "Nutrition5k",
                "action": None,
            }
        )
    ]
    with pytest.raises(RuntimeError, match="used a tool"):
        backend.predict(image)


def test_codex_image_attachment_and_structured_result(
    tmp_path: Path,
    codex_sdk,
):
    async def fake_run(inputs, *, output_schema):
        assert inputs[0] == TextInput("estimate")
        assert isinstance(inputs[1], LocalImageInput)
        assert Path(inputs[1].path).name == "food.jpeg"
        assert Path(inputs[1].path).read_bytes() == b"image"
        assert output_schema == SCHEMA
        return codex_sdk.result

    image = ImageInput(data=b"image", media_type="image/jpeg")
    backend = CodexPredictor("test-model", 5, prompt="estimate")
    codex_sdk.thread.run.side_effect = fake_run
    assert backend.predict(image) == nutrients(100)
    assert backend.config.prompt == "estimate"
    options = codex_sdk.client.thread_start.await_args.kwargs
    assert options["model"] == "test-model"
    assert options["ephemeral"] is True
    assert options["sandbox"] == Sandbox.read_only
    assert options["approval_mode"] == ApprovalMode.deny_all
    assert options["config"]["features"]["shell_tool"] is False
    assert options["config"]["web_search"] == "disabled"
    assert backend.last_metadata is codex_sdk.result
    assert backend.last_metadata.items[0].root.type == "agentMessage"
    codex_sdk.client.__aexit__.assert_awaited_once()


def test_codex_timeout_closes_sdk(tmp_path: Path, codex_sdk):
    async def slow_run(*args, **kwargs):
        await asyncio.sleep(10)

    image = ImageInput(data=b"image", media_type="image/png")
    codex_sdk.thread.run.side_effect = slow_run
    with pytest.raises(TimeoutError):
        CodexPredictor("test-model", 1).predict(image)
    codex_sdk.client.__aexit__.assert_awaited_once()


def test_codex_interrupted_turn_is_a_failure(tmp_path: Path, codex_sdk):
    image = ImageInput(data=b"image", media_type="image/png")
    codex_sdk.result.status = TurnStatus.interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        CodexPredictor("test-model", 5).predict(image)


def test_generated_schema_and_manifest_consistency():
    assert set(SCHEMA["required"]) == set(FIELDS)
    assert SCHEMA["additionalProperties"] is False
    manifest = {
        "dataset": "Nutrition5k",
        "source": "test",
        "license": "CC BY 4.0",
        "split": "rgb_test",
        "seed": 42,
        "n": 1,
        "missing_overhead_images": [],
        "training_mean": nutrients(100),
        "training_label_count": 10,
        "samples": [
            {
                "id": "dish_1",
                "image": "food.png",
                "image_sha256": "0" * 64,
                "target": nutrients(200),
            }
        ],
    }
    Manifest.model_validate(manifest)
    manifest["n"] += 1
    with pytest.raises(ValueError):
        Manifest.model_validate(manifest)


def test_prepare_keeps_test_labels_out_of_training_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    files = {
        "metadata/dish_metadata_cafe1.csv": b"dish_1,100,50,5,20,10,0\ndish_2,400,200,12,45,30,0\n",
        "metadata/dish_metadata_cafe2.csv": b"",
        "dish_ids/splits/rgb_train_ids.txt": b"dish_1\n",
        "dish_ids/splits/rgb_test_ids.txt": b"dish_2\n",
        "imagery/realsense_overhead/dish_1/rgb.png": b"\x89PNG\r\n\x1a\nimage",
        "imagery/realsense_overhead/dish_2/rgb.png": b"\x89PNG\r\n\x1a\nimage",
    }

    def fake_download(relative, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(files[relative])

    monkeypatch.setattr("eval.download", fake_download)
    evaluation.prepare(data_dir=tmp_path, n=1, reference_n=1, seed=42)
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert manifest.training_mean == nutrients(100)
    assert manifest.samples[0].target == nutrients(400, 30, 45, 12)
    assert manifest.references[0].id == "dish_1"
    assert manifest.references[0].target == nutrients(100)


@pytest.mark.parametrize(
    "arguments",
    [
        ["prepare", "--n", "0"],
        ["run", "--timeout", "0"],
        ["run", "--limit", "-1"],
        ["run", "--predictor", "unknown"],
    ],
)
def test_cli_rejects_invalid_options_before_inference(arguments):
    result = CliRunner().invoke(evaluation.app, arguments)
    assert result.exit_code == 2


def test_cli_runs_evaluation_with_in_memory_backend(tmp_path, monkeypatch, codex_sdk):
    image_data = b"test-photo"
    (tmp_path / "photo.png").write_bytes(image_data)
    manifest = Manifest(
        dataset="Nutrition5k",
        source="test",
        license="CC BY 4.0",
        split="rgb_test",
        seed=42,
        n=1,
        missing_overhead_images=[],
        training_mean=nutrients(100),
        training_label_count=1,
        samples=[
            Sample(
                id="secret_id",
                image="photo.png",
                image_sha256=hashlib.sha256(image_data).hexdigest(),
                target=nutrients(200),
            )
        ],
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())

    def predict(self, image):
        assert isinstance(image, ImageInput)
        assert image.data == image_data
        assert image.media_type == "image/png"
        return nutrients(100)

    monkeypatch.setattr("predictors.MeanPredictor.predict", predict)
    output = tmp_path / "results"
    result = CliRunner().invoke(
        evaluation.app,
        [
            "run",
            "--data-dir",
            str(tmp_path),
            "--predictor",
            "mean",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    config = json.loads((output / "config.json").read_text())
    assert "prompt" not in config
    assert "prompt" not in config["predictor"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["succeeded"] == 1
    assert summary["metrics"]["calories_kcal"]["mae"] == 100
    records = json.loads((output / "predictions.json").read_text())
    assert records[0]["prediction"]["calories_kcal"] == 100
    assert set(path.suffix for path in output.iterdir()) == {".json"}

    codex_output = tmp_path / "codex-results"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("estimate portions")
    result = CliRunner().invoke(
        evaluation.app,
        [
            "run",
            "--data-dir",
            str(tmp_path),
            "--predictor",
            "codex",
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(codex_output),
        ],
    )
    assert result.exit_code == 0, result.output
    codex_config = json.loads((codex_output / "config.json").read_text())
    assert codex_config["predictor"]["prompt"] == "estimate portions"
    records = json.loads((codex_output / "predictions.json").read_text())
    assert records[0]["prediction"]["calories_kcal"] == 100
    metadata = records[0]["backend_metadata"]
    assert metadata["items"][0]["type"] == "agentMessage"
    assert metadata["id"] == "turn-test"

    def fail(self, image):
        raise RuntimeError("baseline failed")

    monkeypatch.setattr("predictors.MeanPredictor.predict", fail)
    failure_output = tmp_path / "failed-results"
    result = CliRunner().invoke(
        evaluation.app,
        [
            "run",
            "--data-dir",
            str(tmp_path),
            "--predictor",
            "mean",
            "--output",
            str(failure_output),
        ],
    )
    assert result.exit_code == 1
    assert json.loads((failure_output / "summary.json").read_text())["failed"] == 1
