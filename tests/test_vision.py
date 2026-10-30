"""Offline architecture, split isolation, and train/checkpoint/evaluation checks."""

import hashlib
import json
import os
import subprocess
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

pytest.importorskip("timm")
from PIL import Image
from timm.data import resolve_model_data_config

import eval as evaluation
import train
from schemas import FIELDS, Manifest, Nutrition, Sample
from timing import estimate_runtime
from tracking import wandb_metrics, wandb_summary
from vision import (
    BACKBONES,
    TARGETS,
    NutritionRegressor,
    TrainConfig,
    VisionPredictor,
    model_data_config,
)
from vision_data import (
    Quantities,
    TrainingManifest,
    TrainingSample,
    prepare_training,
    verify_images,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_real_encoder_forward_and_frozen_gradients(backbone):
    config = TrainConfig(
        backbone=backbone, pretrained=False, device="cpu", hidden_dim=8, dropout=0
    )
    model = NutritionRegressor(config)
    model.train()
    assert model.data_config == model_data_config(config)
    assert not model.backbone.training
    batch_norms = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    before = []
    for module in batch_norms:
        assert module.running_mean is not None
        before.append(module.running_mean.clone())
    output = model(torch.randn(1, *model.data_config["input_size"]))
    assert output.shape == (1, len(TARGETS))
    output.sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.heads.parameters())
    for saved, module in zip(before, batch_norms, strict=True):
        assert module.running_mean is not None
        assert torch.equal(saved, module.running_mean)
    assert len({id(head) for head in model.heads.values()}) == len(TARGETS)


def quantities(value=100):
    return Quantities(
        calories_kcal=value, protein_g=10, carbs_g=20, fat_g=5, mass_g=150
    )


def sample(root, name, value=100):
    path = root / f"{name}.png"
    Image.new("RGB", (24, 24), color=(value % 255, 30, 60)).save(path)
    return TrainingSample(
        id=name,
        image=path.name,
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target=quantities(value),
    )


def test_prepare_training_metadata_portable_paths_and_split_isolation(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    for directory in ("metadata", "dish_ids/splits", "images"):
        (root / directory).mkdir(parents=True)
    train_ids = [f"dish_{i}" for i in range(7)]
    test_ids = ["dish_test"]
    (root / "metadata/dish_metadata_cafe1.csv").write_text(
        "\n".join(f"{dish},400,250,12,45,30" for dish in train_ids + test_ids)
    )
    (root / "metadata/dish_metadata_cafe2.csv").write_text("")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        (root / f"dish_ids/splits/rgb_{split}_ids.txt").write_text("\n".join(ids))
    for dish in train_ids[:-1]:
        Image.new("RGB", (20, 20)).save(root / f"images/{dish}.png")

    def offline_download(relative, destination):
        if not destination.exists():
            assert train_ids[-1] in relative  # Never request a test photo.
            raise HTTPError(relative, 404, "missing", Message(), None)

    monkeypatch.setattr(evaluation, "download", offline_download)
    output = tmp_path / "prepared/manifest.json"
    manifest = prepare_training(root, output, 42, 0.2)
    again = prepare_training(root, tmp_path / "other/manifest.json", 42, 0.2)
    assert manifest == again
    assert len(manifest.train) == 4 and len(manifest.validation) == 2
    assert manifest.missing_overhead_images == [train_ids[-1]]
    assert manifest.train[0].target == Quantities(
        calories_kcal=400, mass_g=250, fat_g=12, carbs_g=45, protein_g=30
    )
    verify_images(manifest, output.parent)
    assert all(not Path(s.image).is_absolute() for s in manifest.train)
    with pytest.raises(ValueError, match="already exists"):
        prepare_training(root, output, 42, 0.2)
    contaminated = manifest.model_dump()
    contaminated["official_test_ids"] = [manifest.train[0].id]
    with pytest.raises(ValueError, match="exclude official test"):
        TrainingManifest.model_validate(contaminated)
    contaminated = manifest.model_dump()
    contaminated["validation"][0] = contaminated["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        TrainingManifest.model_validate(contaminated)
    path = output.parent / manifest.train[0].image
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_images(manifest, output.parent)


class TinyBackbone(nn.Module):
    num_features = 8
    pretrained_cfg = {
        "input_size": (3, 16, 16),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "crop_pct": 1.0,
        "interpolation": "bilinear",
    }

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, images):
        return self.layers(images).flatten(1)


def test_finetuning_enables_encoder_gradients(monkeypatch):
    monkeypatch.setattr("timm.create_model", lambda *a, **kw: TinyBackbone())
    config = TrainConfig(backbone="resnet", pretrained=False, freeze_backbone=False)
    model = NutritionRegressor(config).train()
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    assert model.backbone.training
    assert all(p.grad is not None for p in model.backbone.parameters())


@pytest.mark.parametrize(
    "loss,segmented",
    [
        ("standardized_smooth_l1", False),
        ("gaussian_nll", False),
        ("gaussian_nll", True),
    ],
)
def test_tiny_training_checkpoint_and_evaluator_roundtrip(
    tmp_path, monkeypatch, loss, segmented
):
    if segmented:
        import importlib.metadata

        import segmentation
        from segmentation import SegmentationConfig, prepare_masks

        class StubSegmenter:
            def __init__(self, config, device):
                self.config = config
                torch.rand(7)  # Simulate RNG consumed by pretrained model loading.

            def mask(self, image):
                mask = Image.new("L", image.size)
                mask.paste(255, (0, 0, 4, 4))
                mask.paste(
                    255, (image.width - 4, image.height - 4, image.width, image.height)
                )
                return mask

        monkeypatch.setattr(segmentation, "FoodSegmenter", StubSegmenter)
        monkeypatch.setattr("vision.FoodSegmenter", StubSegmenter)
        original_version = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: "test" if name == "transformers" else original_version(name),
        )
        monkeypatch.setattr(train, "version", importlib.metadata.version)
        monkeypatch.setattr(segmentation, "version", lambda name: "test")

    def tiny_model(name, pretrained, num_classes):
        assert name == BACKBONES["resnet"]
        assert pretrained is False and num_classes == 0
        return TinyBackbone()

    monkeypatch.setattr("timm.create_model", tiny_model)
    monkeypatch.setattr(
        train,
        "model_data_config",
        lambda config: resolve_model_data_config(TinyBackbone()),
    )
    data = TrainingManifest(
        seed=42,
        validation_fraction=0.2,
        train=[sample(tmp_path, f"train{i}", 100 + 10 * i) for i in range(4)],
        validation=[sample(tmp_path, "validation", 900)],
        missing_overhead_images=[],
        missing_labels=[],
        official_test_ids=["test"],
    )
    manifest_path = tmp_path / "training.json"
    manifest_path.write_text(data.model_dump_json())
    config_path = Path(__file__).resolve().parents[1] / "configs/resnet.toml"
    output = tmp_path / "run"
    args = [
        "fit",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    ]
    for override in (
        "pretrained=false",
        "device=cpu",
        "epochs=1",
        "workers=0",
        "batch_size=2",
        "hidden_dim=8",
        "dropout=0",
        "compile_backend=eager",
        "wandb_mode=offline",
        f"loss={loss}",
    ):
        args.extend(["--set", override])
    if segmented:
        args.extend(["--set", 'segmentation={revision="' + "a" * 40 + '"}'])
    runner = CliRunner()
    preview = runner.invoke(train.app, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert not output.exists()
    result = runner.invoke(train.app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    saved = json.loads((output / "config.json").read_text())
    assert saved["changes_from_reference"]["device"] == {
        "reference": "cuda",
        "chosen": "cpu",
    }
    # Held-out validation calories (900) must not affect training normalization.
    assert saved["target_mean"][0] == 115
    assert saved["target_std"][0] == pytest.approx((125) ** 0.5)
    assert saved["target_std"][1] == pytest.approx(1e-6)
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert set(metrics["validation"]["mae"]) == set(TARGETS)
    checkpoint = output / "best.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert not any("_orig_mod" in key for key in state["model"])
    assert state["train_config"]["compile_model"] is True
    assert (output / "wandb.json").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["first_epoch_seconds"] > 0
    assert summary["total_seconds"] >= summary["training_seconds"]
    assert metrics["train"]["first_batch_seconds"] > 0
    prediction_backend = VisionPredictor(checkpoint)
    if segmented:
        assert state["train_config"]["segmentation"]["revision"] == "a" * 40
        assert saved["segmentation"]["samples"][0]["foreground_fraction"] > 0
        _, mask_paths, _ = prepare_masks(
            data.train,
            tmp_path,
            SegmentationConfig(revision="a" * 40),
            tmp_path / "segmentation-masks",
            "cpu",
        )
        from segmentation import apply_food_mask
        from vision_data import NutritionDataset

        dataset = NutritionDataset(
            data.train, tmp_path, prediction_backend.transform, masks=mask_paths
        )
        assert prediction_backend.segmenter is not None
        with Image.open(tmp_path / data.train[0].image) as image:
            online = prediction_backend.transform(
                apply_food_mask(image, prediction_backend.segmenter.mask(image))
            )
        torch.testing.assert_close(dataset[0][0], online)
    else:
        # Simulate a checkpoint written before the segmentation field existed.
        state["train_config"].pop("segmentation", None)
        torch.save(state, output / "legacy.pt")
        assert VisionPredictor(output / "legacy.pt").segmenter is None
    test = sample(tmp_path, "test", 200)
    test_manifest = Manifest(
        dataset="Nutrition5k",
        source="synthetic",
        license="test",
        split="rgb_test",
        seed=42,
        n=1,
        missing_overhead_images=[],
        training_label_count=4,
        training_mean=Nutrition(**{key: 1.0 for key in FIELDS}),
        samples=[
            Sample.model_validate(
                {
                    **test.model_dump(),
                    "target": {key: getattr(test.target, key) for key in FIELDS},
                }
            )
        ],
    )
    (tmp_path / "manifest.json").write_text(test_manifest.model_dump_json())
    evaluation_output = tmp_path / "evaluation"
    result = runner.invoke(
        evaluation.app,
        [
            "run",
            "--predictor",
            "vision",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(evaluation_output),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    records = json.loads((evaluation_output / "predictions.json").read_text())
    assert records[0]["status"] == "ok"
    assert records[0]["backend_metadata"]["mass_g"] >= 0
    if loss == "gaussian_nll":
        if segmented:
            assert (
                records[0]["backend_metadata"]["segmentation"]["foreground_fraction"]
                > 0
            )
        assert records[0]["prediction_interval"] is not None
        for key in FIELDS:
            std = records[0]["backend_metadata"]["gaussian_std"][key]
            assert std > 0
            assert records[0]["prediction"]["sigma"][key] == pytest.approx(
                1.959963984540054 * std
            )
        assert "nll" in metrics["validation"]
        assert "mean_std" in metrics["validation"]
    assert all(value >= 0 for value in records[0]["prediction"]["mean"].values())
    recorded = json.loads((evaluation_output / "config.json").read_text())
    assert (
        recorded["predictor"]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert prediction_backend.config.backbone == BACKBONES["resnet"]
    if segmented:
        warm_output = tmp_path / "warm-run"
        warm_args = list(args)
        warm_args[warm_args.index("--output") + 1] = str(warm_output)
        warm_args.extend(["--set", "wandb_mode=disabled"])
        result = runner.invoke(train.app, warm_args)
        assert result.exit_code == 0, result.output + repr(result.exception)
        warm = torch.load(warm_output / "best.pt", weights_only=True)
        for name, tensor in state["model"].items():
            torch.testing.assert_close(tensor, warm["model"][name], rtol=0, atol=0)
    repeated = runner.invoke(train.app, args)
    assert repeated.exit_code != 0


def test_gaussian_head_likelihood_positive_scales_and_frozen_encoder(monkeypatch):
    monkeypatch.setattr("timm.create_model", lambda *a, **kw: TinyBackbone())
    config = TrainConfig(
        backbone="resnet", pretrained=False, loss="gaussian_nll", dropout=0
    )
    model = NutritionRegressor(config).train()
    output = model(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, len(TARGETS), 2)
    mean, std = model.gaussian_parameters(output)
    assert torch.allclose(std, torch.ones_like(std))
    expected = torch.full_like(mean, 3)
    nll = nn.functional.gaussian_nll_loss(
        mean, expected, std.square(), full=True, reduction="none"
    )
    assert torch.allclose(
        nll, -torch.distributions.Normal(mean, std).log_prob(expected)
    )
    nll.mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.heads.parameters()
    )
    extreme = torch.zeros(2, len(TARGETS), 2)
    extreme[..., 1] = -1e4
    _, floor = model.gaussian_parameters(extreme)
    assert torch.all(floor >= config.gaussian_min_std)


def test_gaussian_intervals_preserve_signed_raw_bounds():
    from schemas import NutritionInterval, SignedNutrition

    bounds = NutritionInterval(
        lower=SignedNutrition(**{key: -1.0 for key in FIELDS}),
        upper=SignedNutrition(**{key: 2.0 for key in FIELDS}),
    )
    assert bounds.lower.calories_kcal == -1
    assert (
        NutritionInterval.model_validate_json(bounds.model_dump_json()).model_dump()
        == bounds.model_dump()
    )


def test_configs_only_differ_in_backbone_and_reject_unknown_overrides():
    root = Path(__file__).resolve().parents[1]
    values = []
    for name in BACKBONES:
        config, changes = train.load_config(root / f"configs/{name}.toml", [])
        assert not changes
        values.append(config.model_dump(exclude={"backbone"}))
        gaussian, _ = train.load_config(root / f"configs/{name}-gaussian.toml", [])
        differences = {
            key
            for key in type(config).model_fields
            if getattr(config, key) != getattr(gaussian, key)
        }
        assert differences == {"loss", "wandb_group"}
        assert gaussian.freeze_backbone is True
    assert values[0] == values[1] == values[2]
    with pytest.raises(ValueError, match="known_config_key"):
        train.load_config(root / "configs/resnet.toml", ["batch_szie=4"])


def test_wandb_sections_and_warmup_aware_estimates():
    metrics = {
        "epoch": 2,
        "learning_rates": [0.001],
        "train": {
            "loss": 0.2,
            "mae": {"mass_g": 5},
            "seconds": 9,
            "first_batch_seconds": 1,
            "images_per_second": 30,
        },
        "validation": {
            "loss": 0.3,
            "mae": {"mass_g": 6},
            "seconds": 3,
            "first_batch_seconds": 0.2,
            "images_per_second": 20,
        },
        "timing": {"epoch_seconds": 12},
    }
    flattened = wandb_metrics(metrics)
    assert flattened["train/loss"] == 0.2
    assert flattened["val/loss"] == 0.3
    assert flattened["val/mae/mass_g"] == 6
    assert flattened["timing/train_seconds"] == 9
    assert flattened["timing/val_images_per_second"] == 20
    assert not any(k.startswith("validation/") for k in flattened)
    assert all(
        k.startswith("timing/")
        for k in flattened
        if k.endswith(("seconds", "images_per_second"))
    )
    summary = wandb_summary(
        {
            "total_seconds": 300,
            "training_seconds": 280,
            "best_validation_loss": 0.1,
            "best_epoch": 9,
            "best_validation_mae": {"calories_kcal": 60.0},
        }
    )
    assert summary == {
        "timing/total_seconds": 300,
        "timing/training_seconds": 280,
        "val/best_loss": 0.1,
        "val/best_epoch": 9,
        "val/best_checkpoint_mae/calories_kcal": 60.0,
    }
    estimate = estimate_runtime([90, 9, 11, 10], 30, 130)
    assert estimate["steady_epoch_seconds"] == 10
    assert estimate["estimated_remaining_seconds"] == 260
    assert estimate["estimated_total_seconds"] == 390
    assert estimate_runtime([90, 9], 2, 100)["estimated_remaining_seconds"] == 0


def test_deployment_shell_syntax_and_psc_arguments(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob("deploy/*.sh"), *root.glob("deploy/*.sbatch")]:
        subprocess.run(["bash", "-n", str(path)], check=True)
    log = tmp_path / "commands"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
    )
    fake_python.chmod(0o755)
    sbatch = tmp_path / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("COMMAND_LOG", str(log))
    monkeypatch.setenv("PYTHON", str(fake_python))
    monkeypatch.setenv("PSC_ACCOUNT", "test-allocation")
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    subprocess.run(
        [
            "bash",
            "deploy/bridges2.sh",
            "manifest.json",
            str(tmp_path / "runs"),
            "--set",
            "epochs=7",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commands = log.read_text().splitlines()
    assert len(commands) == 4
    assert all("--dry-run" in cmd and "epochs=7" in cmd for cmd in commands[:3])
    assert "--gpus=h100-80:1" in commands[3]
    assert "--array=0-2" in commands[3]
    assert "--account=test-allocation" in commands[3]
    assert "--partition=GPU-shared" in commands[3]
    assert "--set epochs=7" in commands[3]


def test_pruned_training_finalizes_logs_and_saves_calorie_selected_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("timm.create_model", lambda *a, **kw: TinyBackbone())
    monkeypatch.setattr(
        train, "model_data_config", lambda c: resolve_model_data_config(TinyBackbone())
    )
    data = TrainingManifest(
        seed=42,
        validation_fraction=0.2,
        train=[sample(tmp_path, "train0", 100), sample(tmp_path, "train1", 200)],
        validation=[sample(tmp_path, "val", 300)],
        missing_overhead_images=[],
        missing_labels=[],
        official_test_ids=["test"],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(data.model_dump_json())
    calls = iter(
        [
            {"loss": 1, "mae": {}},
            {"loss": 1, "mae": {}, "nll": {"calories_kcal": 6}},
            {"loss": 2, "mae": {}},
            {"loss": 2, "mae": {}, "nll": {"calories_kcal": 5}},
        ]
    )

    def fake_pass(*args):
        if len(args) > 4:
            args[4].step()
        return next(calls)

    monkeypatch.setattr(train, "epoch_pass", fake_pass)
    output = tmp_path / "run"
    summary = train.run_training(
        Path("configs/resnet-gaussian.toml"),
        manifest,
        output,
        [
            "device=cpu",
            "pretrained=false",
            "compile_model=false",
            "epochs=3",
            "workers=0",
            "wandb_mode=disabled",
            "selection_metric=calorie_nll",
        ],
        epoch_callback=lambda row: row["epoch"] == 2,
    )
    assert summary["stopped_early"] and summary["epochs"] == 2
    assert summary["max_epochs"] == 3 and summary["best_epoch"] == 2
    assert summary["best_selection_score"] == 5
    assert summary["best_validation_loss"] == 2
    state = torch.load(output / "best.pt", weights_only=True)
    assert state["epoch"] == 2 and state["best_selection_score"] == 5
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="requires Gaussian"):
        TrainConfig(backbone="resnet", selection_metric="calorie_nll")


@pytest.mark.parametrize("repair_append", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_epoch_resume_matches_uninterrupted_adam_dropout_and_shuffle(
    tmp_path, monkeypatch, repair_append, segmented
):
    if segmented:
        import segmentation

        resolutions = []

        def resolve(config):
            resolutions.append(config.revision)
            return config.model_copy(update={"revision": "a" * 40})

        class StubSegmenter:
            def __init__(self, config, device):
                assert config.revision == "a" * 40
                torch.rand(7)

            def mask(self, image):
                mask = Image.new("L", image.size)
                mask.paste(255, (0, 0, 12, 12))
                return mask

        monkeypatch.setattr(segmentation, "resolve_segmentation", resolve)
        monkeypatch.setattr(segmentation, "FoodSegmenter", StubSegmenter)
        monkeypatch.setattr(segmentation, "version", lambda name: "test")
        original_version = train.version
        monkeypatch.setattr(
            train,
            "version",
            lambda name: "test" if name == "transformers" else original_version(name),
        )
    monkeypatch.setattr("timm.create_model", lambda *a, **kw: TinyBackbone())
    monkeypatch.setattr(
        train, "model_data_config", lambda c: resolve_model_data_config(TinyBackbone())
    )
    data = TrainingManifest(
        seed=42,
        validation_fraction=0.2,
        train=[sample(tmp_path, f"train{i}", 80 + i * 25) for i in range(6)],
        validation=[sample(tmp_path, "val", 175)],
        missing_overhead_images=[],
        missing_labels=[],
        official_test_ids=["test"],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(data.model_dump_json())
    reference = Path("configs/resnet-gaussian.toml")
    overrides = [
        "device=cpu",
        "pretrained=false",
        "compile_backend=eager",
        "workers=0",
        "epochs=3",
        "hidden_dim=8",
        "batch_size=2",
        "dropout=0.25",
        "wandb_mode=disabled",
        "selection_metric=calorie_nll",
    ]
    if segmented:
        overrides.append('segmentation={revision="main"}')
    full = tmp_path / "full"
    train.run_training(reference, manifest, full, overrides)
    split = tmp_path / "split"

    def interrupt(row):
        if row["epoch"] == 2:
            raise InterruptedError("simulated preemption after committed epoch")
        return False

    with pytest.raises(InterruptedError):
        train.run_training(
            reference, manifest, split, overrides, epoch_callback=interrupt
        )
    if repair_append:
        rows = (split / "metrics.jsonl").read_text().splitlines()
        (split / "metrics.jsonl").write_text(rows[0] + '\n{"epoch":2')
        (
            split / "best.pt"
        ).unlink()  # Crash after last.pt commit, before publishing best.
    if not segmented:
        # Runs created before CLIPSeg had neither configuration field.
        original = json.loads((split / "config.json").read_text())
        original.pop("requested_train_config")
        original["train_config"].pop("segmentation")
        (split / "config.json").write_text(json.dumps(original))
        checkpoint = torch.load(split / "last.pt", weights_only=True)
        checkpoint["train_config"].pop("segmentation")
        torch.save(checkpoint, split / "last.pt")
    summary = train.run_training(reference, manifest, split, overrides, resume=True)
    if segmented:
        assert resolutions == ["main", "main", "a" * 40]
        resumed = json.loads((split / "resume-0001.json").read_text())
        assert resumed["train_config"]["segmentation"]["revision"] == "a" * 40
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(split / "last.pt", weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    for key, value in a["training_state"]["optimizer"]["state"].items():
        assert all(
            torch.equal(v, b["training_state"]["optimizer"]["state"][key][k])
            for k, v in value.items()
        )
    assert a["training_state"]["scheduler"] == b["training_state"]["scheduler"]
    assert summary["resumes"] == 1 and summary["epochs"] == 3
    rows = [
        json.loads(line) for line in (split / "metrics.jsonl").read_text().splitlines()
    ]
    baseline = [
        json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert [row["validation"]["nll"] for row in rows] == [
        row["validation"]["nll"] for row in baseline
    ]
    assert [row["learning_rates"] for row in rows] == [
        row["learning_rates"] for row in baseline
    ]
    assert (split / "best.pt").is_file()
    assert len(list(split.glob(".best-epoch-*.pt"))) == 1
    if repair_append:
        assert rows[1]["timing"]["checkpoint_seconds"] is None
    with pytest.raises(ValueError, match="config differs"):
        train.run_training(
            reference, manifest, split, [*overrides, "head_lr=0.002"], resume=True
        )
