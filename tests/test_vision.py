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


def test_tiny_training_checkpoint_and_evaluator_roundtrip(tmp_path, monkeypatch):
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
    ):
        args.extend(["--set", override])
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
    prediction_backend = VisionPredictor(checkpoint)
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
    assert all(value >= 0 for value in records[0]["prediction"]["mean"].values())
    recorded = json.loads((evaluation_output / "config.json").read_text())
    assert (
        recorded["predictor"]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert prediction_backend.config.backbone == BACKBONES["resnet"]
    repeated = runner.invoke(train.app, args)
    assert repeated.exit_code != 0


def test_configs_only_differ_in_backbone_and_reject_unknown_overrides():
    root = Path(__file__).resolve().parents[1]
    values = []
    for name in BACKBONES:
        config, changes = train.load_config(root / f"configs/{name}.toml", [])
        assert not changes
        values.append(config.model_dump(exclude={"backbone"}))
    assert values[0] == values[1] == values[2]
    with pytest.raises(ValueError, match="known_config_key"):
        train.load_config(root / "configs/resnet.toml", ["batch_szie=4"])


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
