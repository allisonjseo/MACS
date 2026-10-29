#!/usr/bin/env python3
"""Train Nutrition5k image regression heads. See deploy/README.md."""

import hashlib
import json
import os
import platform
import random
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from determinism import configure_determinism
from run import PROJECT_DIR, GitRevision
from vision import (
    BACKBONES,
    TARGETS,
    NutritionRegressor,
    TrainConfig,
    create_transform,
    model_data_config,
)
from vision_data import (
    NutritionDataset,
    TrainingManifest,
    prepare_training,
    verify_images,
)

app = typer.Typer(no_args_is_help=True, help=__doc__)


def save_json(path: Path, value: Any):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path, overrides: list[str]):
    reference = TrainConfig.model_validate(tomllib.loads(path.read_text()))
    values = reference.model_dump()
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or key not in values:
            raise ValueError("Use --set known_config_key=value: " + override)
        try:
            parsed = tomllib.loads("value = " + value)["value"]
        except tomllib.TOMLDecodeError:
            parsed = value
        values[key] = parsed
    config = TrainConfig.model_validate(values)
    changes = {
        key: {"reference": getattr(reference, key), "chosen": getattr(config, key)}
        for key in values
        if getattr(reference, key) != getattr(config, key)
    }
    return config, changes


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@app.command()
def prepare(
    source_dir: Path = Path("data/nutrition5k"),
    output: Path = Path("data/nutrition5k-training/manifest.json"),
    seed: Annotated[int, typer.Option(min=0, max=2**31 - 1)] = 42,
    validation_fraction: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.2,
):
    """Cache all available official training RGBs; create a train/validation split."""
    manifest = prepare_training(source_dir, output, seed, validation_fraction)
    print(
        f"Prepared {output}: {len(manifest.train)} train, {len(manifest.validation)} validation"
    )


def epoch_pass(model, loader, device, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    errors = torch.zeros(len(TARGETS), dtype=torch.float64)
    dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
    for images, targets in tqdm(loader, desc="Train" if training else "Validation"):
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=config.precision != "float32",
            ),
        ):
            normalized = model(images)
            expected = (targets - model.target_mean) / model.target_std
            loss = nn.functional.smooth_l1_loss(
                normalized.float(), expected, beta=config.huber_beta
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Nonfinite loss; run stopped without changing configuration"
            )
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
        count += len(images)
        total_loss += loss.item() * len(images)
        errors += (
            (model.quantities(normalized.detach()) - targets)
            .abs()
            .double()
            .sum(0)
            .cpu()
        )
    return {
        "loss": total_loss / count,
        "mae": dict(zip(TARGETS, (errors / count).tolist(), strict=True)),
    }


@app.command()
def fit(
    config: Annotated[Path, typer.Option(help="Reference TOML configuration")],
    manifest: Annotated[Path, typer.Option(help="Supervised training manifest")],
    output: Annotated[Path, typer.Option(help="Fresh run directory")],
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Explicit key=value override; repeatable"),
    ] = None,
    dry_run: bool = False,
):
    """Print the exact resolved configuration, then train (or only print with --dry-run)."""
    settings, changes = load_config(config, overrides or [])
    manifest = manifest.resolve()
    raw_manifest = manifest.read_bytes()
    data = TrainingManifest.model_validate_json(raw_manifest)
    preprocessing = model_data_config(settings)
    chosen = {
        "config_file": str(config.resolve()),
        "changes_from_reference": changes,
        "train_config": settings.model_dump(mode="json"),
        "backbone_id": BACKBONES[settings.backbone],
        "data_config": preprocessing,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "split_seed": data.seed,
        "validation_fraction": data.validation_fraction,
        "train_count": len(data.train),
        "validation_count": len(data.validation),
        "output": str(output.resolve()),
        "target_normalization": "training-only population mean/std; std floor 1e-6",
        "prediction_postprocessing": "inverse standardization, clamp minimum to zero",
        "selection": "minimum validation standardized Smooth L1 loss",
        "batch_policy": "shuffle train; keep incomplete batches; no gradient accumulation",
        "monitoring": "stdout, metrics.jsonl, config.json, best.pt, last.pt",
        "hardware": {
            "device": settings.device,
            "gpu_count": 1 if settings.device.startswith("cuda") else 0,
        },
    }
    print(json.dumps(chosen, indent=2), flush=True)
    if dry_run:
        return
    if output.exists():
        raise ValueError("Output already exists; use a fresh run directory")
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no automatic CPU fallback")
    if settings.precision != "float32" and device.type != "cuda":
        raise ValueError(
            "Mixed precision requires CUDA; explicitly choose float32 for CPU/MPS"
        )
    if settings.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "GPU does not support bfloat16; explicitly choose another precision"
        )
    determinism = configure_determinism(settings.seed)
    verify_images(data, manifest.parent)
    model = NutritionRegressor(settings)
    # Resolve before initialization/download and check the result is identical.
    if model.data_config != preprocessing:
        raise ValueError("Resolved preprocessing changed during model creation")
    targets = torch.tensor(
        [[getattr(s.target, key) for key in TARGETS] for s in data.train]
    )
    model.target_mean.copy_(targets.mean(0))
    model.target_std.copy_(targets.std(0, unbiased=False).clamp_min(1e-6))
    model.to(device)
    transform = create_transform(preprocessing)
    generator = torch.Generator().manual_seed(settings.seed)
    loaders = [
        DataLoader(
            NutritionDataset(samples, manifest.parent, transform),
            batch_size=settings.batch_size,
            shuffle=training,
            num_workers=settings.workers,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        for samples, training in ((data.train, True), (data.validation, False))
    ]
    groups = [{"params": model.heads.parameters(), "lr": settings.head_lr}]
    if not settings.freeze_backbone:
        groups.append(
            {"params": model.backbone.parameters(), "lr": settings.backbone_lr}
        )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=settings.weight_decay,
        betas=settings.betas,
        eps=settings.eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=settings.precision == "float16")
    chosen.update(
        {
            "determinism": determinism.model_dump(),
            "git": GitRevision.capture(PROJECT_DIR).model_dump(),
            "python_version": platform.python_version(),
            "dependencies": {
                name: version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "timm",
                    "pillow",
                    "numpy",
                    "pydantic",
                )
            },
            "train_ids": [s.id for s in data.train],
            "validation_ids": [s.id for s in data.validation],
            "target_mean": model.target_mean.cpu().tolist(),
            "target_std": model.target_std.cpu().tolist(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    if device.type == "cuda":
        chosen["hardware"]["gpu_name"] = torch.cuda.get_device_name(device)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", chosen)
    print(
        "Runtime and target normalization: "
        + json.dumps(
            {
                key: chosen[key]
                for key in ("hardware", "determinism", "target_mean", "target_std")
            }
        ),
        flush=True,
    )
    best = float("inf")
    for epoch in range(1, settings.epochs + 1):
        rates = scheduler.get_last_lr()
        training = epoch_pass(model, loaders[0], device, settings, optimizer, scaler)
        validation = epoch_pass(model, loaders[1], device, settings)
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "learning_rates": rates,
            "train": training,
            "validation": validation,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        improved = validation["loss"] < best
        best = min(best, validation["loss"])
        checkpoint = {
            "format_version": 1,
            "model": model.state_dict(),
            "train_config": settings.model_dump(mode="json"),
            "data_config": preprocessing,
            "epoch": epoch,
            "validation": validation,
            "best_validation_loss": best,
            "manifest_sha256": chosen["manifest_sha256"],
        }
        for name in ["last.pt", "best.pt"] if improved else ["last.pt"]:
            temporary = output / (name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output / name)
        print(json.dumps(metrics), flush=True)
    save_json(
        output / "summary.json",
        {
            "epochs": settings.epochs,
            "best_validation_loss": best,
            "checkpoint": "best.pt",
        },
    )


def main():
    try:
        app()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
