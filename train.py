#!/usr/bin/env python3
"""Train Nutrition5k image regression heads. See deploy/README.md."""

import hashlib
import json
import os
import platform
import random
import time
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
from timing import estimate_runtime, phase_timer
from tracking import wandb_metrics, wandb_session, wandb_summary
from vision import (
    BACKBONES,
    NORMAL_95_Z,
    TARGETS,
    NutritionRegressor,
    TrainConfig,
    compile_for_training,
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


def gaussian_totals_keys():
    return (
        "nll_standardized",
        "nll",
        "mean_std",
        "coverage_95_pct",
        "mean_interval_width_95",
    )


@app.command()
def prepare(
    source_dir: Path = Path("data/nutrition5k"),
    output: Path = Path("data/nutrition5k-training/manifest.json"),
    seed: Annotated[int, typer.Option(min=0, max=2**31 - 1)] = 42,
    validation_fraction: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.2,
    download_workers: Annotated[int, typer.Option(min=1, max=64)] = 1,
):
    """Cache all available official training RGBs; create a train/validation split."""
    manifest = prepare_training(
        source_dir, output, seed, validation_fraction, download_workers
    )
    print(
        f"Prepared {output}: {len(manifest.train)} train, {len(manifest.validation)} validation"
    )


def epoch_pass(model, loader, device, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    errors = torch.zeros(len(TARGETS), dtype=torch.float64)
    gaussian_totals = {
        key: torch.zeros(len(TARGETS), dtype=torch.float64)
        for key in gaussian_totals_keys()
    }
    dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timer = phase_timer("train" if training else "val")
    first_batch_timer = phase_timer(
        "train_first_batch" if training else "val_first_batch"
    )
    timer.start()
    first_batch_timer.start()
    first_batch_seconds = None
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
            output = model(images)
            expected = (targets - model.target_mean) / model.target_std
            if config.loss == "gaussian_nll":
                normalized, std = model.gaussian_parameters(output)
                nll = nn.functional.gaussian_nll_loss(
                    normalized,
                    expected,
                    std.square(),
                    full=True,
                    eps=config.gaussian_min_std**2,
                    reduction="none",
                )
                loss = nll.mean()
            else:
                normalized = output
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
        if config.loss == "gaussian_nll":
            with torch.no_grad():
                physical_std = std.detach() * model.target_std
                covered = (
                    expected - normalized.detach()
                ).abs() <= NORMAL_95_Z * std.detach()
                values = {
                    "nll_standardized": nll.detach(),
                    "nll": nll.detach() + model.target_std.log(),
                    "mean_std": physical_std,
                    "coverage_95_pct": covered.float() * 100,
                    "mean_interval_width_95": 2 * NORMAL_95_Z * physical_std,
                }
                for key, value in values.items():
                    gaussian_totals[key] += value.double().sum(0).cpu()
        if first_batch_seconds is None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            first_batch_seconds = first_batch_timer.stop()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = timer.stop()
    metrics = {
        "loss": total_loss / count,
        "mae": dict(zip(TARGETS, (errors / count).tolist(), strict=True)),
        "seconds": elapsed,
        "first_batch_seconds": first_batch_seconds,
        "images_per_second": count / elapsed,
    }
    if config.loss == "gaussian_nll":
        metrics.update(
            {
                key: dict(zip(TARGETS, (value / count).tolist(), strict=True))
                for key, value in gaussian_totals.items()
            }
        )
    return metrics


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
    fit_started = time.perf_counter()
    settings, changes = load_config(config, overrides or [])
    manifest = manifest.resolve()
    raw_manifest = manifest.read_bytes()
    data = TrainingManifest.model_validate_json(raw_manifest)
    preprocessing = model_data_config(settings)
    chosen: dict[str, Any] = {
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
        "selection": "minimum validation standardized Gaussian NLL"
        if settings.loss == "gaussian_nll"
        else "minimum validation standardized Smooth L1 loss",
        "distribution": "independent untruncated Gaussian per target; unconstrained mean; std=softplus(raw_std)+floor; full NLL including constant; mean over samples and targets"
        if settings.loss == "gaussian_nll"
        else "point estimates only",
        "batch_policy": "shuffle train; keep incomplete batches; no gradient accumulation",
        "monitoring": "stdout, metrics.jsonl, config.json, best.pt, last.pt, W&B (explicit mode)",
        "timing_policy": "CUDA-synchronized train/validation passes; first batches include lazy compilation; epoch wall time includes checkpoint I/O",
        "timing_tool": "codetiming named phase timers + perf_counter wall clock",
        "compiler_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "eta_policy": "median of last five completed epochs; excludes first epoch after warm-up; low/high use recent min/max, not confidence bounds",
        "hardware": {
            "device": settings.device,
            "gpu_count": 1 if settings.device.startswith("cuda") else 0,
            "hostname": platform.node(),
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
    compile_for_training(model, settings)
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
            "parameters": {
                "backbone_trainable": sum(
                    p.numel() for p in model.backbone.parameters() if p.requires_grad
                ),
                "heads_trainable": sum(
                    p.numel() for p in model.heads.parameters() if p.requires_grad
                ),
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    if device.type == "cuda":
        chosen["hardware"]["gpu_name"] = torch.cuda.get_device_name(device)
    if settings.wandb_mode != "disabled":
        chosen["dependencies"]["wandb"] = version("wandb")
    chosen["dependencies"]["codetiming"] = version("codetiming")
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
    with wandb_session(settings, chosen, output) as wb_run:
        if wb_run is not None:
            save_json(
                output / "wandb.json",
                {
                    "url": wb_run.url,
                    "id": wb_run.id,
                    "entity": wb_run.entity,
                    "project": wb_run.project,
                },
            )
        setup_seconds = time.perf_counter() - fit_started
        training_started = time.perf_counter()
        epoch_seconds = []
        best = float("inf")
        best_epoch = None
        best_validation_mae = None
        best_validation_distribution = {}
        for epoch in range(1, settings.epochs + 1):
            epoch_timer = phase_timer("epoch")
            epoch_timer.start()
            rates = scheduler.get_last_lr()
            training = epoch_pass(
                model, loaders[0], device, settings, optimizer, scaler
            )
            validation = epoch_pass(model, loaders[1], device, settings)
            scheduler.step()
            improved = validation["loss"] < best
            if improved:
                best, best_epoch = validation["loss"], epoch
                best_validation_mae = validation["mae"]
                best_validation_distribution = {
                    key: validation[key]
                    for key in gaussian_totals_keys()
                    if key in validation
                }
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
            with phase_timer("checkpoint") as checkpoint_timer:
                for name in ["last.pt", "best.pt"] if improved else ["last.pt"]:
                    temporary = output / (name + ".tmp")
                    torch.save(checkpoint, temporary)
                    temporary.replace(output / name)
            checkpoint_seconds = checkpoint_timer.last
            elapsed = epoch_timer.stop()
            epoch_seconds.append(elapsed)
            metrics = {
                "epoch": epoch,
                "learning_rates": rates,
                "train": training,
                "validation": validation,
                "timing": {
                    "epoch_seconds": elapsed,
                    "checkpoint_seconds": checkpoint_seconds,
                    "elapsed_training_seconds": time.perf_counter() - training_started,
                    "elapsed_total_seconds": time.perf_counter() - fit_started,
                    **estimate_runtime(
                        epoch_seconds,
                        settings.epochs,
                        time.perf_counter() - fit_started,
                    ),
                },
            }
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if wb_run is not None:
                wb_run.log(wandb_metrics(metrics), step=epoch)
            print(json.dumps(metrics), flush=True)
        summary = {
            "epochs": settings.epochs,
            "best_validation_loss": best,
            "best_epoch": best_epoch,
            "best_validation_mae": best_validation_mae,
            "best_validation_distribution": best_validation_distribution,
            "checkpoint": "best.pt",
            "setup_seconds": setup_seconds,
            "training_seconds": time.perf_counter() - training_started,
            "total_seconds": time.perf_counter() - fit_started,
            "first_epoch_seconds": epoch_seconds[0],
            "mean_steady_epoch_seconds": sum(epoch_seconds[1:]) / len(epoch_seconds[1:])
            if len(epoch_seconds) > 1
            else None,
        }
        save_json(output / "summary.json", summary)
        if wb_run is not None:
            wb_run.summary.update(wandb_summary(summary))
        print("Training summary: " + json.dumps(summary), flush=True)


def main():
    try:
        app()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
