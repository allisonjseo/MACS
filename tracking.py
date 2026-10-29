"""W&B lifecycle and flattened epoch metrics; no image or credential uploads."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vision import TrainConfig


def configure_metrics(run):
    run.define_metric("epoch")
    for prefix in ("train", "val", "timing"):
        run.define_metric(f"{prefix}/*", step_metric="epoch")
    run.define_metric("val/loss", summary="min")


@contextmanager
def wandb_session(
    settings: TrainConfig, config: dict[str, Any], output: Path, *, resume=False
):
    if settings.wandb_mode == "disabled":
        yield None
        return
    import json

    import wandb

    identity = (
        json.loads((output / "wandb.json").read_text())
        if resume and (output / "wandb.json").exists()
        else {}
    )

    # Use an explicit mode; failure to connect must not silently disable logging.
    with wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        group=settings.wandb_group,
        name=output.name,
        job_type="train",
        mode=settings.wandb_mode,
        id=identity.get("id"),
        resume="allow" if resume and settings.wandb_mode == "online" else None,
        config=config,
        dir=str(output.resolve()),
        settings=wandb.Settings(init_timeout=60),
    ) as run:
        configure_metrics(run)
        run.save(
            str((output / "config.json").resolve()),
            base_path=str(output.resolve()),
            policy="now",
        )
        print(f"W&B run: {run.url}", flush=True)
        yield run


def wandb_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = {"epoch": metrics["epoch"]}
    for split in ("train", "validation"):
        prefix = "val" if split == "validation" else "train"
        for key, value in metrics[split].items():
            if isinstance(value, dict):
                flattened.update(
                    {
                        f"{prefix}/{key}/{target}": error
                        for target, error in value.items()
                    }
                )
            elif key in ("seconds", "first_batch_seconds", "images_per_second"):
                flattened[f"timing/{prefix}_{key}"] = value
            else:
                flattened[f"{prefix}/{key}"] = value
    flattened.update(
        {f"timing/{key}": value for key, value in metrics["timing"].items()}
    )
    flattened.update(
        {
            f"train/lr/group_{i}": value
            for i, value in enumerate(metrics["learning_rates"])
        }
    )
    return flattened


def wandb_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in summary.items():
        if key.endswith("_seconds"):
            result[f"timing/{key}"] = value
        elif key == "best_validation_loss":
            result["val/best_loss"] = value
        elif key == "best_selection_score":
            result["val/best_selection_score"] = value
        elif key == "best_epoch":
            result["val/best_epoch"] = value
        elif key == "best_validation_mae":
            result.update(
                {
                    f"val/best_checkpoint_mae/{target}": error
                    for target, error in value.items()
                }
            )
        elif key == "best_validation_distribution":
            for metric, targets in value.items():
                result.update(
                    {
                        f"val/best_checkpoint_{metric}/{target}": score
                        for target, score in targets.items()
                    }
                )
        else:
            result[key] = value
    return result
