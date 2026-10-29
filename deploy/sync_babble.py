"""Upload completed offline runs with train/val/timing metric sections.

Replay the authoritative JSON logs using local credentials. This also organizes
older runs made before the reporting schema changed. GPU monitoring stays in the
original offline records; local replay disables local-machine system monitoring.
"""

import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timing import estimate_runtime
from tracking import configure_metrics, wandb_metrics, wandb_summary


def upload_run(output: Path):
    marker = output / "wandb-uploaded.json"
    if marker.exists():
        print(f"Already uploaded: {json.loads(marker.read_text())['url']}")
        return
    config = json.loads((output / "config.json").read_text())
    settings = config["train_config"]
    identity = json.loads((output / "wandb.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    history = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    best_row = next(row for row in history if row["epoch"] == summary["best_epoch"])
    summary["best_validation_mae"] = best_row["validation"]["mae"]
    with wandb.init(
        project=settings["wandb_project"],
        entity=settings.get("wandb_entity"),
        group=settings["wandb_group"],
        id=identity["id"],
        resume="allow",
        name=output.name,
        job_type="train",
        mode="online",
        config={
            **config,
            "logging": {
                "uploaded_from_offline_json": True,
                "system_monitoring": "remote records retained locally; local replay disabled",
            },
        },
        dir=str(output.resolve()),
        settings=wandb.Settings(x_disable_stats=True),
    ) as run:
        configure_metrics(run)
        durations = []
        for row in history:
            durations.append(row["timing"]["epoch_seconds"])
            row["timing"].update(
                estimate_runtime(
                    durations,
                    settings["epochs"],
                    row["timing"]["elapsed_total_seconds"],
                )
            )
            run.log(wandb_metrics(row), step=row["epoch"])
        run.summary.update(wandb_summary(summary))
        run.save(
            str((output / "config.json").resolve()),
            base_path=str(output.resolve()),
            policy="now",
        )
        url = run.url
    marker.write_text(json.dumps({"url": url, "id": identity["id"]}, indent=2) + "\n")
    print(f"Uploaded: {url}", flush=True)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/babel")
    for summary in sorted(root.glob("*-*/summary.json")):
        if (summary.parent / "wandb.json").exists():
            upload_run(summary.parent)


if __name__ == "__main__":
    main()
