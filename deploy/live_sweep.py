"""Stream append-only remote epoch records into persistent online W&B runs."""

import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timing import estimate_runtime
from tracking import configure_metrics, wandb_metrics, wandb_summary


def read_history(path: Path):
    """Ignore an unfinished append at EOF, but reject malformed completed records."""
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_bytes().splitlines(keepends=True)
        if line.endswith(b"\n")
    ]
    if [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Epoch history is not contiguous and unique: {path}")
    return rows


def save_marker(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class LiveRelay:
    def __init__(self):
        self.runs = {}

    def update(self, output: Path):
        if (output / "wandb-uploaded.json").exists():
            return
        if not all((output / name).exists() for name in ("config.json", "wandb.json")):
            return
        config = json.loads((output / "config.json").read_text())
        settings = config["train_config"]
        if output not in self.runs:
            identity = json.loads((output / "wandb.json").read_text())
            run = wandb.init(
                project=settings["wandb_project"],
                entity=settings.get("wandb_entity"),
                group=settings["wandb_group"],
                id=identity["id"],
                resume="allow",
                reinit="create_new",
                name=f"{settings['backbone']}-{output.name}",
                job_type="sweep-trial",
                tags=[settings["backbone"], "optuna", "running"],
                mode="online",
                dir=str(output.resolve()),
                config={
                    **config,
                    "logging": {
                        "uploaded_from_offline_json": True,
                        "live_epoch_relay": True,
                        "system_monitoring": "remote records retained locally; local relay disabled",
                    },
                },
                settings=wandb.Settings(x_disable_stats=True, console="off"),
            )
            configure_metrics(run)
            run.summary["relay/status"] = "streaming"
            self.runs[output] = run
            run.save(
                str((output / "config.json").resolve()),
                base_path=str(output.resolve()),
                policy="now",
            )
        run = self.runs[output]
        history = read_history(output / "metrics.jsonl")
        next_step = max(run.step, run.starting_step)
        durations = []
        for row in history:
            durations.append(row["timing"]["epoch_seconds"])
            if row["epoch"] < next_step:
                continue
            row["timing"].update(
                estimate_runtime(
                    durations,
                    settings["epochs"],
                    row["timing"]["elapsed_total_seconds"],
                )
            )
            # Explicit commit flushes the latest epoch without waiting for the next one.
            run.log(wandb_metrics(row), step=row["epoch"], commit=True)
        if history:
            run.summary["relay/last_epoch"] = history[-1]["epoch"]
            best = min(
                history, key=lambda row: row["validation"]["nll"]["calories_kcal"]
            )
            run.summary["val/running_best_calorie_nll"] = best["validation"]["nll"][
                "calories_kcal"
            ]
            run.summary["val/running_best_epoch"] = best["epoch"]
        identity = {
            "id": run.id,
            "url": run.url,
            "last_epoch": history[-1]["epoch"] if history else 0,
        }
        save_marker(output / "wandb-live.json", identity)
        summary_path = output / "summary.json"
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text())
        if len(history) < summary["epochs"]:
            return  # rsync can observe the summary after its earlier metrics snapshot.
        if len(history) != summary["epochs"]:
            raise ValueError("Final summary and epoch history disagree")
        run.summary.update(wandb_summary(summary))
        status = "pruned" if summary.get("stopped_early") else "complete"
        run.summary["relay/status"] = status
        run.tags = (settings["backbone"], "optuna", status)
        run.finish()
        del self.runs[output]
        save_marker(output / "wandb-uploaded.json", identity)
        print(f"Finalized {output}: {identity['url']}", flush=True)

    def close(self):
        for run in self.runs.values():
            run.summary["relay/status"] = "relay_stopped_without_source_summary"
            run.finish(exit_code=1)
        self.runs.clear()
