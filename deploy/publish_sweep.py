"""Publish an Optuna plan/status overview into the trials' shared W&B group."""

import json
import sys
from pathlib import Path
from typing import Any

import wandb


def publish(plan_file: Path, results: Path):
    plan = json.loads(plan_file.read_text())
    settings = next(iter(plan["studies"].values()))["trials"][0]["train_config"]
    results.mkdir(parents=True, exist_ok=True)
    marker = results / "wandb-overview.json"
    identity = json.loads(marker.read_text()) if marker.exists() else {}
    rows: list[Any] = []
    counts = dict.fromkeys(["PLANNED", "RUNNING", "COMPLETE", "PRUNED"], 0)
    best = {}
    total_seconds = 0.0
    for backbone, study in plan["studies"].items():
        for proposal in study["trials"]:
            output = results / backbone / f"trial-{proposal['number']:03d}"
            path = output / "summary.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            state = (
                ("PRUNED" if summary.get("stopped_early") else "COMPLETE")
                if summary
                else "RUNNING"
                if (output / "config.json").exists()
                else "PLANNED"
            )
            counts[state] += 1
            params = proposal["params"]
            url_file = output / "wandb-uploaded.json"
            url = json.loads(url_file.read_text())["url"] if url_file.exists() else None
            score = summary.get("best_selection_score")
            seconds = summary.get("total_seconds")
            if state == "COMPLETE":
                assert score is not None
                best[backbone] = min(best.get(backbone, float("inf")), score)
            total_seconds += seconds or 0
            rows.append(
                [
                    backbone,
                    proposal["number"],
                    params["head_lr"],
                    params["weight_decay"],
                    params["hidden_dim"],
                    params["dropout"],
                    state,
                    summary.get("epochs"),
                    summary.get("best_epoch"),
                    score,
                    seconds,
                    url,
                ]
            )
    with wandb.init(
        project=settings["wandb_project"],
        entity=settings.get("wandb_entity"),
        group=settings["wandb_group"],
        name="optuna-pilot-overview",
        job_type="sweep-overview",
        tags=["optuna", "overview"],
        id=identity.get("id"),
        resume="allow" if identity else None,
        config=plan,
        mode="online",
        dir=str(results.resolve()),
        settings=wandb.Settings(x_disable_stats=True),
    ) as run:
        table = wandb.Table(
            columns=[
                "backbone",
                "trial",
                "head_lr",
                "weight_decay",
                "hidden_dim",
                "dropout",
                "state",
                "epochs",
                "best_epoch",
                "val/calorie_nll",
                "timing/total_seconds",
                "run_url",
            ],
            data=rows,
        )
        run.log({"sweep/trials": table})
        run.summary.update(
            {f"sweep/{key.lower()}": value for key, value in counts.items()}
        )
        run.summary["timing/finished_trial_seconds"] = total_seconds
        run.summary["sweep/planned_trials"] = len(rows)
        run.summary["sweep/status_source"] = (
            "fetched JSON snapshots; not live Slurm state"
        )
        for backbone, score in best.items():
            run.summary[f"val/best_calorie_nll/{backbone}"] = score
        artifact = wandb.Artifact("optuna-gaussian-pilot-plan", type="sweep-plan")
        artifact.add_file(str(plan_file.resolve()), name="plan.json")
        run.log_artifact(artifact)
        url = run.url
        assert url is not None
        identity = {
            "id": run.id,
            "url": url,
            "group": settings["wandb_group"],
            "group_url": url.rsplit("/runs/", 1)[0]
            + "/groups/"
            + settings["wandb_group"],
        }
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
