#!/usr/bin/env python3
"""Plan exact, matched Optuna trials, then execute one frozen-backbone study."""

import csv
import gc
import hashlib
import json
import math
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import optuna
import torch
import typer
from optuna.trial import TrialState

from train import load_config, run_training, save_json
from vision import BACKBONES

app = typer.Typer(no_args_is_help=True)
PROJECT = Path(__file__).resolve().parent
SOURCES = (
    "sweep.py",
    "train.py",
    "vision.py",
    "vision_data.py",
    "tracking.py",
    "timing.py",
    "determinism.py",
    "run.py",
    "uv.lock",
    "pyproject.toml",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suggest(trial, search):
    params = {}
    for name, domain in search.items():
        if "choices" in domain:
            params[name] = trial.suggest_categorical(name, domain["choices"])
        else:
            params[name] = trial.suggest_float(
                name, domain["low"], domain["high"], log=domain.get("log", False)
            )
    return params


def overrides_for(values):
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def make_plan(spec: Path):
    settings = tomllib.loads(spec.read_text())
    if (
        settings["pruning_epochs"] != [10, 20]
        or settings["reduction_factor"] != 2
        or settings["max_epochs"] != 30
        or settings["trials_per_backbone"] < 2
        or set(settings["search"])
        != {"head_lr", "weight_decay", "hidden_dim", "dropout"}
    ):
        raise ValueError("Unsupported pilot protocol; update implementation explicitly")
    studies = {}
    for backbone in settings["backbones"]:
        reference_path = PROJECT / f"configs/{backbone}-gaussian.toml"
        reference, _ = load_config(reference_path, [])
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=settings["seed"]),
            direction="minimize",
        )
        baseline = {name: getattr(reference, name) for name in settings["search"]}
        study.enqueue_trial(baseline)
        trials = []
        for _ in range(settings["trials_per_backbone"]):
            trial = study.ask()
            params = suggest(trial, settings["search"])
            overrides = {**settings["overrides"], **params}
            chosen, changes = load_config(reference_path, overrides_for(overrides))
            if not (
                chosen.freeze_backbone
                and chosen.pretrained
                and chosen.compile_model
                and chosen.compile_backend == "inductor"
                and chosen.compile_fullgraph
                and chosen.compile_dynamic
                and chosen.selection_metric == "calorie_nll"
                and chosen.loss == "gaussian_nll"
                and chosen.epochs == settings["max_epochs"]
            ):
                raise ValueError(
                    "Pilot requires compiled frozen Gaussian reference models"
                )
            trials.append(
                {
                    "number": trial.number,
                    "params": params,
                    "overrides": overrides,
                    "train_config": chosen.model_dump(mode="json"),
                    "changes_from_reference": changes,
                }
            )
            study.tell(trial, state=TrialState.FAIL)
        studies[backbone] = {
            "reference": str(reference_path.relative_to(PROJECT)),
            "reference_sha256": sha256(reference_path),
            "backbone_id": BACKBONES[backbone],
            "trials": trials,
        }
    return {
        "spec": str(spec.resolve()),
        "spec_sha256": sha256(spec),
        "protocol": settings,
        "sampler": "seeded RandomSampler; preplanned matched proposals, not adaptive TPE",
        "pruner": "SuccessiveHalvingPruner(min_resource=10, reduction_factor=2, min_early_stopping_rate=0, bootstrap_count=0)",
        "pruning_score": "best calorie-only validation NLL through current epoch",
        "checkpoint_selection": "minimum calorie-only validation NLL, including pruned trials",
        "sources": {name: sha256(PROJECT / name) for name in SOURCES},
        "studies": studies,
    }


@app.command()
def plan(
    spec: Path = Path("configs/sweep-gaussian.toml"),
    output: Path = Path("runs/optuna-pilot-plan.json"),
):
    """Materialize every exact configuration for review before any GPU launch."""
    if output.exists():
        raise ValueError("Plan exists; use a new path to preserve provenance")
    chosen = make_plan(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, chosen)
    print(json.dumps(chosen, indent=2))


def pruning_callback(trial, protocol):
    best = math.inf

    def after_epoch(row):
        nonlocal best
        score = float(row["validation"]["nll"]["calories_kcal"])
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite calorie validation NLL")
        best = min(best, score)
        epoch = row["epoch"]
        trial.report(best, step=epoch)
        if epoch not in protocol["pruning_epochs"]:
            return False
        prune = trial.should_prune()
        # Trial zero is the full-budget reference; it still populates rung scores.
        decision = prune and not trial.user_attrs.get("reference_trial", False)
        print(
            json.dumps(
                {
                    "optuna_trial": trial.number,
                    "epoch": epoch,
                    "val/best_calorie_nll": best,
                    "prune": decision,
                }
            ),
            flush=True,
        )
        return decision

    return after_epoch


def export_study(study, output, started):
    rows = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": json.dumps(trial.params),
            "user_attrs": json.dumps(trial.user_attrs),
            "intermediate_values": json.dumps(trial.intermediate_values),
        }
        for trial in study.trials
    ]
    temporary = output / "trials.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "number",
                "state",
                "value",
                "params",
                "user_attrs",
                "intermediate_values",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "trials.csv")
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    save_json(
        output / "study-summary.json",
        {
            "study_name": study.study_name,
            "counts": {
                state.name: sum(t.state == state for t in study.trials)
                for state in TrialState
            },
            "best_trial": study.best_trial.number if complete else None,
            "best_calorie_nll": study.best_value if complete else None,
            "best_params": study.best_params if complete else None,
            "timing": {"study_seconds": time.perf_counter() - started},
            "objective": "best calorie-only validation Gaussian NLL, nats/example",
            "test_evaluated": False,
        },
    )


@app.command()
def run(plan_file: Path, backbone: str, manifest: Path, output: Path):
    """Execute a fresh study from a reviewed immutable plan; no automatic restarts."""
    chosen = json.loads(plan_file.read_text())
    settings = chosen["protocol"]
    entry = chosen["studies"][backbone]
    if sha256(manifest) != settings["manifest_sha256"]:
        raise ValueError("Training population differs from the reviewed plan")
    for name, digest in chosen["sources"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"Source changed since planning: {name}")
    reference = PROJECT / entry["reference"]
    if sha256(reference) != entry["reference_sha256"]:
        raise ValueError("Reference configuration changed since planning")
    if output.exists():
        raise ValueError("Output exists; refusing to restart or overwrite a study")
    output.mkdir(parents=True)
    save_json(output / "plan.json", chosen)
    study = optuna.create_study(
        study_name=f"gaussian-pilot-{backbone}",
        storage=f"sqlite:///{output.resolve() / 'study.db'}",
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=settings["seed"]),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10,
            reduction_factor=2,
            min_early_stopping_rate=0,
            bootstrap_count=0,
        ),
    )
    for proposal in entry["trials"]:
        study.enqueue_trial(
            proposal["params"], user_attrs={"reference_trial": proposal["number"] == 0}
        )
    started = time.perf_counter()

    def objective(trial):
        params = suggest(trial, settings["search"])
        proposal = entry["trials"][trial.number]
        if params != proposal["params"]:
            raise ValueError("Optuna parameters differ from reviewed proposal")
        overrides = overrides_for(proposal["overrides"])
        resolved, _ = load_config(reference, overrides)
        if resolved.model_dump(mode="json") != proposal["train_config"]:
            raise ValueError("Resolved settings differ from reviewed proposal")
        trial_output = output / f"trial-{trial.number:03d}"
        trial.set_user_attr("output", str(trial_output))
        summary = None
        try:
            summary = run_training(
                reference,
                manifest,
                trial_output,
                overrides,
                epoch_callback=pruning_callback(trial, settings),
                experiment_metadata={
                    "study": study.study_name,
                    "trial": trial.number,
                    "plan_sha256": sha256(plan_file),
                    "objective": settings["objective"],
                    "pruner": chosen["pruner"],
                    "pruning_score": chosen["pruning_score"],
                    "sampler": chosen["sampler"],
                    "optuna_version": version("optuna"),
                },
            )
        finally:
            gc.collect()
            torch.compiler.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        assert summary is not None
        trial.set_user_attr("epochs", summary["epochs"])
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("best_calorie_nll", summary["best_selection_score"])
        trial.set_user_attr("timing/total_seconds", summary["total_seconds"])
        if summary["stopped_early"]:
            raise optuna.TrialPruned(f"Stopped at epoch {summary['epochs']}")
        return summary["best_selection_score"]

    try:
        study.optimize(
            objective,
            n_trials=settings["trials_per_backbone"],
            n_jobs=1,
            callbacks=[lambda study, trial: export_study(study, output, started)],
        )
    finally:
        export_study(study, output, started)


if __name__ == "__main__":
    app()
