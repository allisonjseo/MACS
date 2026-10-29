#!/usr/bin/env python3
"""Plan exact, matched Optuna trials, then execute one frozen-backbone study."""

import csv
import fcntl
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
    "checkpointing.py",
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


def pruning_callback(trial, protocol, initial_best=math.inf):
    best = initial_best

    def after_epoch(row):
        nonlocal best
        score = float(row["validation"]["nll"]["calories_kcal"])
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite calorie validation NLL")
        best = min(best, score)
        epoch = row["epoch"]
        frozen = trial.study.trials[trial.number]
        if epoch in frozen.intermediate_values:
            if not math.isclose(
                frozen.intermediate_values[epoch], best, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "Resumed validation score disagrees with Optuna history"
                )
        else:
            trial.report(best, step=epoch)
        if epoch not in protocol["pruning_epochs"]:
            return False
        key = f"prune_at_epoch_{epoch}"
        if key in frozen.user_attrs:
            return frozen.user_attrs[key]
        rung = protocol["pruning_epochs"].index(epoch)
        if f"completed_rung_{rung}" in frozen.system_attrs:
            raise ValueError(
                "Interrupted while persisting a pruning decision; manual inspection required"
            )
        prune = trial.should_prune()
        # Trial zero is the full-budget reference; it still populates rung scores.
        decision = prune and not trial.user_attrs.get("reference_trial", False)
        trial.set_user_attr(key, decision)
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
def refresh_plan(plan_file: Path, output: Path):
    """Explicitly migrate source provenance, preserving every reviewed setting/proposal."""
    if output.exists():
        raise ValueError("Migration output already exists")
    chosen = json.loads(plan_file.read_text())
    for entry in chosen["studies"].values():
        reference = PROJECT / entry["reference"]
        if sha256(reference) != entry["reference_sha256"]:
            raise ValueError(
                "Reference changed; migration must not change training settings"
            )
        for proposal in entry["trials"]:
            resolved, _ = load_config(reference, overrides_for(proposal["overrides"]))
            if resolved.model_dump(mode="json") != proposal["train_config"]:
                raise ValueError(
                    "Proposal changed; migration must not change training settings"
                )
    chosen["source_migration"] = {
        "previous_plan_sha256": sha256(plan_file),
        "previous_sources": chosen["sources"],
        "reason": "full-state epoch checkpoints and study resume; no training/pruning parameter changes",
    }
    chosen["sources"] = {name: sha256(PROJECT / name) for name in SOURCES}
    save_json(output, chosen)
    print(
        json.dumps(
            {
                "plan": str(output),
                "sha256": sha256(output),
                "changes_to_training_config": {},
            },
            indent=2,
        )
    )


def drive_study(study, objective, output, started, expected_trials):
    """Resume an interrupted RUNNING trial before asking for the next queued trial."""
    try:
        while True:
            trials = study.trials
            if len(trials) != expected_trials:
                raise ValueError("Study trial count differs from reviewed plan")
            if any(t.state == TrialState.FAIL for t in trials):
                raise ValueError(
                    "Study contains a failed trial; inspect it instead of silently retrying"
                )
            running = [t for t in trials if t.state == TrialState.RUNNING]
            if len(running) > 1:
                raise ValueError("Multiple RUNNING trials in a serial study")
            if running:
                trial = optuna.trial.Trial(study, running[0]._trial_id)
            elif any(t.state == TrialState.WAITING for t in trials):
                trial = study.ask()
            else:
                break
            try:
                value = objective(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=TrialState.PRUNED)
            except (KeyboardInterrupt, SystemExit):
                raise  # Keep RUNNING: a recovery process reconnects to this same ID.
            except Exception:
                study.tell(trial, state=TrialState.FAIL)
                raise
            else:
                study.tell(trial, value)
            export_study(study, output, started)
    finally:
        export_study(study, output, started)


@app.command()
def run(
    plan_file: Path, backbone: str, manifest: Path, output: Path, resume: bool = False
):
    """Run/resume the same reviewed serial study; protect it from concurrent controllers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another controller owns this study") from error
        execute_study(plan_file, backbone, manifest, output, resume)


def execute_study(plan_file, backbone, manifest, output, resume):
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
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Output exists; refusing to restart or overwrite a study")
    if resume:
        recorded = json.loads((output / "plan.json").read_text())
        if (
            recorded["protocol"] != chosen["protocol"]
            or recorded["studies"] != chosen["studies"]
        ):
            raise ValueError(
                "Resumed study settings or proposals differ from original plan"
            )
        if sha256(output / "plan.json") != sha256(plan_file):
            if chosen.get("source_migration", {}).get("previous_plan_sha256") != sha256(
                output / "plan.json"
            ):
                raise ValueError(
                    "Source migration is not linked to the original study plan"
                )
            save_json(output / "plan-before-checkpointing.json", recorded)
    else:
        output.mkdir(parents=True, exist_ok=True)
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
        load_if_exists=resume,
    )
    existing = study.trials
    for recorded in existing:
        proposal = entry["trials"][recorded.number]
        for key, value in recorded.params.items():
            if proposal["params"].get(key) != value:
                raise ValueError("Persisted trial parameters differ from reviewed plan")
        if recorded.system_attrs.get("fixed_params") != proposal["params"]:
            raise ValueError("Persisted queued parameters differ from reviewed plan")
    for proposal in entry["trials"][len(existing) :]:
        study.enqueue_trial(
            proposal["params"],
            user_attrs={"reference_trial": proposal["number"] == 0},
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
        if (trial_output / "summary.json").exists():
            summary = json.loads((trial_output / "summary.json").read_text())
            if summary["stopped_early"]:
                raise optuna.TrialPruned("Recover finalized pruned trial")
            return summary["best_selection_score"]
        initial_best = math.inf
        if (trial_output / "last.pt").exists():
            checkpoint = torch.load(
                trial_output / "last.pt", map_location="cpu", weights_only=True
            )
            if "training_state" not in checkpoint:
                raise ValueError("Legacy weights-only trial cannot faithfully resume")
            initial_best = checkpoint["training_state"]["progress"].get(
                "best", math.inf
            )
        summary = None
        try:
            summary = run_training(
                reference,
                manifest,
                trial_output,
                overrides,
                epoch_callback=pruning_callback(trial, settings, initial_best),
                resume=trial_output.exists(),
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

    drive_study(study, objective, output, started, settings["trials_per_backbone"])


if __name__ == "__main__":
    app()
