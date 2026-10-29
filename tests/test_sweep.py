"""Selection semantics and the actual Optuna pruning protocol."""

import json
from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("timm")
import optuna
from optuna.trial import TrialState

from sweep import export_study, make_plan, pruning_callback
from vision import TrainConfig


def test_matched_plan_includes_reference_and_preserves_frozen_compiled_settings():
    chosen = make_plan(Path("configs/sweep-gaussian.toml"))
    designs = []
    for backbone, entry in chosen["studies"].items():
        designs.append([row["params"] for row in entry["trials"]])
        assert len(entry["trials"]) == 12
        assert entry["trials"][0]["params"] == {
            "head_lr": 0.001,
            "weight_decay": 0.01,
            "hidden_dim": 256,
            "dropout": 0.1,
        }
        for row in entry["trials"]:
            config = TrainConfig.model_validate(row["train_config"])
            assert config.backbone == backbone
            assert config.freeze_backbone and config.compile_model
            assert config.loss == "gaussian_nll"
            assert config.selection_metric == "calorie_nll"
            assert config.epochs == 30 and config.seed == 42
            assert config.batch_size == 32 and config.precision == "float32"
    assert designs[0] == designs[1] == designs[2]


def test_real_pruner_cuts_bad_trial_at_ten_and_compares_best_so_far(tmp_path):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10, reduction_factor=2
        ),
    )
    stopped = []

    def objective(trial):
        callback = pruning_callback(trial, {"pruning_epochs": [10, 20]})
        # Trial 2 is best at epoch 1 but worsens: comparisons use its best checkpoint.
        for epoch in range(1, 31):
            score = (
                [10, 11, 9][trial.number] if epoch == 1 else [10, 11, 12][trial.number]
            )
            if callback(
                {"epoch": epoch, "validation": {"nll": {"calories_kcal": score}}}
            ):
                stopped.append((trial.number, epoch))
                raise optuna.TrialPruned()
        return [10, 11, 9][trial.number]

    study.optimize(objective, n_trials=3)
    assert stopped == [(1, 10)]
    assert [t.state for t in study.trials] == [
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.COMPLETE,
    ]
    assert study.best_trial.number == 2
    export_study(study, tmp_path, 0)
    assert json.loads((tmp_path / "study-summary.json").read_text())["best_trial"] == 2
