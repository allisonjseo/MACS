"""Streaming appends, W&B resume steps, and final-summary consistency."""

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("wandb")
from deploy.live_sweep import LiveRelay, read_history


class FakeRun:
    def __init__(self, step=0):
        self.id = "same1234"
        self.url = "https://example.test/runs/same1234"
        self.step = step
        self.starting_step = step
        self.summary = {}
        self.tags = ()
        self.logged = []
        self.finished = []

    def define_metric(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def log(self, data, step, commit):
        assert step >= self.step and commit is True
        self.logged.append((step, data))
        self.step = step + 1

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


def row(epoch):
    return {
        "epoch": epoch,
        "train": {"loss": 2},
        "validation": {"loss": 1, "nll": {"calories_kcal": 6 - epoch / 10}},
        "timing": {"epoch_seconds": 1, "elapsed_total_seconds": epoch + 1},
        "learning_rates": [0.001],
    }


def setup_output(tmp_path):
    output = tmp_path / "trial-003"
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps(
            {
                "train_config": {
                    "backbone": "dino",
                    "epochs": 30,
                    "wandb_project": "project",
                    "wandb_group": "group",
                }
            }
        )
    )
    (output / "wandb.json").write_text(json.dumps({"id": "same1234"}))
    return output


def write_history(output, epochs):
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row(e)) + "\n" for e in epochs)
    )


def test_live_epochs_append_without_reopening_or_duplicate_logging(
    tmp_path, monkeypatch
):
    output = setup_output(tmp_path)
    run = FakeRun()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setattr("deploy.live_sweep.wandb.init", init)
    relay = LiveRelay()
    write_history(output, [1])
    relay.update(output)
    relay.update(output)
    write_history(output, [1, 2])
    relay.update(output)
    assert len(calls) == 1
    assert calls[0]["id"] == "same1234"
    assert calls[0]["reinit"] == "create_new"
    assert [step for step, _ in run.logged] == [1, 2]
    assert run.finished == []
    assert not (output / "wandb-uploaded.json").exists()
    assert run.summary["relay/last_epoch"] == 2
    assert "val/nll/calories_kcal" in run.logged[0][1]
    assert "timing/epoch_seconds" in run.logged[0][1]


def test_resume_uses_server_next_step_and_finalizes_only_after_all_epochs(
    tmp_path, monkeypatch
):
    output = setup_output(tmp_path)
    run = FakeRun(step=2)
    run.step = 0  # The asynchronous history service can initially lag resume metadata.
    monkeypatch.setattr("deploy.live_sweep.wandb.init", lambda **kw: run)
    relay = LiveRelay()
    write_history(output, [1])
    (output / "summary.json").write_text(
        json.dumps({"epochs": 2, "stopped_early": True})
    )
    relay.update(output)
    assert not run.logged and not run.finished
    write_history(output, [1, 2])
    relay.update(output)
    assert [step for step, _ in run.logged] == [2]
    assert run.finished == [0]
    assert run.tags == ("dino", "optuna", "pruned")
    assert (output / "wandb-uploaded.json").exists()
    relay.update(output)
    assert run.finished == [0]


def test_partial_append_is_deferred_but_corrupt_completed_records_rejected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row(1)) + '\n{"epoch":2')
    assert [r["epoch"] for r in read_history(path)] == [1]
    path.write_text(json.dumps(row(1)) + '\n{"epoch":2\n')
    with pytest.raises(json.JSONDecodeError):
        read_history(path)
    path.write_text(json.dumps(row(2)) + "\n")
    with pytest.raises(ValueError, match="contiguous"):
        read_history(path)


def test_rsync_filters_exclude_live_checkpoint_temps_and_include_nested_metrics(
    tmp_path, monkeypatch
):
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("rsync is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "deploy"))
    filters = importlib.import_module("watch_sweep").SYNC_FILTERS
    source = tmp_path / "source"
    trial = source / "resnet/trial-000"
    (trial / "wandb").mkdir(parents=True)
    for name in [
        "config.json",
        "metrics.jsonl",
        "summary.json",
        "best.pt",
        "last.pt.tmp",
        "study.db",
        "config.json.tmp",
        "wandb/debug.json",
    ]:
        (trial / name).write_text("test")
    target = tmp_path / "target"
    subprocess.run([rsync, "-az", *filters, str(source) + "/", str(target)], check=True)
    assert {p.name for p in (target / "resnet/trial-000").iterdir()} == {
        "config.json",
        "metrics.jsonl",
        "summary.json",
    }
