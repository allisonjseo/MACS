"""Run artifact persistence and Git provenance, without model inference."""

import json
import subprocess
from pathlib import Path

import pytest

from eval import score
from predictors import MeanConfig
from run import EvaluationRun, GitRevision, RunConfig
from schemas import EvaluationRecord, Nutrition


def config(project_dir: Path) -> RunConfig:
    return RunConfig.capture(
        predictor=MeanConfig(
            values=Nutrition(calories_kcal=100, protein_g=10, carbs_g=20, fat_g=5)
        ),
        manifest=b"manifest bytes",
        sample_ids=["sample-1", "sample-2"],
        timeout_seconds=5,
        project_dir=project_dir,
    )


def test_git_records_evaluator_commit_and_uncommitted_changes(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    (repo / "baseline.py").write_text("# baseline\n")
    git("add", "baseline.py")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "-m",
        "Initial baseline",
    )
    expected_commit = git("rev-parse", "HEAD")
    # Launching elsewhere must not pick up the wrong checkout.
    monkeypatch.chdir(tmp_path)
    assert config(repo).git == GitRevision(commit=expected_commit, dirty=False)
    (repo / "baseline.py").write_text("# changed baseline\n")
    assert config(repo).git == GitRevision(commit=expected_commit, dirty=True)
    git("restore", "baseline.py")
    (repo / "new-baseline.py").write_text("# untracked baseline\n")
    assert config(repo).git == GitRevision(commit=expected_commit, dirty=True)


def test_missing_git_does_not_prevent_evaluation(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("run.subprocess.run", unavailable)
    assert config(tmp_path).git == GitRevision(commit=None, dirty=None)


def test_source_without_git_metadata(tmp_path):
    assert config(tmp_path).git == GitRevision(commit=None, dirty=None)


def test_run_checkpoints_results_and_exports_successes_and_failures(tmp_path):
    cfg = config(tmp_path)
    recording = EvaluationRun(cfg, score, output=tmp_path / "results")
    saved_config = RunConfig.model_validate_json(
        (recording.output / "config.json").read_text()
    )
    assert saved_config.predictor.backend == cfg.predictor.backend
    assert saved_config.git == cfg.git
    target = Nutrition(calories_kcal=200, protein_g=20, carbs_g=30, fat_g=10)
    success = EvaluationRecord(
        id="sample-1",
        target=target,
        status="ok",
        latency_seconds=1.0,
        prediction=Nutrition(calories_kcal=100, protein_g=20, carbs_g=30, fat_g=10),
    )
    failure = EvaluationRecord(
        id="sample-2",
        target=target,
        status="error",
        latency_seconds=2.0,
        error="RuntimeError: baseline failed",
    )
    recording.record(success)
    # Artifacts are readable after each record, so interruption preserves progress.
    assert json.loads((recording.output / "summary.json").read_text())["attempted"] == 1
    assert json.loads((recording.output / "predictions.json").read_text()) == [
        success.model_dump(mode="json", exclude_none=True)
    ]
    recording.record(failure)
    rows = json.loads((recording.output / "predictions.json").read_text())
    assert rows == [
        row.model_dump(mode="json", exclude_none=True) for row in (success, failure)
    ]
    assert recording.records == [success, failure]
    summary = json.loads((recording.output / "summary.json").read_text())
    assert summary["attempted"] == 2
    assert summary["failed"] == 1
    assert summary["metrics"]["calories_kcal"]["mae"] == 100
    assert "prediction" not in rows[1]
    assert rows[1]["error"] == failure.error
    assert set(path.suffix for path in recording.output.iterdir()) == {".json"}
    with pytest.raises(FileExistsError):
        EvaluationRun(cfg, score, output=recording.output)
    assert json.loads((recording.output / "config.json").read_text()) == cfg.model_dump(
        mode="json"
    )
