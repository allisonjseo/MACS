"""Capture run provenance and persist configuration, predictions, and metrics."""

import hashlib
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, TypeAdapter

from schemas import EvaluationRecord, EvaluationSummary, PredictorConfig

PROJECT_DIR = Path(__file__).resolve().parent
RECORDS_ADAPTER = TypeAdapter(list[EvaluationRecord])


def write_json(path: Path, value: BaseModel | list[EvaluationRecord]) -> None:
    if isinstance(value, BaseModel):
        content = value.model_dump_json(indent=2).encode()
    else:
        content = RECORDS_ADAPTER.dump_json(value, indent=2, exclude_none=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content + b"\n")
    temporary.replace(path)


class GitRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str | None = None
    dirty: bool | None = None

    @classmethod
    def capture(cls, project_dir: Path) -> "GitRevision":
        """Inspect the evaluator's checkout, independent of the launch directory."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return cls()
        return cls(commit=commit, dirty=bool(status.strip()))


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictor: SerializeAsAny[PredictorConfig]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids: list[str]
    timeout_seconds: int = Field(gt=0)
    started_at: str
    dependencies: dict[str, str]
    git: GitRevision

    @classmethod
    def capture(
        cls,
        *,
        predictor: PredictorConfig,
        manifest: bytes,
        sample_ids: list[str],
        timeout_seconds: int,
        project_dir: Path = PROJECT_DIR,
    ) -> "RunConfig":
        return cls(
            predictor=predictor,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            sample_ids=sample_ids,
            timeout_seconds=timeout_seconds,
            started_at=datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            dependencies={
                name: version(name)
                for name in (
                    "numpy",
                    "pydantic",
                    "polars",
                    "openai-codex",
                    "opencv-python-headless",
                    "typer",
                )
            },
            git=GitRevision.capture(project_dir),
        )


class EvaluationRun:
    """Own a fresh run directory and checkpoint each result as it arrives."""

    def __init__(
        self,
        config: RunConfig,
        summarize: Callable[[list[EvaluationRecord]], EvaluationSummary],
        output: Path | None = None,
    ):
        self.config = config
        self.output = output or Path("runs") / (
            config.started_at + "-" + config.predictor.backend
        )
        self.output.mkdir(parents=True, exist_ok=False)
        self.records: list[EvaluationRecord] = []
        self.summarize = summarize
        self.summary = summarize(self.records)
        write_json(self.output / "config.json", config)
        write_json(self.output / "summary.json", self.summary)

    def record(self, row: EvaluationRecord) -> None:
        self.records.append(row)
        write_json(self.output / "predictions.json", self.records)
        self.summary = self.summarize(self.records)
        write_json(self.output / "summary.json", self.summary)
