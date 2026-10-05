"""Shared data contracts and generated prediction JSON schema."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonnegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class Nutrition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calories_kcal: NonnegativeNumber
    protein_g: NonnegativeNumber
    carbs_g: NonnegativeNumber
    fat_g: NonnegativeNumber


FIELDS = tuple(Nutrition.model_fields)
SCHEMA = Nutrition.model_json_schema()


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: Nutrition


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    source: str
    license: str
    split: str
    seed: int
    n: int = Field(gt=0)
    missing_overhead_images: list[str]
    training_mean: Nutrition
    training_label_count: int = Field(gt=0)
    samples: list[Sample]
    references: list[Sample] = Field(default_factory=list)
    missing_reference_images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_sample_count(self):
        if len(self.samples) != self.n:
            raise ValueError("Manifest sample count does not match n")
        if len({sample.id for sample in self.samples}) != self.n:
            raise ValueError("Manifest sample IDs must be unique")
        sample_ids = {sample.id for sample in self.samples}
        reference_ids = {sample.id for sample in self.references}
        if len(reference_ids) != len(self.references):
            raise ValueError("Manifest reference IDs must be unique")
        if sample_ids & reference_ids:
            raise ValueError("Test samples and training references must not overlap")
        return self


class ImageInput(BaseModel):
    """Encoded image content, independent of dataset storage or API transport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: bytes
    media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"]


class PredictorConfig(BaseModel):
    """Common configuration identity; backends add their own typed fields."""

    model_config = ConfigDict(extra="allow", frozen=True)

    backend: str


class EvaluationRecord(BaseModel):
    """One saved result, with optional prediction or failure diagnostics."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    target: Nutrition
    status: Literal["ok", "error"] = "error"
    latency_seconds: NonnegativeNumber = 0.0
    prediction: Nutrition | None = None
    error: str | None = None
    backend_metadata: Any = None


class Metric(BaseModel):
    mae: NonnegativeNumber | None = None


class CalorieAccuracy(BaseModel):
    count: int
    eligible_successes: int
    percent: float | None


class EvaluationSummary(BaseModel):
    attempted: int
    succeeded: int
    failed: int
    metrics: dict[str, Metric]
    calories_within_20pct: CalorieAccuracy
    mean_latency_seconds: NonnegativeNumber | None
