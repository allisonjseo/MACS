"""Prediction backends. Each exposes predict(image) -> Nutrition."""

import asyncio
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    CodexConfig,
    LocalImageInput,
    Sandbox,
    TextInput,
)

from schemas import FIELDS, SCHEMA, ImageInput, Nutrition, PredictorConfig


class MeanConfig(PredictorConfig):
    backend: Literal["mean"] = "mean"
    values: Nutrition


class CodexPredictorConfig(PredictorConfig):
    backend: Literal["codex"] = "codex"
    model: str | None
    prompt: str
    sdk_version: str
    runtime_version: str


class SiftPredictorConfig(PredictorConfig):
    """Reproducible settings for the classical image-retrieval baseline."""

    backend: Literal["sift"] = "sift"
    reference_count: int
    neighbors: int
    ratio_threshold: float
    max_features: int
    fallback: Nutrition


class Predictor(ABC):
    """Common interface for image-to-nutrition baselines."""

    config: PredictorConfig
    last_metadata: Any = None

    @abstractmethod
    def predict(self, image: ImageInput) -> Nutrition: ...


PROMPT = (
    "Estimate the total nutrition of ALL food shown in the attached photograph. "
    "Infer portion sizes from the image. Return your best point estimates of "
    "calories in kcal and protein, carbohydrates, and fat in grams. "
    "Use only the photograph; do not search, inspect other files, or use tools. "
    "Return only a JSON object with these numeric keys: " + ", ".join(FIELDS) + "."
)


class MeanPredictor(Predictor):
    """A fixed prediction computed from training labels only."""

    config: MeanConfig

    def __init__(self, values: Nutrition):
        self.config = MeanConfig(values=values)

    def predict(self, image: ImageInput) -> Nutrition:
        return self.config.values.model_copy()


@dataclass(frozen=True)
class _SiftReference:
    id: str
    target: Nutrition
    descriptors: np.ndarray[Any, Any] | None


class SiftPredictor(Predictor):
    """Retrieve visually similar, labeled training dishes using SIFT matches.

    This is a classical computer-vision baseline: it learns no weights.  Each
    query is assigned the similarity-weighted nutrition of its closest training
    references.  A fixed training-label mean is used only when no reliable SIFT
    matches are available.
    """

    config: SiftPredictorConfig
    detector: Any
    matcher: Any

    def __init__(
        self,
        references: list[tuple[str, ImageInput, Nutrition]],
        fallback: Nutrition,
        *,
        neighbors: int = 5,
        ratio_threshold: float = 0.75,
        max_features: int = 500,
    ):
        if not references:
            raise ValueError("SIFT predictor needs at least one training reference")
        if neighbors < 1:
            raise ValueError("SIFT neighbors must be at least 1")
        if not 0 < ratio_threshold < 1:
            raise ValueError("SIFT ratio threshold must be between 0 and 1")
        if max_features < 1:
            raise ValueError("SIFT max features must be at least 1")
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Installed OpenCV build does not provide SIFT")
        self.detector = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references = [
            _SiftReference(id, target, self._descriptors(image))
            for id, image, target in references
        ]
        self.config = SiftPredictorConfig(
            reference_count=len(references),
            neighbors=neighbors,
            ratio_threshold=ratio_threshold,
            max_features=max_features,
            fallback=fallback,
        )

    def _descriptors(self, image: ImageInput) -> np.ndarray[Any, Any] | None:
        encoded = np.frombuffer(image.data, dtype=np.uint8)
        grayscale = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise ValueError("SIFT predictor received an invalid image")
        height, width = grayscale.shape
        longest_side = max(height, width)
        if longest_side > 512:
            scale = 512 / longest_side
            grayscale = cv2.resize(
                grayscale,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        _, descriptors = self.detector.detectAndCompute(grayscale, None)
        return descriptors

    def _similarity(
        self, query: np.ndarray[Any, Any], reference: np.ndarray[Any, Any]
    ) -> float:
        matches = self.matcher.knnMatch(query, reference, k=2)
        good_matches = sum(
            1
            for pair in matches
            if len(pair) == 2
            and pair[0].distance < self.config.ratio_threshold * pair[1].distance
        )
        # Normalizing prevents images with many texture keypoints from winning
        # purely because they expose more possible correspondences.
        return good_matches / sqrt(len(query) * len(reference))

    def predict(self, image: ImageInput) -> Nutrition:
        query = self._descriptors(image)
        if query is None or len(query) < 2:
            self.last_metadata = {
                "query_keypoints": 0 if query is None else len(query),
                "neighbors": [],
            }
            return self.config.fallback.model_copy()
        scored = [
            (self._similarity(query, reference.descriptors), reference)
            for reference in self.references
            if reference.descriptors is not None and len(reference.descriptors) >= 2
        ]
        scored = [(score, reference) for score, reference in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        nearest = scored[: self.config.neighbors]
        self.last_metadata = {
            "query_keypoints": len(query),
            "neighbors": [
                {"id": reference.id, "similarity": score}
                for score, reference in nearest
            ],
        }
        if not nearest:
            return self.config.fallback.model_copy()
        weight = sum(score for score, _ in nearest)
        return Nutrition(
            **{
                field: sum(
                    score * getattr(reference.target, field)
                    for score, reference in nearest
                )
                / weight
                for field in FIELDS
            }
        )


class CodexPredictor(Predictor):
    """A fresh, image-only thread through the official Codex Python SDK."""

    config: CodexPredictorConfig

    def __init__(self, model: str | None, timeout: int, *, prompt: str = PROMPT):
        self.timeout = timeout
        self.config = CodexPredictorConfig(
            model=model,
            prompt=prompt,
            sdk_version=version("openai-codex"),
            runtime_version=version("openai-codex-cli-bin"),
        )

    def predict(self, image: ImageInput) -> Nutrition:
        with tempfile.TemporaryDirectory(prefix="nutrition-codex-") as name:
            root = Path(name)
            extension = image.media_type.split("/", 1)[1]
            photo = root / f"food.{extension}"
            photo.write_bytes(image.data)

            async def infer():
                async with AsyncCodex(CodexConfig(cwd=str(root))) as codex:
                    thread = await codex.thread_start(
                        model=self.config.model,
                        cwd=str(root),
                        ephemeral=True,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        config={
                            "web_search": "disabled",
                            "features": dict.fromkeys(
                                (
                                    "shell_tool unified_exec view_image apps plugins memories "
                                    "multi_agent browser_use computer_use code_mode "
                                    "image_generation skill_search hooks"
                                ).split(),
                                False,
                            ),
                        },
                    )
                    return await thread.run(
                        [TextInput(self.config.prompt), LocalImageInput(str(photo))],
                        output_schema=SCHEMA,
                    )

            async def bounded_infer():
                return await asyncio.wait_for(infer(), timeout=self.timeout)

            result = asyncio.run(bounded_infer())
            self.last_metadata = result
            if result.status.value != "completed":
                raise RuntimeError(f"Codex turn {result.status.value}: {result.error}")
            for item in result.items:
                if item.root.type not in {"userMessage", "agentMessage", "reasoning"}:
                    raise RuntimeError("Image-only run used a tool: " + item.root.type)
            return Nutrition.model_validate_json(result.final_response or "")
