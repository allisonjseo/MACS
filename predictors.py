"""Prediction backends. Each exposes predict(image) -> Nutrition."""

import asyncio
import hashlib
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    CodexConfig,
    LocalImageInput,
    Sandbox,
    TextInput,
)
from tqdm import tqdm

from schemas import (
    FIELDS,
    SCHEMA,
    ImageInput,
    Nutrition,
    NutritionInterval,
    PredictorConfig,
)

SIFT_INTERVAL_FRACTION = 0.25
SIFT_INTERVAL_MINIMUM = Nutrition(
    calories_kcal=50.0, protein_g=3.0, carbs_g=5.0, fat_g=3.0
)


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
    matcher_device: str
    interval_method: Literal["weighted_neighbor_spread"] = "weighted_neighbor_spread"
    interval_fraction: float = SIFT_INTERVAL_FRACTION
    interval_minimum_half_width: Nutrition = SIFT_INTERVAL_MINIMUM
    fallback: Nutrition


class Predictor(ABC):
    """Common interface for image-to-nutrition baselines."""

    config: PredictorConfig
    last_metadata: Any = None
    last_interval: NutritionInterval | None = None

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


class _TorchBFMatcher:
    """A BF-style L2 matcher backed by PyTorch, using CUDA when available."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reference_descriptors: torch.Tensor | None = None
        self.reference_indices: torch.Tensor | None = None

    def add(self, references: list[np.ndarray[Any, Any]]) -> None:
        descriptors = []
        indices = []
        for index, reference in enumerate(references):
            descriptors.append(torch.from_numpy(reference).to(dtype=torch.float32))
            indices.extend([index] * len(reference))
        if descriptors:
            self.reference_descriptors = torch.cat(descriptors).to(self.device)
            self.reference_indices = torch.tensor(indices, device=self.device)

    @staticmethod
    def _ratio_matches(distances: torch.Tensor, ratio_threshold: float) -> torch.Tensor:
        nearest = torch.topk(distances, k=2, dim=1, largest=False).values
        return nearest[:, 0] < ratio_threshold * nearest[:, 1]

    def similarity(
        self,
        query: np.ndarray[Any, Any],
        reference: np.ndarray[Any, Any],
        ratio_threshold: float,
    ) -> int:
        query_tensor = torch.from_numpy(query).to(self.device, dtype=torch.float32)
        reference_tensor = torch.from_numpy(reference).to(
            self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            distances = torch.cdist(query_tensor, reference_tensor, p=2)
            return int(self._ratio_matches(distances, ratio_threshold).sum().item())

    def global_matches(self, query: np.ndarray[Any, Any], ratio_threshold: float):
        if self.reference_descriptors is None or self.reference_indices is None:
            return {}
        query_tensor = torch.from_numpy(query).to(self.device, dtype=torch.float32)
        counts: dict[int, int] = {}
        # Keep the distance matrix bounded while retaining the full index on GPU.
        with torch.inference_mode():
            for query_chunk in query_tensor.split(64):
                distances = torch.cdist(query_chunk, self.reference_descriptors, p=2)
                good = self._ratio_matches(distances, ratio_threshold)
                nearest = torch.topk(distances, k=1, dim=1, largest=False).indices[:, 0]
                for reference_index in self.reference_indices[nearest[good]].tolist():
                    counts[reference_index] = counts.get(reference_index, 0) + 1
        return counts


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
        references: Iterable[tuple[str, ImageInput, Nutrition]],
        fallback: Nutrition,
        *,
        neighbors: int = 5,
        ratio_threshold: float = 0.75,
        max_features: int = 500,
        feature_cache: Path | None = None,
    ):
        if neighbors < 1:
            raise ValueError("SIFT neighbors must be at least 1")
        if not 0 < ratio_threshold < 1:
            raise ValueError("SIFT ratio threshold must be between 0 and 1")
        if max_features < 1:
            raise ValueError("SIFT max features must be at least 1")
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Installed OpenCV build does not provide SIFT")
        self.detector = cv2.SIFT_create(nfeatures=max_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = _TorchBFMatcher(self.device)
        self.max_features = max_features
        self.feature_cache = feature_cache
        if self.feature_cache is not None:
            self.feature_cache.mkdir(parents=True, exist_ok=True)
        self.references = []
        for id, image, target in tqdm(
            references, desc="Extracting SIFT features", unit="reference"
        ):
            self.references.append(_SiftReference(id, target, self._descriptors(image)))
        if not self.references:
            raise ValueError("SIFT predictor needs at least one training reference")
        self.fallback_spread = {
            field: float(
                np.std(
                    [getattr(reference.target, field) for reference in self.references]
                )
            )
            for field in FIELDS
        }
        self.index_references = [
            reference
            for reference in self.references
            if reference.descriptors is not None and len(reference.descriptors) >= 2
        ]
        self.index_matcher = _TorchBFMatcher(self.device)
        if self.index_references:
            self.index_matcher.add(
                [
                    reference.descriptors
                    for reference in self.index_references
                    if reference.descriptors is not None
                ]
            )
        self.config = SiftPredictorConfig(
            reference_count=len(self.references),
            neighbors=neighbors,
            ratio_threshold=ratio_threshold,
            max_features=max_features,
            matcher_device=str(self.device),
            fallback=fallback,
        )

    def _descriptors(self, image: ImageInput) -> np.ndarray[Any, Any] | None:
        cache_path = None
        if self.feature_cache is not None:
            digest = hashlib.sha256(image.data).hexdigest()
            cache_path = self.feature_cache / f"{digest}-f{self.max_features}.npy"
            if cache_path.exists():
                descriptors = np.load(cache_path, allow_pickle=False)
                return descriptors if len(descriptors) else None
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
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp.npy")
            np.save(
                temporary,
                descriptors
                if descriptors is not None
                else np.empty((0, 128), dtype=np.float32),
            )
            temporary.replace(cache_path)
        return descriptors

    def _similarity(
        self, query: np.ndarray[Any, Any], reference: np.ndarray[Any, Any]
    ) -> float:
        good_matches = self.matcher.similarity(
            query, reference, self.config.ratio_threshold
        )
        # Normalizing prevents images with many texture keypoints from winning
        # purely because they expose more possible correspondences.
        return good_matches / sqrt(len(query) * len(reference))

    def cache_features(self, image: ImageInput) -> None:
        """Extract and persist descriptors without running retrieval."""
        self._descriptors(image)

    def _interval(
        self, prediction: Nutrition, spread: dict[str, float]
    ) -> NutritionInterval:
        lower, upper = {}, {}
        for field in FIELDS:
            value = getattr(prediction, field)
            half_width = max(
                spread[field],
                self.config.interval_fraction * value,
                getattr(self.config.interval_minimum_half_width, field),
            )
            lower[field] = max(0.0, value - half_width)
            upper[field] = value + half_width
        return NutritionInterval(lower=Nutrition(**lower), upper=Nutrition(**upper))

    def predict(self, image: ImageInput) -> Nutrition:
        self.last_interval = None
        query = self._descriptors(image)
        if query is None or len(query) < 2:
            self.last_metadata = {
                "query_keypoints": 0 if query is None else len(query),
                "neighbors": [],
            }
            prediction = self.config.fallback.model_copy()
            self.last_interval = self._interval(prediction, self.fallback_spread)
            return prediction
        scored_by_reference = {}
        if self.index_references:
            # One global match operation replaces thousands of per-reference
            # BFMatcher calls. imgIdx identifies the added reference.
            scored_by_reference = self.index_matcher.global_matches(
                query, self.config.ratio_threshold
            )
        scored = [
            (
                count / sqrt(len(query) * len(reference.descriptors)),
                reference,
            )
            for index, reference in enumerate(self.index_references)
            if reference.descriptors is not None
            and (count := scored_by_reference.get(index, 0)) > 0
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
            prediction = self.config.fallback.model_copy()
            self.last_interval = self._interval(prediction, self.fallback_spread)
            return prediction
        weight = sum(score for score, _ in nearest)
        prediction = Nutrition(
            **{
                field: sum(
                    score * getattr(reference.target, field)
                    for score, reference in nearest
                )
                / weight
                for field in FIELDS
            }
        )
        spread = {
            field: sqrt(
                sum(
                    score
                    / weight
                    * (getattr(reference.target, field) - getattr(prediction, field))
                    ** 2
                    for score, reference in nearest
                )
            )
            for field in FIELDS
        }
        self.last_interval = self._interval(prediction, spread)
        return prediction


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
