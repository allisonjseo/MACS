"""Prediction backends. Each exposes predict(image) -> Nutrition."""

import asyncio
import tempfile
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

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
