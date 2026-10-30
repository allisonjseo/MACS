"""Frozen CLIPSeg food masks shared by training and checkpoint inference."""

import hashlib
import json
import re
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional as F
from tqdm import tqdm


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: Literal["CIDAS/clipseg-rd64-refined"] = "CIDAS/clipseg-rd64-refined"
    revision: str = Field(default="main", min_length=1)
    prompt: str = Field(default="food", min_length=1)
    threshold: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)


def resolve_segmentation(config: SegmentationConfig) -> SegmentationConfig:
    """Resolve a branch once; pinned revisions also work from the Hub cache offline."""
    if re.fullmatch(r"[0-9a-f]{40}", config.revision):
        return config
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(config.model_id, "config.json", revision=config.revision)
    revision = Path(path).parent.name
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve an immutable CLIPSeg revision")
    return config.model_copy(update={"revision": revision})


def mask_from_logits(logits: torch.Tensor, size: tuple[int, int], threshold: float):
    """Return an original-size binary mask, retaining every disconnected region."""
    if logits.ndim != 3 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite CLIPSeg logits with shape [1, height, width]")
    width, height = size
    resized = F.interpolate(
        logits.float().unsqueeze(1),
        (height, width),
        mode="bilinear",
        align_corners=False,
    )
    pixels = (resized[0, 0].sigmoid() >= threshold).cpu().numpy()
    if not pixels.any():
        raise ValueError("Empty food mask")
    return Image.fromarray(pixels.astype(np.uint8) * 255)


def apply_food_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    if mask.size != image.size:
        raise ValueError("Food mask dimensions do not match the image")
    pixels = np.asarray(mask.convert("L"))
    if not np.isin(pixels, [0, 255]).all():
        raise ValueError("Food mask must be binary (0 or 255)")
    if not pixels.any():
        raise ValueError("Empty food mask")
    return Image.composite(image.convert("RGB"), Image.new("RGB", image.size), mask)


class FoodSegmenter:
    def __init__(self, config: SegmentationConfig, device: str = "cpu"):
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        self.config = resolve_segmentation(config)
        self.device = torch.device(device)
        kwargs = {"revision": self.config.revision}
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, use_fast=False, **kwargs
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.config.model_id, **kwargs
        ).to(self.device)
        self.model.requires_grad_(False).eval()

    def mask(self, image: Image.Image) -> Image.Image:
        inputs = self.processor(
            text=[self.config.prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        return mask_from_logits(logits, image.size, self.config.threshold)


def prepare_masks(
    samples,
    root: Path,
    config: SegmentationConfig,
    cache: Path,
    device: str,
    *,
    previews: Path | None = None,
):
    """Run CLIPSeg once in the parent process; workers only read binary PNGs."""
    config = resolve_segmentation(config)
    identity = {
        "config": config.model_dump(),
        "transformers": version("transformers"),
        "mask_policy": "bilinear-logits-original-size-v1",
    }
    namespace = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    directory = cache / namespace
    directory.mkdir(parents=True, exist_ok=True)
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
    segmenter = None
    paths, diagnostics = {}, []
    started = time.perf_counter()
    for sample in tqdm(samples, desc="Food masks"):
        path = directory / f"{sample.image_sha256}.png"
        sample_started = time.perf_counter()
        cached = path.exists()
        try:
            with Image.open(root / sample.image) as source:
                image = source.convert("RGB")
            if cached:
                with Image.open(path) as stored:
                    mask = stored.convert("L")
            else:
                if segmenter is None:
                    segmenter = FoodSegmenter(config, device)
                mask = segmenter.mask(image)
            masked = apply_food_mask(image, mask)
            if not cached:
                # Atomic and safe when independent runs share a cache.
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix=".png", delete=False
                ) as tmp:
                    temporary = Path(tmp.name)
                try:
                    mask.save(temporary, format="PNG")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            fraction = float((np.asarray(mask) > 0).mean())
            if previews is not None:
                overlay = Image.blend(
                    image,
                    Image.composite(
                        Image.new("RGB", image.size, (0, 255, 0)), image, mask
                    ),
                    0.35,
                )
                panel = Image.new("RGB", (image.width * 3, image.height))
                for index, view in enumerate((image, overlay, masked)):
                    panel.paste(view, (index * image.width, 0))
                panel.save(previews / f"{sample.id}.png")
        except (ValueError, OSError) as error:
            raise ValueError(
                f"Food segmentation failed for {sample.id}: {error}"
            ) from error
        paths[sample.id] = path
        diagnostics.append(
            {
                "id": sample.id,
                "foreground_fraction": fraction,
                "cached": cached,
                "seconds": time.perf_counter() - sample_started,
            }
        )
    return (
        config,
        paths,
        {
            **identity,
            "cache": str(directory.resolve()),
            "seconds": time.perf_counter() - started,
            "samples": diagnostics,
        },
    )
