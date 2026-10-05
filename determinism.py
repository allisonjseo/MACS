"""Shared reproducibility settings for dataset preparation and evaluation."""

import os
import random
from typing import Literal

import cv2
import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field

MAX_SEED = 2**31 - 1


class DeterminismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, le=MAX_SEED)
    opencv_threads: Literal[0] = 0  # Disable parallel regions, including macOS GCD.
    opencv_opencl: Literal[False] = False
    opencv_optimized: Literal[False] = False
    torch_deterministic_algorithms: Literal[True] = True
    cublas_workspace_config: str = ":4096:8"


def configure_determinism(seed: int) -> DeterminismConfig:
    """Seed library RNGs and avoid OpenCV parallel/hardware-dependent paths.

    New independent RNG objects (e.g. numpy.random.default_rng) must also receive
    this seed explicitly. Python hash order is deliberately never used for sampling.
    """
    settings = DeterminismConfig(
        seed=seed,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    )
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    pl.set_random_seed(settings.seed)
    cv2.setRNGSeed(settings.seed)
    cv2.setNumThreads(settings.opencv_threads)
    cv2.ocl.setUseOpenCL(settings.opencv_opencl)
    cv2.setUseOptimized(settings.opencv_optimized)
    # Set before CUDA is initialized by the SIFT matcher.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", settings.cublas_workspace_config)
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(settings.torch_deterministic_algorithms)
    return settings
