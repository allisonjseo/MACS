"""Pretrained image encoders with an independent regression head per quantity.

Install the optional `vision` dependencies for training or checkpoint inference.
"""

import hashlib
import io
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from predictors import Predictor
from schemas import FIELDS, ImageInput, Nutrition, Prediction, PredictorConfig

BACKBONES = {
    "resnet": "resnet152.a1_in1k",
    "swin": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "dino": "vit_base_patch16_dinov3.lvd1689m",
}
TARGETS = (*FIELDS, "mass_g")


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    backbone: Literal["resnet", "swin", "dino"]
    pretrained: bool = True
    freeze_backbone: bool = True
    hidden_dim: int = Field(default=256, ge=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=32, gt=0)
    workers: int = Field(default=4, ge=0)
    device: str = "cuda"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    compile_model: bool = True
    compile_backend: Literal["inductor", "eager"] = "inductor"
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_fullgraph: bool = True
    compile_dynamic: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "macs-nutrition5k"
    wandb_entity: str | None = None
    wandb_group: str = "pretrained-heads"
    head_lr: float = Field(default=1e-3, gt=0)
    backbone_lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    optimizer: Literal["adamw"] = "adamw"
    betas: tuple[
        Annotated[float, Field(ge=0, lt=1)],
        Annotated[float, Field(ge=0, lt=1)],
    ] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0)
    scheduler: Literal["cosine"] = "cosine"
    loss: Literal["standardized_smooth_l1"] = "standardized_smooth_l1"
    huber_beta: float = Field(default=1.0, gt=0)
    grad_clip: float = Field(default=1.0, gt=0)
    augmentation: Literal["none"] = "none"
    targets: tuple[str, ...] = TARGETS

    @model_validator(mode="after")
    def check_targets(self):
        if self.targets != TARGETS:
            raise ValueError(f"Expected targets in order {TARGETS}")
        return self


def create_transform(data_config: dict[str, Any]):
    from timm.data import create_transform as timm_transform

    # Match each encoder's pretrained normalization, size and interpolation.
    # Use identical deterministic preprocessing for training and validation.
    return timm_transform(**data_config, is_training=False)


def model_data_config(config: TrainConfig) -> dict[str, Any]:
    import timm
    from timm.data import resolve_data_config

    pretrained = timm.get_pretrained_cfg(BACKBONES[config.backbone])
    if pretrained is None:
        raise ValueError("Backbone is unavailable in the installed timm")
    return resolve_data_config({}, pretrained_cfg=pretrained.to_dict())


def compile_for_training(model: nn.Module, config: TrainConfig):
    if config.compile_model:
        # Compile in place: state_dict keys stay compatible with eager inference.
        # fullgraph=True makes graph breaks fail instead of silently running eager.
        model.compile(
            backend=config.compile_backend,
            mode=config.compile_mode,
            fullgraph=config.compile_fullgraph,
            dynamic=config.compile_dynamic,
        )


class NutritionRegressor(nn.Module):
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def __init__(self, config: TrainConfig, *, load_pretrained: bool | None = None):
        super().__init__()
        import timm
        from timm.data import resolve_model_data_config

        self.config = config
        self.backbone = timm.create_model(
            BACKBONES[config.backbone],
            pretrained=config.pretrained
            if load_pretrained is None
            else load_pretrained,
            num_classes=0,
        )
        self.data_config = resolve_model_data_config(self.backbone)
        features = cast(int, self.backbone.num_features)
        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(features, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                )
                if config.hidden_dim
                else nn.Linear(features, 1)
                for target in TARGETS
            }
        )
        self.register_buffer("target_mean", torch.zeros(len(TARGETS)))
        self.register_buffer("target_std", torch.ones(len(TARGETS)))
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()  # Also freeze BatchNorm statistics and dropout.
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return torch.cat([self.heads[target](features) for target in TARGETS], dim=1)

    def quantities(self, normalized: torch.Tensor) -> torch.Tensor:
        return (normalized.float() * self.target_std + self.target_mean).clamp_min(0)


class VisionPredictorConfig(PredictorConfig):
    backend: Literal["vision"] = "vision"
    backbone: str
    checkpoint: str
    checkpoint_sha256: str
    device: str
    data_config: dict[str, Any]
    targets: tuple[str, ...] = TARGETS
    versions: dict[str, str]


class VisionPredictor(Predictor):
    """Self-contained checkpoint adapter for the existing evaluation harness."""

    config: VisionPredictorConfig

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        from importlib.metadata import version

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported vision checkpoint format")
        config = TrainConfig.model_validate(saved["train_config"])
        self.model = NutritionRegressor(config, load_pretrained=False)
        self.model.load_state_dict(saved["model"])
        self.model.to(device).eval()
        self.transform = create_transform(saved["data_config"])
        self.device = torch.device(device)
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.config = VisionPredictorConfig(
            backbone=BACKBONES[config.backbone],
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=digest,
            device=device,
            data_config=saved["data_config"],
            versions={n: version(n) for n in ("timm", "torchvision", "pillow")},
        )

    def predict(self, image: ImageInput) -> Prediction:
        from PIL import Image

        with Image.open(io.BytesIO(image.data)) as decoded:
            tensor = self.transform(decoded.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            values = self.model.quantities(self.model(tensor))[0].cpu().tolist()
        quantities = dict(zip(TARGETS, values, strict=True))
        # The evaluator's established schema has four nutrients; preserve it.
        self.last_metadata = {"mass_g": quantities["mass_g"]}
        return Prediction(mean=Nutrition(**{key: quantities[key] for key in FIELDS}))
