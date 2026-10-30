"""Offline checks for multi-food masking, caching, and pretrained-model wiring."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import segmentation
from segmentation import (
    FoodSegmenter,
    SegmentationConfig,
    apply_food_mask,
    mask_from_logits,
    prepare_masks,
    resolve_segmentation,
)

REVISION = "a" * 40


def test_revision_is_resolved_once_and_pinned_without_network(monkeypatch):
    import sys

    calls = []

    def download(model_id, filename, revision):
        calls.append((model_id, filename, revision))
        return f"/tmp/hub/snapshots/{REVISION}/config.json"

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    resolved = resolve_segmentation(SegmentationConfig())
    assert resolved.revision == REVISION
    assert resolve_segmentation(resolved) == resolved
    assert calls == [(resolved.model_id, "config.json", "main")]


def test_mask_retains_multiple_foods_and_original_dimensions():
    logits = torch.full((1, 3, 7), -12.0)
    logits[0, 0, 0] = 12
    logits[0, 2, 6] = 12
    mask = mask_from_logits(logits, (7, 3), 0.5)
    image = Image.new("RGB", (7, 3), (80, 120, 200))
    masked = np.asarray(apply_food_mask(image, mask))
    assert mask.size == image.size
    np.testing.assert_array_equal(masked[0, 0], [80, 120, 200])
    np.testing.assert_array_equal(masked[2, 6], [80, 120, 200])
    assert not masked[1, 3].any()
    resized = mask_from_logits(logits, (14, 9), 0.5)
    assert resized.size == (14, 9)
    assert resized.getpixel((0, 0)) == resized.getpixel((13, 8)) == 255


@pytest.mark.parametrize(
    "logits", [torch.full((1, 2, 2), -10.0), torch.full((1, 2, 2), float("nan"))]
)
def test_invalid_or_empty_masks_fail(logits):
    with pytest.raises(ValueError):
        mask_from_logits(logits, (10, 4), 0.5)


def test_mask_validation():
    image = Image.new("RGB", (5, 3))
    with pytest.raises(ValueError, match="dimensions"):
        apply_food_mask(image, Image.new("L", (3, 5), 255))
    with pytest.raises(ValueError, match="binary"):
        apply_food_mask(image, Image.new("L", image.size, 128))
    with pytest.raises(ValueError, match="Empty"):
        apply_food_mask(image, Image.new("L", image.size))


def test_cache_reuse_invalidation_and_previews(tmp_path, monkeypatch):
    calls = []

    class StubSegmenter:
        def __init__(self, config, device):
            calls.append(config)

        def mask(self, image):
            mask = Image.new("L", image.size)
            mask.putpixel((0, 0), 255)
            mask.putpixel((image.width - 1, image.height - 1), 255)
            return mask

    monkeypatch.setattr(segmentation, "FoodSegmenter", StubSegmenter)
    monkeypatch.setattr(segmentation, "version", lambda name: "test")
    source = tmp_path / "food.png"
    Image.new("RGB", (7, 3), (255, 100, 50)).save(source)
    sample = SimpleNamespace(
        id="dish_1",
        image=source.name,
        image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    config = SegmentationConfig(revision=REVISION)
    _, paths, report = prepare_masks(
        [sample],
        tmp_path,
        config,
        tmp_path / "cache",
        "cpu",
        previews=tmp_path / "previews",
    )
    assert len(calls) == 1
    assert report["samples"][0]["foreground_fraction"] == pytest.approx(2 / 21)
    assert not report["samples"][0]["cached"]
    with Image.open(paths[sample.id]) as mask:
        original_mask = np.asarray(mask).copy()
        online = apply_food_mask(Image.open(source), mask)
    assert (tmp_path / "previews/dish_1.png").exists()
    _, reused, report = prepare_masks(
        [sample], tmp_path, config, tmp_path / "cache", "cpu"
    )
    assert len(calls) == 1 and reused == paths
    assert report["samples"][0]["cached"]
    with Image.open(reused[sample.id]) as cached:
        np.testing.assert_array_equal(np.asarray(cached), original_mask)
        np.testing.assert_array_equal(
            np.asarray(online), np.asarray(apply_food_mask(Image.open(source), cached))
        )
    for changes in ({"threshold": 0.7}, {"prompt": "a meal"}, {"revision": "b" * 40}):
        _, changed, _ = prepare_masks(
            [sample],
            tmp_path,
            config.model_copy(update=changes),
            tmp_path / "cache",
            "cpu",
        )
        assert changed != paths
    sample.image_sha256 = "c" * 64
    _, changed, _ = prepare_masks([sample], tmp_path, config, tmp_path / "cache", "cpu")
    assert changed != paths
    assert len(calls) == 5


def test_empty_mask_identifies_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(segmentation, "version", lambda name: "test")

    class EmptySegmenter:
        def __init__(self, *args):
            pass

        def mask(self, image):
            return Image.new("L", image.size)

    monkeypatch.setattr(segmentation, "FoodSegmenter", EmptySegmenter)
    Image.new("RGB", (4, 3)).save(tmp_path / "food.png")
    sample = SimpleNamespace(id="dish_empty", image="food.png", image_sha256="d" * 64)
    with pytest.raises(ValueError, match="dish_empty: Empty food mask"):
        prepare_masks(
            [sample],
            tmp_path,
            SegmentationConfig(revision=REVISION),
            tmp_path / "cache",
            "cpu",
        )


def test_segmenter_loads_pinned_weights_and_freezes_every_parameter(monkeypatch):
    import sys

    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["food"]
            return Batch(pixel_values=torch.ones(1, 3, 3, 7))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

        def forward(self, pixel_values):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(logits=pixel_values[:, 0])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=Processor, CLIPSegForImageSegmentation=Model),
    )
    adapter = FoodSegmenter(SegmentationConfig(revision=REVISION))
    assert not adapter.model.training
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert calls[0][1] == {"revision": REVISION, "use_fast": False}
    assert calls[1][1] == {"revision": REVISION}
    assert adapter.mask(Image.new("RGB", (7, 3))).size == (7, 3)
