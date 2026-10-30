# Pretrained food masking with Gaussian nutrition heads

This variant keeps every detected food region in one mask, including disconnected
foods on the same plate. It predicts one independent Gaussian per **whole-dish**
quantity: calories, protein, carbohydrates, fat, and mass. It does not identify
ingredients or predict separate nutrition values for individual foods.

```text
RGB image -> frozen CLIPSeg (prompt: food) -> black background
          -> frozen pretrained ResNet-152 -> five Gaussian mean/std heads
```

Only the regression heads train. The segmenter is
`CIDAS/clipseg-rd64-refined`, from the CVPR 2022 paper
[Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg).
It has an established [Transformers implementation](https://huggingface.co/docs/transformers/model_doc/clipseg).
This uses the segmentation-before-regression idea from
[Zhao et al.](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1469878/full),
but replaces their trained U-Net with a fully pretrained general segmenter.
Food-mask quality on Nutrition5k is an experimental question: the model can omit
food or include plates. Inspect overlays before launching a long run.

## Install and preview

```sh
uv sync --locked --extra vision --extra segmentation

# Uses only training images; writes original / mask overlay / masked RGB panels.
.venv/bin/python train.py segment-preview \
  --config configs/resnet-clipseg-gaussian.toml \
  --manifest data/nutrition5k-training/manifest.json \
  --output previews/clipseg --limit 32 --device cuda
```

The preview directory must be new. `segmentation.json` records the immutable model
revision, foreground fractions, cache hits, and wall times. Use `--device cpu`
for a CPU preview. The first uncached image's time includes loading the model.

## Train and evaluate

```sh
.venv/bin/python train.py fit \
  --config configs/resnet-clipseg-gaussian.toml \
  --manifest data/nutrition5k-training/manifest.json \
  --output runs/resnet-clipseg-gaussian --dry-run

.venv/bin/python train.py fit \
  --config configs/resnet-clipseg-gaussian.toml \
  --manifest data/nutrition5k-training/manifest.json \
  --output runs/resnet-clipseg-gaussian

.venv/bin/python eval.py run --predictor vision \
  --checkpoint runs/resnet-clipseg-gaussian/best.pt --device cuda \
  --data-dir data/nutrition5k-test --output runs/clipseg-evaluation
```

Training resolves the requested Hub revision to an immutable commit and caches
binary PNG masks under `segmentation-masks/` beside the training manifest. Each
cache namespace includes the pinned revision, prompt, threshold, Transformers
version, and masking policy; filenames contain the original image checksum.
Changed images or settings cannot reuse incompatible masks. DataLoader workers
read those masks without loading CLIPSeg. The segmenter is released before the
regression model is loaded, and no segmentation training is performed.

CLIPSeg logits are resized to the original image dimensions, passed through
sigmoid, and thresholded at 0.05. Every retained region stays at its original
location and scale, with background pixels set to black. Existing ResNet
preprocessing follows. Empty masks stop preparation or fail that evaluation
sample explicitly; there is no unmasked fallback.

The threshold is intentionally below 0.5: in a check on the first 32 sorted
available training-reference images, the fixed `food` prompt gave five empty
masks at 0.5 and none at 0.05. These scores are not calibrated probabilities.
This is a starting setting, not a segmentation-accuracy claim; inspect overlays
and validate downstream performance before relying on it.
Visual inspection of that sample found coarse boundaries and plate pixels around
mixed foods, and missed some isolated olives even at 0.05. Retaining multiple
regions does not guarantee that every food item was detected.

The checkpoint records the pinned segmentation settings and library version.
Inference loads those same pretrained weights, computes a fresh mask for each
new image, and records its foreground fraction in backend metadata. Evaluation
latency includes segmentation. Model weights use the normal Hugging Face cache;
offline inference requires both processor and weights to be cached first (set
`HF_HUB_OFFLINE=1`). Gaussian checkpoints are therefore not standalone bundles
of segmenter weights. Existing checkpoints without segmentation work as before.

The [Gaussian output conventions](gaussian-heads.md) remain unchanged: actual
standard deviations live in `backend_metadata.gaussian_std`, while
`Prediction.sigma` is the 95% interval half-width. Compare against
`configs/resnet-gaussian.toml` on identical splits and seeds using validation NLL,
MAE, coverage, interval width, and runtime. Do not tune prompts or thresholds on
the official test partition. No accuracy or calibration improvement is assumed.
