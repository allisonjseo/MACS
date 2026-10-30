# Image-to-nutrition evaluation

A small evaluation of total calories (kcal), protein (g), carbohydrates (g), and
fat (g) from one RGB food photograph. Nutrition5k remains the default benchmark;
other datasets plug into the same runner through a common manifest or the
`import-csv` command. See [dataset recommendations and the import interface](docs/datasets.md)
for SNAPMe, SimpleFood45, MetaFood3D, label requirements, and example commands.
Dependencies are managed by
[`uv`](https://docs.astral.sh/uv/), with exact versions in `uv.lock` and Python
3.13 selected by `.python-version` (Python 3.12+ supported). Pydantic validates
labels, manifests, and predictions and generates the JSON schema. Stable sums
compute training means and aggregate metrics.
Data stays in typed Python objects;
JSON is used at the persistence and model-response boundaries.

```sh
uv sync --locked
uv run eval.py prepare --n 200 --seed 42
uv run eval.py run --predictor mean
uv run eval.py prepare --data-dir data/nutrition5k-sift --n 200 --reference-n 500 --seed 42
uv run eval.py run --data-dir data/nutrition5k-sift --predictor sift
uv run eval.py run --predictor codex --model gpt-6-luna --reasoning-effort low
```

The `codex` backend uses the [official Codex Python SDK](https://developers.openai.com/codex/sdk/),
which includes a pinned Codex runtime. Authenticate with `uv run codex login`.
Use `--model MODEL` and `--reasoning-effort LEVEL` to pin the model and reasoning
effort (defaults: `gpt-6-luna`, `low`), `--limit 1` for a smoke test, or
`--timeout 300` to bound each prediction. Both the model ID and reasoning effort
are saved in `predictor.model`, for example
`{"id": "gpt-6-luna", "reasoning_effort": "low"}`. Each image consumes a separate
inference run.

Preparation downloads the official RGB train/test ID lists, dish metadata, and
only the requested test photographs. It shuffles sorted test IDs using the seed,
skips missing overhead images (404s), and saves the selected IDs, image checksums,
labels, and skipped IDs in `data/nutrition5k-200/manifest.json`. The default
benchmark fixes the first 200 available images in the seed-42 shuffled test IDs.
Missing overhead images are recorded and skipped, continuing until 200 images
are selected. Training labels are used solely to compute the fixed mean
baseline unless `--reference-n` is set.
An existing manifest is never silently replaced; use another `--data-dir` to
prepare a different subset.

Pass `--reference-n` to additionally download that many labeled photos from the
official RGB training split. This enables the `sift` predictor, a classical
computer-vision retrieval baseline: it extracts SIFT keypoints from the test
photo, finds training photos with the most unambiguous descriptor matches, then
returns their similarity-weighted nutrition average. It never uses test labels
while predicting and falls back to the full training-label mean if no descriptors
match. `--sift-neighbors`, `--sift-ratio-threshold`, and `--sift-max-features`
are recorded in the run configuration; tune them on a held-out part of the
training split, not on the test subset.

SIFT also reports a heuristic range for each nutrient. Its center is the
similarity-weighted estimate and its half-width is the largest of the weighted
standard deviation of selected neighbor labels, 25% of the estimate, or a fixed
minimum (50 kcal, 3 g protein, 5 g carbohydrates, 3 g fat). Lower bounds are
clipped at zero. With no reliable matches, the center is the training-label mean
and the spread is the standard deviation of reference labels. These ranges are
not calibrated confidence or prediction intervals.

Method notes for the implemented evaluation backends are in [`docs/`](docs/):
the [training mean](docs/mean.md), [SIFT retrieval](docs/sift.md), and
[Codex vision baseline](docs/codex.md).
The [data directory layout](docs/data-layout.md) records the local Nutrition5k
source and prepared SIFT cache structure.
To exclude test dishes whose exact ingredient set appears in official training,
use [`filter_test_recipes.py`](filter_test_recipes.py); see the
[recipe-disjoint test subset](docs/recipe-disjoint-test.md) instructions.

Supervised image baselines use **ResNet-152, Swin V2 Base, and DINOv3 ViT-B/16**
with separate regression heads for calories, protein, carbohydrates, fat, and
total mass. Install them with `uv sync --locked --extra vision`. See
[`deploy/README.md`](deploy/README.md) for training, checkpoint evaluation,
the shared train/validation split, and Babble/PSC launch scripts.

The [calorie task-metric plotting guide](docs/task-metrics.md) documents plotnine
quantile trade-offs, empirical underestimation risk, conditional excess calories,
calibration, and calorie-only validation NLL on the full available overhead test set.

The [CLIPSeg Gaussian variant](docs/clipseg-gaussian.md) adds fully pretrained
food/background segmentation, retaining multiple food regions together before
the frozen ResNet encoder. Training caches masks and learns only Gaussian heads.

Preparation and evaluation seed Python, NumPy, Polars, OpenCV, and PyTorch using the
manifest's seed (default 42; supported range 0–2147483647). Runs use the saved
sample order, and `--limit` always takes its first N images. OpenCV threading,
OpenCL, and optimized paths are disabled. PyTorch uses deterministic algorithms
and records the cuBLAS workspace configuration for GPU matching. With the same cached data,
code, Python, and locked dependencies, sample selection, mean/classical
predictions, and accuracy metrics are repeatable. Codex responses, latency,
timestamps, and temporary paths can vary. Reusing the manifest and its checked
image hashes fixes the evaluation set even if remote image availability changes.
New baselines that create independent RNGs must pass the recorded seed to them;
for example, `numpy.random.default_rng(seed)`.

The CLI uses Typer; run `uv run eval.py --help` to see commands and options.

Every backend implements the same `predict(image) -> Prediction` contract in
`predictors.py`. Dataset loading, output validation, and metrics live in `eval.py`.
`run.py` captures run configuration and provenance and serializes all run artifacts.
The included backends are `codex`, `mean` (training-set mean), `classical`
(white-plate coverage), `sift` (SIFT training-image retrieval), and `vision`
(trained image-backbone checkpoint; requires `--checkpoint`). All inherit
`Predictor`, a small shared interface in `predictors.py`. Pydantic data contracts
are in `schemas.py`. The image is an `ImageInput` containing encoded image bytes
(`data`) and a MIME type (`media_type`). The Codex adapter writes its own temporary
attachment.

To add a baseline, subclass `Predictor`, implement `predict(image)`, and
select it in `eval.py`. Return a `Prediction` object:

```python
Prediction(
    mean=Nutrition(calories_kcal=450, protein_g=25, carbs_g=50, fat_g=17),
    sigma=Nutrition(calories_kcal=150, protein_g=10, carbs_g=20, fat_g=8),
)
```

The Codex prompt is configured when constructing `CodexPredictor`; custom prompts
can be supplied with `--prompt-file` for Codex runs.

Codex must return a `mean` and a `sigma` for every nutrient, enforced by its
structured output schema. Here `sigma` is the **95% interval half-width**, so
the interval is literally `mean ± sigma`, using kcal for calories and grams
for macros. It is not a standard deviation. The prompt asks the model to account
for uncertain food identity, hidden ingredients, and portions. These are
model-estimated prediction intervals; their actual coverage is measured against
the labels, not assumed to be 95%. Bounds are symmetric and are not clipped at
zero. Mean/classical baselines return `sigma=None`. SIFT also returns
`sigma=None`, but reports its separate heuristic bounds as `prediction_interval`;
those ranges do not claim 95% coverage.

## Classical computer vision baseline

```sh
uv run eval.py run --predictor classical --limit 10
```

This is a fixed-rule OpenCV baseline for overhead photos on white plates. It
detects a bright, low-saturation plate region, fills its convex hull, excludes
the rim, and segments food by Lab color/lightness contrast. Morphology and
connected components remove small noise. Predictions scale the training-set
mean nutrition by `food_coverage / 0.35`. The 35% reference coverage is an
assumption, not fitted to test labels; all thresholds and versions are saved.
Plate-detection failures use the mean baseline and record a fallback reason.

No learned food recognition, depth estimation, or regression is used. This is a
coverage heuristic: pale food, shadows, plate geometry, food thickness, and
different calorie densities can defeat it, and all macro ratios stay fixed.
Classical geometric food-volume methods do exist, such as
[Chae et al. (2011)](https://jchae21.github.io/assets/data/paper/tada.pdf), but
require segmentation, food identity, and camera/scale information. This baseline
is not a reproduction of that paper.

## Run the full test split

Use a new cache directory to preserve the current subset:

```sh
uv run eval.py prepare --all --data-dir data/nutrition5k-full
uv run eval.py run --data-dir data/nutrition5k-full --predictor classical
uv run eval.py run --data-dir data/nutrition5k-full --predictor codex \
  --model gpt-6-luna --reasoning-effort low
```

`--all` considers all official RGB test IDs, skips missing overhead images, and
records exclusions in the manifest. Omitting `--limit` evaluates that entire
manifest. The RGB test list has 709 IDs. An earlier 10-image Sol/high run averaged
16.0 seconds per image, suggesting about 3 hours 10 minutes for 709 images when
run sequentially, plus downloads; actual runtime and image availability vary.

## Results and comparison

Each run creates a new `runs/<timestamp>-<backend>/` directory containing:

- `manifest.json`: exact input manifest, including dataset/split identity, labels,
  image checksums, and preparation provenance.
- `config.json`: backend configuration (including `predictor.prompt` for Codex), sample IDs, timeout, and
  dataset/split identity, manifest fingerprint, seed and OpenCV settings (`determinism`), Python and
  dependency versions, and Git provenance (`git.commit` and
  `git.dirty`, including untracked files). Git is read from the evaluator checkout,
  even when launched elsewhere; unavailable provenance is recorded as null.
- `predictions.json`: a readable JSON array of records, updated after every image,
  with prediction or error,
  reference labels, and latency. Codex records also include typed SDK items and
  token usage.
- `summary.json`: mean absolute error and observed-label `count` for each target, calorie estimates within
  ±20%, success/failure counts, and average latency. Updated after every image.
  Each nutrient also records `interval_count`, `interval_coverage_pct` (the
  percentage of labels inside the reported interval), and `mean_interval_width`.
  Codex uses `mean ± sigma` with width `2 * sigma`; SIFT uses its explicit bounds
  with width `upper - lower` after clipping. Only successful predictions with uncertainty enter
  interval metrics; unavailable metrics are null.

Datasets may provide only some nutrition labels. Each nutrient is scored only
where its target is observed; missing labels are never treated as zero. The
prediction contract still requires all four nutrients. Checkpoint and Codex runs
need no training statistics; training-dependent baselines require separate
training labels, as described in [the dataset guide](docs/datasets.md).

MAE and the ±20% rate use successful predictions only; inspect failure counts
when comparing backends. Zero-calorie references are excluded from the percentage
metric, but included in MAE. All-failed runs report null metrics and exit with
status 1. Any failed prediction gives the run a nonzero exit status. Reusing an
output directory is rejected to protect existing results.
Interval coverage is the percentage of true nutrient values inside the reported
bounds, counted separately for each nutrient. If explicit bounds are provided,
they take precedence over sigma-derived bounds. Point-only backends report zero interval count and null interval
metrics.

Use the same manifest and sample limit for comparisons, and the same prompt when
comparing Codex models. Pin the Codex
model for published runs. Codex uses a fresh ephemeral session per photograph,
an anonymized image, structured output, and disabled shell, browser, app, memory,
plugin, and image-reading tools. Any detected tool invocation invalidates that
prediction. Attached images still go directly to the model. Reference labels are
never included in its request. Tool settings are overridden per thread; other user
configuration and managed system instructions may still apply. The SDK handles
runtime startup, typed inputs/results, and cleanup, including when a run times out.

This is a small sanity check, not a comprehensive benchmark. Nutrition5k consists
of cafeteria dishes; single RGB views leave portion size and hidden ingredients
ambiguous. The public dataset may have appeared in model training. Dish IDs can
include incremental scans of a plate, so a random ID subset is not necessarily
20 independent physical meals.

## Validation and source

Development dependencies (Ruff, ty, pytest, and pre-commit) are locked with uv.
Install the commit hook once per checkout:

```sh
uv sync --locked
uv run pre-commit install
uv run pre-commit run --all-files
```

On commit, pre-commit checks lockfile consistency, runs Ruff lint fixes and
formatting, type-checks the project with ty, and runs the offline pytest suite in
`tests/`. All hooks use the tools from `uv.lock`. If Ruff modifies a file, review
and stage the changes before committing again.

Run individual checks during development:

```sh
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Dataset: [Google Research Nutrition5k](https://github.com/google-research-datasets/Nutrition5k),
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Labels combine weighed ingredient portions with USDA nutrition values.
See Thames et al., *Nutrition5k: Towards Automatic Nutritional Understanding of
Generic Food*, CVPR 2021. The Codex adapter follows the
[official SDK documentation](https://developers.openai.com/codex/sdk/).
Downloaded data and generated run artifacts are gitignored.

For matched frozen-head hyperparameter searches with early pruning, see the
[Optuna pilot protocol](docs/optuna-sweep.md) and
[sweep configuration](configs/sweep-gaussian.toml).
