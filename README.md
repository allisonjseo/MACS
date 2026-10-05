# Nutrition5k image evaluation

A small evaluation of total calories (kcal), protein (g), carbohydrates (g), and
fat (g) from one overhead RGB food photograph. Dependencies are managed by
[`uv`](https://docs.astral.sh/uv/), with exact versions in `uv.lock` and Python
3.13 selected by `.python-version` (Python 3.12+ supported). Pydantic validates
labels, manifests, and predictions and generates the JSON schema. Polars computes
training statistics and evaluation metrics. Data stays in typed Python objects;
JSON is used at the persistence and model-response boundaries.

```sh
uv sync --locked
uv run eval.py prepare --n 20 --seed 42
uv run eval.py run --predictor mean
uv run eval.py prepare --data-dir data/nutrition5k-sift --n 200 --reference-n 500 --seed 42
uv run eval.py run --data-dir data/nutrition5k-sift --predictor sift
uv run eval.py run --predictor codex --model gpt-6.1-sol
```

The `codex` backend uses the [official Codex Python SDK](https://developers.openai.com/codex/sdk/),
which includes a pinned Codex runtime. Authenticate with `uv run codex login`.
Use `--model MODEL`
to pin a model, `--limit 1` for a smoke test, or `--timeout 300` to bound each
prediction. Without `--model`, Codex uses its configured default model. Each image
consumes a separate inference run.

Preparation downloads the official RGB train/test ID lists, dish metadata, and
only the requested test photographs. It shuffles sorted test IDs using the seed,
skips missing overhead images (404s), and saves the selected IDs, image checksums,
labels, and skipped IDs in `data/nutrition5k/manifest.json`. Training labels are
used solely to compute the fixed mean baseline unless `--reference-n` is set.
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

The CLI uses Typer; run `uv run eval.py --help` to see commands and options.

Method notes for the implemented evaluation backends are in [`docs/`](docs/):
the [training mean](docs/mean.md), [SIFT retrieval](docs/sift.md), and
[Codex vision baseline](docs/codex.md).
The [data directory layout](docs/data-layout.md) records the local Nutrition5k
source and prepared SIFT cache structure.

Every backend implements the same `predict(image) -> Nutrition` contract in
`predictors.py`. Dataset loading, output validation, and metrics live in `eval.py`.
`run.py` captures run configuration and provenance and serializes all run artifacts.
The included backends are `codex`, `mean` (training-set mean), and `sift`
(SIFT training-image retrieval). All inherit
`Predictor`, a small shared interface in `predictors.py`. Pydantic data contracts
are in `schemas.py`. The image is an `ImageInput` containing encoded image bytes
(`data`) and a MIME type (`media_type`). The Codex adapter writes its own temporary
attachment.

To add a baseline, subclass `Predictor`, implement `predict(image)`, and
select it in `eval.py`. Return a `Nutrition` object:

```python
Nutrition(calories_kcal=450, protein_g=25, carbs_g=50, fat_g=17)
```

The Codex prompt is configured when constructing `CodexPredictor`; custom prompts
can be supplied with `--prompt-file` for Codex runs.

## Results and comparison

Each run creates a new `runs/<timestamp>-<backend>/` directory containing:

- `config.json`: backend configuration (including `predictor.prompt` for Codex), sample IDs, timeout, and
  manifest fingerprint, dependency versions, and Git provenance (`git.commit` and
  `git.dirty`, including untracked files). Git is read from the evaluator checkout,
  even when launched elsewhere; unavailable provenance is recorded as null.
- `predictions.json`: a readable JSON array of records, updated after every image,
  with prediction or error,
  reference labels, and latency. Codex records also include typed SDK items and
  token usage.
- `summary.json`: mean absolute error for each target, calorie estimates within
  ±20%, interval coverage and mean width for backends that return ranges,
  success/failure counts, and average latency. Updated after every image.

MAE and the ±20% rate use successful predictions only; inspect failure counts
when comparing backends. Zero-calorie references are excluded from the percentage
metric, but included in MAE. All-failed runs report null metrics and exit with
status 1. Any failed prediction gives the run a nonzero exit status. Reusing an
output directory is rejected to protect existing results.
Interval coverage is the percentage of true nutrient values inside the reported
bounds, counted separately for each nutrient; mean width is upper minus lower
after clipping. Point-only backends report zero interval count and null interval
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
