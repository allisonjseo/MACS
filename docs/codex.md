# Codex vision baseline

## Purpose

The `codex` backend asks a configured Codex model to estimate total calories,
protein, carbohydrates, and fat from one food photograph. It is the project's
vision-language-model baseline; unlike the mean and SIFT methods, it relies on
the model's visual interpretation rather than retrieving labeled images.

## Prompt and output

The default prompt asks for best point estimates for all food in the photograph,
including inferred portion sizes. It directs the model to use only the photo and
return a JSON object containing `mean` and `sigma` objects, each with all four
nutrients. The output schema requires finite, nonnegative values for both.
A custom prompt can be supplied with
`--prompt-file`; the selected prompt is recorded in the run configuration.

Each image is sent in a fresh ephemeral Codex thread. The image is attached
directly, and the thread runs with a read-only sandbox, no approval, and web
search and tool features disabled. The evaluator also rejects turns that report
tool use. A timeout applies separately to each image. Failed or timed-out turns
are recorded as failed predictions.

Defaults are `gpt-6-luna` with `low` reasoning effort. Use `--model` and
`--reasoning-effort` to select and record both settings explicitly.

## Ranges

For each nutrient, `sigma` is the model-estimated 95% prediction interval
half-width, in the same units as its mean. The interval is `mean ± sigma`, with
no clipping; sigma is not a standard deviation. The evaluator measures coverage
against the labels rather than assuming 95% coverage. The heuristic SIFT ranges
described in [`sift.md`](sift.md) remain separate from these model estimates.

## Evaluation

The shared evaluator reports MAE separately for each nutrient and the percentage
of eligible calorie predictions within 20% of the true calories. Zero-calorie
examples remain in MAE but are excluded from the percentage. It also records
successes, failures, latency, prompt and model configuration, and available SDK
usage metadata. Accuracy metrics use successful predictions only.

## Reproduce

Authenticate with `uv run codex login`, then run:

```sh
uv run eval.py run \
  --data-dir ./data/nutrition5k-200 \
  --predictor codex \
  --model gpt-6-luna --reasoning-effort low \
  --limit 2 \
  --output ./codex-smoke-results
```

This invokes the model for each selected image. Keep the same manifest, model,
prompt, and sample limit when comparing model runs.
