# Codex vision baseline

## Purpose

The `codex` backend asks a configured Codex model to estimate total calories,
protein, carbohydrates, and fat from one food photograph. It is the project's
vision-language-model baseline; unlike the mean and SIFT methods, it relies on
the model's visual interpretation rather than retrieving labeled images.

## Prompt and output

The default prompt asks for best point estimates for all food in the photograph,
including inferred portion sizes. It directs the model to use only the photo and
return a JSON object containing four numeric values. The output schema validates
those values as finite, nonnegative numbers. A custom prompt can be supplied with
`--prompt-file`; the selected prompt is recorded in the run configuration.

Each image is sent in a fresh ephemeral Codex thread. The image is attached
directly, and the thread runs with a read-only sandbox, no approval, and web
search and tool features disabled. The evaluator also rejects turns that report
tool use. A timeout applies separately to each image. Failed or timed-out turns
are recorded as failed predictions.

By default, the Codex SDK uses its configured model. Set `--model` to record and
use a specific model. The current project examples use `gpt-6.1-sol`.

## Ranges

The current schema and default prompt request a single point estimate per
nutrient. The Codex backend does not produce a prediction interval, so its
interval coverage and mean-width metrics are null. The SIFT ranges described in
[`sift.md`](sift.md) are specific to that retrieval baseline and are not applied
to Codex results.

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
  --data-dir ./nutrition5k-sift-full \
  --predictor codex \
  --model gpt-6.1-sol \
  --limit 2 \
  --output ./codex-smoke-results
```

This invokes the model for each selected image. Keep the same manifest, model,
prompt, and sample limit when comparing model runs.
