# Training mean baseline

## Purpose

The `mean` backend is a simple non-image baseline. It predicts the same nutrition
values for every test image: the arithmetic mean of the available training-set
labels for calories, protein, carbohydrates, and fat.

## How the values are calculated

During `prepare`, the evaluator reads the Nutrition5k cafe 1 and cafe 2 dish
metadata and the official RGB train/test ID lists. It rejects overlapping train
and test IDs. For each nutrient, it averages that nutrient across every labeled
dish in the RGB training split. This mean uses labels, so it does not require the
training dish to have an overhead image. The resulting values and label count
are stored in `manifest.json` as `training_mean` and `training_label_count`.

For each test image, `MeanPredictor` returns a copy of `training_mean` without
opening or examining the image. It produces point estimates only; no interval is
attached.

## Evaluation

The shared evaluator compares each successful prediction with that image's
ground-truth labels. It reports mean absolute error (MAE) separately for each
nutrient. It also reports the fraction of eligible images whose calorie
prediction is within 20% of the true calories. Zero-calorie examples are excluded
from that percentage, but remain included in MAE. Interval metrics are null
because this backend does not provide ranges.

## Reproduce

```sh
uv run eval.py prepare --data-dir data/nutrition5k --n 20 --seed 42
uv run eval.py run --data-dir data/nutrition5k --predictor mean
```

The `n` and `seed` options select test images. They do not alter the training mean
when the same manifest inputs are used.
