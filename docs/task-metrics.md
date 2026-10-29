# Calorie decision metrics and plotnine figures

Generate the plots from completed prediction sets with:

```sh
uv sync --locked --extra vision --extra analysis
uv run --locked --extra analysis task_plots.py \
  --spec configs/task-plots.toml --output runs/babel/task-plots
```

`task_plots.py` uses plotnine for every figure and exports PNG, vector PDF, and
SVG versions, plus the underlying CSVs and provenance in `metadata.json`.
The spec names saved prediction directories; plotting does not train or select
new checkpoints. Paths inside the spec are relative to the project root.

## Test population

The main figures use **all 507 available overhead RGB test photographs** from
the official 709-ID RGB test split. The 202 unavailable overhead photographs are
listed in the prepared manifest and figure metadata. Zero-calorie labels stay in
the population. These results describe the available overhead-image population,
not predictions for the unavailable images or other camera modalities.

All Gaussian models, original point heads, and the official training-label mean
were evaluated on this same population. The plotter requires exact manifest
fingerprints, complete successful predictions, unique matching IDs, and matching
labels. It rejects partial/failed runs rather than quietly changing denominators.
Prediction-file and checkpoint fingerprints are recorded.

The separate `saved_200/` figures use only the earlier 200-image test subset.
Those include the saved Luna/low point estimates; no new paid inference was
performed. Its reported interval half-widths are not assumed to define a Gaussian
distribution, so Luna has a single operating point rather than a quantile curve.

## Task metrics

For deployed calorie estimate `X` and true calories `Y`, compute:

```text
p_under = count(X < Y) / N
mean_over_kcal = sum((X - Y)[X >= Y]) / count(X >= Y)
```

The excess is **conditional on overestimation**, not averaged over all examples.
Ties enter the overestimation set with zero excess. If there are no overestimates,
the conditional mean is undefined: CSV output leaves it missing and the plot
omits that operating point. Counts of under/overestimates accompany every point.
The x-axis is a unitless probability between zero and one; both axes are minimized.

For each Gaussian model, use its actual per-image raw mean and standard deviation:

```text
Q(q) = gaussian_mean + gaussian_std * Normal.ppf(q)
X(q) = max(0, Q(q))                 # default deployment policy
```

The 95th percentile uses approximately `mu + 1.645 * std`. It is different from
the existing two-sided 95% interval half-width (`1.960 * std`). The plotter never
uses the legacy `Prediction.sigma` field as if it were a standard deviation.
Set `clip_zero = false` in the spec to inspect raw Gaussian quantile decisions.

The default grid spans 0.01 through 0.9999 with denser central coverage and named
operating levels. The complete grid is recorded in metadata. No quantile is fitted
or selected using test labels. The sweeps describe observed test-set trade-offs;
a nominal model quantile does not guarantee the same empirical underestimation
probability. The calibration diagnostic therefore plots empirical
`P(Y <= Q(q))` against `q`, using raw quantiles before the zero floor.

A conditional mean can decrease when new, small-error examples enter its
conditioning set. Therefore a quantile sweep need not be monotone or entirely
Pareto-efficient. The plots show the raw paths and mark sampled operating points
that are not dominated by another sampled model/quantile or point-only baseline.
No smoothing, convexification, or interpolation is used to claim new feasible
operating points. Duplicate points remain nondominated; undefined points do not.

## Model metric

`validation_calorie_nll` plots the full **calorie-only Gaussian NLL in physical
kcal units**, averaged over the same 551 validation images, in nats per example:

```text
0.5 * log(2*pi) + log(std_kcal) + 0.5 * ((Y - mean_kcal) / std_kcal)^2
```

The source is each epoch's `validation.nll.calories_kcal`, not the standardized
five-target training loss. The code checks the normalization Jacobian against
the recorded calorie standard deviation.

Existing checkpoints were optimized and selected using mean standardized NLL
across calories, protein, carbs, fat, and mass. Circles mark those saved
checkpoints; triangles show each run's minimum calorie-only validation NLL for
diagnosis. Those minima were not used to refit or silently replace the test
checkpoints. A future calorie-only training/selection experiment would be a
separate, explicitly configured change.

## Outputs

- `task_tradeoff`: full test-set quantile trade-offs and point-only references.
- `task_tradeoff_low_risk`: the same data zoomed to underestimation probability
  0–0.2, without dropping data before computing metrics or dominance.
- `quantile_calibration`: empirical coverage versus nominal Gaussian quantile.
- `validation_calorie_nll`: calorie-only validation likelihood over epochs.
- `quantile_sweep.csv`, `operating_points.csv`, `point_baselines.csv`, and
  `pareto_frontier.csv`: all metric values and their denominators.
- `validation_checkpoint_summary.csv`: saved-checkpoint and minimum-calorie-NLL
  epochs/scores, preserving the distinction between selection rules.
- `metadata.json`: population/exclusion IDs, exact quantiles, clipping policy,
  dependencies, code/spec fingerprints, and prediction-source fingerprints.

Tests cover the conditional denominator, ties and zero-calorie meals, undefined
conditional means, one-sided Gaussian quantiles, dominance, and population checks.
