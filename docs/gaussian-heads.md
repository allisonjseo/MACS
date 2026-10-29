# Gaussian regression heads

The `*-gaussian.toml` configs are variants of the original point-head references.
The pretrained encoder remains **frozen**, including BatchNorm statistics and
encoder dropout. Only the five per-quantity heads receive gradients and optimizer
updates. The hidden Linear/ReLU/Dropout layers are unchanged; each final linear
layer now returns two values: a mean and an unconstrained scale parameter.

For standardized target `z`, each head models an independent, untruncated
Gaussian with mean `mu` and standard deviation
`s = softplus(raw_scale) + gaussian_min_std`. The default floor is 0.001 in
standardized target units. Scale-output weights start at zero and their biases
are initialized to give `s = 1.0` for every image.

The optimized per-target loss is

```text
NLL = 0.5 * log(2*pi) + log(s) + (z - mu)^2 / (2*s^2)
```

The trainer uses PyTorch's full Gaussian NLL, averaged equally over examples and
targets. The existing AdamW weight decay remains 0.01. See the
[PyTorch GaussianNLLLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GaussianNLLLoss.html).
Normalization means and population standard deviations still come only from the
training partition. In physical units, `mean = mu * target_std + target_mean`
and `std = s * target_std`. Physical-unit NLL adds `log(target_std)` per target;
that constant does not change training or checkpoint ordering. NLL can be negative.

The Gaussian mean is unconstrained for fitting. The reported nutrition point
estimate is clipped at zero, as in the point-only reference, for MAE comparisons.
Uncertainty coverage uses the **raw Gaussian mean**, not that clipped point
estimate. Explicit 95% bounds are `mean +/- 1.959963984540054 * std`, and can be
negative because the fitted distribution is untruncated. Signed interval bounds
are allowed without changing the nonnegative point-label contract.

`backend_metadata.gaussian_std` stores the actual learned standard deviation in
kcal/grams, including mass; `gaussian_mean` stores the raw mean. The existing
`Prediction.sigma` convention stays a **95% half-width**, not a standard deviation.
The evaluator's explicit bounds take precedence over `mean +/- sigma`. These
nominal Gaussian intervals are evaluated for coverage; 95% coverage is not assumed.

The trainer records MAE, standardized NLL, physical-unit NLL, mean predicted std,
95% coverage and mean 95% interval width for each target under `train/` and `val/`.
All timing remains under `timing/`. Checkpoints are selected only by minimum
validation standardized NLL; selected-checkpoint uncertainty metrics are recorded
alongside its MAEs. No fitting or selection uses the test images.

```sh
# Inspect before launch; other settings match the corresponding point reference.
.venv/bin/python train.py fit --config configs/dino-gaussian.toml \
  --manifest data/nutrition5k-training/manifest.json --output runs/dino-gaussian \
  --dry-run

# On Babel: select Gaussian configs while retaining the same frozen encoders.
MACS_MODELS=dino MACS_CONFIG_SUFFIX=-gaussian GPU_CONSTRAINT=L40S \
  WALLTIME=00:15:00 bash deploy/babble.sh MANIFEST OUTPUT_ROOT \
  --set wandb_mode=offline --set wandb_group=gaussian-mle-20261029
```

The original point checkpoints remain loadable. The checkpoint's saved loss
selects the one-output or two-output head architecture at inference.
