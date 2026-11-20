# Leakage audit of the test filter and validation split

These metadata-only checks respond to TA feedback on the Initial ML Solution
report. They use no model predictions and no retraining. Scripts live in
`analysis/leakage/`; all of them read one directory prepared by
`fetch_metadata.sh`:

```sh
bash analysis/leakage/fetch_metadata.sh /tmp/n5k-meta
python -I analysis/leakage/leakage_analysis.py /tmp/n5k-meta  # exact-set vs 0.8 overlap
python -I analysis/leakage/cutoff_check.py /tmp/n5k-meta      # pairs near the 0.8 cutoff
python -I analysis/leakage/same_plate.py /tmp/n5k-meta        # same-session rule
python -I analysis/leakage/subset200.py /tmp/n5k-meta         # why the table has 96 images
python -I analysis/leakage/task_metric.py PREDICTIONS.csv /tmp/n5k-meta  # task metric
```

The validation split is reproduced from `vision_data.py` (seeded dish-ID shuffle,
seed 42, 20%), giving 2,204 training and 551 validation photos.

## Findings

**Exact ingredient-set filter** (`filter_test_recipes.py`): removes 254 of 507
test photos and keeps 253. It treats the same ingredients in different amounts as
a duplicate (for example, a 207 kcal chicken plate matched to a 44 kcal plate),
and misses the same food photographed again after another item was added.

**Weighted overlap at 0.8** (sum of per-ingredient minimum grams / sum of
maximum grams) keeps 315. Rejected: photos near the cutoff show it removes
separate servings of simple foods weeks apart (strawberries, 0.89) while keeping
the same plate photographed minutes apart (0.75). 120 of its 192 removals are
not same-session matches, 91 of them dishes with one or two ingredients.

**Same-session rule** (proposed): a test dish is leaked when a training dish was
scanned within 4 hours (dish IDs are Unix timestamps) and their shared
ingredients have the same grams, within max(2 g, 3%), for at least 80% of the
smaller dish's mass.

| Window | Test photos flagged | Kept |
| --- | --- | --- |
| 10 min | 56 | 451 |
| 1 h | 101 | 406 |
| 2 h | 113 | 394 |
| **4 h** | **114** | **393** |
| 8 h | 116 | 391 |
| 24 h | 132 | 375 |
| 72 h | 180 | 327 |

The count plateaus from 2 to 8 hours, then grows as servings from different days
match by chance. The exact-set rule catches 72 of the 114; 182 of its 254
removals are not same-session matches. A photo spot check of 8 random flagged
pairs found about 5 clearly the same food and 3 same-session single vegetables
that the photos could not confirm.

**Validation split**: 514 of 551 validation photos (93%) share same-session
food with the training portion; 259 (47%) have an exact ingredient-set match.
Validation NLL therefore mostly measures recognition of food already seen.
Splitting train/validation by day would keep each session on one side.

**The 96-image results table**: GPT-Luna was only run on the 200-image
benchmark (`eval.py prepare --n 200 --seed 42`); 96 of those 200 are
recipe-disjoint, so the table restricted every model to them. There is no other
reason for 96.

**GPT-Luna comparison on those 96** (Wilson intervals, unpaired): DINOv3
underestimates 13/96 vs. 18/96, a 5.2-point (28% relative) reduction with 95% CI
−15.7 to +5.3 points; mean excess falls 38% (188.7 to 117.2 kcal). A paired test
needs the per-image predictions.

## Task metric on the leak-free subset

`task_metric.py` computes the report's single task metric from per-image Gaussian
predictions: mean excess when overestimating, at the quantile tuned on validation
to underestimate at most 10% of meals. Selected checkpoints (ResNet-152 trial 3
epoch 12, Swin V2 trial 3 epoch 12, DINOv3 trial 0 epoch 18) on the 393 leak-free
test photos:

| Model | Tuned q | Test underestimation (95% CI) | Mean excess, kcal (95% CI) | MAE |
| --- | --- | --- | --- | --- |
| ResNet-152 | 0.936 | 17.6% (14.1 to 21.6%) | 135.9 (125.1 to 147.7) | 75.0 |
| Swin V2 | 0.917 | 13.7% (10.7 to 17.5%) | 108.9 (100.9 to 118.6) | 58.9 |
| DINOv3 | 0.916 | 13.7% (10.7 to 17.5%) | 96.6 (88.3 to 105.8) | 53.2 |
| Training-label Gaussian | 0.886 | 8.9% (6.5 to 12.1%) | 307.0 (292.0 to 321.7) | 159.0 |

Paired bootstrap: DINOv3's mean excess is 12.3 kcal below Swin V2 (95% CI 3.8 to
21.3) and 39.3 kcal below ResNet-152 (28.6 to 50.6). No model keeps test
underestimation within the 10% budget tuned on the leaky validation split, and
DINOv3's nominal 95% quantile covers 88% of test meals.

## Not yet done

- Re-evaluate SIFT and GPT-Luna on the same-session subset (393).
- Re-split train/validation by day and repeat model selection.
- Thresholds (4 h, 3%, 80%) are judgment calls; check more flagged pairs.
