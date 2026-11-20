"""Single task metric on the leak-free test subset, from per-image Gaussian predictions.

Usage: python -I task_metric.py PREDICTIONS_CSV DATA_DIR  (see fetch_metadata.sh)

PREDICTIONS_CSV has one row per model, split (validation/test) and dish_id, with
target_calories_kcal, gaussian_mean_calories_kcal and gaussian_std_calories_kcal.

Task metric: for each model, pick the smallest quantile q whose clipped estimate
X = max(0, mean + std * z_q) underestimates at most 10% of validation meals; apply
that q on the test subset and report mean excess E[X - Y | X >= Y] (kcal) with the
test underestimation rate P(X < Y) as a guardrail. 95% CIs: Wilson for rates,
bootstrap (2,000 resamples, seed 0) for mean excess and paired differences.
"""

import contextlib
import csv
import io
import random
import runpy
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import NormalDist, fmean, pstdev

PREDICTIONS, DATA = sys.argv[1], sys.argv[2]
HERE = Path(__file__).parent
BUDGET = 0.10
Z = NormalDist()

# Leak-free test subset (same-session rule, 4-hour window) and the training portion.
sys.argv = ["same_plate.py", DATA]
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(HERE / "same_plate.py"))
g = ns["g"]
flagged = set(ns["same_plate_matches"](g["test_photos"], g["train_all"], 4 * 3600))
leak_free = set(g["test_photos"]) - flagged
exact_set_kept = set(g["test_photos"]) - {
    d for d in g["test_photos"] if g["test_report"]["_by_set"].get(g["meta"][d]["set"])
}
train_kcal = [g["meta"][d]["kcal"] for d in g["training"]]

rows = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(PREDICTIONS)):
    rows[r["model"]][r["split"]].append(
        (r["dish_id"], float(r["target_calories_kcal"]),
         float(r["gaussian_mean_calories_kcal"]), float(r["gaussian_std_calories_kcal"]))
    )
# Training-label Gaussian baseline: the same constant distribution for every image.
mu, sd = fmean(train_kcal), pstdev(train_kcal)
for split in ("validation", "test"):
    rows["training_gaussian"][split] = [(d, y, mu, sd) for d, y, _, _ in rows["dino"][split]]


def estimates(data, q):
    z = Z.inv_cdf(q)
    return [(d, y, max(0.0, m + s * z)) for d, y, m, s in data]


def under_rate(est):
    return sum(x < y for _, y, x in est) / len(est)


def mean_excess(est):
    over = [x - y for _, y, x in est if x >= y]
    return fmean(over) if over else float("nan")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def tune(val):
    for i in range(500, 1000):
        q = i / 1000
        if under_rate(estimates(val, q)) <= BUDGET:
            return q
    return 0.999


def bootstrap(f, ids, reps=2000, seed=0):
    rng = random.Random(seed)
    vals = sorted(f([rng.choice(ids) for _ in ids]) for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def report(subset_name, subset):
    print(f"\n=== Test subset: {subset_name} (n = {len(subset)}) ===")
    print(f"{'model':18s} {'q':>6s} {'val under':>9s} {'test under (95% CI)':>26s} "
          f"{'mean excess kcal (95% CI)':>28s} {'MAE':>6s}")
    tuned = {}
    for model in ("resnet", "swin", "dino", "training_gaussian"):
        val, test = rows[model]["validation"], rows[model]["test"]
        test = [r for r in test if r[0] in subset]
        q = tune(val)
        est = estimates(test, q)
        k = sum(x < y for _, y, x in est)
        lo, hi = wilson(k, len(est))
        by_id = {d: (y, x) for d, y, x in est}
        ids = list(by_id)
        exc = mean_excess(est)
        elo, ehi = bootstrap(
            lambda s: fmean([by_id[i][1] - by_id[i][0] for i in s if by_id[i][1] >= by_id[i][0]]
                            or [float("nan")]), ids)
        mae = fmean(abs(max(0.0, m) - y) for _, y, m, _ in test)
        tuned[model] = by_id
        print(f"{model:18s} {q:6.3f} {under_rate(estimates(val, q)):9.1%} "
              f"{k / len(est):7.1%} ({lo:5.1%} to {hi:5.1%}) "
              f"{exc:9.1f} ({elo:6.1f} to {ehi:6.1f}) {mae:6.1f}")
    # Paired bootstrap: DINO minus each other model, on the same images.
    ids = list(tuned["dino"])
    for other in ("resnet", "swin", "training_gaussian"):
        def diff(s, a=tuned["dino"], b=tuned[other]):
            ea = [a[i][1] - a[i][0] for i in s if a[i][1] >= a[i][0]]
            eb = [b[i][1] - b[i][0] for i in s if b[i][1] >= b[i][0]]
            return fmean(ea) - fmean(eb)
        point = diff(ids)
        lo, hi = bootstrap(diff, ids)
        print(f"  DINO - {other:17s} mean excess difference {point:7.1f} kcal (95% CI {lo:6.1f} to {hi:6.1f})")
    return tuned


# Sanity check against Cyril's summary: all 507 test images at the nominal 95% quantile.
print("Check vs. shared summary table (all test images, q = 0.95):")
for model in ("resnet", "swin", "dino"):
    est = estimates(rows[model]["test"], 0.95)
    print(f"  {model:6s} n={len(est)} under={under_rate(est):.2%} excess={mean_excess(est):.1f} "
          f"test MAE={fmean(abs(max(0.0, m) - y) for _, y, m, _ in rows[model]['test']):.1f}")

print(f"\nTraining-label Gaussian: mean {mu:.1f} kcal, sd {sd:.1f} kcal (n = {len(train_kcal)})")
report("same-session leak-free (primary)", leak_free)
report("exact ingredient-set disjoint (previous report)", exact_set_kept)

# Calibration: empirical coverage of nominal quantiles on the leak-free subset.
print("\nCalibration on leak-free test subset (share of meals with Y <= quantile):")
for model in ("resnet", "swin", "dino"):
    test = [r for r in rows[model]["test"] if r[0] in leak_free]
    cells = []
    for q in (0.5, 0.8, 0.9, 0.95):
        z = Z.inv_cdf(q)
        cells.append(f"q={q:.2f}: {fmean(y <= m + s * z for _, y, m, s in test):.1%}")
    print(f"  {model:6s} " + "  ".join(cells))
