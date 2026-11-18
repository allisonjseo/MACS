"""Audit exact-ingredient-set matching on Nutrition5k test and validation splits.

Usage: python -I leakage_analysis.py DATA_DIR
"""

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

data = Path(sys.argv[1])

meta = {}
for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
    with (data / name).open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].startswith("dish_"):
                continue
            weights = defaultdict(float)
            names = {}
            for i in range(6, len(row) - 6, 7):
                weights[row[i]] += float(row[i + 2])
                names[row[i]] = row[i + 1]
            meta[row[0]] = {
                "kcal": float(row[1]),
                "mass": float(row[2]),
                "weights": dict(weights),
                "names": names,
                "set": frozenset(weights),
            }

train_ids = set((data / "rgb_train_ids.txt").read_text().split())
test_ids = set((data / "rgb_test_ids.txt").read_text().split())
available = {
    line.split()[0] for line in (data / "avail.txt").read_text().splitlines()
    if line.split()[1] == "200"
}


def weighted_overlap(a, b):
    """Weighted Jaccard over grams: sum(min)/sum(max). 1.0 = same amounts."""
    keys = a.keys() | b.keys()
    top = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in keys) / top if top else 0.0


def set_jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def best_match(dish, pool):
    """Best reference by weighted overlap; also report its set Jaccard."""
    best = (0.0, 0.0, None)
    for other in pool:
        if not (meta[dish]["set"] & meta[other]["set"]):
            continue
        w = weighted_overlap(meta[dish]["weights"], meta[other]["weights"])
        if w > best[0]:
            best = (w, set_jaccard(meta[dish]["set"], meta[other]["set"]), other)
    return best


def audit(name, queries, reference_pool, near=0.8):
    by_set = defaultdict(list)
    for dish in reference_pool:
        by_set[meta[dish]["set"]].append(dish)
    exact = [d for d in queries if by_set.get(meta[d]["set"])]
    kept = [d for d in queries if not by_set.get(meta[d]["set"])]
    # For exact-set matches: how similar are the amounts, really?
    exact_best = {
        d: max(weighted_overlap(meta[d]["weights"], meta[o]["weights"])
               for o in by_set[meta[d]["set"]])
        for d in exact
    }
    false_removals = [d for d, w in exact_best.items() if w < near]
    # For kept dishes: is there a near-duplicate (different set, similar grams)?
    kept_best = {d: best_match(d, reference_pool) for d in kept}
    missed = [d for d, (w, _, _) in kept_best.items() if w >= near]
    weighted_dupes = [d for d in queries
                      if (exact_best.get(d, 0) >= near) or (d in missed)]
    hist = Counter(min(int(w * 10), 9) for w in exact_best.values())
    return {
        "split": name,
        "queries": len(queries),
        "exact_set_matches": len(exact),
        "exact_kept": len(kept),
        "exact_matches_with_amount_overlap_below_threshold": len(false_removals),
        "kept_with_weighted_near_duplicate": len(missed),
        "weighted_rule_flagged": len(weighted_dupes),
        "weighted_rule_kept": len(queries) - len(weighted_dupes),
        "exact_match_amount_overlap_hist_by_decile": dict(sorted(hist.items())),
        "_false_removals": false_removals,
        "_missed": missed,
        "_by_set": by_set,
        "_kept_best": kept_best,
    }


def describe(dish):
    m = meta[dish]
    parts = sorted(m["weights"].items(), key=lambda kv: -kv[1])
    return f"{dish} ({m['kcal']:.0f} kcal, {m['mass']:.0f} g): " + ", ".join(
        f"{m['names'][k]} {g:.0f}g" for k, g in parts[:6])


# Test audit: available test photos vs the ENTIRE official training split.
test_photos = sorted(test_ids & available & meta.keys())
train_all = sorted(train_ids & meta.keys())
test_report = audit("test vs all official train", test_photos, train_all)

# Validation audit: reproduce the seeded dish-ID split from vision_data.py.
candidates = sorted(train_ids & meta.keys())
samples = [d for d in candidates if d in available]
rng_samples = list(samples)
random.Random(42).shuffle(rng_samples)
count = min(len(rng_samples) - 1, max(1, math.ceil(len(rng_samples) * 0.2)))
validation, training = rng_samples[:count], rng_samples[count:]
val_report = audit("validation vs training portion", validation, training)

out = {}
for report in (test_report, val_report):
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    out[report["split"]] = public
    print(json.dumps(public, indent=1))

print("\nsplit sizes:", len(training), "train /", len(validation), "validation /",
      len(test_photos), "test photos")

# Examples: same ingredient set, very different amounts (cake = cookie case).
print("\n=== Test dishes removed by exact-set rule but amounts differ a lot ===")
examples = sorted(test_report["_false_removals"],
                  key=lambda d: max(weighted_overlap(meta[d]["weights"], meta[o]["weights"])
                                    for o in test_report["_by_set"][meta[d]["set"]]))
for d in examples[:3]:
    o = max(test_report["_by_set"][meta[d]["set"]],
            key=lambda o: weighted_overlap(meta[d]["weights"], meta[o]["weights"]))
    print(" test :", describe(d))
    print(" train:", describe(o),
          f"| amount overlap {weighted_overlap(meta[d]['weights'], meta[o]['weights']):.2f}")

print("\n=== Test dishes kept by exact-set rule but with a near-duplicate ===")
for d in test_report["_missed"][:3]:
    w, j, o = test_report["_kept_best"][d]
    print(" test :", describe(d))
    print(" train:", describe(o), f"| amount overlap {w:.2f}, set Jaccard {j:.2f}")

json.dump(out, open(data.parent / "leakage_summary.json", "w"), indent=1)
