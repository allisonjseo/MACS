"""Detect test dishes that are another scan of the same physical plate as a training dish.

Usage: python -I same_plate.py DATA_DIR  (see fetch_metadata.sh)
Same plate = dish IDs (Unix timestamps) within GAP seconds AND the shared
ingredients carry the same grams (within max(2 g, 3%)) for at least 80% of the
smaller dish's mass. Compares counts against the exact-set and 0.8-overlap rules.
"""

import contextlib
import io
import runpy
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

data = sys.argv[1]
sys.argv = ["leakage_analysis.py", data]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("leakage_analysis.py")))
meta, best_match = g["meta"], g["best_match"]


def stamp(dish):
    return int(dish.split("_")[1])


def matched_fraction(a, b):
    """Share of the smaller dish's grams in ingredients whose amounts agree."""
    wa, wb = meta[a]["weights"], meta[b]["weights"]
    small = wa if sum(wa.values()) <= sum(wb.values()) else wb
    total = sum(small.values())
    if not total:
        return 0.0
    agree = sum(
        min(wa[k], wb[k])
        for k in wa.keys() & wb.keys()
        if abs(wa[k] - wb[k]) <= max(2.0, 0.03 * max(wa[k], wb[k]))
    )
    return agree / total


def same_plate_matches(queries, pool, gap):
    ordered = sorted(pool, key=stamp)
    stamps = [stamp(d) for d in ordered]
    found = {}
    for dish in queries:
        t = stamp(dish)
        near = ordered[bisect_left(stamps, t - gap): bisect_right(stamps, t + gap)]
        hits = [(matched_fraction(dish, o), o) for o in near]
        hits = [h for h in hits if h[0] >= 0.8]
        if hits:
            found[dish] = max(hits)
    return found


by_set = g["test_report"]["_by_set"]
test = g["test_photos"]
exact = {d for d in test if by_set.get(meta[d]["set"])}
weighted = {d for d in test if best_match(d, g["train_all"])[0] >= 0.8}

for gap in (600, 3600, 2 * 3600, 4 * 3600, 8 * 3600, 24 * 3600, 72 * 3600):
    same = same_plate_matches(test, g["train_all"], gap)
    s = set(same)
    print(f"\ngap <= {gap / 60:.0f} min: {len(s)} test dishes share a plate with training")
    print(f"  also removed by exact-set rule: {len(s & exact)}; by 0.8 overlap: {len(s & weighted)}")
    print(f"  same-plate dishes the 0.8 rule KEEPS: {len(s - weighted)}; the exact rule KEEPS: {len(s - exact)}")

same = same_plate_matches(test, g["train_all"], 4 * 3600)
print("\n0.8-rule removals that are NOT same-plate:", len(weighted - set(same)),
      "of", len(weighted))
single = [d for d in weighted - set(same) if len(meta[d]["weights"]) <= 2]
print("  of those, dishes with 1-2 ingredients (common simple foods):", len(single))

print("\nExamples of same-plate pairs the 0.8 rule keeps:")
for d in sorted(set(same) - weighted)[:6]:
    frac, o = same[d]
    print(f"  {d} vs {o}: overlap {best_match(d, [o])[0]:.2f}, matched mass {frac:.2f}, "
          f"gap {abs(stamp(d) - stamp(o)) / 60:.0f} min")

val_same = same_plate_matches(g["validation"], g["training"], 4 * 3600)
print(f"\nvalidation: {len(val_same)} of {len(g['validation'])} share a plate with the training portion")
