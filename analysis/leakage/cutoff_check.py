"""Sample test dishes whose closest training dish is near the 0.8 overlap cutoff.

Usage: python -I cutoff_check.py DATA_DIR
Prints each pair with ingredient grams, kcal, and the time gap between dish IDs
(Nutrition5k dish IDs are Unix timestamps, so scans of one plate are minutes apart).
"""

import contextlib
import io
import json
import random
import runpy
import sys
from pathlib import Path

data = sys.argv[1]
sys.argv = ["leakage_analysis.py", data]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("leakage_analysis.py")))
meta, best_match = g["meta"], g["best_match"]

pairs = []
for dish in g["test_photos"]:
    overlap, jaccard, other = best_match(dish, g["train_all"])
    if other and 0.65 <= overlap < 0.95:
        pairs.append((overlap, jaccard, dish, other))

bins = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
rng = random.Random(0)
chosen = []
for low, high in bins:
    inside = [p for p in pairs if low <= p[0] < high]
    picks = rng.sample(inside, min(6, len(inside)))
    print(f"\n##### overlap {low:.2f}-{high:.2f}: {len(inside)} test dishes, showing {len(picks)}")
    for overlap, jaccard, dish, other in sorted(picks):
        gap = abs(int(dish.split("_")[1]) - int(other.split("_")[1]))
        print(f"\n[{overlap:.2f}] set Jaccard {jaccard:.2f}, ID time gap {gap / 3600:.1f} h")
        for label, d in (("test ", dish), ("train", other)):
            m = meta[d]
            parts = sorted(m["weights"].items(), key=lambda kv: -kv[1])
            print(f"  {label} {d} {m['kcal']:.0f} kcal {m['mass']:.0f} g: "
                  + ", ".join(f"{m['names'][k]} {w:.0f}" for k, w in parts))
        chosen.append({"overlap": overlap, "test": dish, "train": other, "gap_h": gap / 3600})

# Time-gap summary across ALL near-cutoff pairs, by bin.
print("\n##### share of pairs whose IDs are within 1 hour (likely same plate), by bin")
for low, high in bins:
    inside = [p for p in pairs if low <= p[0] < high]
    close = sum(abs(int(p[2].split("_")[1]) - int(p[3].split("_")[1])) < 3600 for p in inside)
    print(f"  {low:.2f}-{high:.2f}: {close}/{len(inside)}")

json.dump(chosen, open(Path(data).parent / "cutoff_pairs.json", "w"), indent=1)
