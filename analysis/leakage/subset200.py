import csv, random, sys
from pathlib import Path
d = Path(sys.argv[1])
labels, sets = set(), {}
for n in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
    for r in csv.reader(open(d / n)):
        if r and r[0].startswith("dish_"):
            labels.add(r[0]); sets[r[0]] = frozenset(r[6::7])
train = set((d / "rgb_train_ids.txt").read_text().split())
test = set((d / "rgb_test_ids.txt").read_text().split())
avail = {l.split()[0] for l in (d / "avail.txt").read_text().splitlines() if l.split()[1] == "200"}
c = sorted(test & labels); random.Random(42).shuffle(c)
sub = [x for x in c if x in avail][:200]
train_sets = {sets[x] for x in train if x in sets}
print("200-subset recipe-disjoint:", sum(sets[x] not in train_sets for x in sub))
