# Local ingredient overlap viewer

From the repository root, run:

```bash
python3 image_browser.py --host 0.0.0.0 --port 8000
```

Open `http://rhea.int.autonlab.org:8000` from the lab network or VPN. On this
machine, `http://192.168.6.79:8000` is the IP alternative. For access through
SSH, run this on your laptop using your normal SSH username/host:

```bash
ssh -N -L 8000:localhost:8000 mhua2@rhea.int.autonlab.org
```

Then open `http://localhost:8000`. If your usual SSH connection uses a jump host
or alias, use that connection in the command above.

The viewer uses only Python's standard library and reads
`data/nutrition5k` by default; override with `--data-dir PATH`. It indexes the
official RGB train/test ID lists, ingredient metadata, and available overhead
RGB photos. Dishes without photos are included in ingredient comparisons.
Train refers to the official split before any
training/validation partition. It never downloads or edits the dataset.

The default view shows test dishes with identical ingredient sets in training
and requires local photos for both dishes. Switch Photo availability to
“All dishes” to include metadata-only dishes. Summary counts, groups, and CSV
exports follow this setting. Each recipe group also shows photo coverage and
all-metadata totals separately; its comparison button prefers a test dish with
a photo, and training choices prefer photos when ingredient similarity ties.
Choose a test dish, then select a matching training dish to compare ingredient
weights. Lower the overlap threshold to find near matches. Search by dish ID
or ingredient, or use Exact recipe groups to see shared ingredient sets with
their train/test dish counts. Export pairs downloads the current filtered
matches as CSV. Photos are expanded below each ingredient comparison.
Each card in Exact recipe groups displays up to five available train photos
and five test photos, labeled with dish IDs and total ingredient weights.
Click a thumbnail to open the original image. The separate ingredient-weight
comparison button opens one train/test pair for a detailed quantity comparison.

Ingredient overlap is Jaccard similarity over official ingredient IDs. All
ingredients, including seasonings, count equally. Quantity overlap is the sum
of minimum ingredient gram weights divided by the sum of maximum weights.
Exact ingredient sets can still have different quantities; the same-amount
count requires no more than 0.1 g difference per ingredient. Repeated IDs are
summed. Every exact ingredient-set pair is retained, plus the top 12 overall
matches and top 12 matches with training photos per test dish. This lets the
photo filter find its nearest matches even when closer metadata-only matches
exist. Ranking uses ingredient overlap, photo availability, and quantity overlap.
Ingredient overlap alone does not establish data leakage.

The server serves only the viewer, `/api/dataset`, `/api/ingredients`, and indexed photos.
There is no authentication; network access to port 8000 permits viewing.
