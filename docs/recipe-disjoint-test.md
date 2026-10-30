# Test subset without exact training recipe matches

Create a new evaluation dataset directory:

```bash
uv run filter_test_recipes.py \
  --manifest data/nutrition5k-sift-full/manifest.json \
  --source-dir data/nutrition5k \
  --output-dir data/nutrition5k-test-recipe-disjoint
```

The script excludes a test dish when its exact set of official ingredient IDs
appears in **any official RGB training dish**, including dishes without local
photos. Ingredient order, duplicate IDs, names, and weights do not affect set
equality. All ingredients, including seasonings, are included. Partial ingredient
overlap remains allowed; there is no minimum ingredient count.

The current full overhead test manifest retains **253 of 507** test dishes and
excludes **254**. This measures performance on ingredient combinations unseen
in the official training split; it does not establish the absence of all forms
of data leakage.

Output files:

- `manifest.json`: compatible with the existing evaluator, with retained samples
  in their original order and the split labeled `rgb_test_exact_recipe_disjoint`.
- `test_ids.txt`: retained test IDs.
- `removed_test_ids.txt`: excluded test IDs.
- `filter_report.json`: rule, counts, input checksums, and matching training IDs
  for every excluded test dish.

Source files and image files are preserved. Training references, their targets
and checksums, training means, and skipped-image records remain intact.
Relative sample and reference image paths are rebased to the new directory.
Missing ingredient metadata causes an error, and an existing output directory
is never overwritten.

Run the evaluator with the new directory:

```bash
uv run eval.py run --data-dir data/nutrition5k-test-recipe-disjoint --predictor mean
```

Keep the original official test manifest for benchmark comparisons, and apply
the same fixed filtered subset to every model being compared.
