# Evaluating other datasets

The common task is **one RGB image → totals for the pictured portion**, in kcal
and grams. Keep Nutrition5k as the existing benchmark and report external
datasets separately. The table below is a researched shortlist, not a claim
that these datasets have been downloaded or evaluated in this checkout.

## Recommended datasets

| Priority | Dataset and labels | Why add it | Protocol and limitations |
| --- | --- | --- | --- |
| 1 | **[SNAPMe](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708545/)**: 3,311 food photos from 95 participants, linked to ASA24 dietary records; includes before/after meal photos and packaged-food photos. [Data DOI](https://doi.org/10.15482/USDA.ADC/1528346), [authors' analysis repository](https://github.com/JulesLarke-USDA/SNAPMe). | My first choice for testing transfer to ordinary phone photos and everyday meals. | Start with non-packaged before-meal images. Audit image-to-food-record aggregation and whether the recorded amount represents food served or consumed. These are dietary records, not independently weighed plate labels. Keep participants together if creating development/test splits. |
| 2 | **[SimpleFood45](https://arxiv.org/html/2404.12257v2#S3.SS5)**: 513 smartphone images of 45 food items across 12 food types, with volume, weight, and energy labels. [Dataset site](https://lorenz.ecn.purdue.edu/~gvinod/simplefood45/). | A small, interpretable test of portion and camera-angle generalization. | Evaluate calories; leave macros unobserved. Images include a checkerboard. Energy is calculated from measured weight and FNDDS food matching. There are multiple views of each physical item: group by item, and disclose whether using the authors' split or a new object-disjoint protocol. |
| 3 | **[MetaFood3D](https://arxiv.org/html/2409.01966v2)**: 743 scanned food objects across 131 categories, with RGB-D video, weights, calories, protein, carbohydrate, and fat. [Dataset site](https://lorenz.ecn.purdue.edu/~food3d/). | Broader controlled food coverage with all four existing prediction targets; also tests viewpoint sensitivity. | Use a fixed real RGB frame per object for the primary single-image comparison. Keep all views of an object together. Nutrition totals derive from measured weight and FNDDS nutrient densities. Report rendered-image experiments separately from real photographs. |
| Optional | **[ECUSTFD](https://github.com/Liang-yc/ECUSTFD-resized-)**: 2,978 images, 19 food types, top/side views, and weight/volume information. | A complementary test with simple foods and a coin for scale. | Lower priority: inspect the supplied tables before deriving calorie targets. A weight/volume label alone does not supply calories or macros. Any nutrient lookup must be explicit, versioned, and reported as a derived label. Group related views of each portion. |

These choices test different kinds of transfer; that rationale is our proposed
evaluation design, not a demonstrated result. Food classification accuracy on
Food-101 or UECFood is a different task: class names alone cannot establish the
nutrition of a pictured portion. Likewise, recipe nutrition must be aligned to
the portion in an image before it can serve as a regression target.

The sizes above follow the authors' papers. The [MetaFood3D project
page](https://lorenz.ecn.purdue.edu/~food3d/) describes an earlier 637-object,
108-category release, requires an access request and download password, and
states a CC BY-NC 4.0 license. Record which release is actually obtained.
The Purdue pages were reachable through search indexing after direct page
fetches failed. Obtain the source archives and their actual data license before
preparing them; a paper's license is not evidence of the dataset's license.
No native archive parser or automatic downloader for the shortlisted datasets
is included yet.

## Shared evaluation interface

`schemas.Manifest` is the adapter contract, independent of both the dataset and
predictor. An adapter produces samples with a stable `id`, an image path and
SHA-256 checksum, and a nutrition `target`. Optional `group_id` identifies a
participant, physical portion, or object. PNG, JPEG, WebP, and GIF inputs have
their corresponding MIME type; existing PNG manifests still work.

Predictions retain the strict four-nutrient `Nutrition` contract. Evaluation
targets may contain any nonempty subset through `NutritionTarget`; missing
labels are `null` or omitted, never filled with zero. A measured zero is valid.
This lets a calorie-only dataset score the same models without inventing macro
labels or changing the prediction prompt.

`eval.py prepare` remains the existing Nutrition5k preparation command, with the
same defaults and sampling. For a new dataset, export a normalized CSV and use
`eval.py import-csv`, or write an adapter that returns the same typed manifest.
The shared `run` command requires no dataset-specific scoring or predictor code.

### CSV format

For example, a full-label export could be:

```csv
id,image,group_id,calories_kcal,protein_g,carbs_g,fat_g
meal_001,photos/meal_001.jpg,participant_01,420,24,50,14
meal_002,photos/meal_002.jpg,participant_02,310,18,37,10
```

A calorie-only export can instead be:

```csv
id,image,group_id,calories_kcal
object_001_view_01,images/object_001_view_01.png,object_001,125
object_002_view_01,images/object_002_view_01.png,object_002,270
```

Requirements:

- `id` and `image` columns are required; `group_id` is optional. At least one
  nutrient must be observed per row. Missing columns or empty nutrient cells
  mean unobserved. Non-finite, negative, or nonnumeric labels are rejected.
- All labels are **totals for the visible portion**, not per 100 g, per recipe,
  per day, or per serving unless that serving is what the image shows. Perform
  and document any necessary conversion upstream. No units are guessed.
- IDs are unique. Rename native fields to the canonical names and remove other
  columns; unknown/duplicate columns and malformed rows are rejected.
- Images are local. Relative paths resolve from the CSV's directory, or an
  explicit `--image-root`. Absolute paths also work. The importer records
  absolute paths and does not copy images; moving the archive requires a new
  manifest. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`.

After exporting real labels and obtaining the corresponding images:

```sh
uv run eval.py import-csv \
  --labels /path/to/external-test.csv \
  --image-root /path/to/dataset \
  --data-dir data/external-test \
  --dataset 'DatasetName-release-version' \
  --source 'https://dataset-source.example/release' \
  --license 'actual source data license' \
  --split 'published-test-or-described-protocol' \
  --seed 42

uv run eval.py run \
  --data-dir data/external-test \
  --predictor vision \
  --checkpoint /path/to/existing-checkpoint.pt \
  --device cpu \
  --output runs/external-test-checkpoint
```

These are templates, not commands already executed. Use the existing checkpoint
and its training-time normalization for a transfer evaluation. Changing training
data or fitting an external dataset's labels would be a separate experiment.
For Codex, the same manifest works with `--predictor codex --model MODEL
--reasoning-effort LEVEL`; pin the same prompt and model configuration across
datasets. Before an actual experiment, state the full chosen runtime
configuration and differences from the reference run.

Import defaults are all CSV rows, seed 42, and zero training references. IDs are
sorted, then shuffled with that seed; optional `--n N` fixes a subset.
References use seed+1. A missing selected image fails preparation, rather than
silently changing the evaluation population. Existing manifests are never
replaced. CSV hashes, source identity, split, requested counts, roots, ordering,
units, selected samples, and image checksums are saved in the manifest.

### Training-dependent baselines

Checkpoint and Codex evaluations do not require any training labels in the
evaluation manifest. Mean, classical, and SIFT baselines require a complete
four-field training mean; SIFT also requires labeled image references. Supply
an independent normalized training CSV:

```sh
uv run eval.py import-csv \
  --labels /path/to/test.csv \
  --training-labels /path/to/train.csv \
  --training-split train \
  --reference-n 500 \
  --data-dir data/external-with-references \
  --dataset 'DatasetName-version' \
  --source 'https://dataset-source.example/release' \
  --license 'actual source data license' \
  --split test
```

The mean uses **every row in that training CSV**, not the evaluation samples or
just the retrieval subset. All training labels must be complete. Training image
paths may be empty when `--reference-n 0`; otherwise selected references must
exist. `--training-image-root` sets their root separately. Evaluation/training
ID and supplied group overlaps are rejected across the entire tables before
subsetting. Identical image bytes across selected evaluation samples and cached
references are also rejected. These checks cannot detect related samples when
the adapter omits their group identities.

For transfer comparisons, declare whether the baseline's training statistics
and retrieval pool come from Nutrition5k or the external dataset. An externally
adapted baseline and a frozen Nutrition5k checkpoint answer different questions.
Custom adapters may explicitly populate `training_mean`, `training_label_count`,
`training_split`, and `references` from the intended reference corpus; never
estimate them from the evaluation labels.

### Results and reproducibility

Every run saves the exact input `manifest.json` alongside `config.json`,
`predictions.json`, and `summary.json`. Config records the dataset and split as
well as the manifest hash, selected IDs, backend settings, seed, versions, and
Git provenance. The effective run config is printed before predictions start.

Each nutrient metric reports `count`, the number of successful predictions with
that label, and MAE. Interval count, coverage, and width include only rows with
both that label and uncertainty. If no labels qualify, count is zero and metrics
are null. Calorie ±20% accuracy excludes missing and zero-calorie targets.
Failures are counted separately; always compare success rates alongside errors.
The prior Nutrition5k metric definitions are preserved.

Report one result per dataset and protocol, including label counts, image
selection, checkpoint/model, input modality, and exclusions. For multi-view
datasets, the default scorer weights images equally; export one fixed view per
object for an object-weighted comparison, or state the different weighting.
Do not pool heterogeneous datasets into one headline MAE. Keep model selection
and prompt tuning separate from each final evaluation split.
