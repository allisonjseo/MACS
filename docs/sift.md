# SIFT image retrieval baseline

## Purpose and inputs

The `sift` backend is a classical computer-vision baseline. It extracts local
SIFT descriptors from a test food image, finds visually similar labeled training
images, and predicts nutrition from those images' labels. It trains no model
weights and never uses test labels to make predictions.

The manifest must contain labeled training references. `prepare --reference-n N`
selects `N` images from the official RGB training split that have overhead
photos and labels; preparation errors if fewer than `N` are available. In the
full Nutrition5k manifest used for this evaluation, all 2,755 available training
overhead photos are references and all 507 available test overhead photos are
evaluated. Other split IDs lack those overhead photos.

## Feature extraction

OpenCV decodes each image as grayscale. If its longest side exceeds 512 pixels,
it is resized proportionally to a 512-pixel longest side. OpenCV's SIFT detector
then extracts at most 500 descriptors by default; each SIFT descriptor has 128
components. Descriptors are cached as `.npy` files keyed by the image-content
SHA-256 and feature limit, so repeated runs can skip extraction.

Images with fewer than two descriptors cannot participate in matching. Invalid
image bytes raise an error and cause that sample's prediction to be recorded as
failed.

## Matching and neighbor ranking

The evaluator concatenates descriptors from all eligible reference images into a
single index. PyTorch calculates Euclidean (L2) distances from each query
descriptor to that full pool. It uses CUDA when `torch.cuda.is_available()` and
otherwise CPU. Query descriptors are processed in chunks of 64 to bound the
temporary distance matrix. Descriptor matching uses the Lowe ratio test with a
default threshold of 0.75: a query descriptor is accepted when its nearest
reference descriptor is less than 0.75 times as far away as its second-nearest
reference descriptor.

This ratio test is applied across the combined training pool. Each accepted
query descriptor votes for the reference image containing its nearest descriptor.
For a reference image `i`, the vote count is normalized as:

```text
similarity_i = accepted_votes_i / sqrt(query_descriptor_count * reference_descriptor_count_i)
```

Images with no accepted votes are discarded. The remaining references are sorted
by similarity, and up to five are retained. Thus, the method can use fewer than
five neighbors when fewer than five references receive accepted votes.

## Point prediction

For each nutrient, the predicted value is the similarity-weighted mean of the
retained neighbors' training labels:

```text
prediction = sum(similarity_i * nutrient_label_i) / sum(similarity_i)
```

The predictor records the selected reference IDs and their similarity values in
each result's `backend_metadata.neighbors` field.

## Heuristic nutrient ranges

The evaluator also returns a range for each nutrient. First it calculates the
similarity-weighted population standard deviation of the selected neighbors'
labels around the weighted point prediction. The half-width is the largest of
that spread, 25% of the point prediction, and the nutrient-specific minimum:

| Nutrient | Minimum half-width |
|---|---:|
| Calories | 50 kcal |
| Protein | 3 g |
| Carbohydrates | 5 g |
| Fat | 3 g |

The lower endpoint is `max(0, prediction - half_width)`; the upper endpoint is
`prediction + half_width`. This creates nonzero ranges even when the selected
neighbors agree or only one neighbor is available. These widths are a simple
uncertainty heuristic, not calibrated 95% prediction intervals.

## Fallbacks

If an image has fewer than two query descriptors, or no reference image passes
the matching process, the point prediction is the training-label mean saved in
the manifest. Its range uses the per-nutrient standard deviation across all
references as the spread, subject to the same 25% rule and minimum half-width.

## Evaluation metrics

The shared evaluator reports per-nutrient MAE for the point predictions. For
each nutrient, it also reports interval coverage (the percentage of successful
predictions whose range contains the ground truth), mean interval width (upper
minus lower endpoint), and the number of intervals scored. The calorie-specific
within-20%-of-truth percentage is reported as a separate metric. Failed
predictions are excluded from accuracy and interval metrics.

## Reproduce

Starting with the raw Nutrition5k tree at `./data/nutrition5k`, prepare a full
RGB test manifest and training reference cache. The prepared SIFT data is written
to `./data/nutrition5k-sift-full`; preparation refuses to overwrite an existing
manifest, so use another data directory when starting over.

```sh
uv run eval.py prepare \
  --source-dir ./data/nutrition5k \
  --data-dir ./data/nutrition5k-sift-full \
  --n 507 \
  --reference-n 2755 \
  --seed 42
```

Then smoke-test two images and evaluate the full test set. These commands put
results under `/tmp`, outside the repository, so prediction and metrics
directories are not added to a clone:

```sh
uv run eval.py run \
  --data-dir ./data/nutrition5k-sift-full \
  --predictor sift \
  --limit 2 \
  --output /tmp/sift-smoke-results

uv run eval.py run \
  --data-dir ./data/nutrition5k-sift-full \
  --predictor sift \
  --output /tmp/sift-full-results
```

The `--limit 2` invocation is a smoke test. `prepare` needs the raw metadata,
RGB split lists, and overhead images; the `run` commands need the resulting
manifest, reference/test images, and can reuse the `sift-features` cache. The
same manifest, seed, and SIFT options make the evaluation inputs reproducible.
