#!/usr/bin/env bash
# Download the public Nutrition5k metadata and split lists, and record which
# dishes have an overhead RGB photo (avail.txt: "<dish> <HTTP status>").
set -euo pipefail
out=${1:?usage: fetch_metadata.sh OUTPUT_DIR}
base=https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset
mkdir -p "$out"
for f in metadata/dish_metadata_cafe1.csv metadata/dish_metadata_cafe2.csv \
  dish_ids/splits/rgb_train_ids.txt dish_ids/splits/rgb_test_ids.txt; do
  curl -sf -o "$out/$(basename "$f")" "$base/$f"
done
cat "$out/rgb_train_ids.txt" "$out/rgb_test_ids.txt" | xargs -P 48 -I{} sh -c \
  'echo "{} $(curl -s -o /dev/null -I -w "%{http_code}" '"$base"'/imagery/realsense_overhead/{}/rgb.png)"' \
  > "$out/avail.txt"
