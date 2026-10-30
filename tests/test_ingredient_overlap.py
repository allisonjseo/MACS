"""Ingredient comparison correctness independent of the real dataset."""

import pytest

from image_browser import load_dataset
from ingredient_overlap import build_report, compare_weights, ingredient_weights


def dish(dish_id, split, weights):
    return {
        "id": dish_id,
        "split": split,
        "ingredients": [
            {"id": key, "name": key, "grams": grams} for key, grams in weights.items()
        ],
    }


def test_ingredient_identity_is_distinct_from_quantities():
    overlap, quantity, same, shared, union = compare_weights(
        {"rice": 100, "beans": 50}, {"rice": 200, "beans": 100}
    )
    assert overlap == 1
    assert quantity == 0.5
    assert not same
    assert (shared, union) == (2, 2)


def test_partial_overlap_and_same_amount_tolerance():
    overlap, quantity, same, shared, union = compare_weights(
        {"rice": 100, "beans": 50}, {"rice": 100, "chicken": 50}
    )
    assert overlap == pytest.approx(1 / 3)
    assert quantity == 0.5
    assert not same
    assert (shared, union) == (1, 3)
    assert compare_weights({"rice": 100}, {"rice": 100.05})[2]
    assert not compare_weights({"rice": 100}, {"rice": 100.2})[2]


def test_all_exact_pairs_are_retained_beyond_top_twelve():
    items = [dish(f"train_{i}", "train", {"rice": 100 + i}) for i in range(20)]
    items += [dish("test", "test", {"rice": 100})]
    result = build_report({"items": items})
    assert result["summary"]["exact_pairs"] == 20
    assert result["summary"]["exact_test_dishes"] == 1
    assert result["summary"]["same_amount_pairs"] == 1
    assert len(result["rows"][0]["matches"]) == 20
    assert result["rows"][0]["matches"][0]["train_id"] == "train_0"
    assert result["groups"][0]["pairs"] == 20


def test_repeated_ids_are_summed_and_names_do_not_merge_ids():
    item = {
        "ingredients": [
            {"id": "a", "name": "rice", "grams": 25},
            {"id": "a", "name": "rice", "grams": 75},
            {"id": "b", "name": "rice", "grams": 10},
        ]
    }
    assert ingredient_weights(item) == {"a": 100, "b": 10}
    assert compare_weights({"a": 100}, {"b": 100})[0] == 0


def test_dishes_without_photos_are_included(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "dish_metadata_cafe1.csv").write_text(
        "dish_a,100,50,1,2,3,ingr_1,rice,50,100,1,2,3\n"
        "dish_b,100,50,1,2,3,ingr_1,rice,50,100,1,2,3\n"
    )
    splits = tmp_path / "dish_ids/splits"
    splits.mkdir(parents=True)
    (splits / "rgb_train_ids.txt").write_text("dish_a\n")
    (splits / "rgb_test_ids.txt").write_text("dish_b\n")
    data, images = load_dataset(tmp_path)
    assert images == {}
    assert len(data["items"]) == 2
    assert all(not item["has_image"] for item in data["items"])
    result = build_report(data)
    assert result["summary"]["exact_pairs"] == 1


def test_missing_ingredient_metadata_is_not_an_exact_empty_recipe():
    result = build_report(
        {"items": [dish("train", "train", {}), dish("test", "test", {})]}
    )
    assert result["summary"]["exact_pairs"] == 0
    assert result["summary"]["missing_ingredients"] == 2
    assert result["rows"] == []


def test_photo_matches_survive_metadata_only_top_twelve():
    items = [
        dish(f"metadata_{i}", "train", {"rice": 100, "salt": 1}) for i in range(15)
    ]
    items += [dish("photo_train", "train", {"rice": 100, "beans": 50})]
    items += [dish("photo_test", "test", {"rice": 100, "salt": 1})]
    items[-2]["has_image"] = items[-1]["has_image"] = True
    result = build_report({"items": items})
    matches = result["rows"][0]["matches"]
    assert any(match["train_id"] == "photo_train" for match in matches)
    assert result["summary"]["exact_pairs"] == 15
    assert result["summary"]["photo_exact_pairs"] == 0


def test_recipe_groups_report_photo_coverage_separately():
    items = [
        dish("missing_train", "train", {"rice": 100}),
        dish("photo_train", "train", {"rice": 110}),
        dish("missing_test", "test", {"rice": 90}),
        dish("photo_test", "test", {"rice": 100}),
    ]
    items[1]["has_image"] = items[3]["has_image"] = True
    result = build_report({"items": items})
    group = result["groups"][0]
    assert group["pairs"] == 4
    assert group["photo_train_ids"] == ["photo_train"]
    assert group["photo_test_ids"] == ["photo_test"]
    assert result["summary"]["photo_exact_pairs"] == 1
