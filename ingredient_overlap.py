"""Compare official train/test ingredient identities and measured quantities."""

from collections import defaultdict
from heapq import nlargest
from typing import Any


def ingredient_weights(item):
    weights = defaultdict(float)
    for ingredient in item.get("ingredients", []):
        weights[ingredient["id"]] += ingredient["grams"]
    return dict(weights)


def compare_weights(left, right):
    shared = left.keys() & right.keys()
    union = left.keys() | right.keys()
    score = len(shared) / len(union) if union else 0
    maximum = sum(max(left.get(key, 0), right.get(key, 0)) for key in union)
    quantity = (
        sum(min(left[key], right[key]) for key in shared) / maximum if maximum else 0
    )
    same_amounts = (
        bool(union)
        and left.keys() == right.keys()
        and all(abs(left[key] - right[key]) <= 0.1 for key in union)
    )
    return score, quantity, same_amounts, len(shared), len(union)


def build_report(dataset):
    """Retain exact pairs and nearest matches with and without available photos."""
    train, test = {}, {}
    photos = {"train": set(), "test": set()}
    ingredient_names = {}
    for item in dataset["items"]:
        if item.get("has_image"):
            photos[item["split"]].add(item["id"])
        for ingredient in item.get("ingredients", []):
            ingredient_names[ingredient["id"]] = ingredient["name"]
        weights = ingredient_weights(item)
        if weights:
            (train if item["split"] == "train" else test)[item["id"]] = weights
    groups = defaultdict(lambda: {"train_ids": [], "test_ids": []})
    for split, collection in (("train", train), ("test", test)):
        for dish, weights in collection.items():
            groups[tuple(sorted(weights))][split + "_ids"].append(dish)
    exact_groups: list[dict[str, Any]] = [
        {
            "ingredient_ids": list(ids),
            **group,
            "pairs": len(group["train_ids"]) * len(group["test_ids"]),
            "photo_train_ids": [
                dish for dish in group["train_ids"] if dish in photos["train"]
            ],
            "photo_test_ids": [
                dish for dish in group["test_ids"] if dish in photos["test"]
            ],
        }
        for ids, group in groups.items()
        if group["train_ids"] and group["test_ids"]
    ]
    exact_groups.sort(key=lambda group: (-group["pairs"], group["ingredient_ids"]))
    rows: list[dict[str, Any]] = []
    same_amount_pairs = 0
    for test_id, weights in sorted(test.items()):
        candidates = []
        for train_id, other in train.items():
            score, quantity, same, shared, union = compare_weights(weights, other)
            if not shared:
                continue
            candidates.append((score, quantity, same, train_id, shared, union))
        nearest = nlargest(12, candidates)
        keep = {value[3]: value for value in nearest}
        # Metadata-only matches must not crowd photos out of the nearest results.
        keep.update(
            (value[3], value)
            for value in nlargest(
                12, (value for value in candidates if value[3] in photos["train"])
            )
        )
        keep.update((value[3], value) for value in candidates if value[0] == 1)
        matches = [
            {
                "train_id": train_id,
                "similarity": score,
                "quantity_similarity": quantity,
                "same_amounts": same,
                "shared": shared,
                "union": union,
            }
            for score, quantity, same, train_id, shared, union in sorted(
                keep.values(), reverse=True
            )
        ]
        exact = sum(value[0] == 1 for value in candidates)
        same_amount_pairs += sum(value[2] for value in candidates)
        rows.append({"test_id": test_id, "exact_count": exact, "matches": matches})
    rows.sort(
        key=lambda row: (
            -(row["matches"][0]["similarity"] if row["matches"] else 0),
            -(row["matches"][0]["quantity_similarity"] if row["matches"] else 0),
            row["test_id"],
        )
    )
    train_types = set().union(*(weights.keys() for weights in train.values()))
    test_types = set().union(*(weights.keys() for weights in test.values()))
    summary = {
        "train_dishes": len(train),
        "test_dishes": len(test),
        "exact_pairs": sum(group["pairs"] for group in exact_groups),
        "exact_test_dishes": sum(row["exact_count"] > 0 for row in rows),
        "near_test_dishes": sum(
            bool(row["matches"]) and row["matches"][0]["similarity"] >= 0.8
            for row in rows
        ),
        "exact_groups": len(exact_groups),
        "same_amount_pairs": same_amount_pairs,
        "photo_exact_pairs": sum(
            len(group["photo_train_ids"]) * len(group["photo_test_ids"])
            for group in exact_groups
        ),
        "shared_ingredient_types": len(train_types & test_types),
        "train_ingredient_types": len(train_types),
        "test_ingredient_types": len(test_types),
        "missing_ingredients": sum(
            not item.get("ingredients") for item in dataset["items"]
        ),
        "id_overlap": dataset.get("id_overlap", []),
    }
    print("Ingredient overlap:", summary, flush=True)
    return {
        "summary": summary,
        "rows": rows,
        "groups": exact_groups,
        "ingredient_names": ingredient_names,
    }
