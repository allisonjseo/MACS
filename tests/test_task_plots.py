"""Task definitions, quantile semantics, dominance, and population integrity."""

import hashlib
import json
import math

import numpy as np
import pytest

pytest.importorskip("plotnine")
import pandas as pd

from task_plots import (
    load_population,
    nondominated,
    quantile_sweep,
    read_predictions,
    task_metrics,
    validation_data,
)


def test_conditional_excess_includes_ties_and_does_not_average_over_underestimates():
    result = task_metrics([4, 1, 0], [2, 1, 1])
    assert result == {
        "p_under": 1 / 3,
        "mean_over_kcal": 1.0,
        "n_over": 2,
        "n_under": 1,
        "n": 3,
    }


def test_no_overestimates_is_undefined_and_zero_ties_are_not_underestimates():
    assert task_metrics([0, 1], [1, 2])["mean_over_kcal"] is None
    assert task_metrics([0, 0], [0, 0])["mean_over_kcal"] == 0
    assert task_metrics([0, 0], [0, 0])["p_under"] == 0


def test_quantile_95_is_one_sided_and_zero_floor_is_explicit():
    result = quantile_sweep([100], [10], [0], [0.95])
    assert result.iloc[0]["mean_over_kcal"] == pytest.approx(116.4485362695)
    raw = quantile_sweep([-1], [1], [0], [0.5], clip_zero=False)
    clipped = quantile_sweep([-1], [1], [0], [0.5], clip_zero=True)
    assert raw.iloc[0]["p_under"] == 1
    assert clipped.iloc[0]["p_under"] == 0
    assert clipped.iloc[0]["mean_over_kcal"] == 0
    assert clipped.iloc[0]["raw_quantile_coverage"] == 0
    assert clipped.iloc[0]["n_floored"] == 1


@pytest.mark.parametrize(
    "std,q", [(0, 0.5), (-1, 0.5), (1, 0), (1, 1), (float("nan"), 0.5)]
)
def test_invalid_gaussian_parameters_and_quantile_endpoints_rejected(std, q):
    with pytest.raises(ValueError):
        quantile_sweep([1], [std], [1], [q])


def test_pareto_dominance_preserves_ties_and_omits_undefined_points():
    frame = pd.DataFrame(
        {
            "p_under": [0.1, 0.2, 0.3, 0.1, 0.1, 1],
            "mean_over_kcal": [100, 110, 90, 100, 101, None],
        }
    )
    assert np.array_equal(nondominated(frame), [True, False, True, True, False, False])


def test_full_population_required_even_when_a_subset_has_good_scores(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n": 2,
                "missing_overhead_images": ["missing"],
                "samples": [
                    {"id": "a", "target": {"calories_kcal": 10}},
                    {"id": "b", "target": {"calories_kcal": 20}},
                ],
            }
        )
    )
    population = load_population(manifest)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "predictor": {},
            }
        )
    )
    records = [
        {"id": "a", "status": "ok", "target": {"calories_kcal": 10}},
        {"id": "b", "status": "ok", "target": {"calories_kcal": 20}},
    ]
    (run / "predictions.json").write_text(json.dumps(records[::-1]))
    ordered, _ = read_predictions(run, population)
    assert [r["id"] for r in ordered] == ["a", "b"]
    (run / "predictions.json").write_text(json.dumps(records[:1]))
    with pytest.raises(ValueError, match="ENTIRE population"):
        read_predictions(run, population)
    records[1]["status"] = "error"
    (run / "predictions.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="ENTIRE population"):
        read_predictions(run, population)


def test_wrong_labels_cannot_be_hidden_by_matching_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n": 1,
                "missing_overhead_images": [],
                "samples": [{"id": "a", "target": {"calories_kcal": 10}}],
            }
        )
    )
    population = load_population(manifest)
    (tmp_path / "config.json").write_text(
        json.dumps({"manifest_sha256": population.sha256, "predictor": {}})
    )
    (tmp_path / "predictions.json").write_text(
        json.dumps([{"id": "a", "status": "ok", "target": {"calories_kcal": 11}}])
    )
    with pytest.raises(ValueError, match="labels differ"):
        read_predictions(tmp_path, population)


def test_validation_metric_is_calorie_nll_and_keeps_original_selection(
    tmp_path, monkeypatch
):
    import task_plots

    monkeypatch.setattr(task_plots, "PROJECT", tmp_path)
    run = tmp_path / "training"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "validation_ids": ["heldout"],
                "manifest_sha256": "fixed",
                "target_std": [10.0],
                "selection": "five-target NLL",
            }
        )
    )
    (run / "summary.json").write_text(json.dumps({"best_epoch": 1}))
    history = [
        {
            "epoch": 1,
            "validation": {
                "loss": 0.1,
                "nll": {"calories_kcal": 5.0},
                "nll_standardized": {"calories_kcal": 5.0 - math.log(10)},
            },
        },
        {
            "epoch": 2,
            "validation": {
                "loss": 0.2,
                "nll": {"calories_kcal": 4.0},
                "nll_standardized": {"calories_kcal": 4.0 - math.log(10)},
            },
        },
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in history))
    curve, _, summary = validation_data([{"name": "Model", "training_run": "training"}])
    assert curve["calorie_nll"].tolist() == [5.0, 4.0]
    assert summary.iloc[0]["saved_epoch"] == 1
    assert summary.iloc[0]["minimum_calorie_nll_epoch"] == 2
