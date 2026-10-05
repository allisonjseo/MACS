"""Behavior of the fixed-rule CV baseline on controlled images."""

import cv2
import numpy as np
import pytest

from classical import ClassicalPredictor
from schemas import ImageInput, Nutrition


def photo(food_radius: int, size: int = 320) -> ImageInput:
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 160), 130, (230, 230, 230), -1)
    if food_radius:
        cv2.circle(frame, (160, 160), food_radius, (20, 80, 140), -1)
    frame = cv2.resize(frame, (size, size))
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return ImageInput(data=encoded.tobytes(), media_type="image/png")


def baseline() -> ClassicalPredictor:
    return ClassicalPredictor(
        Nutrition(calories_kcal=200, protein_g=20, carbs_g=30, fat_g=10)
    )


def test_larger_food_region_increases_estimate_without_changing_macro_ratios():
    predictor = baseline()
    small = predictor.predict(photo(35)).mean
    large = predictor.predict(photo(70)).mean
    assert large.calories_kcal > 3 * small.calories_kcal
    assert large.protein_g / small.protein_g == pytest.approx(
        large.calories_kcal / small.calories_kcal
    )
    assert predictor.last_metadata.fallback is None
    assert 0 < predictor.last_metadata.food_coverage < 1


def test_estimate_is_independent_of_image_resolution():
    predictor = baseline()
    small = predictor.predict(photo(60, size=320)).mean
    large = predictor.predict(photo(60, size=1280)).mean
    assert large.calories_kcal == pytest.approx(small.calories_kcal, rel=0.03)


def test_empty_plate_produces_zero_nutrition():
    predictor = baseline()
    assert predictor.predict(photo(0)).mean == Nutrition(
        calories_kcal=0, protein_g=0, carbs_g=0, fat_g=0
    )
    assert predictor.last_metadata.fallback is None


def test_missing_plate_uses_explicit_mean_fallback():
    ok, encoded = cv2.imencode(".png", np.zeros((320, 320, 3), dtype=np.uint8))
    assert ok
    predictor = baseline()
    prediction = predictor.predict(
        ImageInput(data=encoded.tobytes(), media_type="image/png")
    )
    assert prediction.mean == predictor.config.reference_nutrition
    assert prediction.sigma is None
    assert predictor.last_metadata.fallback == "no_white_plate"


def test_invalid_image_is_a_failure():
    with pytest.raises(ValueError, match="decodable image"):
        baseline().predict(ImageInput(data=b"not a PNG", media_type="image/png"))
