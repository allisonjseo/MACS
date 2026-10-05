"""A fixed-rule, white-plate coverage baseline for overhead RGB images."""

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from predictors import Predictor
from schemas import FIELDS, ImageInput, Nutrition, Prediction, PredictorConfig


class ClassicalConfig(PredictorConfig):
    backend: Literal["classical"] = "classical"
    algorithm: str = "white_plate_coverage_v1"
    reference_nutrition: Nutrition
    reference_coverage: float = Field(default=0.35, gt=0, le=1)
    plate_min_brightness: int = 80
    plate_max_saturation: int = 50
    food_chroma_distance: float = 12
    food_lightness_difference: float = 25
    opencv_version: str = cv2.__version__
    numpy_version: str = np.__version__


class ClassicalDiagnostics(BaseModel):
    plate_pixels: int = 0
    food_pixels: int = 0
    food_coverage: float | None = None
    portion_multiplier: float = 1
    fallback: str | None = None


class ClassicalPredictor(Predictor):
    """Scale training-mean nutrition by visible food coverage; no fitted model."""

    config: ClassicalConfig

    def __init__(self, reference_nutrition: Nutrition):
        self.config = ClassicalConfig(reference_nutrition=reference_nutrition)

    def predict(self, image: ImageInput) -> Prediction:
        bgr = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or min(bgr.shape[:2]) < 32:
            raise ValueError(
                "Classical baseline needs a decodable image at least 32px wide/high"
            )
        # Fixed working resolution bounds runtime and normalizes morphology sizes.
        height, width = bgr.shape[:2]
        scale = 640 / max(height, width)
        bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)))
        height, width = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        neutral = cv2.inRange(
            hsv,
            (0, 0, self.config.plate_min_brightness),
            (179, self.config.plate_max_saturation, 255),
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        neutral = cv2.morphologyEx(neutral, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            neutral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if not 0.15 * height * width <= area <= 0.95 * height * width:
                continue
            moments = cv2.moments(hull)
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            distance = (cx / width - 0.5) ** 2 + (cy / height - 0.5) ** 2
            candidates.append((area / (1 + 4 * distance), hull))
        if not candidates:
            self.last_metadata = ClassicalDiagnostics(fallback="no_white_plate")
            return Prediction(mean=self.config.reference_nutrition.model_copy())
        hull = max(candidates, key=lambda candidate: candidate[0])[1]
        plate = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(plate, [hull], -1, 255, cv2.FILLED)
        # Exclude the rim, whose shadows otherwise look like food.
        rim = max(3, round(0.02 * min(height, width))) | 1
        plate = cv2.erode(plate, np.ones((rim, rim), dtype=np.uint8))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        plate_colors = lab[(plate > 0) & (neutral > 0)]
        if not len(plate_colors):
            self.last_metadata = ClassicalDiagnostics(fallback="no_exposed_plate")
            return Prediction(mean=self.config.reference_nutrition.model_copy())
        reference = np.median(plate_colors, axis=0)
        chroma = np.linalg.norm(lab[:, :, 1:] - reference[1:], axis=2)
        food = (
            (plate > 0)
            & (
                (chroma >= self.config.food_chroma_distance)
                | (reference[0] - lab[:, :, 0] >= self.config.food_lightness_difference)
            )
        ).astype(np.uint8) * 255
        food = cv2.morphologyEx(food, cv2.MORPH_OPEN, kernel)
        food = cv2.morphologyEx(food, cv2.MORPH_CLOSE, kernel)
        food = cv2.bitwise_and(food, plate)
        plate_pixels = cv2.countNonZero(plate)
        _, _, stats, _ = cv2.connectedComponentsWithStats(food)
        # Ignore crumbs/noise smaller than 0.2% of plate area.
        food_pixels = sum(
            int(area)
            for area in stats[1:, cv2.CC_STAT_AREA]
            if area >= 0.002 * plate_pixels
        )
        coverage = food_pixels / plate_pixels
        multiplier = coverage / self.config.reference_coverage
        self.last_metadata = ClassicalDiagnostics(
            plate_pixels=plate_pixels,
            food_pixels=food_pixels,
            food_coverage=coverage,
            portion_multiplier=multiplier,
        )
        return Prediction(
            mean=Nutrition(
                **{
                    field: getattr(self.config.reference_nutrition, field) * multiplier
                    for field in FIELDS
                }
            )
        )
