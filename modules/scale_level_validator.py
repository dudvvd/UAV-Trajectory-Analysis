"""Preflight validation for scale-estimation levels."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import config
from modules.feature_tracker import FeatureTracker
from modules.scale_estimator import ScaleEstimator

LOGGER = logging.getLogger(__name__)


class ScaleLevelValidator:
    """Measure early confidence for scale levels and disable dead paths."""

    def __init__(self, estimator: ScaleEstimator) -> None:
        self.estimator = estimator

    def validate(self, image_paths: list[Path], start: int, end: int, altitude: float) -> set[int]:
        sample_end = min(end, start + config.SCALE_VALIDATION_FRAMES)
        if sample_end - start < 3:
            LOGGER.warning("Scale level validation skipped: not enough frames")
            return set(self.estimator.enabled_levels)

        tracker = FeatureTracker()
        first_gray = tracker.read_gray(image_paths[start])
        tracker.initialize(first_gray)
        scores: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}

        for idx in range(start + 1, sample_end):
            gray = tracker.read_gray(image_paths[idx])
            if FeatureTracker.blur_score(gray) < config.BLUR_LAPLACIAN_THRESHOLD:
                tracker.initialize(gray)
                continue
            tracking = tracker.track(gray)
            frame_scores = self.estimator.score_levels(
                tracking.prev_points,
                tracking.curr_points,
                tracking.homography,
                gray.shape[:2],
                0.0,
                0.0,
                altitude,
            )
            for level, score in frame_scores.items():
                scores[level].append(float(score))

        enabled = set()
        report = {}
        for level, values in scores.items():
            mean_conf = float(np.mean(values)) if values else 0.0
            report[level] = mean_conf
            if level == 4 or mean_conf >= config.SCALE_MIN_LEVEL_CONFIDENCE:
                enabled.add(level)
        LOGGER.info(
            "Scale level validation: L1=%.2f L2=%.2f L3=%.2f L4=%.2f enabled=%s",
            report[1],
            report[2],
            report[3],
            report[4],
            sorted(enabled),
        )
        return enabled
