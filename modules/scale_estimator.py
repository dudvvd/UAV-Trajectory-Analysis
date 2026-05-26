"""Scale and altitude estimation from homography geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaleResult:
    estimated_altitude: float
    pixels_per_meter: float
    scale_confidence: float
    homography_inliers: int


class ScaleEstimator:
    """Recover a practical monocular scale proxy using homography decomposition."""

    def __init__(self, K: np.ndarray, initial_altitude: float) -> None:
        self.K = np.asarray(K, dtype=float)
        self.fx = float(self.K[0, 0])
        self.initial_altitude = float(initial_altitude)
        self.previous_altitude = float(initial_altitude)
        self.reference_distance: float | None = None

    def estimate(self, prev_points: np.ndarray, curr_points: np.ndarray, homography: np.ndarray | None) -> ScaleResult:
        H = homography
        inliers = 0
        if H is None and len(prev_points) >= 4:
            H, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
            inliers = 0 if mask is None else int(mask.sum())
        elif len(prev_points) >= 4:
            _, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
            inliers = 0 if mask is None else int(mask.sum())

        if H is None or inliers < config.MIN_HOMOGRAPHY_INLIERS:
            return self._fallback(inliers)

        try:
            count, rotations, translations, normals = cv2.decomposeHomographyMat(H.astype(float), self.K)
        except cv2.error as exc:
            LOGGER.debug("Homography decomposition failed: %s", exc)
            return self._fallback(inliers)

        candidates: list[float] = []
        for i in range(count):
            normal = normals[i].reshape(3)
            t = translations[i].reshape(3)
            if normal[2] < config.GROUND_NORMAL_MIN_Z:
                continue
            t_norm = float(np.linalg.norm(t))
            if t_norm <= 1e-9:
                continue
            candidates.append(1.0 / t_norm)

        if not candidates:
            return self._fallback(inliers)

        distance = max(candidates)
        if self.reference_distance is None:
            self.reference_distance = distance
        ratio = distance / max(self.reference_distance, 1e-9)
        altitude = float(np.clip(self.initial_altitude * ratio, self.initial_altitude * 0.5, self.initial_altitude * 2.0))
        if abs(altitude - self.previous_altitude) > config.MAX_ALTITUDE_STEP_M:
            LOGGER.info(
                "Scale degraded: altitude jump %.2f m exceeds %.2f m",
                altitude - self.previous_altitude,
                config.MAX_ALTITUDE_STEP_M,
            )
            return self._fallback(inliers)
        self.previous_altitude = altitude
        confidence = float(np.clip(inliers / max(len(prev_points), 1), 0.0, 1.0))
        return ScaleResult(altitude, self.fx / max(altitude, 1e-6), confidence, inliers)

    def _fallback(self, inliers: int) -> ScaleResult:
        return ScaleResult(
            self.previous_altitude,
            self.fx / max(self.previous_altitude, 1e-6),
            0.0,
            int(inliers),
        )
