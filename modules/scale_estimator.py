"""Multi-level scale and altitude estimation for monocular UAV imagery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaleCandidate:
    level: int
    altitude: float
    confidence: float
    raw_value: float


@dataclass(frozen=True)
class ScaleResult:
    estimated_altitude: float
    pixels_per_meter: float
    scale_confidence: float
    homography_inliers: int
    scale_method: int
    divergence_ratio: float
    affine_scale: float
    raw_alt_estimate: float


class ScaleEstimator:
    """Estimate altitude with a robust four-level fallback structure."""

    def __init__(
        self,
        K: np.ndarray,
        initial_altitude: float,
        calibration_confidence: float = 0.0,
        enabled_levels: set[int] | None = None,
    ) -> None:
        self.K = np.asarray(K, dtype=float)
        self.fx = float(self.K[0, 0])
        self.initial_altitude = float(initial_altitude)
        self.previous_altitude = float(initial_altitude)
        self.calibration_confidence = float(calibration_confidence)
        self.enabled_levels = enabled_levels or {1, 2, 3, 4}

    def estimate(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        homography: np.ndarray | None,
        image_shape: tuple[int, int],
        dt: float,
        v_alt: float,
        current_altitude: float | None = None,
    ) -> ScaleResult:
        """Return weighted altitude estimate and diagnostics."""

        base_alt = float(self.previous_altitude if current_altitude is None else current_altitude)
        candidates: list[ScaleCandidate] = []
        divergence_ratio = float("nan")
        affine_scale = float("nan")
        homography_inliers = 0

        if 1 in self.enabled_levels:
            candidate = self._divergence_level(prev_points, curr_points, image_shape, base_alt)
            if candidate is not None:
                divergence_ratio = candidate.raw_value
                candidates.append(candidate)

        if 2 in self.enabled_levels:
            candidate = self._affine_level(prev_points, curr_points, base_alt)
            if candidate is not None:
                affine_scale = candidate.raw_value
                candidates.append(candidate)

        if 3 in self.enabled_levels and self.calibration_confidence > 0.8:
            candidate, homography_inliers = self._homography_level(prev_points, curr_points, homography, base_alt)
            if candidate is not None:
                candidates.append(candidate)

        if 4 in self.enabled_levels:
            fallback_alt = base_alt + float(v_alt) * max(float(dt), 0.0)
            candidates.append(ScaleCandidate(4, fallback_alt, 0.05, float(v_alt)))

        strong = [c for c in candidates if c.confidence >= config.SCALE_MIN_LEVEL_CONFIDENCE]
        if not strong:
            raw_altitude = base_alt + float(v_alt) * max(float(dt), 0.0)
            method = 4
            confidence = 0.0
        else:
            weighted_altitudes = []
            weights = []
            for candidate in strong:
                weight = config.SCALE_LEVEL_WEIGHTS.get(candidate.level, 0.0) * candidate.confidence
                if weight <= 0:
                    continue
                weighted_altitudes.append(candidate.altitude)
                weights.append(weight)
            if not weights:
                raw_altitude = base_alt
                method = 4
                confidence = 0.0
            else:
                raw_altitude = float(np.average(weighted_altitudes, weights=weights))
                best = max(strong, key=lambda c: c.confidence)
                method = best.level
                confidence = float(best.confidence)

        if abs(raw_altitude - base_alt) > config.MAX_ALTITUDE_STEP_M:
            LOGGER.info(
                "Scale altitude jump clipped: %.2f m -> max %.2f m",
                raw_altitude - base_alt,
                config.MAX_ALTITUDE_STEP_M,
            )
            raw_altitude = base_alt + np.sign(raw_altitude - base_alt) * config.MAX_ALTITUDE_STEP_M
            confidence *= 0.5

        self.previous_altitude = float(raw_altitude)
        return ScaleResult(
            estimated_altitude=float(raw_altitude),
            pixels_per_meter=self.fx / max(float(raw_altitude), 1e-6),
            scale_confidence=confidence,
            homography_inliers=homography_inliers,
            scale_method=method,
            divergence_ratio=float(divergence_ratio),
            affine_scale=float(affine_scale),
            raw_alt_estimate=float(raw_altitude),
        )

    def score_levels(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        homography: np.ndarray | None,
        image_shape: tuple[int, int],
        dt: float,
        v_alt: float,
        altitude: float,
    ) -> dict[int, float]:
        """Return one-frame confidence scores for validation without state updates."""

        original_altitude = self.previous_altitude
        result = self.estimate(prev_points, curr_points, homography, image_shape, dt, v_alt, altitude)
        scores = {level: 0.0 for level in (1, 2, 3, 4)}
        scores[result.scale_method] = result.scale_confidence
        if not np.isnan(result.divergence_ratio):
            scores[1] = max(scores[1], self._divergence_confidence(prev_points, image_shape, result.divergence_ratio))
        if not np.isnan(result.affine_scale):
            scores[2] = max(scores[2], self._affine_confidence(result.affine_scale, 0.0))
        self.previous_altitude = original_altitude
        return scores

    def _divergence_level(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        image_shape: tuple[int, int],
        base_altitude: float,
    ) -> ScaleCandidate | None:
        if len(prev_points) < 8 or len(curr_points) < 8:
            return None
        h, w = image_shape[:2]
        center = np.array([w * 0.5, h * 0.5], dtype=float)
        p0 = np.asarray(prev_points, dtype=float).reshape(-1, 2)
        p1 = np.asarray(curr_points, dtype=float).reshape(-1, 2)
        translation = np.median(p1 - p0, axis=0)
        p1 = p1 - translation
        d0 = np.linalg.norm(p0 - center, axis=1)
        d1 = np.linalg.norm(p1 - center, axis=1)
        valid = d0 > 3.0
        if int(valid.sum()) < 8:
            return None
        ratios = d1[valid] / d0[valid]
        r = float(np.median(ratios))
        if not np.isfinite(r) or r <= 1e-6:
            return None
        altitude = base_altitude * r
        confidence = self._divergence_confidence(p0[valid], image_shape, r)
        return ScaleCandidate(1, float(altitude), confidence, r)

    def _divergence_confidence(self, points: np.ndarray, image_shape: tuple[int, int], ratio: float) -> float:
        confidence = 0.8
        if len(points) < config.SCALE_MIN_DIVERGENCE_POINTS:
            confidence *= 0.5
        if ratio < config.SCALE_DIVERGENCE_MIN or ratio > config.SCALE_DIVERGENCE_MAX:
            confidence *= 0.3
        h, w = image_shape[:2]
        center = np.array([w * 0.5, h * 0.5], dtype=float)
        offsets = np.asarray(points, dtype=float).reshape(-1, 2) - center
        quadrants = [
            np.sum((offsets[:, 0] >= 0) & (offsets[:, 1] >= 0)),
            np.sum((offsets[:, 0] < 0) & (offsets[:, 1] >= 0)),
            np.sum((offsets[:, 0] < 0) & (offsets[:, 1] < 0)),
            np.sum((offsets[:, 0] >= 0) & (offsets[:, 1] < 0)),
        ]
        occupied = sum(q > 0 for q in quadrants)
        if occupied < 3 or (max(quadrants) / max(sum(quadrants), 1)) > 0.65:
            confidence *= 0.7
        return float(np.clip(confidence, 0.0, 1.0))

    def _affine_level(self, prev_points: np.ndarray, curr_points: np.ndarray, base_altitude: float) -> ScaleCandidate | None:
        if len(prev_points) < 8 or len(curr_points) < 8:
            return None
        A, _ = cv2.estimateAffinePartial2D(
            np.asarray(prev_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(curr_points, dtype=np.float32).reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
            maxIters=1000,
            confidence=0.99,
        )
        if A is None:
            return None
        linear = A[:, :2]
        det = float(np.linalg.det(linear))
        if det <= 1e-9:
            return None
        scale = float(np.sqrt(abs(det)))
        rotation = degrees(atan2(linear[1, 0], linear[0, 0]))
        confidence = self._affine_confidence(scale, rotation)
        return ScaleCandidate(2, float(base_altitude * scale), confidence, scale)

    @staticmethod
    def _affine_confidence(scale: float, rotation_deg: float) -> float:
        confidence = 0.7
        if abs(rotation_deg) > config.SCALE_AFFINE_ROTATION_MAX_DEG:
            confidence *= 0.3
        if scale < config.SCALE_DIVERGENCE_MIN or scale > config.SCALE_DIVERGENCE_MAX:
            confidence *= 0.3
        return float(np.clip(confidence, 0.0, 1.0))

    def _homography_level(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        homography: np.ndarray | None,
        base_altitude: float,
    ) -> tuple[ScaleCandidate | None, int]:
        H = homography
        inliers = 0
        if len(prev_points) >= 4:
            if H is None:
                H, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
            else:
                _, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
            inliers = 0 if mask is None else int(mask.sum())
        if H is None or inliers < config.MIN_HOMOGRAPHY_INLIERS:
            return None, inliers

        try:
            count, _, translations, normals = cv2.decomposeHomographyMat(H.astype(float), self.K)
        except cv2.error as exc:
            LOGGER.debug("Homography decomposition failed: %s", exc)
            return None, inliers

        distances = []
        for i in range(count):
            normal = normals[i].reshape(3)
            translation = translations[i].reshape(3)
            if normal[2] < config.GROUND_NORMAL_MIN_Z:
                continue
            t_norm = float(np.linalg.norm(translation))
            if t_norm > 1e-9:
                distances.append(1.0 / t_norm)
        if not distances:
            return None, inliers
        relative = float(np.median(distances))
        altitude = np.clip(base_altitude * relative, base_altitude * 0.8, base_altitude * 1.25)
        confidence = float(np.clip((inliers / max(len(prev_points), 1)) * 0.5, 0.0, 0.5))
        return ScaleCandidate(3, float(altitude), confidence, relative), inliers
