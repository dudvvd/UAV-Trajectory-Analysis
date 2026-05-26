"""One-time yaw calibration from early GPS direction and visual motion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np

import config
from modules.feature_tracker import FeatureTracker
from utils.geo_utils import lonlat_to_local_m

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class YawCalibrationResult:
    yaw_deg: float
    visual_vectors: list[tuple[float, float]]
    gps_vectors: list[tuple[float, float]]
    used_frames: int


class YawCalibrator:
    """Estimate camera-to-ENU yaw by aligning visual directions to GPS directions."""

    def __init__(self, fx: float) -> None:
        self.fx = float(fx)

    def calibrate(
        self,
        image_paths: list[Path],
        gt_lon: list[float],
        gt_lat: list[float],
        altitude_m: float,
        start: int,
        end: int,
    ) -> YawCalibrationResult:
        sample_end = min(end, start + config.YAW_CALIBRATION_FRAMES)
        if sample_end - start < 3:
            LOGGER.warning("Yaw calibration skipped: not enough frames")
            return YawCalibrationResult(config.CAMERA_YAW_DEG, [], [], 0)

        east, north = lonlat_to_local_m(
            np.asarray(gt_lon[start:sample_end]),
            np.asarray(gt_lat[start:sample_end]),
            float(gt_lon[start]),
            float(gt_lat[start]),
        )
        gps_steps = np.column_stack([np.diff(east), np.diff(north)])
        total_gps = float(np.sum(np.linalg.norm(gps_steps, axis=1)))
        if total_gps < config.YAW_MIN_GPS_DISPLACEMENT_M:
            LOGGER.warning("Yaw calibration skipped: first %d frames move only %.2f m", sample_end - start, total_gps)
            return YawCalibrationResult(config.CAMERA_YAW_DEG, [], [], 0)

        tracker = FeatureTracker()
        first_gray = tracker.read_gray(image_paths[start])
        tracker.initialize(first_gray)
        visual_vectors: list[tuple[float, float]] = []
        gps_vectors: list[tuple[float, float]] = []
        meters_per_pixel = altitude_m / max(self.fx, 1e-6)

        for idx in range(start + 1, sample_end):
            gray = tracker.read_gray(image_paths[idx])
            if FeatureTracker.blur_score(gray) < config.BLUR_LAPLACIAN_THRESHOLD:
                tracker.initialize(gray)
                continue
            tracking = tracker.track(gray)
            gps_step = gps_steps[idx - start - 1]
            if tracking.inlier_ratio <= 0.5:
                continue
            if np.linalg.norm(gps_step) < 0.05:
                continue
            visual_vectors.append((tracking.pixel_dx * meters_per_pixel, tracking.pixel_dy * meters_per_pixel))
            gps_vectors.append((float(gps_step[0]), float(gps_step[1])))

        if len(visual_vectors) < 3:
            LOGGER.warning("Yaw calibration skipped: only %d usable visual/GPS pairs", len(visual_vectors))
            return YawCalibrationResult(config.CAMERA_YAW_DEG, visual_vectors, gps_vectors, len(visual_vectors))

        visual = np.asarray(visual_vectors, dtype=float)
        gps = np.asarray(gps_vectors, dtype=float)

        def objective(theta_deg: float) -> float:
            theta = radians(theta_deg)
            R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], dtype=float)
            rotated = visual @ R.T
            dot = np.sum(rotated * gps, axis=1)
            denom = np.linalg.norm(rotated, axis=1) * np.linalg.norm(gps, axis=1)
            valid = denom > 1e-9
            cosang = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
            return float(np.mean(1.0 - cosang))

        try:
            from scipy.optimize import minimize_scalar

            result = minimize_scalar(objective, bounds=(-180.0, 180.0), method="bounded")
            yaw = float(result.x)
        except Exception as exc:
            LOGGER.warning("SciPy yaw optimization unavailable; using grid fallback: %s", exc)
            coarse = np.linspace(-180.0, 180.0, 721)
            best = float(coarse[int(np.argmin([objective(v) for v in coarse]))])
            fine = np.linspace(best - 1.0, best + 1.0, 401)
            yaw = float(fine[int(np.argmin([objective(v) for v in fine]))])

        LOGGER.info("Yaw calibration: yaw_deg=%.2f using %d pairs", yaw, len(visual_vectors))
        return YawCalibrationResult(yaw, visual_vectors, gps_vectors, len(visual_vectors))
