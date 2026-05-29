"""Multi-frame scale and altitude estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class ScaleResult:
    estimated_altitude: float
    scale_confidence: float
    scale_method: int
    divergence_ratio: float
    affine_scale: float
    consistency_diff: float
    pixels_per_meter_used: float
    is_alt_fixed: bool
    raw_alt_estimate: float
    cumulative_window_n: int
    delta_alt_cumulative: float


class ScaleEstimator:
    """Estimate altitude from compensated divergence over a sliding window."""

    def __init__(self, fx: float, initial_altitude: float, calibration_confidence: float) -> None:
        self.fx = float(fx)
        self.initial_altitude = float(initial_altitude)
        self.prev_altitude = float(initial_altitude)
        self.calibration_confidence = float(calibration_confidence)
        self.alt_deltas: deque[float] = deque(maxlen=len(config.ALT_SMOOTH_WEIGHTS))
        self.trend_history: deque[float] = deque(maxlen=config.ALT_TREND_WINDOW)
        self.ratio_history: deque[float] = deque(maxlen=max(config.N_WINDOW_CRUISE, config.CALIB_WINDOW))

    def estimate(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        affine_matrix: np.ndarray | None,
        prev_altitude: float,
        phase: str,
        dt: float,
        vertical_velocity: float,
    ) -> ScaleResult:
        div_ratio, div_conf = self._divergence_ratio(prev_points, curr_points, affine_matrix)
        aff_scale, aff_conf = self._affine_scale(affine_matrix)
        window_n = self._window_n(phase)

        if np.isfinite(div_ratio) and div_ratio > 1e-6:
            self.ratio_history.append(div_ratio)
        usable_n = min(window_n, len(self.ratio_history))
        cumulative_ratio = self._cumulative_ratio(usable_n)
        if np.isfinite(cumulative_ratio) and cumulative_ratio > 1e-6:
            delta_alt_total = prev_altitude * (1.0 / cumulative_ratio - 1.0)
            alt1 = prev_altitude + delta_alt_total / max(usable_n, 1)
        elif np.isfinite(div_ratio) and div_ratio > 1e-6:
            delta_alt_total = prev_altitude * (1.0 / div_ratio - 1.0)
            alt1 = prev_altitude + delta_alt_total
            usable_n = 1
        else:
            delta_alt_total = np.nan
            alt1 = np.nan

        if len(prev_points) < config.MIN_CUMULATIVE_POINTS:
            div_conf *= 0.3
        if usable_n < window_n:
            div_conf *= max(0.3, usable_n / max(window_n, 1))

        alt2 = prev_altitude / aff_scale if np.isfinite(aff_scale) and aff_scale > 1e-6 else np.nan
        diff = abs(div_ratio - aff_scale) if np.isfinite(div_ratio) and np.isfinite(aff_scale) else np.inf

        if np.isfinite(alt1) and np.isfinite(alt2):
            if diff < 0.005:
                raw_alt = 0.5 * alt1 + 0.5 * alt2
                conf = min(1.0, max(div_conf, aff_conf) * 1.2)
                method = 1
            elif diff < 0.02:
                raw_alt = 0.3 * alt1 + 0.7 * alt2
                conf = max(div_conf, aff_conf) * 0.7
                method = 2
            else:
                raw_alt = self._homography_or_predict(prev_points, curr_points, prev_altitude, dt, vertical_velocity)
                conf = 0.2
                method = 3 if self.calibration_confidence > config.HOMOGRAPHY_ENABLED_CONFIDENCE else 4
        elif np.isfinite(alt1):
            raw_alt, conf, method = alt1, div_conf, 1
        elif np.isfinite(alt2):
            raw_alt, conf, method = alt2, aff_conf, 2
        else:
            raw_alt, conf, method = prev_altitude + vertical_velocity * dt, 0.05, 4

        raw_alt, conf = self._phase_direction_check(prev_altitude, raw_alt, conf, phase)
        raw_alt, conf = self._trend_check(prev_altitude, raw_alt, conf)
        smooth_alt = self._smooth(prev_altitude, raw_alt)
        smooth_alt = float(np.clip(smooth_alt, config.ALT_MIN, config.ALT_MAX))
        self.prev_altitude = smooth_alt
        ppm = self.fx / max(smooth_alt, 1e-6)
        return ScaleResult(
            estimated_altitude=smooth_alt,
            scale_confidence=float(np.clip(conf, 0.0, 1.0)),
            scale_method=method,
            divergence_ratio=float(div_ratio),
            affine_scale=float(aff_scale),
            consistency_diff=float(diff),
            pixels_per_meter_used=float(ppm),
            is_alt_fixed=False,
            raw_alt_estimate=float(raw_alt),
            cumulative_window_n=int(usable_n),
            delta_alt_cumulative=float(delta_alt_total),
        )

    def _divergence_ratio(self, p0: np.ndarray, p1: np.ndarray, A: np.ndarray | None) -> tuple[float, float]:
        if len(p0) < 3 or len(p1) < 3:
            return np.nan, 0.2
        center0 = np.median(p0, axis=0)
        trans = np.array([A[0, 2], A[1, 2]], dtype=float) if A is not None else np.median(p1 - p0, axis=0)
        p1_comp = p1 - trans
        d0 = np.linalg.norm(p0 - center0, axis=1)
        d1 = np.linalg.norm(p1_comp - center0, axis=1)
        valid = d0 > 5.0
        if np.count_nonzero(valid) < 3:
            return np.nan, 0.2
        ratio = float(np.median(d1[valid] / d0[valid]))
        conf = 0.8
        if len(p0) < config.SCALE_MIN_DIVERGENCE_POINTS:
            conf *= 0.5
        if ratio < config.SCALE_DIVERGENCE_MIN or ratio > config.SCALE_DIVERGENCE_MAX:
            conf *= 0.3
        return ratio, conf

    @staticmethod
    def _affine_scale(A: np.ndarray | None) -> tuple[float, float]:
        if A is None:
            return np.nan, 0.1
        linear = A[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        rotation = abs(float(np.rad2deg(np.arctan2(A[0, 1], A[0, 0]))))
        conf = 0.7
        if rotation > config.SCALE_AFFINE_ROTATION_MAX_DEG:
            conf *= 0.3
        if scale < config.SCALE_DIVERGENCE_MIN or scale > config.SCALE_DIVERGENCE_MAX:
            conf *= 0.3
        return scale, conf

    def _homography_or_predict(self, p0: np.ndarray, p1: np.ndarray, prev_alt: float, dt: float, vz: float) -> float:
        if self.calibration_confidence <= config.HOMOGRAPHY_ENABLED_CONFIDENCE or len(p0) < 8:
            return prev_alt + vz * dt
        H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if H is None:
            return prev_alt + vz * dt
        scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        return prev_alt / max(scale, 1e-6)

    def _trend_check(self, prev_alt: float, raw_alt: float, conf: float) -> tuple[float, float]:
        delta = raw_alt - prev_alt
        if len(self.trend_history) >= config.ALT_TREND_WINDOW:
            signs = np.sign(np.asarray(self.trend_history))
            pos = np.mean(signs > 0)
            neg = np.mean(signs < 0)
            if (pos >= config.ALT_TREND_RATIO and delta < 0) or (neg >= config.ALT_TREND_RATIO and delta > 0):
                delta = float(np.mean(self.trend_history)) * 0.3
                raw_alt = prev_alt + delta
                conf *= 0.4
        self.trend_history.append(delta)
        return raw_alt, conf

    def _phase_direction_check(self, prev_alt: float, raw_alt: float, conf: float, phase: str) -> tuple[float, float]:
        delta = raw_alt - prev_alt
        if (
            config.ASSUME_TAKEOFF_CLIMB
            and phase.startswith("A")
            and prev_alt < self.initial_altitude + config.TAKEOFF_CLIMB_MIN_GAIN_M
        ):
            raw_alt = prev_alt + abs(delta)
            return raw_alt, conf * 0.7
        if phase == "A_up" and delta < 0:
            raw_alt = prev_alt + abs(delta)
            conf *= 0.7
        elif phase == "A_down" and delta > 0:
            raw_alt = prev_alt - abs(delta)
            conf *= 0.7
        return raw_alt, conf

    def _smooth(self, prev_alt: float, raw_alt: float) -> float:
        self.alt_deltas.append(raw_alt - prev_alt)
        weights = np.asarray(config.ALT_SMOOTH_WEIGHTS[-len(self.alt_deltas) :], dtype=float)
        weights /= weights.sum()
        delta = float(np.dot(weights, np.asarray(self.alt_deltas)))
        return prev_alt + delta

    def _cumulative_ratio(self, n: int) -> float:
        if n <= 0:
            return np.nan
        ratios = np.asarray(list(self.ratio_history)[-n:], dtype=float)
        ratios = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
        if len(ratios) == 0:
            return np.nan
        return float(np.prod(ratios))

    @staticmethod
    def _window_n(phase: str) -> int:
        if phase.startswith("A"):
            return config.N_WINDOW_CLIMB
        if phase == "B":
            return config.N_WINDOW_CRUISE
        return config.N_WINDOW_TRANS
