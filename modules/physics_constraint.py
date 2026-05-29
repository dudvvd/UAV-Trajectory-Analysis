"""Physical plausibility constraints for visual motion estimates."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class PhysicsResult:
    pixel_dx: float
    pixel_dy: float
    altitude: float
    confidence_multiplier: float
    flag: int
    predict_only: bool


class PhysicsConstraint:
    """Clip impossible motion and down-weight suspicious frames."""

    def __init__(self) -> None:
        self.motion_history: deque[np.ndarray] = deque(maxlen=config.CONSISTENCY_HISTORY)

    def reset(self) -> None:
        self.motion_history.clear()

    def apply(
        self,
        pixel_dx: float,
        pixel_dy: float,
        prev_altitude: float,
        estimated_altitude: float,
        fx: float,
        phase: str,
    ) -> PhysicsResult:
        conf = 1.0
        flag = 0
        predict_only = False
        disp_px = float(np.hypot(pixel_dx, pixel_dy))
        max_px = config.MAX_HORIZ_DISP_M / max(prev_altitude / max(fx, 1e-6), 1e-6)
        if disp_px > max_px > 0:
            if config.CLIP_HORIZ_DISP:
                scale = max_px / disp_px
                pixel_dx *= scale
                pixel_dy *= scale
                logging.info("Horizontal physics constraint clipped displacement %.2fpx to %.2fpx", disp_px, max_px)
            else:
                logging.info("Horizontal physics constraint down-weighted displacement %.2fpx above %.2fpx", disp_px, max_px)
            conf *= 0.2
            flag = 1
            predict_only = config.HORIZ_CLIP_PREDICT_ONLY

        alt_limit = self._alt_limit(phase)
        delta_alt = estimated_altitude - prev_altitude
        if abs(delta_alt) > alt_limit:
            estimated_altitude = prev_altitude + np.sign(delta_alt) * alt_limit
            conf *= 0.3
            flag = 2 if flag == 0 else flag
            logging.info("Altitude physics constraint clipped delta %.2fm to %.2fm", delta_alt, alt_limit)

        vec = np.array([pixel_dx, pixel_dy], dtype=float)
        if len(self.motion_history) >= 3:
            hist = np.vstack(self.motion_history)
            mean = hist.mean(axis=0)
            std = np.maximum(hist.std(axis=0), 1.0)
            if np.any(np.abs(vec - mean) > 3.0 * std):
                if config.CONSISTENCY_REPLACE_MOTION:
                    vec = mean
                    pixel_dx, pixel_dy = float(vec[0]), float(vec[1])
                conf *= 0.3
                flag = 3 if flag == 0 else flag
                logging.info("3-sigma motion consistency constraint down-weighted current displacement")
        self.motion_history.append(vec)

        if estimated_altitude < config.ALT_MIN or estimated_altitude > config.ALT_MAX:
            estimated_altitude = float(np.clip(estimated_altitude, config.ALT_MIN, config.ALT_MAX))
            conf *= 0.2
            flag = 2 if flag == 0 else flag
            logging.warning("Altitude hard boundary applied: %.2fm", estimated_altitude)

        return PhysicsResult(float(pixel_dx), float(pixel_dy), float(estimated_altitude), conf, flag, predict_only)

    @staticmethod
    def _alt_limit(phase: str) -> float:
        if phase.startswith("A"):
            return config.MAX_DELTA_ALT_CLIMB
        if phase == "B":
            return config.MAX_DELTA_ALT_CRUISE
        return config.MAX_DELTA_ALT_TRANS
