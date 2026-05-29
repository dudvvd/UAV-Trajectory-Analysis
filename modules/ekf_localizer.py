"""Extended Kalman filter for lon/lat/alt visual localization."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class EKFState:
    longitude: float
    latitude: float
    altitude: float
    v_lon: float
    v_lat: float
    v_alt: float

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.longitude, self.latitude, self.altitude, self.v_lon, self.v_lat, self.v_alt], dtype=float)


@dataclass
class EKFUpdateResult:
    state: EKFState
    update_mode: int
    alt_innovation: float


class EKFLocalizer:
    """Fuse visual lon/lat/alt observations with a phase-aware motion model."""

    def __init__(self, lon: float, lat: float, alt: float) -> None:
        self.x = np.array([lon, lat, alt, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([config.EKF_INIT_POS_VAR, config.EKF_INIT_POS_VAR, config.EKF_INIT_ALT_VAR, config.EKF_INIT_VEL_VAR, config.EKF_INIT_VEL_VAR, config.EKF_INIT_VEL_VAR])
        self.Q_current = np.diag([config.EKF_Q_POS_TRANS, config.EKF_Q_POS_TRANS, config.EKF_Q_ALT_TRANS, 1e-10, 1e-10, config.EKF_Q_ALT_TRANS])
        self.innovations: deque[float] = deque(maxlen=config.INNOVATION_WINDOW)
        self.alt_bad_count = 0
        self.predicted_alt = alt

    @property
    def state(self) -> EKFState:
        return EKFState(*self.x.tolist())

    def predict(self, dt: float, phase: str) -> EKFState:
        dt = max(float(dt), 1e-3)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        if phase == "B":
            F[5, 5] = config.EKF_V_ALT_DECAY_CRUISE
        else:
            F[2, 5] = dt
        self.x = F @ self.x
        self.x[2] = float(np.clip(self.x[2], config.ALT_MIN, config.ALT_MAX))
        self.predicted_alt = float(self.x[2])
        self.P = F @ self.P @ F.T + self._phase_q(phase)
        return self.state

    def update_visual(
        self,
        obs_lon: float,
        obs_lat: float,
        obs_alt: float,
        inlier_ratio: float,
        scale_confidence: float,
        phase: str,
        force_predict_only: bool = False,
    ) -> EKFUpdateResult:
        combined = max(float(inlier_ratio) * float(scale_confidence), 0.0)
        alt_innovation = float(obs_alt - self.predicted_alt)
        self._monitor_alt_reset(obs_alt, alt_innovation)
        if force_predict_only or inlier_ratio <= 0.5:
            return EKFUpdateResult(self.state, 3, alt_innovation)
        if scale_confidence > 0.3:
            z = np.array([obs_lon, obs_lat, obs_alt], dtype=float)
            H = np.zeros((3, 6))
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            R = np.diag([config.EKF_R_BASE_LON_LAT, config.EKF_R_BASE_LON_LAT, self._phase_r_alt(phase)]) / (combined + config.CONFIDENCE_EPS)
            mode = 1
        else:
            z = np.array([obs_lon, obs_lat], dtype=float)
            H = np.zeros((2, 6))
            H[0, 0] = H[1, 1] = 1.0
            R = np.diag([config.EKF_R_BASE_LON_LAT, config.EKF_R_BASE_LON_LAT]) / (max(inlier_ratio, 0.0) + config.CONFIDENCE_EPS)
            mode = 2
        self._kalman_update(z, H, R)
        self._adapt_q(abs(alt_innovation), phase)
        self.x[2] = float(np.clip(self.x[2], config.ALT_MIN, config.ALT_MAX))
        return EKFUpdateResult(self.state, mode, alt_innovation)

    def reset_altitude(self, altitude: float) -> None:
        self.x[2] = float(np.clip(altitude, config.ALT_MIN, config.ALT_MAX))
        self.x[5] = 0.0
        self.P[2, 2] = 10.0
        self.P[5, 5] = 1.0

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def _monitor_alt_reset(self, raw_alt: float, innovation: float) -> None:
        limit = getattr(config, "EKF_ALT_INNOVATION_THRESH", config.ALT_INNOVATION_LIMIT_M)
        if abs(innovation) > limit:
            self.alt_bad_count += 1
        else:
            self.alt_bad_count = 0
        if self.alt_bad_count >= config.ALT_INNOVATION_RESET_COUNT:
            logging.warning("EKF altitude local reset after persistent innovation %.2fm", innovation)
            self.reset_altitude(raw_alt)
            self.alt_bad_count = 0

    def _adapt_q(self, innovation_abs: float, phase: str) -> None:
        self.innovations.append(innovation_abs)
        base = self._phase_q(phase)
        if len(self.innovations) < config.INNOVATION_WINDOW:
            self.Q_current = base
            return
        values = np.asarray(self.innovations)
        mean = float(values.mean())
        std = float(values.std() + 1e-6)
        if values[-1] > mean + 2.0 * std:
            self.Q_current = base * (1.0 + 0.1 * values[-1])
        elif values[-1] < mean + std:
            self.Q_current = 0.95 * self.Q_current + 0.05 * base

    @staticmethod
    def _phase_q(phase: str) -> np.ndarray:
        if phase == "B":
            return np.diag([config.EKF_Q_POS_CRUISE, config.EKF_Q_POS_CRUISE, config.EKF_Q_ALT_CRUISE, 1e-10, 1e-10, config.EKF_Q_ALT_CRUISE])
        if phase.startswith("A"):
            return np.diag([config.EKF_Q_POS_CLIMB, config.EKF_Q_POS_CLIMB, config.EKF_Q_ALT_CLIMB, 1e-10, 1e-10, config.EKF_Q_ALT_CLIMB])
        return np.diag([config.EKF_Q_POS_TRANS, config.EKF_Q_POS_TRANS, config.EKF_Q_ALT_TRANS, 1e-10, 1e-10, config.EKF_Q_ALT_TRANS])

    @staticmethod
    def _phase_r_alt(phase: str) -> float:
        if phase == "B":
            return config.EKF_R_ALT_CRUISE
        if phase.startswith("A"):
            return config.EKF_R_ALT_CLIMB
        return config.EKF_R_ALT_TRANS
