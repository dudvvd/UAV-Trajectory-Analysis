"""Extended Kalman filter for visual dead reckoning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class EKFState:
    longitude: float
    latitude: float
    altitude: float
    velocity: np.ndarray
    covariance: np.ndarray


class EKFLocalizer:
    """Six-state constant-velocity EKF over lon, lat, altitude and velocities."""

    def __init__(self, longitude: float, latitude: float, altitude: float) -> None:
        self.x = np.array([longitude, latitude, altitude, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([1e-12, 1e-12, 1.0, 1e-10, 1e-10, 1.0]).astype(float)

    def predict(self, dt: float) -> EKFState:
        dt = max(float(dt), 0.0)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        q = np.diag(
            [
                config.EKF_PROCESS_NOISE,
                config.EKF_PROCESS_NOISE,
                config.EKF_ALT_PROCESS_NOISE,
                config.EKF_PROCESS_NOISE,
                config.EKF_PROCESS_NOISE,
                config.EKF_ALT_PROCESS_NOISE,
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q
        return self.state

    def update_visual(
        self,
        delta_lon: float,
        delta_lat: float,
        delta_alt: float,
        inlier_ratio: float,
        scale_confidence: float,
        dt: float,
    ) -> tuple[EKFState, int]:
        """Fuse visual deltas using selective observation dimensions.

        Returns update mode:
        1 = position and altitude, 2 = position only, 3 = prediction only.
        """

        if inlier_ratio <= 0.5:
            return self.state, 3

        if scale_confidence > 0.3:
            z = np.array([self.x[0] + delta_lon, self.x[1] + delta_lat, self.x[2] + delta_alt], dtype=float)
            H = np.zeros((3, 6), dtype=float)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 2] = 1.0
            R = np.diag([config.EKF_R_BASE_LON_LAT, config.EKF_R_BASE_LON_LAT, config.EKF_R_BASE_ALT])
            conf = max(float(inlier_ratio * scale_confidence), 0.0)
            mode = 1
        else:
            z = np.array([self.x[0] + delta_lon, self.x[1] + delta_lat], dtype=float)
            H = np.zeros((2, 6), dtype=float)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            R = np.diag([config.EKF_R_BASE_LON_LAT, config.EKF_R_BASE_LON_LAT])
            conf = max(float(inlier_ratio), 0.0)
            mode = 2

        R = R / (conf + config.CONFIDENCE_EPS)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        if dt > 0:
            self.x[3] = delta_lon / dt
            self.x[4] = delta_lat / dt
            if mode == 1:
                self.x[5] = delta_alt / dt
        return self.state, mode

    @property
    def state(self) -> EKFState:
        return EKFState(
            longitude=float(self.x[0]),
            latitude=float(self.x[1]),
            altitude=float(self.x[2]),
            velocity=self.x[3:6].copy(),
            covariance=self.P.copy(),
        )
