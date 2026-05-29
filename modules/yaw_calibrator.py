"""Camera yaw calibration from GPS direction and visual direction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from utils.geo_utils import lonlat_to_local_m


@dataclass
class YawCalibrationResult:
    yaw_deg: float
    used: bool
    total_gps_displacement_m: float
    mean_direction_error_deg: float


class YawCalibrator:
    """Estimate pixel-to-East/North rotation using direction angles only."""

    def __init__(self, fx: float) -> None:
        self.fx = float(fx)

    def calibrate(
        self,
        image_paths: list[Path],
        gt_lon: np.ndarray,
        gt_lat: np.ndarray,
        altitude_m: float,
        start_index: int,
        frame_count: int = config.YAW_CALIBRATION_FRAMES,
    ) -> YawCalibrationResult:
        end = min(len(image_paths), start_index + frame_count)
        if end - start_index < 3:
            return YawCalibrationResult(0.0, False, 0.0, 180.0)

        east, north = lonlat_to_local_m(gt_lon[start_index:end], gt_lat[start_index:end], gt_lon[start_index], gt_lat[start_index])
        gps_vec = np.column_stack([np.diff(east), np.diff(north)])
        total_gps = float(np.linalg.norm([east[-1] - east[0], north[-1] - north[0]]))
        if total_gps < config.YAW_MIN_GPS_DISPLACEMENT_M:
            logging.warning("Yaw calibration skipped: GPS direction baseline %.2fm is too small", total_gps)
            return YawCalibrationResult(0.0, False, total_gps, 180.0)

        vis_vec = self._visual_vectors(image_paths[start_index:end])
        n = min(len(gps_vec), len(vis_vec))
        gps_vec, vis_vec = gps_vec[:n], vis_vec[:n]
        valid = (np.linalg.norm(gps_vec, axis=1) > 0.2) & (np.linalg.norm(vis_vec, axis=1) > 0.2)
        if np.count_nonzero(valid) < 3:
            logging.warning("Yaw calibration skipped: not enough visual/GPS direction pairs")
            return YawCalibrationResult(0.0, False, total_gps, 180.0)

        gps_unit = self._unit(gps_vec[valid])
        vis_unit = self._unit(vis_vec[valid])

        def objective(theta_deg: float) -> float:
            theta = np.deg2rad(theta_deg)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pred = vis_unit @ rot.T
            dot = np.clip(np.sum(pred * gps_unit, axis=1), -1.0, 1.0)
            return float(np.mean(1.0 - dot))

        yaw = self._minimize_angle(objective)
        mean_error = float(np.rad2deg(np.arccos(np.clip(1.0 - objective(yaw), -1.0, 1.0))))
        if config.PERSIST_CALIBRATED_YAW:
            self._persist_yaw(yaw)
        logging.info("Yaw calibrated: %.2f deg, mean direction error %.2f deg", yaw, mean_error)
        return YawCalibrationResult(yaw, True, total_gps, mean_error)

    def _visual_vectors(self, image_paths: list[Path]) -> np.ndarray:
        vectors: list[list[float]] = []
        prev = self._read_gray(image_paths[0])
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=config.MAX_CORNERS, qualityLevel=config.QUALITY_LEVEL, minDistance=config.MIN_DISTANCE)
        for path in image_paths[1:]:
            curr = self._read_gray(path)
            if pts is None or len(pts) < 30:
                pts = cv2.goodFeaturesToTrack(prev, maxCorners=config.MAX_CORNERS, qualityLevel=config.QUALITY_LEVEL, minDistance=config.MIN_DISTANCE)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
            if p1 is None or st is None:
                vectors.append([0.0, 0.0])
            else:
                valid = st.reshape(-1) == 1
                delta = p1.reshape(-1, 2)[valid] - pts.reshape(-1, 2)[valid]
                vectors.append(np.median(delta, axis=0).tolist() if len(delta) else [0.0, 0.0])
                pts = p1[valid].reshape(-1, 1, 2) if np.any(valid) else None
            prev = curr
        return np.asarray(vectors, dtype=float)

    @staticmethod
    def _unit(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _minimize_angle(objective) -> float:
        candidates = np.linspace(-180.0, 180.0, 361)
        values = np.asarray([objective(v) for v in candidates])
        best = float(candidates[int(np.argmin(values))])
        for step in (0.5, 0.1, 0.02):
            candidates = np.arange(best - 1.0, best + 1.0 + step, step)
            values = np.asarray([objective(v) for v in candidates])
            best = float(candidates[int(np.argmin(values))])
        return best

    @staticmethod
    def _persist_yaw(yaw_deg: float) -> None:
        config_path = Path(config.__file__).resolve()
        text = config_path.read_text(encoding="utf-8")
        updated = re.sub(r"^CAMERA_YAW_DEG\s*=\s*[-+0-9.eE]+", f"CAMERA_YAW_DEG = {yaw_deg:.8f}", text, flags=re.MULTILINE)
        if updated != text:
            config_path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[1] > config.MAX_PROCESS_WIDTH:
            scale = config.MAX_PROCESS_WIDTH / img.shape[1]
            img = cv2.resize(img, (config.MAX_PROCESS_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return img
