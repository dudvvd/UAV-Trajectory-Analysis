"""Feature tracking with LK optical flow, validation, and ORB fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class TrackingResult:
    pixel_dx: float
    pixel_dy: float
    inlier_ratio: float
    affine_matrix: np.ndarray | None
    is_keyframe_reset: bool
    num_tracked: int
    prev_points: np.ndarray
    curr_points: np.ndarray
    residual_px: float
    method: str
    affine_rotation_deg: float
    phase_dx: float
    phase_dy: float
    phase_response: float
    fourier_rotation_deg: float
    fourier_response: float


class FeatureTracker:
    """Maintain sparse features and estimate frame-to-frame affine motion."""

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES, scoreType=cv2.ORB_HARRIS_SCORE)

    def initialize(self, gray: np.ndarray) -> None:
        self.prev_gray = gray
        self.points = self._detect_grid(gray)

    def track(self, curr_gray: np.ndarray) -> TrackingResult:
        if self.prev_gray is None:
            self.initialize(curr_gray)
            return self._empty_result(True)
        if self.points is None or len(self.points) < config.MIN_POINTS:
            self.points = self._detect_grid(self.prev_gray)

        phase_dx, phase_dy, phase_response = self._phase_correlate(self.prev_gray, curr_gray)
        fourier_rotation_deg, fourier_response = self._fourier_mellin_rotation(self.prev_gray, curr_gray)
        result = self._track_lk(curr_gray)
        if result.residual_px > config.ROTATION_RESIDUAL_THRESHOLD_PX or result.num_tracked < 20:
            logging.info("LK residual %.2fpx; using ORB fallback", result.residual_px)
            result = self._track_orb(curr_gray)
        result.phase_dx = phase_dx
        result.phase_dy = phase_dy
        result.phase_response = phase_response
        result.fourier_rotation_deg = fourier_rotation_deg
        result.fourier_response = fourier_response
        if self._should_use_phase_direction(result, phase_dx, phase_dy, phase_response):
            result.pixel_dx = phase_dx
            result.pixel_dy = phase_dy
            result.method = f"{result.method}+phase"

        reset = result.is_keyframe_reset or result.inlier_ratio < config.MIN_TRACK_RATIO
        self.prev_gray = curr_gray
        if reset:
            self.points = self._detect_grid(curr_gray)
            result.is_keyframe_reset = True
            logging.info("Feature keyframe reset with %d points", 0 if self.points is None else len(self.points))
        else:
            self.points = result.curr_points.reshape(-1, 1, 2).astype(np.float32)
            if len(self.points) < config.MIN_POINTS:
                self.points = self._merge_points(curr_gray, self.points)
        return result

    def _track_lk(self, curr_gray: np.ndarray) -> TrackingResult:
        assert self.prev_gray is not None
        pts0 = self.points
        if pts0 is None or len(pts0) == 0:
            return self._empty_result(True)
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, pts0, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
        if pts1 is None or st is None:
            return self._empty_result(True)
        back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, pts1, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
        valid = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
        p0 = pts0.reshape(-1, 2)[valid]
        p1 = pts1.reshape(-1, 2)[valid]
        b0 = back.reshape(-1, 2)[valid]
        fb = np.linalg.norm(p0 - b0, axis=1) if len(p0) else np.array([])
        keep = fb <= config.FB_CHECK_THRESHOLD_PX
        return self._estimate_motion(p0[keep], p1[keep], len(pts0), float(np.mean(fb)) if len(fb) else 999.0, "lk", curr_gray.shape[:2])

    def _track_orb(self, curr_gray: np.ndarray) -> TrackingResult:
        assert self.prev_gray is not None
        kp0, des0 = self.orb.detectAndCompute(self.prev_gray, None)
        kp1, des1 = self.orb.detectAndCompute(curr_gray, None)
        if des0 is None or des1 is None:
            return self._empty_result(True, "orb")
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
        good = [m for m, n in pairs if m.distance < config.ORB_RATIO_TEST * n.distance]
        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])
        return self._estimate_motion(p0, p1, max(len(kp0), 1), 0.0, "orb", curr_gray.shape[:2])

    def _estimate_motion(self, p0: np.ndarray, p1: np.ndarray, total: int, residual: float, method: str, image_shape: tuple[int, int]) -> TrackingResult:
        if len(p0) < 3:
            return self._empty_result(True, method)
        A, mask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD)
        if A is None or mask is None:
            delta = np.median(p1 - p0, axis=0)
            inliers = np.ones(len(p0), dtype=bool)
        else:
            inliers = mask.reshape(-1).astype(bool)
            h, w = image_shape
            center = np.array([w * 0.5, h * 0.5, 1.0], dtype=float)
            warped_center = A @ center
            delta = warped_center - center[:2]
        ratio = float(np.count_nonzero(inliers) / max(total, 1))
        rotation = self._rotation_deg(A)
        return TrackingResult(float(delta[0]), float(delta[1]), ratio, A, False, int(np.count_nonzero(inliers)), p0[inliers], p1[inliers], residual, method, rotation, np.nan, np.nan, 0.0, np.nan, 0.0)

    def _phase_correlate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float, float]:
        w, h = config.PHASE_CORR_SIZE
        prev = cv2.resize(prev_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        curr = cv2.resize(curr_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        prev = prev - cv2.GaussianBlur(prev, (0, 0), 3)
        curr = curr - cv2.GaussianBlur(curr, (0, 0), 3)
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(prev, curr, window)
        sx = prev_gray.shape[1] / float(w)
        sy = prev_gray.shape[0] / float(h)
        return float(dx * sx), float(dy * sy), float(response)

    def _fourier_mellin_rotation(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
        if not config.COMPUTE_FOURIER_MELLIN:
            return np.nan, 0.0
        w, h = config.FOURIER_MELLIN_SIZE
        prev = cv2.resize(prev_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        curr = cv2.resize(curr_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        prev = self._normalize_for_frequency(prev)
        curr = self._normalize_for_frequency(curr)
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        mag0 = self._fft_log_magnitude(prev * window)
        mag1 = self._fft_log_magnitude(curr * window)
        center = (w * 0.5, h * 0.5)
        max_radius = min(center)
        flags = cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS
        polar0 = cv2.warpPolar(mag0, (w, h), center, max_radius, flags)
        polar1 = cv2.warpPolar(mag1, (w, h), center, max_radius, flags)
        (shift_x, shift_y), response = cv2.phaseCorrelate(polar0.astype(np.float32), polar1.astype(np.float32))
        rotation = -shift_y * 360.0 / float(h)
        if rotation > 180.0:
            rotation -= 360.0
        if rotation < -180.0:
            rotation += 360.0
        if abs(rotation) > config.FOURIER_MELLIN_MAX_ABS_ROTATION_DEG:
            return np.nan, float(response)
        return float(rotation), float(response)

    @staticmethod
    def _normalize_for_frequency(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        high = gray - blurred
        high -= float(high.mean())
        std = float(high.std())
        if std > 1e-6:
            high /= std
        return high

    @staticmethod
    def _fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft2(gray)
        spectrum = np.fft.fftshift(spectrum)
        magnitude = np.ascontiguousarray(np.log1p(np.abs(spectrum)).astype(np.float32))
        return cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)

    @staticmethod
    def _should_use_phase_direction(result: TrackingResult, phase_dx: float, phase_dy: float, response: float) -> bool:
        if not config.PHASE_CORR_PREFER_FOR_DIRECTION or response < config.PHASE_CORR_MIN_RESPONSE:
            return False
        phase_norm = float(np.hypot(phase_dx, phase_dy))
        lk_norm = float(np.hypot(result.pixel_dx, result.pixel_dy))
        if phase_norm < 1.0:
            return False
        if lk_norm < 1.0:
            return True
        dot = np.clip((phase_dx * result.pixel_dx + phase_dy * result.pixel_dy) / (phase_norm * lk_norm), -1.0, 1.0)
        diff = float(np.rad2deg(np.arccos(dot)))
        return diff > config.PHASE_CORR_DIRECTION_DIFF_DEG or result.inlier_ratio < config.MIN_TRACK_RATIO

    @staticmethod
    def _rotation_deg(A: np.ndarray | None) -> float:
        if A is None:
            return np.nan
        return float(np.rad2deg(np.arctan2(A[1, 0], A[0, 0])))

    def _detect_grid(self, gray: np.ndarray) -> np.ndarray | None:
        h, w = gray.shape[:2]
        points: list[np.ndarray] = []
        per_cell = max(8, config.MAX_CORNERS // (config.GRID_ROWS * config.GRID_COLS))
        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                y0, y1 = r * h // config.GRID_ROWS, (r + 1) * h // config.GRID_ROWS
                x0, x1 = c * w // config.GRID_COLS, (c + 1) * w // config.GRID_COLS
                roi = gray[y0:y1, x0:x1]
                pts = cv2.goodFeaturesToTrack(roi, maxCorners=per_cell, qualityLevel=config.QUALITY_LEVEL, minDistance=config.MIN_DISTANCE)
                if pts is not None:
                    pts[:, 0, 0] += x0
                    pts[:, 0, 1] += y0
                    points.append(pts)
        if not points:
            return None
        return np.vstack(points).astype(np.float32)[: config.MAX_CORNERS]

    def _merge_points(self, gray: np.ndarray, existing: np.ndarray) -> np.ndarray:
        new = self._detect_grid(gray)
        if new is None:
            return existing
        merged = np.vstack([existing.reshape(-1, 1, 2), new.reshape(-1, 1, 2)])
        return merged[: config.MAX_CORNERS].astype(np.float32)

    @staticmethod
    def read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[1] > config.MAX_PROCESS_WIDTH:
            scale = config.MAX_PROCESS_WIDTH / img.shape[1]
            img = cv2.resize(img, (config.MAX_PROCESS_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def blur_score(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _empty_result(reset: bool, method: str = "lk") -> TrackingResult:
        empty = np.empty((0, 2), dtype=np.float32)
        return TrackingResult(0.0, 0.0, 0.0, None, reset, 0, empty, empty, 999.0, method, np.nan, np.nan, np.nan, 0.0, np.nan, 0.0)
