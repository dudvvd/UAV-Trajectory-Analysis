"""Automatic camera intrinsic estimation with cache-backed fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class CalibrationResult:
    K: list[list[float]]
    fx: float
    fy: float
    cx: float
    cy: float
    confidence: float
    method: str
    image_size: tuple[int, int]


class CameraCalibrator:
    """Estimate K using sampled frame pairs, falling back to a bounded heuristic."""

    def __init__(self, cache_path: Path = config.CALIBRATION_PATH, recalibrate: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.recalibrate = recalibrate
        self.orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES, scoreType=cv2.ORB_HARRIS_SCORE)

    def estimate(self, image_paths: list[Path]) -> CalibrationResult:
        if config.CALIBRATION_CACHE_ENABLED and self.cache_path.exists() and not self.recalibrate:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return CalibrationResult(**data)

        first = self._read_gray(image_paths[0])
        h, w = first.shape[:2]
        if config.MANUAL_FOCAL_PX is not None:
            logging.info("Using manual focal length from config: %.2fpx", float(config.MANUAL_FOCAL_PX))
            return self._save(self._make_result(float(config.MANUAL_FOCAL_PX), w, h, "manual", 1.0))

        focals_f, sv_errors = self._estimate_from_fundamental(image_paths, w, h)
        valid_f = self._filter_focals(focals_f, w, h)
        if len(valid_f) >= config.CALIBRATION_MIN_VALID_ESTIMATES:
            fx = float(np.median(valid_f))
            sv_error = float(np.median(sv_errors)) if sv_errors else 1.0
            if self._valid_focal(fx, w, h) and not self._near_boundary(fx, w, h) and sv_error <= config.ESSENTIAL_SINGULAR_VALUE_TOL:
                return self._save(self._make_result(fx, w, h, "fundamental", max(0.72, 1.0 - sv_error)))
            logging.warning("Fundamental focal estimate rejected: fx=%.2f count=%d sv_error=%.3f", fx, len(valid_f), sv_error)

        logging.warning("Fundamental calibration degraded; trying homography statistics")
        valid_h = self._filter_focals(self._estimate_from_homography(image_paths, w, h), w, h)
        if len(valid_h) >= config.CALIBRATION_MIN_VALID_ESTIMATES:
            fx = float(np.median(valid_h))
            if self._valid_focal(fx, w, h) and not self._near_boundary(fx, w, h):
                return self._save(self._make_result(fx, w, h, "homography", 0.68))
            logging.warning("Homography focal estimate rejected: fx=%.2f count=%d", fx, len(valid_h))

        logging.warning("Camera calibration fell back to empirical focal length")
        fx = max(w, h) * config.EMPIRICAL_FOCAL_FACTOR
        return self._save(self._make_result(float(fx), w, h, "empirical", 0.45))

    def _estimate_from_fundamental(self, image_paths: list[Path], w: int, h: int) -> tuple[list[float], list[float]]:
        focals: list[float] = []
        sv_errors: list[float] = []
        for p0, p1 in self._sample_pairs(image_paths, config.CALIBRATION_PAIR_COUNT_F):
            pts0, pts1 = self._match_orb(self._read_gray(p0), self._read_gray(p1))
            if len(pts0) < config.CALIBRATION_MIN_MATCHES:
                continue
            F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.5, 0.99)
            if F is None or mask is None or int(mask.sum()) < config.CALIBRATION_MIN_INLIERS:
                continue
            focals.append(self._focal_from_f_rank(F, w, h))
            f_guess = max(w, h) * config.EMPIRICAL_FOCAL_FACTOR
            K = np.array([[f_guess, 0, w / 2], [0, f_guess, h / 2], [0, 0, 1]], dtype=float)
            s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
            if s[1] > 1e-9:
                sv_errors.append(abs(s[0] - s[1]) / s[1])
        return [f for f in focals if np.isfinite(f)], sv_errors

    def _estimate_from_homography(self, image_paths: list[Path], w: int, h: int) -> list[float]:
        focals: list[float] = []
        center = np.array([w / 2, h / 2], dtype=float)
        for p0, p1 in self._sample_pairs(image_paths, config.CALIBRATION_PAIR_COUNT_H):
            pts0, pts1 = self._match_orb(self._read_gray(p0), self._read_gray(p1))
            if len(pts0) < config.CALIBRATION_MIN_MATCHES:
                continue
            H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
            if H is None or mask is None or int(mask.sum()) < config.CALIBRATION_MIN_INLIERS:
                continue
            warped_center = cv2.perspectiveTransform(center.reshape(1, 1, 2), H).reshape(2)
            motion = float(np.linalg.norm(warped_center - center))
            focals.append(max(w, h) * np.clip(1.0 + motion / max(w, h), 0.8, 1.8))
        return focals

    def _focal_from_f_rank(self, F: np.ndarray, w: int, h: int) -> float:
        candidates = np.linspace(max(w, h) * config.FOCAL_MIN_FACTOR, max(w, h) * config.FOCAL_MAX_FACTOR, 40)
        best_f, best_err = candidates[0], float("inf")
        for f in candidates:
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
            s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
            if s[1] <= 1e-9:
                continue
            err = abs(s[0] - s[1]) / s[1] + abs(s[2]) / s[1]
            if err < best_err:
                best_f, best_err = f, err
        return float(best_f)

    def _match_orb(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kp0, des0 = self.orb.detectAndCompute(img0, None)
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        if des0 is None or des1 is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
        good = [m for m, n in pairs if m.distance < config.ORB_RATIO_TEST * n.distance]
        return np.float32([kp0[m.queryIdx].pt for m in good]), np.float32([kp1[m.trainIdx].pt for m in good])

    def _sample_pairs(self, image_paths: list[Path], count: int) -> list[tuple[Path, Path]]:
        max_start = max(0, len(image_paths) - 2)
        idx = np.linspace(0, max_start, min(count, max_start + 1), dtype=int)
        return [(image_paths[i], image_paths[i + 1]) for i in idx]

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[1] > config.MAX_PROCESS_WIDTH:
            scale = config.MAX_PROCESS_WIDTH / img.shape[1]
            img = cv2.resize(img, (config.MAX_PROCESS_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _filter_focals(focals: list[float], w: int, h: int) -> list[float]:
        arr = np.asarray(focals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-6
        filtered = arr[np.abs(arr - med) < 3.5 * mad]
        bounded = filtered[(filtered >= max(w, h) * config.FOCAL_MIN_FACTOR) & (filtered <= max(w, h) * config.FOCAL_MAX_FACTOR)]
        return bounded.tolist()

    @staticmethod
    def _valid_focal(fx: float, w: int, h: int) -> bool:
        return max(w, h) * config.FOCAL_MIN_FACTOR <= fx <= max(w, h) * config.FOCAL_MAX_FACTOR

    @staticmethod
    def _near_boundary(fx: float, w: int, h: int) -> bool:
        max_dim = max(w, h)
        lo = max_dim * config.FOCAL_MIN_FACTOR
        hi = max_dim * config.FOCAL_MAX_FACTOR
        margin = max_dim * config.CALIBRATION_BOUNDARY_MARGIN
        return fx <= lo + margin or fx >= hi - margin

    @staticmethod
    def _make_result(fx: float, w: int, h: int, method: str, confidence: float) -> CalibrationResult:
        return CalibrationResult(
            K=[[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
            fx=fx,
            fy=fx,
            cx=w / 2.0,
            cy=h / 2.0,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method=method,
            image_size=(w, h),
        )

    def _save(self, result: CalibrationResult) -> CalibrationResult:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        logging.info("Calibration: method=%s fx=%.2f confidence=%.2f", result.method, result.fx, result.confidence)
        return result
