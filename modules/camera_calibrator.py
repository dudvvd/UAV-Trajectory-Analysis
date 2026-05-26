"""Automatic intrinsic calibration from an uncalibrated image sequence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    K: list[list[float]]
    fx: float
    fy: float
    cx: float
    cy: float
    method: str
    confidence: float


class CameraCalibrator:
    """Estimate or load a reusable pinhole intrinsic matrix."""

    def __init__(self, cache_path: Path = config.CALIBRATION_PATH, recalibrate: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.recalibrate = recalibrate
        self._orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES, fastThreshold=7)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def estimate(self, image_paths: list[Path]) -> CalibrationResult:
        if self.cache_path.exists() and not self.recalibrate:
            LOGGER.info("Loading cached camera calibration from %s", self.cache_path)
            with self.cache_path.open("r", encoding="utf-8") as f:
                return CalibrationResult(**json.load(f))

        first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise FileNotFoundError(image_paths[0])
        h, w = first.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        result = self._estimate_from_fundamental(image_paths, w, h, cx, cy)
        if result is None:
            result = self._estimate_from_homography(image_paths, w, h, cx, cy)
        if result is None or not self._valid_result(result, w, h):
            LOGGER.warning("Camera calibration degraded to empirical focal estimate")
            result = self._empirical(w, h, cx, cy)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        return result

    def _estimate_from_fundamental(
        self, image_paths: list[Path], width: int, height: int, cx: float, cy: float
    ) -> CalibrationResult | None:
        pairs = self._sample_pairs(image_paths, config.CALIBRATION_PAIR_COUNT_F)
        estimates: list[tuple[float, float]] = []
        for prev_path, curr_path in pairs:
            pts1, pts2 = self._orb_matches(prev_path, curr_path)
            if len(pts1) < config.CALIBRATION_MIN_MATCHES:
                continue
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.0, 0.995)
            if F is None or mask is None:
                continue
            inliers = int(mask.sum())
            if inliers < config.CALIBRATION_MIN_INLIERS:
                continue
            f, quality = self._scan_focal_from_f(F, width, height, cx, cy)
            if f > 0:
                estimates.append((f, quality * inliers / len(mask)))

        if len(estimates) < 3:
            LOGGER.info("Fundamental calibration failed: only %d usable estimates", len(estimates))
            return None
        focals = np.asarray([v[0] for v in estimates], dtype=float)
        qualities = np.asarray([v[1] for v in estimates], dtype=float)
        f = float(np.median(focals))
        confidence = float(np.clip(np.median(qualities), 0.0, 1.0))
        LOGGER.info("Camera calibration used fundamental strategy: f=%.2f confidence=%.2f", f, confidence)
        return self._build_result(f, cx, cy, "fundamental", confidence)

    def _estimate_from_homography(
        self, image_paths: list[Path], width: int, height: int, cx: float, cy: float
    ) -> CalibrationResult | None:
        pairs = self._sample_pairs(image_paths, config.CALIBRATION_PAIR_COUNT_H)
        scale_votes: list[float] = []
        confidences: list[float] = []
        for prev_path, curr_path in pairs:
            pts1, pts2 = self._orb_matches(prev_path, curr_path)
            if len(pts1) < config.CALIBRATION_MIN_MATCHES:
                continue
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is None or mask is None:
                continue
            inliers = int(mask.sum())
            if inliers < config.CALIBRATION_MIN_INLIERS:
                continue
            sx = float(np.linalg.norm(H[0:2, 0]))
            sy = float(np.linalg.norm(H[0:2, 1]))
            if 0.8 <= sx <= 1.25 and 0.8 <= sy <= 1.25:
                scale_votes.append((sx + sy) * 0.5)
                confidences.append(inliers / len(mask))
        if len(scale_votes) < 3:
            LOGGER.info("Homography calibration failed: only %d usable estimates", len(scale_votes))
            return None

        f = max(width, height) * config.EMPIRICAL_FOCAL_FACTOR
        confidence = float(np.clip(np.median(confidences) * 0.65, 0.0, 0.75))
        LOGGER.info("Camera calibration used homography fallback: f=%.2f confidence=%.2f", f, confidence)
        return self._build_result(f, cx, cy, "homography", confidence)

    def _orb_matches(self, prev_path: Path, curr_path: Path) -> tuple[np.ndarray, np.ndarray]:
        prev = self._read_gray(prev_path)
        curr = self._read_gray(curr_path)
        prev = self._enhance(prev)
        curr = self._enhance(curr)
        kp1, des1 = self._orb.detectAndCompute(prev, None)
        kp2, des2 = self._orb.detectAndCompute(curr, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        raw = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < config.ORB_RATIO_TEST * n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2

    def _scan_focal_from_f(self, F: np.ndarray, width: int, height: int, cx: float, cy: float) -> tuple[float, float]:
        min_f = max(width, height) * config.FOCAL_MIN_FACTOR
        max_f = max(width, height) * config.FOCAL_MAX_FACTOR
        candidates = np.linspace(min_f, max_f, 80)
        best_f = 0.0
        best_score = float("inf")
        for f in candidates:
            K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
            E = K.T @ F @ K
            s = np.linalg.svd(E, compute_uv=False)
            if s[1] <= 1e-9:
                continue
            score = abs((s[0] / s[1]) - 1.0) + abs(s[2] / s[1])
            if score < best_score:
                best_score = float(score)
                best_f = float(f)
        quality = 1.0 / (1.0 + best_score) if best_f > 0 else 0.0
        return best_f, quality

    def _valid_result(self, result: CalibrationResult, width: int, height: int) -> bool:
        f = float(result.fx)
        min_f = max(width, height) * config.FOCAL_MIN_FACTOR
        max_f = max(width, height) * config.FOCAL_MAX_FACTOR
        if not min_f <= f <= max_f:
            return False
        edge_margin = 0.05 * (max_f - min_f)
        if f <= min_f + edge_margin or f >= max_f - edge_margin:
            LOGGER.warning("Camera calibration focal %.2f is too close to search boundary", f)
            return False
        if result.method == "fundamental" and result.confidence < (1.0 - config.ESSENTIAL_SINGULAR_VALUE_TOL):
            return False
        return True

    def _empirical(self, width: int, height: int, cx: float, cy: float) -> CalibrationResult:
        f = max(width, height) * config.EMPIRICAL_FOCAL_FACTOR
        return self._build_result(float(f), cx, cy, "empirical", 0.45)

    def _build_result(self, f: float, cx: float, cy: float, method: str, confidence: float) -> CalibrationResult:
        K = [[float(f), 0.0, float(cx)], [0.0, float(f), float(cy)], [0.0, 0.0, 1.0]]
        return CalibrationResult(K=K, fx=float(f), fy=float(f), cx=float(cx), cy=float(cy), method=method, confidence=float(confidence))

    def _sample_pairs(self, image_paths: list[Path], count: int) -> list[tuple[Path, Path]]:
        max_start = max(len(image_paths) - 2, 0)
        if max_start == 0:
            return []
        indices = np.linspace(0, max_start, min(count, max_start + 1), dtype=int)
        return [(image_paths[i], image_paths[i + 1]) for i in indices]

    @staticmethod
    def _enhance(gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image
