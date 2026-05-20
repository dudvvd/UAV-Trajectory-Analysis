"""Classical computer-vision motion estimation between adjacent frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMotion:
    dx_px: float
    dy_px: float
    scale: float
    confidence: float
    method: str
    matches: int
    inliers: int


def read_gray(image_path: str | Path, max_width: int = 960) -> np.ndarray:
    """Read an image as grayscale and optionally downscale it for speed."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return image


class OrbAffineEstimator:
    """Estimate image translation and scale using ORB matches + RANSAC affine."""

    def __init__(self, max_features: int = 2500, ratio: float = 0.78) -> None:
        self.max_features = max_features
        self.ratio = ratio
        self._orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FrameMotion:
        kp1, des1 = self._orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self._orb.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return FrameMotion(0.0, 0.0, 1.0, 0.0, "orb_affine", 0, 0)

        pairs = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in pairs if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return FrameMotion(0.0, 0.0, 1.0, 0.0, "orb_affine", len(good), 0)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if matrix is None or inlier_mask is None:
            return FrameMotion(0.0, 0.0, 1.0, 0.0, "orb_affine", len(good), 0)

        a, b = matrix[0, 0], matrix[0, 1]
        scale = float(np.sqrt(a * a + b * b))
        dx_px = float(matrix[0, 2])
        dy_px = float(matrix[1, 2])
        inliers = int(inlier_mask.sum())
        confidence = float(inliers / max(len(good), 1))
        return FrameMotion(dx_px, dy_px, scale, confidence, "orb_affine", len(good), inliers)


class PhaseCorrelationEstimator:
    """Fast translation-only estimator used as an alternative comparison method."""

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FrameMotion:
        h = min(prev_gray.shape[0], curr_gray.shape[0])
        w = min(prev_gray.shape[1], curr_gray.shape[1])
        prev = prev_gray[:h, :w].astype(np.float32)
        curr = curr_gray[:h, :w].astype(np.float32)
        shift, response = cv2.phaseCorrelate(prev, curr)
        return FrameMotion(float(shift[0]), float(shift[1]), 1.0, float(response), "phase", 0, 0)


def create_estimator(method: str):
    if method == "orb_affine":
        return OrbAffineEstimator()
    if method == "phase":
        return PhaseCorrelationEstimator()
    raise ValueError(f"未知方法: {method}")
