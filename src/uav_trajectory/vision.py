"""Classical computer-vision motion estimation between adjacent UAV frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMotion:
    """Image-space motion estimated between two frames."""

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
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return image


def enhance_ir_image(gray: np.ndarray) -> np.ndarray:
    """Improve local contrast before feature extraction on IR images."""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


class OrbAffineEstimator:
    """Estimate image translation and scale using ORB matches + RANSAC affine."""

    def __init__(self, max_features: int = 3000, ratio: float = 0.75) -> None:
        self.max_features = max_features
        self.ratio = ratio
        self._orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FrameMotion:
        prev = enhance_ir_image(prev_gray)
        curr = enhance_ir_image(curr_gray)
        kp1, des1 = self._orb.detectAndCompute(prev, None)
        kp2, des2 = self._orb.detectAndCompute(curr, None)
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
            maxIters=3000,
            confidence=0.995,
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


def create_estimator() -> OrbAffineEstimator:
    """Create the project's main image-motion estimator."""

    return OrbAffineEstimator()
