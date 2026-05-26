"""Long-window LK feature tracking with ORB fallback for difficult frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingResult:
    pixel_dx: float
    pixel_dy: float
    inlier_ratio: float
    is_keyframe_reset: bool
    num_tracked: int
    prev_points: np.ndarray
    curr_points: np.ndarray
    homography: np.ndarray | None
    method: str


class FeatureTracker:
    """Track Shi-Tomasi features using forward-backward LK optical flow."""

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.key_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.key_points: np.ndarray | None = None
        self.frames_since_key = 0
        self._orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES, fastThreshold=7)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def initialize(self, gray: np.ndarray) -> None:
        enhanced = self._enhance(gray)
        points = cv2.goodFeaturesToTrack(
            enhanced,
            maxCorners=config.MAX_CORNERS,
            qualityLevel=config.QUALITY_LEVEL,
            minDistance=config.MIN_DISTANCE,
        )
        self.prev_gray = gray
        self.key_gray = gray
        self.prev_points = points
        self.key_points = points.copy() if points is not None else None
        self.frames_since_key = 0
        LOGGER.info("Feature tracker initialized with %d points", 0 if points is None else len(points))

    def track(self, curr_gray: np.ndarray) -> TrackingResult:
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) < 8:
            self.initialize(curr_gray)
            return self._empty(True, "reinit")

        lk_params = dict(
            winSize=config.LK_WIN_SIZE,
            maxLevel=config.LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_points, None, **lk_params)
        if p1 is None or st1 is None:
            orb_result = self._orb_fallback(curr_gray)
            self._reset_keyframe(curr_gray)
            return orb_result
        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, p1, None, **lk_params)
        if p0_back is None or st2 is None:
            orb_result = self._orb_fallback(curr_gray)
            self._reset_keyframe(curr_gray)
            return orb_result

        p0 = self.prev_points.reshape(-1, 2)
        p1r = p1.reshape(-1, 2)
        p0b = p0_back.reshape(-1, 2)
        status = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
        fb_error = np.linalg.norm(p0 - p0b, axis=1)
        keep = status & (fb_error <= config.FB_CHECK_THRESHOLD_PX)
        old = p0[keep].astype(np.float32)
        new = p1r[keep].astype(np.float32)
        initial_count = len(p0)

        if len(old) < 8:
            orb_result = self._orb_fallback(curr_gray)
            self._reset_keyframe(curr_gray)
            return orb_result

        affine, mask = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
            maxIters=1500,
            confidence=0.99,
        )
        if affine is None or mask is None:
            orb_result = self._orb_fallback(curr_gray)
            self._reset_keyframe(curr_gray)
            return orb_result

        inlier_mask = mask.reshape(-1).astype(bool)
        old_in = old[inlier_mask]
        new_in = new[inlier_mask]
        residual = float(np.mean(np.linalg.norm((old @ affine[:, :2].T + affine[:, 2]) - new, axis=1)))
        if residual > config.ROTATION_RESIDUAL_THRESHOLD_PX:
            LOGGER.info("LK residual %.2f px; using ORB fallback for this frame", residual)
            result = self._orb_fallback(curr_gray)
            self.prev_gray = curr_gray
            self.prev_points = cv2.goodFeaturesToTrack(
                self._enhance(curr_gray),
                maxCorners=config.MAX_CORNERS,
                qualityLevel=config.QUALITY_LEVEL,
                minDistance=config.MIN_DISTANCE,
            )
            return result

        # Optical flow measures ground texture motion in the image. UAV motion
        # over ground is the opposite displacement under a nadir camera model.
        dx = float(-np.median(new_in[:, 0] - old_in[:, 0]))
        dy = float(-np.median(new_in[:, 1] - old_in[:, 1]))
        inlier_ratio = float(len(old_in) / max(initial_count, 1))
        homography = None
        if len(old_in) >= 4:
            homography, _ = cv2.findHomography(old_in, new_in, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)

        self.frames_since_key += 1
        reset = inlier_ratio < config.MIN_TRACK_RATIO or self.frames_since_key >= config.TRACK_WINDOW
        self.prev_gray = curr_gray
        self.prev_points = new_in.reshape(-1, 1, 2)
        if reset:
            LOGGER.info("Keyframe reset: inlier_ratio=%.2f tracked=%d", inlier_ratio, len(old_in))
            self._reset_keyframe(curr_gray)

        return TrackingResult(dx, dy, inlier_ratio, reset, len(old_in), old_in, new_in, homography, "lk")

    def _orb_fallback(self, curr_gray: np.ndarray) -> TrackingResult:
        if self.prev_gray is None:
            return self._empty(False, "orb")
        kp1, des1 = self._orb.detectAndCompute(self._enhance(self.prev_gray), None)
        kp2, des2 = self._orb.detectAndCompute(self._enhance(curr_gray), None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return self._empty(False, "orb")
        matches = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < config.ORB_RATIO_TEST * n.distance]
        if len(good) < 8:
            return self._empty(False, "orb")
        old = np.float32([kp1[m.queryIdx].pt for m in good])
        new = np.float32([kp2[m.trainIdx].pt for m in good])
        affine, mask = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if affine is None or mask is None:
            return self._empty(False, "orb")
        inliers = mask.reshape(-1).astype(bool)
        old_in = old[inliers]
        new_in = new[inliers]
        homography = None
        if len(old_in) >= 4:
            homography, _ = cv2.findHomography(old_in, new_in, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
        return TrackingResult(
            float(-affine[0, 2]),
            float(-affine[1, 2]),
            float(len(old_in) / max(len(good), 1)),
            False,
            int(len(old_in)),
            old_in,
            new_in,
            homography,
            "orb",
        )

    def _reset_keyframe(self, gray: np.ndarray) -> None:
        points = cv2.goodFeaturesToTrack(
            self._enhance(gray),
            maxCorners=config.MAX_CORNERS,
            qualityLevel=config.QUALITY_LEVEL,
            minDistance=config.MIN_DISTANCE,
        )
        self.key_gray = gray
        self.prev_gray = gray
        self.key_points = points
        self.prev_points = points
        self.frames_since_key = 0

    @staticmethod
    def _enhance(gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def read_gray(image_path: Path, max_width: int = config.MAX_PROCESS_WIDTH) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            image = cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def blur_score(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _empty(reset: bool, method: str) -> TrackingResult:
        empty = np.empty((0, 2), dtype=np.float32)
        return TrackingResult(0.0, 0.0, 0.0, reset, 0, empty, empty, None, method)
