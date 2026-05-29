"""Start-frame quality screening."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class StartFrameResult:
    index: int
    blur_score: float
    corner_count: int
    inlier_ratio: float
    accepted: bool


class StartFrameChecker:
    """Find the first frame that is sharp, feature-rich, and trackable."""

    def check(self, image_paths: list[Path], start_index: int = 0) -> StartFrameResult:
        best: StartFrameResult | None = None
        last = min(len(image_paths) - 1, start_index + config.START_MAX_CHECK_FRAMES)
        for idx in range(start_index, last):
            gray0 = self._read_gray(image_paths[idx])
            gray1 = self._read_gray(image_paths[idx + 1])
            result = self._score_pair(idx, gray0, gray1)
            if best is None or self._rank(result) > self._rank(best):
                best = result
            if result.accepted:
                logging.info("Accepted start frame %s", image_paths[idx].name)
                return result
        if best is None:
            raise ValueError("No start frame candidate can be scored")
        logging.warning("No start frame met all thresholds; using best candidate at index %d", best.index)
        return best

    def _score_pair(self, idx: int, gray0: np.ndarray, gray1: np.ndarray) -> StartFrameResult:
        blur = float(cv2.Laplacian(gray0, cv2.CV_64F).var())
        pts = cv2.goodFeaturesToTrack(
            gray0,
            maxCorners=config.MAX_CORNERS,
            qualityLevel=config.QUALITY_LEVEL,
            minDistance=config.MIN_DISTANCE,
        )
        corner_count = 0 if pts is None else len(pts)
        ratio = 0.0
        if pts is not None and len(pts) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
            if p1 is not None and st is not None:
                back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
                valid = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
                if np.any(valid):
                    fb_err = np.linalg.norm(pts.reshape(-1, 2)[valid] - back.reshape(-1, 2)[valid], axis=1)
                    ratio = float(np.mean(fb_err <= config.FB_CHECK_THRESHOLD_PX))
        accepted = (
            blur > config.START_BLUR_THRESHOLD
            and corner_count > config.START_MIN_CORNERS
            and ratio > config.START_MIN_INLIER_RATIO
        )
        return StartFrameResult(idx, blur, corner_count, ratio, accepted)

    @staticmethod
    def _rank(result: StartFrameResult) -> float:
        return result.blur_score / config.START_BLUR_THRESHOLD + result.corner_count / config.START_MIN_CORNERS + result.inlier_ratio

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[1] > config.MAX_PROCESS_WIDTH:
            scale = config.MAX_PROCESS_WIDTH / img.shape[1]
            img = cv2.resize(img, (config.MAX_PROCESS_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return img
