"""Pure-vision flight phase detection.

The detector intentionally does not read predicted altitude.  It classifies
climb, descent, cruise, or transition from affine scale and sparse optical
flow geometry only, avoiding the previous altitude/phase feedback loop.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class PhaseResult:
    phase: str
    confidence: float
    alt_limit_m: float
    scale_trend: str
    up_ratio: float
    down_ratio: float
    stable_ratio: float
    phase_from_vision: str


class FlightPhaseDetector:
    """Classify phase from visual expansion/contraction cues only."""

    def __init__(self) -> None:
        self.phase = "C"
        self.pending_phase = "C"
        self.pending_count = 0
        self.scale_labels: deque[str] = deque(maxlen=config.PHASE_WINDOW)
        self.divergence_labels: deque[str] = deque(maxlen=config.PHASE_WINDOW)
        self.center_labels: deque[str] = deque(maxlen=config.PHASE_WINDOW)

    def update(
        self,
        affine_matrix: np.ndarray | None,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        image_shape: tuple[int, int],
    ) -> PhaseResult:
        """Update the visual phase estimate from one frame pair."""
        affine_scale = self._affine_scale(affine_matrix)
        flow_divergence, center_delta = self._flow_cues(prev_points, curr_points, image_shape)

        scale_label = self._label_scale(affine_scale)
        div_label = self._label_signed(flow_divergence, config.PHASE_DIVERGENCE_THRESH)
        center_label = self._label_signed(center_delta, config.PHASE_CENTER_DELTA_THRESH)
        self.scale_labels.append(scale_label)
        self.divergence_labels.append(div_label)
        self.center_labels.append(center_label)

        candidate, confidence, ratios, trend = self._candidate(scale_label, div_label, center_label)
        if candidate != self.phase:
            if candidate == self.pending_phase:
                self.pending_count += 1
            else:
                self.pending_phase = candidate
                self.pending_count = 1
            if self.pending_count >= config.PHASE_MIN_DURATION:
                logging.info("Visual flight phase switched: %s -> %s", self.phase, candidate)
                self.phase = candidate
                self.pending_count = 0
        else:
            self.pending_count = 0

        return PhaseResult(
            phase=self.phase,
            confidence=confidence,
            alt_limit_m=self._alt_limit(self.phase),
            scale_trend=trend,
            up_ratio=ratios[0],
            down_ratio=ratios[1],
            stable_ratio=ratios[2],
            phase_from_vision=candidate,
        )

    def _candidate(self, scale_label: str, div_label: str, center_label: str) -> tuple[str, float, tuple[float, float, float], str]:
        if len(self.scale_labels) < max(3, config.PHASE_WINDOW // 2):
            votes = [scale_label, div_label, center_label]
            phase = self._majority(votes)
            return phase, 0.5, (0.0, 0.0, 0.0), self._trend_name(phase)

        labels = list(self.scale_labels)
        up_ratio = labels.count("A_up") / len(labels)
        down_ratio = labels.count("A_down") / len(labels)
        stable_ratio = labels.count("B") / len(labels)

        if up_ratio > config.PHASE_UP_RATIO_THRESH:
            scale_phase = "A_up"
        elif down_ratio > config.PHASE_DOWN_RATIO_THRESH:
            scale_phase = "A_down"
        elif stable_ratio > config.PHASE_STABLE_RATIO_THRESH:
            scale_phase = "B"
        else:
            scale_phase = "C"

        votes = [scale_phase, self._majority(self.divergence_labels), self._majority(self.center_labels)]
        phase = self._majority(votes)
        support = votes.count(phase) / max(len(votes), 1)
        ratio_support = max(up_ratio, down_ratio, stable_ratio)
        confidence = float(np.clip(0.35 + 0.35 * support + 0.30 * ratio_support, 0.0, 1.0))
        return phase, confidence, (up_ratio, down_ratio, stable_ratio), self._trend_name(scale_phase)

    @staticmethod
    def _affine_scale(A: np.ndarray | None) -> float:
        if A is None:
            return np.nan
        return float(np.sqrt(abs(np.linalg.det(A[:, :2]))))

    @staticmethod
    def _flow_cues(p0: np.ndarray, p1: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float]:
        if len(p0) < 3 or len(p1) < 3:
            return np.nan, np.nan
        h, w = image_shape
        center = np.array([w * 0.5, h * 0.5], dtype=float)
        radial = p0 - center
        radius = np.linalg.norm(radial, axis=1)
        valid = radius > 5.0
        if np.count_nonzero(valid) < 3:
            return np.nan, np.nan
        unit = radial[valid] / radius[valid, None]
        flow = p1[valid] - p0[valid]
        divergence = float(np.median(np.sum(flow * unit, axis=1) / radius[valid]))
        d0 = np.linalg.norm(p0[valid] - center, axis=1)
        d1 = np.linalg.norm(p1[valid] - center, axis=1)
        center_delta = float(np.median(d1 - d0))
        return divergence, center_delta

    @staticmethod
    def _label_scale(scale: float) -> str:
        if not np.isfinite(scale):
            return "C"
        if config.PHASE_INVERT_VERTICAL_CUES:
            if scale < config.PHASE_SCALE_UP_THRESH:
                return "A_down"
            if scale > config.PHASE_SCALE_DOWN_THRESH:
                return "A_up"
            return "B"
        if scale < config.PHASE_SCALE_UP_THRESH:
            return "A_up"
        if scale > config.PHASE_SCALE_DOWN_THRESH:
            return "A_down"
        return "B"

    @staticmethod
    def _label_signed(value: float, threshold: float) -> str:
        if not np.isfinite(value):
            return "C"
        if config.PHASE_INVERT_VERTICAL_CUES:
            if value < -threshold:
                return "A_down"
            if value > threshold:
                return "A_up"
            return "B"
        if value < -threshold:
            return "A_up"
        if value > threshold:
            return "A_down"
        return "B"

    @staticmethod
    def _majority(labels) -> str:
        labels = list(labels)
        if not labels:
            return "C"
        counts = Counter(labels)
        priority = {"A_up": 3, "A_down": 3, "B": 2, "C": 1}
        return max(counts, key=lambda item: (counts[item], priority.get(item, 0)))

    @staticmethod
    def _trend_name(phase: str) -> str:
        if phase == "A_up":
            return "up"
        if phase == "A_down":
            return "down"
        if phase == "B":
            return "stable"
        return "transition"

    @staticmethod
    def _alt_limit(phase: str) -> float:
        if phase.startswith("A"):
            return config.MAX_DELTA_ALT_CLIMB
        if phase == "B":
            return config.MAX_DELTA_ALT_CRUISE
        return config.MAX_DELTA_ALT_TRANS
