"""Meters-per-pixel management decoupled from raw altitude estimation."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class ScaleManagerResult:
    mpp_for_position: float
    mpp_history_median: float
    mpp_current_raw: float
    mpp_ratio: float
    is_alt_fixed: bool
    alt_fixed_triggered: int


class ScaleManager:
    """Use robust mpp history and guarded cruise fixing for position scale."""

    def __init__(self, fx: float, initial_altitude: float) -> None:
        self.fx = float(fx)
        initial_mpp = float(initial_altitude) / max(self.fx, 1e-6)
        self.mpp_history: deque[float] = deque([initial_mpp], maxlen=config.MPP_HISTORY_LEN)
        self.raw_alt_history: deque[float] = deque([float(initial_altitude)], maxlen=config.CALIB_WINDOW)
        self.affine_scale_history: deque[float] = deque(maxlen=config.CALIB_WINDOW)
        self.delta_alt_history: deque[float] = deque(maxlen=config.CALIB_WINDOW)
        self.fixed_mpp: float | None = None
        self.fixed_ref_alt: float | None = None
        self.exit_count = 0
        self.frame_index = 0

    def update(
        self,
        raw_altitude: float,
        affine_scale: float,
        delta_alt_cumulative: float,
        phase: str,
        frame_index: int,
    ) -> ScaleManagerResult:
        self.frame_index = int(frame_index)
        raw_altitude = float(np.clip(raw_altitude, config.ALT_MIN, config.ALT_MAX))
        mpp_raw = raw_altitude / max(self.fx, 1e-6)
        self.mpp_history.append(mpp_raw)
        self.raw_alt_history.append(raw_altitude)
        if np.isfinite(affine_scale):
            self.affine_scale_history.append(float(affine_scale))
        if np.isfinite(delta_alt_cumulative):
            self.delta_alt_history.append(float(delta_alt_cumulative))

        triggered = 0
        if self.fixed_mpp is None:
            if self._should_enter_fixed(phase):
                alt_stable = float(np.median(self.raw_alt_history))
                self.fixed_mpp = alt_stable / max(self.fx, 1e-6)
                self.fixed_ref_alt = alt_stable
                triggered = frame_index
                logging.info(
                    "Fixed mpp enabled at frame %d: alt_stable=%.2fm mpp=%.6f",
                    frame_index,
                    alt_stable,
                    self.fixed_mpp,
                )
        else:
            if self._should_exit_fixed(affine_scale, delta_alt_cumulative):
                logging.info("Fixed mpp disabled at frame %d", frame_index)
                self.fixed_mpp = None
                self.fixed_ref_alt = None
                self.exit_count = 0

        median_mpp = float(np.median(self.mpp_history))
        mpp_for_position = self.fixed_mpp if self.fixed_mpp is not None else median_mpp
        return ScaleManagerResult(
            mpp_for_position=float(mpp_for_position),
            mpp_history_median=median_mpp,
            mpp_current_raw=float(mpp_raw),
            mpp_ratio=float(mpp_raw / max(median_mpp, 1e-9)),
            is_alt_fixed=self.fixed_mpp is not None,
            alt_fixed_triggered=int(triggered),
        )

    def _should_enter_fixed(self, phase: str) -> bool:
        if phase != "B":
            return False
        if len(self.raw_alt_history) < config.CALIB_WINDOW or len(self.affine_scale_history) < config.CALIB_WINDOW:
            return False
        scale_std = float(np.std(np.asarray(self.affine_scale_history, dtype=float)))
        delta_confirm = float(np.median(np.abs(np.asarray(self.delta_alt_history, dtype=float)))) if self.delta_alt_history else np.inf
        return scale_std < config.SCALE_STD_THRESH and delta_confirm < config.ALT_CHANGE_CONFIRM

    def _should_exit_fixed(self, affine_scale: float, delta_alt_cumulative: float) -> bool:
        if np.isfinite(affine_scale) and abs(float(affine_scale) - 1.0) > 0.01:
            self.exit_count += 1
        else:
            self.exit_count = 0
        if self.exit_count >= 10:
            return True
        return np.isfinite(delta_alt_cumulative) and abs(float(delta_alt_cumulative)) > config.PHASE_CHANGE_THRESH
