"""Plotting utilities for trajectory and diagnostic outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config


def shade_phases(ax: plt.Axes, phases: list[str]) -> None:
    """Draw phase-colored spans behind a time-series plot."""
    if not phases:
        return
    start = 0
    current = phases[0]
    for idx, phase in enumerate(phases[1:], 1):
        if phase != current:
            ax.axvspan(start, idx, color=config.PHASE_COLORS.get(current, "#eeeeee"), alpha=0.28, lw=0)
            start = idx
            current = phase
    ax.axvspan(start, len(phases), color=config.PHASE_COLORS.get(current, "#eeeeee"), alpha=0.28, lw=0)


def save_line_plot(path, x, series, title: str, ylabel: str) -> None:
    """Small helper for simple diagnostic lines."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, y, style in series:
        ax.plot(x, y, style, label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)


def finite_limits(values: np.ndarray, pad_ratio: float = 0.04) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    pad = max((hi - lo) * pad_ratio, 1e-6)
    return lo - pad, hi + pad
