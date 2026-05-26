"""Plotting helpers for UAV localization results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils.geo_utils import lonlat_to_local_m


def plot_trajectory(result: pd.DataFrame, output_path: Path) -> None:
    """Draw predicted and validation trajectories in a local top-down frame."""

    ref_lon = float(result["gt_lon"].iloc[0])
    ref_lat = float(result["gt_lat"].iloc[0])
    gt_e, gt_n = lonlat_to_local_m(result["gt_lon"], result["gt_lat"], ref_lon, ref_lat)
    pred_e, pred_n = lonlat_to_local_m(result["pred_lon"], result["pred_lat"], ref_lon, ref_lat)

    plt.figure(figsize=(9, 7))
    plt.plot(gt_e, gt_n, "b-", linewidth=2.0, label="Ground truth")
    plt.plot(pred_e, pred_n, "r--", linewidth=1.5, label="Predicted")
    plt.scatter(gt_e[0], gt_n[0], marker="*", s=180, c="green", label="Start", zorder=4)
    reset_rows = result[result["is_keyframe_reset"].fillna(False)]
    if not reset_rows.empty:
        reset_e, reset_n = lonlat_to_local_m(reset_rows["pred_lon"], reset_rows["pred_lat"], ref_lon, ref_lat)
        plt.scatter(reset_e, reset_n, marker="o", s=24, c="gold", edgecolors="black", linewidths=0.4, label="Keyframe reset")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.PLOT_DPI)
    plt.close()


def plot_error_over_time(result: pd.DataFrame, output_path: Path) -> None:
    """Draw horizontal error curve with mean and one-sigma band."""

    errors = result["horizontal_error"].to_numpy(dtype=float)
    frames = np.arange(len(errors))
    mean = float(np.mean(errors))
    sigma = float(np.std(errors))
    plt.figure(figsize=(10, 5))
    plt.plot(frames, errors, color="tab:red", linewidth=1.2, label="Horizontal error")
    plt.axhline(mean, color="tab:blue", linestyle="-", linewidth=1.2, label=f"Mean {mean:.2f} m")
    plt.fill_between(frames, mean - sigma, mean + sigma, color="tab:blue", alpha=0.15, label="1 sigma")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frame index")
    plt.ylabel("Horizontal error (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.PLOT_DPI)
    plt.close()


def plot_altitude(result: pd.DataFrame, output_path: Path) -> None:
    """Draw predicted and validation altitude curves."""

    frames = np.arange(len(result))
    plt.figure(figsize=(10, 5))
    plt.plot(frames, result["gt_alt"], "b-", linewidth=1.5, label="Ground truth altitude")
    plt.plot(frames, result["pred_alt"], "r--", linewidth=1.2, label="Predicted altitude")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frame index")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.PLOT_DPI)
    plt.close()
