"""Evaluation, CSV export, and visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils.geo_utils import haversine_m, lonlat_to_local_m
from utils.visualization import shade_phases


class Evaluator:
    """Compute localization errors and write required artifacts."""

    def __init__(self, output_dir: Path = config.OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, float]:
        df = pd.DataFrame(records)
        df["horizontal_error"] = [
            haversine_m(plon, plat, glon, glat)
            for plon, plat, glon, glat in zip(df["pred_lon"], df["pred_lat"], df["gt_lon"], df["gt_lat"])
        ]
        df["altitude_error"] = np.abs(df["pred_alt"].astype(float) - df["gt_alt"].astype(float))
        metrics = self._metrics(df)
        df.to_csv(self.output_dir / "results.csv", index=False, encoding="utf-8-sig")
        self._plot_trajectory(df)
        self._plot_error(df)
        self._plot_altitude(df)
        self._plot_phase_distribution(df)
        return metrics

    @staticmethod
    def _metrics(df: pd.DataFrame) -> dict[str, float]:
        h = df["horizontal_error"].to_numpy(dtype=float)
        a = df["altitude_error"].to_numpy(dtype=float)
        return {
            "horizontal_mae": float(np.mean(h)),
            "horizontal_rmse": float(np.sqrt(np.mean(h**2))),
            "horizontal_max": float(np.max(h)),
            "altitude_mae": float(np.mean(a)),
            "altitude_rmse": float(np.sqrt(np.mean(a**2))),
            "altitude_max": float(np.max(a)),
            "drift_per_frame": float(h[-1] / max(len(h), 1)),
        }

    def _plot_trajectory(self, df: pd.DataFrame) -> None:
        ref_lon, ref_lat = float(df["gt_lon"].iloc[0]), float(df["gt_lat"].iloc[0])
        gt_e, gt_n = lonlat_to_local_m(df["gt_lon"].to_numpy(float), df["gt_lat"].to_numpy(float), ref_lon, ref_lat)
        pr_e, pr_n = lonlat_to_local_m(df["pred_lon"].to_numpy(float), df["pred_lat"].to_numpy(float), ref_lon, ref_lat)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(gt_e, gt_n, "b-", label="Ground Truth", linewidth=1.4)
        ax.plot(pr_e, pr_n, "r--", label="Prediction", linewidth=1.2)
        ax.plot(gt_e[0], gt_n[0], "g*", markersize=12, label="Start")
        resets = df["is_keyframe_reset"].astype(bool).to_numpy()
        if np.any(resets):
            ax.plot(pr_e[resets], pr_n[resets], "yo", markersize=3, label="Keyframe Reset")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Trajectory Comparison")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "trajectory_comparison.png", dpi=config.PLOT_DPI)
        plt.close(fig)

    def _plot_error(self, df: pd.DataFrame) -> None:
        x = np.arange(len(df))
        h = df["horizontal_error"].to_numpy(float)
        a = df["altitude_error"].to_numpy(float)
        phases = df["flight_phase"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(11, 5))
        shade_phases(ax, phases)
        ax.plot(x, h, "r-", label="Horizontal Error", linewidth=1.0)
        ax.plot(x, a, "b-", label="Altitude Error", linewidth=1.0)
        mean = float(np.mean(h))
        std = float(np.std(h))
        ax.axhline(mean, color="r", linestyle="--", linewidth=0.9, label="Horizontal Mean")
        ax.fill_between(x, max(0.0, mean - std), mean + std, color="r", alpha=0.12, label="Horizontal 1 Sigma")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Error (m)")
        ax.set_title("Error Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "error_over_time.png", dpi=config.PLOT_DPI)
        plt.close(fig)

    def _plot_altitude(self, df: pd.DataFrame) -> None:
        x = np.arange(len(df))
        phases = df["flight_phase"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(11, 4))
        shade_phases(ax, phases)
        ax.plot(x, df["gt_alt"], "b-", label="Ground Truth", linewidth=1.2)
        ax.plot(x, df["pred_alt"], "r--", label="Prediction", linewidth=1.2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("Altitude Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "altitude_comparison.png", dpi=config.PLOT_DPI)
        plt.close(fig)

    def _plot_phase_distribution(self, df: pd.DataFrame) -> None:
        counts = df["flight_phase"].value_counts().reindex(["A_up", "A_down", "B", "C"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color=[config.PHASE_COLORS[p] for p in counts.index])
        ax.set_xlabel("Flight Phase")
        ax.set_ylabel("Frames")
        ax.set_title("Phase Distribution")
        fig.tight_layout()
        fig.savefig(self.output_dir / "phase_distribution.png", dpi=config.PLOT_DPI)
        plt.close(fig)
