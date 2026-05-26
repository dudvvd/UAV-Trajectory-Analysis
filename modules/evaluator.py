"""Evaluation and output generation for UAV localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.geo_utils import haversine_m
from utils.visualization import plot_altitude, plot_error_over_time, plot_trajectory


@dataclass(frozen=True)
class EvaluationSummary:
    MAE_horizontal: float
    RMSE_horizontal: float
    MAX_horizontal: float
    MAE_altitude: float
    RMSE_altitude: float
    drift_per_frame: float


class Evaluator:
    """Compute validation metrics and write CSV/plots."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def evaluate(self, records: list[dict[str, object]]) -> tuple[pd.DataFrame, EvaluationSummary]:
        result = pd.DataFrame(records)
        h_errors = [
            haversine_m(float(row.pred_lon), float(row.pred_lat), float(row.gt_lon), float(row.gt_lat))
            for row in result.itertuples(index=False)
        ]
        result["horizontal_error"] = h_errors
        result["altitude_error"] = result["pred_alt"].astype(float) - result["gt_alt"].astype(float)

        h = result["horizontal_error"].to_numpy(dtype=float)
        alt = result["altitude_error"].to_numpy(dtype=float)
        summary = EvaluationSummary(
            MAE_horizontal=float(np.mean(np.abs(h))),
            RMSE_horizontal=float(np.sqrt(np.mean(h * h))),
            MAX_horizontal=float(np.max(np.abs(h))),
            MAE_altitude=float(np.mean(np.abs(alt))),
            RMSE_altitude=float(np.sqrt(np.mean(alt * alt))),
            drift_per_frame=float(np.mean(np.abs(h)) / max(len(result) - 1, 1)),
        )
        return result, summary

    def save(self, result: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_dir / "results.csv", index=False, encoding="utf-8-sig")
        plot_trajectory(result, self.output_dir / "trajectory_comparison.png")
        plot_error_over_time(result, self.output_dir / "error_over_time.png")
        plot_altitude(result, self.output_dir / "altitude_comparison.png")
