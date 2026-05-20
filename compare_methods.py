"""Run several classical motion estimators on the same UAV image segment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uav_trajectory.data import DatasetConfig, load_dataset
from uav_trajectory.estimator import EstimationConfig, estimate_trajectory, save_outputs, summarize_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trajectory estimation methods.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/IR-image"))
    parser.add_argument("--csv", type=Path, default=Path("data/定位数据.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compare"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--horizontal-fov-deg", type=float, default=60.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--calibration-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame_table = load_dataset(DatasetConfig(image_dir=args.image_dir, csv_path=args.csv))
    summaries = []
    for method in ["orb_affine", "phase"]:
        config = EstimationConfig(
            method=method,
            horizontal_fov_deg=args.horizontal_fov_deg,
            yaw_deg=args.yaw_deg,
            max_width=args.max_width,
            max_frames=args.max_frames,
            start_index=args.start_index,
            height_from_scale=False,
            calibration_frames=args.calibration_frames,
        )
        result = estimate_trajectory(frame_table, config)
        prefix = f"{method}_start{args.start_index}_n{len(result)}"
        save_outputs(result, args.output_dir, prefix)
        summary = summarize_errors(result)
        summary["method"] = method
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / "method_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df.to_string(index=False))
    print(f"\n方法对比结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
