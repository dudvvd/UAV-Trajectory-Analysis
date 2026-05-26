"""Command-line entry point for UAV trajectory estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uav_trajectory.data import DatasetConfig, load_dataset
from uav_trajectory.estimator import EstimationConfig, estimate_trajectory, save_outputs, summarize_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate UAV trajectory from the first pose and continuous images.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/IR-image"), help="Image directory.")
    parser.add_argument("--csv", type=Path, default=Path("data/定位数据.csv"), help="Validation CSV path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the initial frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--horizontal-fov-deg", type=float, default=30.0, help="Camera horizontal FOV.")
    parser.add_argument("--yaw-deg", type=float, default=-90.0, help="Image x-axis rotation relative to east.")
    parser.add_argument("--max-width", type=int, default=960, help="Maximum processing image width.")
    parser.add_argument("--estimate-altitude", action="store_true", help="Estimate altitude from affine scale.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Minimum RANSAC inlier ratio.")
    parser.add_argument("--min-inliers", type=int, default=30, help="Minimum RANSAC inlier count.")
    parser.add_argument("--smoothing-alpha", type=float, default=0.75, help="Step smoothing weight for current motion.")
    parser.add_argument("--max-step-m", type=float, default=30.0, help="Maximum allowed single-frame horizontal step.")
    parser.add_argument("--keyframe-max-interval", type=int, default=8, help="Maximum frames kept under one keyframe.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--print-rows", type=int, default=8, help="Print first N result rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame_table = load_dataset(DatasetConfig(image_dir=args.image_dir, csv_path=args.csv))
    config = EstimationConfig(
        horizontal_fov_deg=args.horizontal_fov_deg,
        yaw_deg=args.yaw_deg,
        max_width=args.max_width,
        max_frames=args.max_frames,
        start_index=args.start_index,
        height_from_scale=args.estimate_altitude,
        min_confidence=args.min_confidence,
        min_inliers=args.min_inliers,
        smoothing_alpha=args.smoothing_alpha,
        max_step_m=args.max_step_m,
        keyframe_max_interval=args.keyframe_max_interval,
    )
    result = estimate_trajectory(frame_table, config)
    prefix = args.prefix or f"orb_affine_start{args.start_index}_n{len(result)}"
    csv_path, plot_path = save_outputs(result, args.output_dir, prefix)

    summary = summarize_errors(result)
    print("输出文件:")
    print(f"  CSV : {csv_path}")
    print(f"  图像: {plot_path}")
    print("\n误差统计:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")

    columns = [
        "timestamp",
        "truth_longitude",
        "truth_latitude",
        "truth_altitude",
        "pred_longitude",
        "pred_latitude",
        "pred_altitude",
        "horizontal_error_m",
        "altitude_error_m",
        "confidence",
        "motion_source",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n样例结果:")
        print(result[columns].head(args.print_rows).to_string(index=False))


if __name__ == "__main__":
    main()
