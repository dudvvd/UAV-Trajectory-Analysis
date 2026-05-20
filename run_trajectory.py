"""Command-line entry point for UAV trajectory estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uav_trajectory.data import DatasetConfig, load_dataset
from uav_trajectory.estimator import EstimationConfig, estimate_trajectory, save_outputs, summarize_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate UAV trajectory from adjacent IR images.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/IR-image"), help="图片目录")
    parser.add_argument("--csv", type=Path, default=Path("data/定位数据.csv"), help="定位 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--method", choices=["orb_affine", "phase"], default="orb_affine", help="图像运动估计方法")
    parser.add_argument("--start-index", type=int, default=0, help="从匹配后的第几帧开始")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数；不填则处理全部")
    parser.add_argument("--horizontal-fov-deg", type=float, default=60.0, help="相机水平视场角，用于像素到米换算")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="图像 x 轴相对正东的旋转角假设")
    parser.add_argument("--max-width", type=int, default=960, help="图像处理时的最大宽度")
    parser.add_argument("--estimate-altitude", action="store_true", help="使用仿射尺度估计高度变化；默认保持首帧高度")
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=0,
        help="使用前 N 帧真值标定像素到 ENU 的线性映射；0 表示纯图像递推",
    )
    parser.add_argument("--prefix", default=None, help="输出文件名前缀")
    parser.add_argument("--print-rows", type=int, default=8, help="打印前 N 行结果")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame_table = load_dataset(DatasetConfig(image_dir=args.image_dir, csv_path=args.csv))
    config = EstimationConfig(
        method=args.method,
        horizontal_fov_deg=args.horizontal_fov_deg,
        yaw_deg=args.yaw_deg,
        max_width=args.max_width,
        max_frames=args.max_frames,
        start_index=args.start_index,
        height_from_scale=args.estimate_altitude,
        calibration_frames=args.calibration_frames,
    )
    result = estimate_trajectory(frame_table, config)
    prefix = args.prefix or f"{args.method}_start{args.start_index}_n{len(result)}"
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
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n样例结果:")
        print(result[columns].head(args.print_rows).to_string(index=False))


if __name__ == "__main__":
    main()
