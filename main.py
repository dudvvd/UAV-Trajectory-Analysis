"""Main entry point for pure-vision UAV localization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from modules.camera_calibrator import CameraCalibrator
from modules.data_loader import DataLoader, LoadedData
from modules.ekf_localizer import EKFLocalizer
from modules.evaluator import Evaluator
from modules.feature_tracker import FeatureTracker
from modules.scale_level_validator import ScaleLevelValidator
from modules.scale_estimator import ScaleEstimator
from modules.yaw_calibrator import YawCalibrator
from utils.geo_utils import pixel_to_geo_delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure-vision UAV lon/lat/alt estimation from an initial pose.")
    parser.add_argument("--image-dir", type=Path, default=config.IMAGE_DIR)
    parser.add_argument("--csv", type=Path, default=config.CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--first-image-path", type=Path, default=None)
    parser.add_argument("--initial-longitude", type=float, default=None)
    parser.add_argument("--initial-latitude", type=float, default=None)
    parser.add_argument("--initial-altitude", type=float, default=None)
    parser.add_argument("--recalibrate", action="store_true", help="Ignore cached calibration.json and estimate again.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional development limit.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _resolve_start(data: LoadedData, first_image_path: Path | None) -> int:
    if first_image_path is None:
        return 0
    target = Path(first_image_path).stem.strip()
    for i, timestamp in enumerate(data.timestamps):
        if timestamp == target:
            return i
    raise ValueError(f"first_image_path timestamp {target} was not found in matched data")


def _scaled_intrinsics(calibration_k: list[list[float]], gray_shape: tuple[int, int], calibration_shape: tuple[int, int]) -> np.ndarray:
    K = np.asarray(calibration_k, dtype=float).copy()
    h_cal, w_cal = calibration_shape
    h_proc, w_proc = gray_shape
    sx = w_proc / float(w_cal)
    sy = h_proc / float(h_cal)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def run(args: argparse.Namespace) -> tuple[object, object]:
    loader = DataLoader(args.image_dir, args.csv)
    data = loader.load()
    start = _resolve_start(data, args.first_image_path)
    end = len(data.timestamps) if args.max_frames is None else min(len(data.timestamps), start + args.max_frames)
    if end - start < 2:
        raise ValueError("Selected range must contain at least two frames")

    initial_lon = data.gt_lon[start] if args.initial_longitude is None else args.initial_longitude
    initial_lat = data.gt_lat[start] if args.initial_latitude is None else args.initial_latitude
    initial_alt = data.gt_alt[start] if args.initial_altitude is None else args.initial_altitude
    logging.info("Initial pose: lon=%.8f lat=%.8f alt=%.2f", initial_lon, initial_lat, initial_alt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator = CameraCalibrator(cache_path=output_dir / "calibration.json", recalibrate=args.recalibrate)
    calibration = calibrator.estimate(data.image_paths[start:end])

    tracker = FeatureTracker()
    first_gray = tracker.read_gray(data.image_paths[start])
    import cv2

    full_first = cv2.imread(str(data.image_paths[start]), cv2.IMREAD_GRAYSCALE)
    if full_first is None:
        raise FileNotFoundError(data.image_paths[start])
    K = _scaled_intrinsics(calibration.K, first_gray.shape[:2], full_first.shape[:2])
    yaw_calibrator = YawCalibrator(float(K[0, 0]))
    yaw_result = yaw_calibrator.calibrate(
        data.image_paths,
        data.gt_lon,
        data.gt_lat,
        initial_alt,
        start,
        end,
    )
    config.CAMERA_YAW_DEG = yaw_result.yaw_deg

    scale_estimator = ScaleEstimator(K, initial_alt, calibration_confidence=calibration.confidence)
    enabled_levels = ScaleLevelValidator(scale_estimator).validate(data.image_paths, start, end, initial_alt)
    scale_estimator.enabled_levels = enabled_levels
    ekf = EKFLocalizer(initial_lon, initial_lat, initial_alt)
    tracker.initialize(first_gray)

    records: list[dict[str, object]] = [
        {
            "timestamp": data.timestamps[start],
            "image_path": str(data.image_paths[start]),
            "pred_lon": initial_lon,
            "pred_lat": initial_lat,
            "pred_alt": initial_alt,
            "gt_lon": data.gt_lon[start],
            "gt_lat": data.gt_lat[start],
            "gt_alt": data.gt_alt[start],
            "pixel_dx": 0.0,
            "pixel_dy": 0.0,
            "inlier_ratio": 1.0,
            "scale_confidence": 1.0,
            "homography_inliers": 0,
            "scale_method": 4,
            "divergence_ratio": np.nan,
            "affine_scale": np.nan,
            "raw_alt_estimate": initial_alt,
            "ekf_alt": initial_alt,
            "yaw_deg": config.CAMERA_YAW_DEG,
            "ekf_update_mode": 1,
            "is_keyframe_reset": True,
            "num_tracked": 0,
            "blur_score": FeatureTracker.blur_score(first_gray),
            "skipped_blur": False,
            "method": "initial",
        }
    ]

    for i in tqdm(range(start + 1, end), desc="localizing", unit="frame"):
        dt = data.delta_t[i]
        state = ekf.predict(dt)
        try:
            curr_gray = tracker.read_gray(data.image_paths[i])
            blur = FeatureTracker.blur_score(curr_gray)
            if blur < config.BLUR_LAPLACIAN_THRESHOLD:
                logging.info("Blur frame skipped at %s: laplacian_var=%.2f", data.timestamps[i], blur)
                state = ekf.state
                records.append(
                    _record(
                        data,
                        i,
                        state,
                        blur,
                        skipped_blur=True,
                        yaw_deg=config.CAMERA_YAW_DEG,
                        ekf_update_mode=3,
                    )
                )
                continue

            tracking = tracker.track(curr_gray)
            scale = scale_estimator.estimate(
                tracking.prev_points,
                tracking.curr_points,
                tracking.homography,
                curr_gray.shape[:2],
                dt,
                float(state.velocity[2]),
                state.altitude,
            )
            delta_lon, delta_lat = pixel_to_geo_delta(
                tracking.pixel_dx,
                tracking.pixel_dy,
                scale.estimated_altitude,
                scale_estimator.fx,
                state.latitude,
                config.CAMERA_YAW_DEG,
            )
            delta_alt = scale.estimated_altitude - state.altitude
            state, update_mode = ekf.update_visual(
                delta_lon,
                delta_lat,
                delta_alt,
                tracking.inlier_ratio,
                scale.scale_confidence,
                dt,
            )
            records.append(
                _record(
                    data,
                    i,
                    state,
                    blur,
                    pixel_dx=tracking.pixel_dx,
                    pixel_dy=tracking.pixel_dy,
                    inlier_ratio=tracking.inlier_ratio,
                    scale_confidence=scale.scale_confidence,
                    homography_inliers=scale.homography_inliers,
                    scale_method=scale.scale_method,
                    divergence_ratio=scale.divergence_ratio,
                    affine_scale=scale.affine_scale,
                    raw_alt_estimate=scale.raw_alt_estimate,
                    yaw_deg=config.CAMERA_YAW_DEG,
                    ekf_update_mode=update_mode,
                    is_keyframe_reset=tracking.is_keyframe_reset,
                    num_tracked=tracking.num_tracked,
                    method=tracking.method,
                )
            )
        except Exception as exc:
            logging.exception("Frame %s failed; EKF prediction kept: %s", data.timestamps[i], exc)
            records.append(
                _record(
                    data,
                    i,
                    ekf.state,
                    float("nan"),
                    method="exception",
                    yaw_deg=config.CAMERA_YAW_DEG,
                    ekf_update_mode=3,
                )
            )

    evaluator = Evaluator(output_dir)
    result, summary = evaluator.evaluate(records)
    evaluator.save(result)
    return result, summary


def _record(
    data: LoadedData,
    i: int,
    state: object,
    blur_score: float,
    pixel_dx: float = 0.0,
    pixel_dy: float = 0.0,
    inlier_ratio: float = 0.0,
    scale_confidence: float = 0.0,
    homography_inliers: int = 0,
    scale_method: int = 4,
    divergence_ratio: float = float("nan"),
    affine_scale: float = float("nan"),
    raw_alt_estimate: float | None = None,
    yaw_deg: float = 0.0,
    ekf_update_mode: int = 3,
    is_keyframe_reset: bool = False,
    num_tracked: int = 0,
    skipped_blur: bool = False,
    method: str = "predict",
) -> dict[str, object]:
    return {
        "timestamp": data.timestamps[i],
        "image_path": str(data.image_paths[i]),
        "pred_lon": state.longitude,
        "pred_lat": state.latitude,
        "pred_alt": state.altitude,
        "gt_lon": data.gt_lon[i],
        "gt_lat": data.gt_lat[i],
        "gt_alt": data.gt_alt[i],
        "pixel_dx": pixel_dx,
        "pixel_dy": pixel_dy,
        "inlier_ratio": inlier_ratio,
        "scale_confidence": scale_confidence,
        "homography_inliers": homography_inliers,
        "scale_method": scale_method,
        "divergence_ratio": divergence_ratio,
        "affine_scale": affine_scale,
        "raw_alt_estimate": state.altitude if raw_alt_estimate is None else raw_alt_estimate,
        "ekf_alt": state.altitude,
        "yaw_deg": yaw_deg,
        "ekf_update_mode": ekf_update_mode,
        "is_keyframe_reset": is_keyframe_reset,
        "num_tracked": num_tracked,
        "blur_score": blur_score,
        "skipped_blur": skipped_blur,
        "method": method,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _, summary = run(args)
    print("Output files:")
    print(f"  {Path(args.output_dir) / 'calibration.json'}")
    print(f"  {Path(args.output_dir) / 'trajectory_comparison.png'}")
    print(f"  {Path(args.output_dir) / 'error_over_time.png'}")
    print(f"  {Path(args.output_dir) / 'altitude_comparison.png'}")
    print(f"  {Path(args.output_dir) / 'results.csv'}")
    print("Metrics:")
    for key, value in summary.__dict__.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
