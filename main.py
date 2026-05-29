"""Main entry point for pure-vision UAV localization."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from modules.camera_calibrator import CameraCalibrator
from modules.data_loader import DataLoader, LoadedData
from modules.ekf_localizer import EKFLocalizer, EKFState
from modules.evaluator import Evaluator
from modules.feature_tracker import FeatureTracker
from modules.flight_phase_detector import FlightPhaseDetector
from modules.physics_constraint import PhysicsConstraint
from modules.scale_estimator import ScaleEstimator
from modules.scale_manager import ScaleManager
from modules.start_frame_checker import StartFrameChecker
from modules.yaw_calibrator import YawCalibrator
from utils.geo_utils import pixel_to_geo_delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure-vision UAV lon/lat/alt estimation from the first valid frame.")
    parser.add_argument("--image-dir", type=Path, default=config.IMAGE_DIR)
    parser.add_argument("--csv", type=Path, default=config.CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--initial-longitude", type=float, default=None)
    parser.add_argument("--initial-latitude", type=float, default=None)
    parser.add_argument("--initial-altitude", type=float, default=None)
    parser.add_argument("--first-image-path", type=Path, default=None)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run(args: argparse.Namespace) -> dict[str, float]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(args.image_dir, args.csv)
    all_data = loader.load()
    requested_start = _resolve_start(all_data, args.first_image_path)
    start_result = StartFrameChecker().check(all_data.image_paths, requested_start)
    start = start_result.index
    end = len(all_data.timestamps) if args.max_frames is None else min(len(all_data.timestamps), start + args.max_frames)
    data = loader.slice_from(all_data, start, end)
    yaw_data = loader.slice_from(all_data, start, min(len(all_data.timestamps), start + config.YAW_CALIBRATION_FRAMES))

    initial_lon = data.gt_lon[0] if args.initial_longitude is None else args.initial_longitude
    initial_lat = data.gt_lat[0] if args.initial_latitude is None else args.initial_latitude
    initial_alt = data.gt_alt[0] if args.initial_altitude is None else args.initial_altitude
    logging.info("Initial pose: lon=%.8f lat=%.8f alt=%.2f start_timestamp=%s", initial_lon, initial_lat, initial_alt, data.timestamps[0])

    calibration = CameraCalibrator(output_dir / "calibration.json", args.recalibrate).estimate(data.image_paths)
    calibration = _refine_calibration_with_motion_prior(calibration, yaw_data.image_paths, initial_alt, output_dir / "calibration.json")
    if calibration.confidence <= 0.7:
        logging.warning("Calibration confidence %.2f is below acceptance target 0.7", calibration.confidence)

    yaw = YawCalibrator(calibration.fx).calibrate(yaw_data.image_paths, yaw_data.gt_lon, yaw_data.gt_lat, initial_alt, 0)
    config.CAMERA_YAW_DEG = yaw.yaw_deg
    if not yaw.used:
        logging.warning("Yaw calibration kept theta=0")

    tracker = FeatureTracker()
    first_gray = tracker.read_gray(data.image_paths[0])
    tracker.initialize(first_gray)
    phase_detector = FlightPhaseDetector()
    physics = PhysicsConstraint()
    scale_estimator = ScaleEstimator(calibration.fx, initial_alt, calibration.confidence)
    scale_manager = ScaleManager(calibration.fx, initial_alt)
    ekf = EKFLocalizer(initial_lon, initial_lat, initial_alt)

    records = [_initial_record(data, first_gray, initial_lon, initial_lat, initial_alt, start_result, yaw.yaw_deg)]
    last_phase = phase_detector.update(None, np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), first_gray.shape[:2])
    current_yaw_deg = yaw.yaw_deg
    visual_map_transform = np.eye(3)
    visual_map_center: np.ndarray | None = None

    for i in tqdm(range(1, len(data.timestamps)), desc="localizing", unit="frame"):
        dt = float(data.delta_t[i])
        curr_gray = tracker.read_gray(data.image_paths[i])
        blur = tracker.blur_score(curr_gray)
        if blur < config.BLUR_LAPLACIAN_THRESHOLD:
            pred_state = ekf.predict(dt, last_phase.phase)
            logging.info("Blur frame skipped at %s: laplacian_var=%.2f", data.timestamps[i], blur)
            records.append(_record(data, i, pred_state, last_phase, blur, current_yaw_deg, ekf_update_mode=3, method="blur_predict"))
            continue

        tracking = tracker.track(curr_gray)
        if tracking.is_keyframe_reset:
            physics.reset()
        phase = phase_detector.update(tracking.affine_matrix, tracking.prev_points, tracking.curr_points, curr_gray.shape[:2])
        last_phase = phase
        pred_state = ekf.predict(dt, phase.phase)
        current_yaw_deg = _update_visual_yaw(
            current_yaw_deg,
            tracking.affine_matrix,
            tracking.fourier_rotation_deg,
            tracking.fourier_response,
        )
        map_dx, map_dy, visual_map_transform, visual_map_center = _cumulative_affine_delta(
            visual_map_transform,
            visual_map_center,
            tracking.affine_matrix,
            curr_gray.shape[:2],
        )
        motion_dx = map_dx if np.isfinite(map_dx) else tracking.pixel_dx
        motion_dy = map_dy if np.isfinite(map_dy) else tracking.pixel_dy
        scale = scale_estimator.estimate(
            tracking.prev_points,
            tracking.curr_points,
            tracking.affine_matrix,
            pred_state.altitude,
            phase.phase_from_vision if phase.phase_from_vision.startswith("A") else phase.phase,
            dt,
            pred_state.v_alt,
        )
        scale_state = scale_manager.update(
            scale.raw_alt_estimate,
            scale.affine_scale,
            scale.delta_alt_cumulative,
            phase.phase,
            i,
        )
        physics_result = physics.apply(
            motion_dx,
            motion_dy,
            pred_state.altitude,
            scale.estimated_altitude,
            calibration.fx,
            phase.phase,
        )
        effective_conf = scale.scale_confidence * physics_result.confidence_multiplier
        altitude_for_mpp = scale_state.mpp_for_position * calibration.fx
        d_lon, d_lat, _, _, _ = pixel_to_geo_delta(
            physics_result.pixel_dx,
            physics_result.pixel_dy,
            altitude_for_mpp,
            calibration.fx,
            pred_state.latitude,
            current_yaw_deg,
        )
        obs_lon = pred_state.longitude + d_lon
        obs_lat = pred_state.latitude + d_lat
        update = ekf.update_visual(
            obs_lon,
            obs_lat,
            physics_result.altitude,
            tracking.inlier_ratio,
            effective_conf,
            phase.phase,
            physics_result.predict_only,
        )
        records.append(
            _record(
                data,
                i,
                update.state,
                phase,
                blur,
                current_yaw_deg,
                pixel_dx=physics_result.pixel_dx,
                pixel_dy=physics_result.pixel_dy,
                inlier_ratio=tracking.inlier_ratio,
                scale_confidence=effective_conf,
                scale_method=scale.scale_method,
                divergence_ratio=scale.divergence_ratio,
                affine_scale=scale.affine_scale,
                consistency_diff=scale.consistency_diff,
                pixels_per_meter_used=scale.pixels_per_meter_used,
                is_alt_fixed=scale_state.is_alt_fixed,
                physics_flag=physics_result.flag,
                ekf_innovation_alt=update.alt_innovation,
                ekf_update_mode=update.update_mode,
                is_keyframe_reset=tracking.is_keyframe_reset,
                num_tracked=tracking.num_tracked,
                affine_rotation_deg=tracking.affine_rotation_deg,
                phase_dx=tracking.phase_dx,
                phase_dy=tracking.phase_dy,
                phase_response=tracking.phase_response,
                fourier_rotation_deg=tracking.fourier_rotation_deg,
                fourier_response=tracking.fourier_response,
                map_pixel_dx=map_dx,
                map_pixel_dy=map_dy,
                raw_alt_estimate=scale.raw_alt_estimate,
                cumulative_window_n=scale.cumulative_window_n,
                delta_alt_cumulative=scale.delta_alt_cumulative,
                mpp_history_median=scale_state.mpp_history_median,
                mpp_current_raw=scale_state.mpp_current_raw,
                mpp_ratio=scale_state.mpp_ratio,
                alt_fixed_triggered=scale_state.alt_fixed_triggered,
                ekf_v_alt=update.state.v_alt,
                method=tracking.method,
            )
        )

    metrics = Evaluator(output_dir).evaluate(records)
    logging.info("Metrics: %s", metrics)
    return metrics


def _resolve_start(data: LoadedData, first_image_path: Path | None) -> int:
    if first_image_path is None:
        return 0
    target = Path(first_image_path).stem.strip()
    for idx, ts in enumerate(data.timestamps):
        if ts == target:
            return idx
    raise ValueError(f"first_image_path timestamp {target} was not found in matched data")


def _refine_calibration_with_motion_prior(calibration, image_paths: list[Path], initial_alt: float, calibration_path: Path):
    if not config.ENABLE_MOTION_PRIOR_FOCAL_REFINE or config.MANUAL_FOCAL_PX is not None:
        return calibration
    displacements: list[float] = []
    max_pairs = min(len(image_paths) - 1, config.MOTION_PRIOR_FRAMES)
    if max_pairs < 3:
        return calibration
    for i in range(max_pairs):
        gray0 = FeatureTracker.read_gray(image_paths[i])
        gray1 = FeatureTracker.read_gray(image_paths[i + 1])
        pts0 = cv2.goodFeaturesToTrack(gray0, maxCorners=config.MAX_CORNERS, qualityLevel=config.QUALITY_LEVEL, minDistance=config.MIN_DISTANCE)
        if pts0 is None or len(pts0) < 20:
            continue
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None, winSize=config.LK_WIN_SIZE, maxLevel=config.LK_MAX_LEVEL)
        if pts1 is None or st is None:
            continue
        valid = st.reshape(-1) == 1
        delta = pts1.reshape(-1, 2)[valid] - pts0.reshape(-1, 2)[valid]
        if len(delta) >= 10:
            displacements.append(float(np.linalg.norm(np.median(delta, axis=0))))
    if not displacements:
        return calibration
    px = float(np.percentile(displacements, config.MOTION_PRIOR_PERCENTILE))
    required_fx = px * initial_alt / max(config.MAX_HORIZ_DISP_M, 1e-6) * config.MOTION_PRIOR_MARGIN
    max_fx = max(calibration.image_size) * config.FOCAL_MAX_FACTOR
    refined_fx = float(np.clip(max(calibration.fx, required_fx), calibration.fx, max_fx))
    if refined_fx <= calibration.fx * 1.05:
        return calibration
    logging.warning(
        "Refining focal with motion prior: fx %.2f -> %.2f using %.2fpx percentile displacement",
        calibration.fx,
        refined_fx,
        px,
    )
    calibration.fx = refined_fx
    calibration.fy = refined_fx
    calibration.K[0][0] = refined_fx
    calibration.K[1][1] = refined_fx
    calibration.method = f"{calibration.method}+motion_prior"
    calibration.confidence = max(calibration.confidence, 0.72)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(asdict(calibration), indent=2, ensure_ascii=False), encoding="utf-8")
    return calibration


def _update_visual_yaw(
    current_yaw_deg: float,
    affine_matrix: np.ndarray | None,
    fourier_rotation_deg: float,
    fourier_response: float,
) -> float:
    if not config.ENABLE_VISUAL_YAW_INTEGRATION:
        return current_yaw_deg
    if (
        config.USE_FOURIER_MELLIN_YAW
        and np.isfinite(fourier_rotation_deg)
        and fourier_response >= config.FOURIER_MELLIN_MIN_RESPONSE
    ):
        delta = config.FOURIER_MELLIN_SIGN * float(fourier_rotation_deg)
        delta = float(np.clip(delta, -config.FOURIER_MELLIN_MAX_STEP_DEG, config.FOURIER_MELLIN_MAX_STEP_DEG))
        return current_yaw_deg + delta
    if affine_matrix is None:
        return current_yaw_deg
    a = float(affine_matrix[0, 0])
    c = float(affine_matrix[1, 0])
    delta = config.VISUAL_YAW_SIGN * float(np.rad2deg(np.arctan2(c, a)))
    delta = float(np.clip(delta, -config.MAX_VISUAL_YAW_STEP_DEG, config.MAX_VISUAL_YAW_STEP_DEG))
    return current_yaw_deg + delta


def _cumulative_affine_delta(
    transform_curr_to_start: np.ndarray,
    last_center_start: np.ndarray | None,
    affine_matrix: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[float, float, np.ndarray, np.ndarray | None]:
    if not config.ENABLE_CUMULATIVE_AFFINE_MOTION or affine_matrix is None:
        return np.nan, np.nan, transform_curr_to_start, last_center_start
    h, w = image_shape
    A = np.eye(3, dtype=float)
    A[:2, :] = affine_matrix
    try:
        inv_a = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, transform_curr_to_start, last_center_start
    transform_curr_to_start = transform_curr_to_start @ inv_a
    center = np.array([w * 0.5, h * 0.5, 1.0], dtype=float)
    center_start_h = transform_curr_to_start @ center
    if abs(center_start_h[2]) < 1e-9:
        return np.nan, np.nan, transform_curr_to_start, last_center_start
    center_start = center_start_h[:2] / center_start_h[2]
    if last_center_start is None:
        return np.nan, np.nan, transform_curr_to_start, center_start
    camera_delta = center_start - last_center_start
    apparent_motion = -camera_delta
    return float(apparent_motion[0]), float(apparent_motion[1]), transform_curr_to_start, center_start


def _initial_record(data: LoadedData, gray: np.ndarray, lon: float, lat: float, alt: float, start, yaw_deg: float) -> dict[str, object]:
    phase = type(
        "Phase",
        (),
        {
            "phase": "C",
            "confidence": 1.0,
            "scale_trend": "transition",
            "up_ratio": 0.0,
            "down_ratio": 0.0,
            "stable_ratio": 0.0,
            "phase_from_vision": "C",
        },
    )()
    return _record(
        data,
        0,
        EKFState(lon, lat, alt, 0.0, 0.0, 0.0),
        phase,
        FeatureTracker.blur_score(gray),
        yaw_deg,
        inlier_ratio=start.inlier_ratio,
        is_keyframe_reset=True,
        num_tracked=start.corner_count,
        method="initial",
    )


def _record(
    data: LoadedData,
    i: int,
    state: EKFState,
    phase,
    blur: float,
    yaw_deg: float,
    pixel_dx: float = 0.0,
    pixel_dy: float = 0.0,
    inlier_ratio: float = 0.0,
    scale_confidence: float = 0.0,
    scale_method: int = 4,
    divergence_ratio: float = np.nan,
    affine_scale: float = np.nan,
    consistency_diff: float = np.nan,
    pixels_per_meter_used: float = np.nan,
    is_alt_fixed: bool = False,
    physics_flag: int = 0,
    ekf_innovation_alt: float = 0.0,
    ekf_update_mode: int = 3,
    is_keyframe_reset: bool = False,
    num_tracked: int = 0,
    affine_rotation_deg: float = np.nan,
    phase_dx: float = np.nan,
    phase_dy: float = np.nan,
    phase_response: float = 0.0,
    fourier_rotation_deg: float = np.nan,
    fourier_response: float = 0.0,
    map_pixel_dx: float = np.nan,
    map_pixel_dy: float = np.nan,
    raw_alt_estimate: float | None = None,
    cumulative_window_n: int = 0,
    delta_alt_cumulative: float = np.nan,
    mpp_history_median: float = np.nan,
    mpp_current_raw: float = np.nan,
    mpp_ratio: float = np.nan,
    alt_fixed_triggered: int = 0,
    ekf_v_alt: float = 0.0,
    method: str = "",
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
        "yaw_deg": yaw_deg,
        "blur_score": blur,
        "method": method,
        "raw_alt_estimate": state.altitude if raw_alt_estimate is None else raw_alt_estimate,
        "flight_phase": phase.phase,
        "phase_confidence": phase.confidence,
        "phase_from_vision": getattr(phase, "phase_from_vision", phase.phase),
        "scale_trend": getattr(phase, "scale_trend", ""),
        "scale_trend_ratio": f"up={getattr(phase, 'up_ratio', 0.0):.3f},stable={getattr(phase, 'stable_ratio', 0.0):.3f},down={getattr(phase, 'down_ratio', 0.0):.3f}",
        "up_ratio": getattr(phase, "up_ratio", 0.0),
        "down_ratio": getattr(phase, "down_ratio", 0.0),
        "stable_ratio": getattr(phase, "stable_ratio", 0.0),
        "scale_method": scale_method,
        "divergence_ratio": divergence_ratio,
        "affine_scale": affine_scale,
        "consistency_diff": consistency_diff,
        "pixels_per_meter_used": pixels_per_meter_used,
        "is_alt_fixed": is_alt_fixed,
        "cumulative_window_n": cumulative_window_n,
        "delta_alt_cumulative": delta_alt_cumulative,
        "mpp_history_median": mpp_history_median,
        "mpp_current_raw": mpp_current_raw,
        "mpp_ratio": mpp_ratio,
        "alt_fixed_triggered": alt_fixed_triggered,
        "physics_flag": physics_flag,
        "ekf_innovation_alt": ekf_innovation_alt,
        "ekf_v_alt": ekf_v_alt,
        "ekf_update_mode": ekf_update_mode,
        "is_keyframe_reset": is_keyframe_reset,
        "num_tracked": num_tracked,
        "affine_rotation_deg": affine_rotation_deg,
        "phase_dx": phase_dx,
        "phase_dy": phase_dy,
        "phase_response": phase_response,
        "fourier_rotation_deg": fourier_rotation_deg,
        "fourier_response": fourier_response,
        "map_pixel_dx": map_pixel_dx,
        "map_pixel_dy": map_pixel_dy,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
