"""Trajectory estimation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .geo import GeoReference, enu_to_geodetic, geodetic_to_enu, horizontal_errors_m
from .vision import FrameMotion, create_estimator, read_gray


@dataclass(frozen=True)
class EstimationConfig:
    """Configuration for pure image-based dead reckoning after the first frame."""

    horizontal_fov_deg: float = 30.0
    yaw_deg: float = -90.0
    max_width: int = 960
    max_frames: int | None = None
    start_index: int = 0
    height_from_scale: bool = False
    min_confidence: float = 0.35
    min_inliers: int = 30
    smoothing_alpha: float = 0.75
    max_step_m: float = 30.0
    keyframe_max_interval: int = 8


def _meters_per_pixel(altitude_m: float, width_px: int, horizontal_fov_deg: float) -> float:
    footprint_width_m = 2.0 * altitude_m * tan(radians(horizontal_fov_deg) / 2.0)
    return footprint_width_m / float(width_px)


def _rotate_image_motion_to_enu(dx_m: float, dy_m: float, yaw_deg: float) -> tuple[float, float]:
    """Map image displacement to ENU.

    The image transform describes feature motion from previous to current frame.
    For a nadir-looking camera, UAV motion is opposite along image x and follows
    positive image y for north under the default coordinate convention.
    """

    east_cam = -dx_m
    north_cam = dy_m
    yaw = radians(yaw_deg)
    east = cos(yaw) * east_cam - sin(yaw) * north_cam
    north = sin(yaw) * east_cam + cos(yaw) * north_cam
    return east, north


def _motion_is_reliable(motion: FrameMotion, config: EstimationConfig) -> bool:
    return motion.confidence >= config.min_confidence and motion.inliers >= config.min_inliers


def _motion_to_step(
    motion: FrameMotion,
    altitude_m: float,
    width_px: int,
    config: EstimationConfig,
) -> tuple[float, float, float]:
    mpp = _meters_per_pixel(altitude_m, width_px, config.horizontal_fov_deg)
    east, north = _rotate_image_motion_to_enu(motion.dx_px * mpp, motion.dy_px * mpp, config.yaw_deg)
    return east, north, mpp


def _limit_step(step: np.ndarray, max_step_m: float) -> tuple[np.ndarray, bool]:
    if max_step_m <= 0:
        return step, False
    norm = float(np.linalg.norm(step))
    if norm <= max_step_m or norm == 0.0:
        return step, False
    return step * (max_step_m / norm), True


def _smooth_step(raw_step: np.ndarray, previous_step: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous_step is None:
        return raw_step
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * raw_step + (1.0 - alpha) * previous_step


def estimate_trajectory(frame_table: pd.DataFrame, config: EstimationConfig) -> pd.DataFrame:
    """Estimate trajectory from first-frame location and continuous images only."""

    if config.start_index < 0 or config.start_index >= len(frame_table) - 1:
        raise ValueError("start_index must be within the available sequence range")

    end = len(frame_table)
    if config.max_frames is not None:
        end = min(end, config.start_index + config.max_frames)
    data = frame_table.iloc[config.start_index:end].reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("The selected sequence must contain at least 2 frames")

    reference = GeoReference(
        longitude=float(data.loc[0, "longitude"]),
        latitude=float(data.loc[0, "latitude"]),
        altitude=float(data.loc[0, "altitude"]),
    )

    # Truth is prepared only for validation metrics and never feeds the
    # prediction path after the first frame.
    truth_enu = geodetic_to_enu(
        data["longitude"].to_numpy(),
        data["latitude"].to_numpy(),
        data["altitude"].to_numpy(),
        reference,
    )

    estimator = create_estimator()
    prev_gray = read_gray(data.loc[0, "image_path"], max_width=config.max_width)
    keyframe_gray = prev_gray
    keyframe_index = 0
    keyframe_enu = np.zeros(3, dtype=float)
    width_px = prev_gray.shape[1]

    predicted_enu = np.zeros((len(data), 3), dtype=float)
    previous_step: np.ndarray | None = None
    motion_rows: list[dict[str, float | int | str | bool]] = []

    for i in tqdm(range(1, len(data)), desc="estimating:orb_affine", unit="frame"):
        curr_gray = read_gray(data.loc[i, "image_path"], max_width=config.max_width)
        adjacent_motion = estimator.estimate(prev_gray, curr_gray)
        adjacent_good = _motion_is_reliable(adjacent_motion, config)

        source = "adjacent"
        used_motion = adjacent_motion
        raw_step: np.ndarray | None = None
        meters_per_pixel = float("nan")
        limited = False
        keyframe_good = False

        use_keyframe = i - keyframe_index <= config.keyframe_max_interval
        if use_keyframe:
            keyframe_motion = estimator.estimate(keyframe_gray, curr_gray)
            keyframe_good = _motion_is_reliable(keyframe_motion, config)
            if keyframe_good:
                altitude_abs = reference.altitude + keyframe_enu[2]
                east, north, meters_per_pixel = _motion_to_step(keyframe_motion, altitude_abs, width_px, config)
                candidate = keyframe_enu[:2] + np.array([east, north], dtype=float)
                raw_step = candidate - predicted_enu[i - 1, :2]
                used_motion = keyframe_motion
                source = "keyframe"

        if raw_step is None:
            if not adjacent_good and previous_step is not None:
                raw_step = previous_step.copy()
                source = "motion_model"
            else:
                altitude_abs = reference.altitude + predicted_enu[i - 1, 2]
                east, north, meters_per_pixel = _motion_to_step(adjacent_motion, altitude_abs, width_px, config)
                raw_step = np.array([east, north], dtype=float)

        raw_step, limited = _limit_step(raw_step, config.max_step_m)
        step = _smooth_step(raw_step, previous_step, config.smoothing_alpha)
        predicted_enu[i, 0] = predicted_enu[i - 1, 0] + step[0]
        predicted_enu[i, 1] = predicted_enu[i - 1, 1] + step[1]

        if config.height_from_scale and used_motion.scale > 0.05:
            prev_alt_abs = reference.altitude + predicted_enu[i - 1, 2]
            next_alt_abs = prev_alt_abs / used_motion.scale
            predicted_enu[i, 2] = next_alt_abs - reference.altitude
        else:
            predicted_enu[i, 2] = predicted_enu[i - 1, 2]

        if not keyframe_good or i - keyframe_index >= config.keyframe_max_interval:
            keyframe_gray = curr_gray
            keyframe_index = i
            keyframe_enu = predicted_enu[i].copy()

        previous_step = step
        prev_gray = curr_gray
        motion_rows.append(
            {
                "timestamp": int(data.loc[i, "timestamp"]),
                "dx_px": used_motion.dx_px,
                "dy_px": used_motion.dy_px,
                "scale": used_motion.scale,
                "confidence": used_motion.confidence,
                "matches": used_motion.matches,
                "inliers": used_motion.inliers,
                "adjacent_confidence": adjacent_motion.confidence,
                "adjacent_inliers": adjacent_motion.inliers,
                "motion_source": source,
                "step_limited": limited,
                "meters_per_pixel": meters_per_pixel,
                "step_east_m": step[0],
                "step_north_m": step[1],
                "raw_step_east_m": raw_step[0],
                "raw_step_north_m": raw_step[1],
                "keyframe_index": keyframe_index,
            }
        )

    predicted_geo = enu_to_geodetic(predicted_enu, reference)
    horizontal_error = horizontal_errors_m(predicted_enu, truth_enu)
    altitude_error = predicted_geo[:, 2] - data["altitude"].to_numpy()
    total_error = np.linalg.norm(predicted_enu - truth_enu, axis=1)

    result = data[["timestamp", "image_path", "longitude", "latitude", "altitude"]].copy()
    result = result.rename(
        columns={
            "longitude": "truth_longitude",
            "latitude": "truth_latitude",
            "altitude": "truth_altitude",
        }
    )
    result["pred_longitude"] = predicted_geo[:, 0]
    result["pred_latitude"] = predicted_geo[:, 1]
    result["pred_altitude"] = predicted_geo[:, 2]
    result["pred_east_m"] = predicted_enu[:, 0]
    result["pred_north_m"] = predicted_enu[:, 1]
    result["pred_up_m"] = predicted_enu[:, 2]
    result["truth_east_m"] = truth_enu[:, 0]
    result["truth_north_m"] = truth_enu[:, 1]
    result["truth_up_m"] = truth_enu[:, 2]
    result["horizontal_error_m"] = horizontal_error
    result["altitude_error_m"] = altitude_error
    result["total_error_m"] = total_error

    motion_df = pd.DataFrame(motion_rows)
    if not motion_df.empty:
        result = result.merge(motion_df, on="timestamp", how="left")
    return result


def summarize_errors(result: pd.DataFrame) -> dict[str, float]:
    """Return common error statistics."""

    metrics = {}
    for col in ["horizontal_error_m", "altitude_error_m", "total_error_m"]:
        values = result[col].to_numpy(dtype=float)
        metrics[f"{col}_mean"] = float(np.mean(np.abs(values)))
        metrics[f"{col}_median"] = float(np.median(np.abs(values)))
        metrics[f"{col}_max"] = float(np.max(np.abs(values)))
        metrics[f"{col}_rmse"] = float(np.sqrt(np.mean(values * values)))
    return metrics


def save_outputs(result: pd.DataFrame, output_dir: Path, prefix: str) -> tuple[Path, Path]:
    """Save CSV results and a trajectory plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_results.csv"
    plot_path = output_dir / f"{prefix}_trajectory.png"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))
    plt.plot(result["truth_east_m"], result["truth_north_m"], label="truth", linewidth=2)
    plt.plot(result["pred_east_m"], result["pred_north_m"], label="predicted", linewidth=1.5)
    plt.scatter(result["truth_east_m"].iloc[0], result["truth_north_m"].iloc[0], marker="o", label="start")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("UAV Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return csv_path, plot_path
