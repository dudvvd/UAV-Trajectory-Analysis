"""Trajectory estimation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .geo import GeoReference, enu_to_geodetic, geodetic_to_enu, horizontal_errors_m
from .vision import create_estimator, read_gray


@dataclass(frozen=True)
class EstimationConfig:
    method: str = "orb_affine"
    horizontal_fov_deg: float = 60.0
    yaw_deg: float = 0.0
    max_width: int = 960
    max_frames: int | None = None
    start_index: int = 0
    height_from_scale: bool = False
    calibration_frames: int = 0


def _meters_per_pixel(altitude_m: float, width_px: int, horizontal_fov_deg: float) -> float:
    footprint_width_m = 2.0 * altitude_m * tan(radians(horizontal_fov_deg) / 2.0)
    return footprint_width_m / float(width_px)


def _rotate_image_motion_to_enu(dx_m: float, dy_m: float, yaw_deg: float) -> tuple[float, float]:
    """Map image displacement to ENU.

    The default assumes nadir-looking camera and image up aligned with north.
    The image transform describes feature motion from previous to current frame;
    UAV motion is opposite along image x and follows positive image y for north.
    """

    east_cam = -dx_m
    north_cam = dy_m
    yaw = radians(yaw_deg)
    east = cos(yaw) * east_cam - sin(yaw) * north_cam
    north = sin(yaw) * east_cam + cos(yaw) * north_cam
    return east, north


def estimate_trajectory(frame_table: pd.DataFrame, config: EstimationConfig) -> pd.DataFrame:
    """Estimate a trajectory from the selected image sequence."""

    if config.start_index < 0 or config.start_index >= len(frame_table) - 1:
        raise ValueError("start_index 必须落在可用序列范围内")

    end = len(frame_table)
    if config.max_frames is not None:
        end = min(end, config.start_index + config.max_frames)
    data = frame_table.iloc[config.start_index:end].reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("选定序列不足 2 帧")

    reference = GeoReference(
        longitude=float(data.loc[0, "longitude"]),
        latitude=float(data.loc[0, "latitude"]),
        altitude=float(data.loc[0, "altitude"]),
    )
    truth_enu = geodetic_to_enu(
        data["longitude"].to_numpy(),
        data["latitude"].to_numpy(),
        data["altitude"].to_numpy(),
        reference,
    )

    estimator = create_estimator(config.method)
    prev_gray = read_gray(data.loc[0, "image_path"], max_width=config.max_width)
    width_px = prev_gray.shape[1]
    motion_rows: list[dict[str, float | int | str]] = []
    motions = []

    iterator = range(1, len(data))
    for i in tqdm(iterator, desc=f"estimating:{config.method}", unit="frame"):
        curr_gray = read_gray(data.loc[i, "image_path"], max_width=config.max_width)
        motion = estimator.estimate(prev_gray, curr_gray)
        motions.append(motion)
        motion_rows.append(
            {
                "timestamp": int(data.loc[i, "timestamp"]),
                "dx_px": motion.dx_px,
                "dy_px": motion.dy_px,
                "scale": motion.scale,
                "confidence": motion.confidence,
                "matches": motion.matches,
                "inliers": motion.inliers,
            }
        )
        prev_gray = curr_gray

    pixel_to_enu = _fit_pixel_to_enu(motions, truth_enu, config.calibration_frames)
    predicted_enu = np.zeros((len(data), 3), dtype=float)
    predicted_enu[0, :] = truth_enu[0, :]

    for i, motion in enumerate(motions, start=1):
        prev_alt_abs = reference.altitude + predicted_enu[i - 1, 2]
        if pixel_to_enu is not None:
            step_east, step_north = _apply_pixel_to_enu(motion.dx_px, motion.dy_px, pixel_to_enu)
            mpp = float("nan")
        else:
            mpp = _meters_per_pixel(prev_alt_abs, width_px, config.horizontal_fov_deg)
            step_east, step_north = _rotate_image_motion_to_enu(
                motion.dx_px * mpp,
                motion.dy_px * mpp,
                config.yaw_deg,
            )

        predicted_enu[i, 0] = predicted_enu[i - 1, 0] + step_east
        predicted_enu[i, 1] = predicted_enu[i - 1, 1] + step_north

        if config.height_from_scale and motion.scale > 0.05:
            next_alt_abs = prev_alt_abs / motion.scale
            predicted_enu[i, 2] = next_alt_abs - reference.altitude
        else:
            predicted_enu[i, 2] = predicted_enu[i - 1, 2]

        motion_rows[i - 1]["meters_per_pixel"] = mpp
        motion_rows[i - 1]["step_east_m"] = step_east
        motion_rows[i - 1]["step_north_m"] = step_north

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
    result["calibration_frames"] = config.calibration_frames
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


def _fit_pixel_to_enu(motions, truth_enu: np.ndarray, calibration_frames: int) -> np.ndarray | None:
    """Fit a 2x2 pixel-to-ENU matrix from validation truth.

    This is intentionally optional: it uses ground truth after the initial frame,
    so it is useful for calibration/evaluation, not for a pure deployment run.
    """

    if calibration_frames <= 1:
        return None
    usable = min(calibration_frames - 1, len(motions))
    if usable < 2:
        return None

    pixel_steps = np.array([[m.dx_px, m.dy_px] for m in motions[:usable]], dtype=float)
    truth_steps = np.diff(truth_enu[: usable + 1, :2], axis=0)
    valid = np.isfinite(pixel_steps).all(axis=1) & np.isfinite(truth_steps).all(axis=1)
    if valid.sum() < 2:
        return None

    matrix, *_ = np.linalg.lstsq(pixel_steps[valid], truth_steps[valid], rcond=None)
    return matrix


def _apply_pixel_to_enu(dx_px: float, dy_px: float, matrix: np.ndarray) -> tuple[float, float]:
    step = np.array([dx_px, dy_px], dtype=float) @ matrix
    return float(step[0]), float(step[1])


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
