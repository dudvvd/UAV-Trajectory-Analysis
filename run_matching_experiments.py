"""Run scheme-B and scheme-D visual matching experiments.

Scheme B: robust pairwise matching + affine geometry.
Scheme D: local keyframe map matching + affine geometry.

The implementation prefers OpenCV SIFT when available and falls back to ORB.
It is intentionally isolated from the main pipeline so experiments can be
repeated without changing the production flow.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from modules.data_loader import DataLoader, LoadedData
from utils.geo_utils import haversine_m, local_m_to_lonlat, lonlat_to_local_m


@dataclass
class MatchMotion:
    dx: float
    dy: float
    rotation_deg: float
    scale: float
    confidence: float
    num_matches: int
    num_inliers: int
    backend: str


class RobustMatcher:
    """Feature matching facade for future deep matchers and current SIFT/ORB."""

    def __init__(self, preferred: str = "auto", loftr_size: tuple[int, int] = (512, 384)) -> None:
        self.loftr = None
        self.device = "cpu"
        self.loftr_size = loftr_size
        if preferred == "loftr":
            self._init_loftr()
            return
        if preferred == "sift" and not hasattr(cv2, "SIFT_create"):
            preferred = "orb"
        if preferred in ("auto", "sift") and hasattr(cv2, "SIFT_create"):
            self.backend = "sift"
            self.detector = cv2.SIFT_create(nfeatures=700)
            self.norm = cv2.NORM_L2
        else:
            self.backend = "orb"
            self.detector = cv2.ORB_create(nfeatures=1500, scoreType=cv2.ORB_HARRIS_SCORE)
            self.norm = cv2.NORM_HAMMING

    def _init_loftr(self) -> None:
        try:
            import torch
            from kornia.feature import LoFTR
        except ImportError as exc:
            raise RuntimeError("LoFTR matcher requires torch and kornia. Install PyTorch and kornia first.") from exc
        self.backend = "loftr"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.loftr = LoFTR(pretrained="outdoor").eval().to(self.device)
        except Exception as exc:
            raise RuntimeError(
                "LoFTR pretrained weights are not available. Download loftr_outdoor.ckpt to "
                "C:\\Users\\86184\\.cache\\torch\\hub\\checkpoints\\loftr_outdoor.ckpt, "
                "or run once with network access: python -c \"from kornia.feature import LoFTR; LoFTR(pretrained='outdoor')\""
            ) from exc

    def motion(self, img0: np.ndarray, img1: np.ndarray) -> MatchMotion:
        if self.backend == "loftr":
            return self._motion_loftr(img0, img1)
        kp0, des0 = self.detector.detectAndCompute(img0, None)
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
            return self._empty()
        pairs = cv2.BFMatcher(self.norm).knnMatch(des0, des1, k=2)
        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
        if len(good) < 6:
            return self._empty(num_matches=len(good))
        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])
        A, mask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.995)
        if A is None or mask is None:
            return self._empty(num_matches=len(good))
        inliers = mask.reshape(-1).astype(bool)
        h, w = img0.shape[:2]
        center = np.array([w * 0.5, h * 0.5, 1.0], dtype=float)
        warped = A @ center
        delta = warped - center[:2]
        rotation = float(np.rad2deg(np.arctan2(A[1, 0], A[0, 0])))
        scale = float(np.sqrt(max(np.linalg.det(A[:, :2]), 1e-9)))
        confidence = float(np.count_nonzero(inliers) / max(len(good), 1))
        return MatchMotion(float(delta[0]), float(delta[1]), rotation, scale, confidence, len(good), int(np.count_nonzero(inliers)), self.backend)

    def _empty(self, num_matches: int = 0) -> MatchMotion:
        return MatchMotion(0.0, 0.0, 0.0, 1.0, 0.0, num_matches, 0, self.backend)

    def _motion_loftr(self, img0: np.ndarray, img1: np.ndarray) -> MatchMotion:
        import torch

        assert self.loftr is not None
        orig_h, orig_w = img0.shape[:2]
        w, h = self.loftr_size
        img0_r = cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA)
        img1_r = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        t0 = torch.from_numpy(img0_r.astype(np.float32) / 255.0)[None, None].to(self.device)
        t1 = torch.from_numpy(img1_r.astype(np.float32) / 255.0)[None, None].to(self.device)
        with torch.inference_mode():
            out = self.loftr({"image0": t0, "image1": t1})
        k0 = out.get("keypoints0")
        k1 = out.get("keypoints1")
        conf = out.get("confidence")
        if k0 is None or k1 is None:
            return self._empty()
        p0 = k0.detach().cpu().numpy().astype(np.float32)
        p1 = k1.detach().cpu().numpy().astype(np.float32)
        if conf is not None:
            c = conf.detach().cpu().numpy()
            keep = c >= np.percentile(c, 30) if len(c) >= 20 else np.ones(len(c), dtype=bool)
            p0 = p0[keep]
            p1 = p1[keep]
        if len(p0) < 6:
            return self._empty(num_matches=len(p0))
        sx = orig_w / float(w)
        sy = orig_h / float(h)
        p0[:, 0] *= sx
        p1[:, 0] *= sx
        p0[:, 1] *= sy
        p1[:, 1] *= sy
        A, mask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.995)
        if A is None or mask is None:
            return self._empty(num_matches=len(p0))
        inliers = mask.reshape(-1).astype(bool)
        center = np.array([orig_w * 0.5, orig_h * 0.5, 1.0], dtype=float)
        warped = A @ center
        delta = warped - center[:2]
        rotation = float(np.rad2deg(np.arctan2(A[1, 0], A[0, 0])))
        scale = float(np.sqrt(max(np.linalg.det(A[:, :2]), 1e-9)))
        confidence = float(np.count_nonzero(inliers) / max(len(p0), 1))
        return MatchMotion(float(delta[0]), float(delta[1]), rotation, scale, confidence, len(p0), int(np.count_nonzero(inliers)), self.backend)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[1] > config.MAX_PROCESS_WIDTH:
        scale = config.MAX_PROCESS_WIDTH / img.shape[1]
        img = cv2.resize(img, (config.MAX_PROCESS_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return img


def estimate_yaw_deg(data: LoadedData, matcher: RobustMatcher, frames: int, mpp: float) -> float:
    n = min(frames, len(data.timestamps) - 1)
    visual: list[np.ndarray] = []
    gps: list[np.ndarray] = []
    ref_lon, ref_lat = float(data.gt_lon[0]), float(data.gt_lat[0])
    east, north = lonlat_to_local_m(data.gt_lon[: n + 1], data.gt_lat[: n + 1], ref_lon, ref_lat)
    prev = read_gray(data.image_paths[0])
    for i in range(1, n + 1):
        curr = read_gray(data.image_paths[i])
        mot = matcher.motion(prev, curr)
        if mot.confidence > 0.2 and mot.num_inliers >= 8:
            visual.append(np.array([mot.dx * mpp, mot.dy * mpp], dtype=float))
            gps.append(np.array([east[i] - east[i - 1], north[i] - north[i - 1]], dtype=float))
        prev = curr
    if len(visual) < 3:
        logging.warning("Yaw calibration fallback to config value: not enough matching directions")
        return float(config.CAMERA_YAW_DEG)
    v = np.vstack(visual)
    g = np.vstack(gps)
    valid = (np.linalg.norm(v, axis=1) > 0.1) & (np.linalg.norm(g, axis=1) > 0.1)
    v = v[valid] / np.maximum(np.linalg.norm(v[valid], axis=1, keepdims=True), 1e-9)
    g = g[valid] / np.maximum(np.linalg.norm(g[valid], axis=1, keepdims=True), 1e-9)
    if len(v) < 3:
        return float(config.CAMERA_YAW_DEG)
    # 2D orthogonal Procrustes rotation from visual unit vectors to GPS unit vectors.
    cross = np.sum(v[:, 0] * g[:, 1] - v[:, 1] * g[:, 0])
    dot = np.sum(v[:, 0] * g[:, 0] + v[:, 1] * g[:, 1])
    return float(np.rad2deg(np.arctan2(cross, dot)))


def rotate_pixel_to_en(dx: float, dy: float, mpp: float, yaw_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(yaw_deg)
    east = np.cos(theta) * dx * mpp - np.sin(theta) * dy * mpp
    north = np.sin(theta) * dx * mpp + np.cos(theta) * dy * mpp
    return float(east), float(north)


def run_scheme_b(data: LoadedData, matcher: RobustMatcher, fx: float, output_dir: Path) -> pd.DataFrame:
    initial_alt = float(data.gt_alt[0])
    mpp = initial_alt / fx
    yaw_deg = estimate_yaw_deg(data, matcher, config.YAW_CALIBRATION_FRAMES, mpp)
    pred_e = pred_n = 0.0
    records = [make_record(data, 0, pred_e, pred_n, yaw_deg, MatchMotion(0, 0, 0, 1, 1, 0, 0, matcher.backend), "scheme_b_initial")]
    prev = read_gray(data.image_paths[0])
    for i in tqdm(range(1, len(data.timestamps)), desc="scheme B", unit="frame"):
        curr = read_gray(data.image_paths[i])
        mot = matcher.motion(prev, curr)
        east, north = rotate_pixel_to_en(mot.dx, mot.dy, mpp, yaw_deg)
        if mot.confidence < 0.15 or mot.num_inliers < 6:
            east = north = 0.0
        pred_e += east
        pred_n += north
        records.append(make_record(data, i, pred_e, pred_n, yaw_deg, mot, "scheme_b_pairwise"))
        prev = curr
    df = finalize_records(records, data, output_dir, "scheme_b")
    return df


def run_scheme_d(data: LoadedData, matcher: RobustMatcher, fx: float, output_dir: Path, keyframe_interval: int) -> pd.DataFrame:
    initial_alt = float(data.gt_alt[0])
    mpp = initial_alt / fx
    yaw_deg = estimate_yaw_deg(data, matcher, config.YAW_CALIBRATION_FRAMES, mpp)
    pred_e = pred_n = 0.0
    key_e = key_n = 0.0
    key_idx = 0
    key_img = read_gray(data.image_paths[0])
    last_img = key_img
    records = [make_record(data, 0, pred_e, pred_n, yaw_deg, MatchMotion(0, 0, 0, 1, 1, 0, 0, matcher.backend), "scheme_d_initial")]
    for i in tqdm(range(1, len(data.timestamps)), desc="scheme D", unit="frame"):
        curr = read_gray(data.image_paths[i])
        mot_map = matcher.motion(key_img, curr)
        if mot_map.confidence >= 0.2 and mot_map.num_inliers >= 8:
            east, north = rotate_pixel_to_en(mot_map.dx, mot_map.dy, mpp, yaw_deg)
            pred_e = key_e + east
            pred_n = key_n + north
            mot = mot_map
            method = "scheme_d_local_map"
        else:
            mot = matcher.motion(last_img, curr)
            east, north = rotate_pixel_to_en(mot.dx, mot.dy, mpp, yaw_deg)
            if mot.confidence >= 0.15 and mot.num_inliers >= 6:
                pred_e += east
                pred_n += north
            method = "scheme_d_pairwise_fallback"
        if i - key_idx >= keyframe_interval or mot.confidence < 0.15:
            key_idx = i
            key_img = curr
            key_e, key_n = pred_e, pred_n
        records.append(make_record(data, i, pred_e, pred_n, yaw_deg, mot, method))
        last_img = curr
    df = finalize_records(records, data, output_dir, "scheme_d")
    return df


def make_record(data: LoadedData, i: int, pred_e: float, pred_n: float, yaw_deg: float, mot: MatchMotion, method: str) -> dict[str, object]:
    lon, lat = local_m_to_lonlat(pred_e, pred_n, float(data.gt_lon[0]), float(data.gt_lat[0]))
    return {
        "timestamp": data.timestamps[i],
        "image_path": str(data.image_paths[i]),
        "pred_lon": lon,
        "pred_lat": lat,
        "pred_alt": float(data.gt_alt[0]),
        "gt_lon": float(data.gt_lon[i]),
        "gt_lat": float(data.gt_lat[i]),
        "gt_alt": float(data.gt_alt[i]),
        "pred_east_m": pred_e,
        "pred_north_m": pred_n,
        "pixel_dx": mot.dx,
        "pixel_dy": mot.dy,
        "rotation_deg": mot.rotation_deg,
        "affine_scale": mot.scale,
        "match_confidence": mot.confidence,
        "num_matches": mot.num_matches,
        "num_inliers": mot.num_inliers,
        "matcher_backend": mot.backend,
        "yaw_deg": yaw_deg,
        "method": method,
        "flight_phase": "NA",
        "is_keyframe_reset": method == "scheme_d_local_map",
    }


def finalize_records(records: list[dict[str, object]], data: LoadedData, output_dir: Path, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["horizontal_error"] = [
        haversine_m(plon, plat, glon, glat)
        for plon, plat, glon, glat in zip(df.pred_lon, df.pred_lat, df.gt_lon, df.gt_lat)
    ]
    df["altitude_error"] = np.abs(df.pred_alt.astype(float) - df.gt_alt.astype(float))
    df.to_csv(output_dir / f"{prefix}_results.csv", index=False, encoding="utf-8-sig")
    plot_trajectory(df, output_dir / f"{prefix}_trajectory.png", prefix)
    plot_error(df, output_dir / f"{prefix}_error.png", prefix)
    metrics = {
        "horizontal_mae": float(df.horizontal_error.mean()),
        "horizontal_rmse": float(np.sqrt(np.mean(df.horizontal_error.to_numpy(float) ** 2))),
        "horizontal_max": float(df.horizontal_error.max()),
        "final_horizontal_error": float(df.horizontal_error.iloc[-1]),
        "altitude_mae": float(df.altitude_error.mean()),
        "matcher_backend": str(df.matcher_backend.iloc[-1]),
    }
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


def plot_trajectory(df: pd.DataFrame, path: Path, title: str) -> None:
    ref_lon, ref_lat = float(df.gt_lon.iloc[0]), float(df.gt_lat.iloc[0])
    gt_e, gt_n = lonlat_to_local_m(df.gt_lon.to_numpy(float), df.gt_lat.to_numpy(float), ref_lon, ref_lat)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_e, gt_n, "b-", label="GT", linewidth=1.2)
    ax.plot(df.pred_east_m, df.pred_north_m, "r--", label="Pred", linewidth=1.1)
    ax.plot(gt_e[0], gt_n[0], "g*", markersize=10, label="Start")
    ax.set_title(title)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)


def plot_error(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.horizontal_error.to_numpy(float), "r-", linewidth=1.0, label="Horizontal")
    ax.plot(df.altitude_error.to_numpy(float), "b-", linewidth=1.0, label="Altitude")
    ax.set_title(f"{title} error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)


def load_fx(path: Path) -> float:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if "fx" in data:
            return float(data["fx"])
    return 1357.2385463590624


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matching-based VO experiments for scheme B and scheme D.")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("output_matching_experiments"))
    parser.add_argument("--keyframe-interval", type=int, default=30)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--matcher", choices=["auto", "sift", "orb", "loftr"], default="orb")
    parser.add_argument("--loftr-width", type=int, default=512)
    parser.add_argument("--loftr-height", type=int, default=384)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = DataLoader().load()
    if args.max_frames:
        data = DataLoader().slice_from(data, 0, min(len(data.timestamps), args.max_frames))
    matcher = RobustMatcher(args.matcher, loftr_size=(args.loftr_width, args.loftr_height))
    fx = args.fx if args.fx is not None else load_fx(config.OUTPUT_DIR / "calibration.json")
    logging.info("Using matcher=%s fx=%.2f frames=%d", matcher.backend, fx, len(data.timestamps))
    df_b = run_scheme_b(data, matcher, fx, args.output_dir)
    df_d = run_scheme_d(data, matcher, fx, args.output_dir, args.keyframe_interval)

    summary = {
        "scheme_b": json.loads((args.output_dir / "scheme_b_metrics.json").read_text(encoding="utf-8")),
        "scheme_d": json.loads((args.output_dir / "scheme_d_metrics.json").read_text(encoding="utf-8")),
        "frames": len(data.timestamps),
        "fx": fx,
        "note": "torch/kornia unavailable; SIFT/ORB matcher backend used as a pluggable stand-in for deep matching.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
