"""Data loading for the UAV image and location dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    image_dir: Path
    csv_path: Path


def read_location_csv(csv_path: Path) -> pd.DataFrame:
    """Read location CSV and normalize column names/types."""

    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise UnicodeDecodeError(
            last_error.encoding if last_error else "unknown",
            last_error.object if last_error else b"",
            last_error.start if last_error else 0,
            last_error.end if last_error else 0,
            f"无法用常见编码读取 CSV: {encodings}",
        )
    rename_map = {
        "时间戳": "timestamp",
        "经度": "longitude",
        "纬度": "latitude",
        "高度": "altitude",
    }
    df = df.rename(columns=rename_map)
    required = {"timestamp", "longitude", "latitude", "altitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {sorted(missing)}")

    df["timestamp"] = df["timestamp"].astype(str).str.strip().astype("int64")
    for col in ["longitude", "latitude", "altitude"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def collect_image_index(image_dir: Path) -> pd.DataFrame:
    """Build a timestamp-to-image-path table from JPG files."""

    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        try:
            timestamp = int(image_path.stem)
        except ValueError:
            continue
        rows.append({"timestamp": timestamp, "image_path": str(image_path)})
    if not rows:
        raise FileNotFoundError(f"未在 {image_dir} 找到 jpg 图片")
    return pd.DataFrame(rows)


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Join image paths with ground-truth locations by timestamp."""

    locations = read_location_csv(config.csv_path)
    images = collect_image_index(config.image_dir)
    merged = images.merge(locations, on="timestamp", how="inner").sort_values("timestamp")
    if len(merged) < 2:
        raise ValueError("匹配到的图片/定位记录不足 2 条，无法估计相邻运动")
    return merged.reset_index(drop=True)
