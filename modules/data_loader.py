"""Timestamp-exact data loading for infrared frames and GPS truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class LoadedData:
    timestamps: list[str]
    image_paths: list[Path]
    gt_lon: np.ndarray
    gt_lat: np.ndarray
    gt_alt: np.ndarray
    delta_t: np.ndarray
    start_index: int = 0


class DataLoader:
    """Load image paths and ground truth by exact timestamp string matching."""

    def __init__(self, image_dir: Path = config.IMAGE_DIR, csv_path: Path = config.CSV_PATH) -> None:
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)

    def load(self) -> LoadedData:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = self._read_csv()
        ts_col = self._find_column(df.columns, config.TIMESTAMP_COLUMNS, "timestamp")
        lon_col = self._find_column(df.columns, config.LON_COLUMNS, "longitude")
        lat_col = self._find_column(df.columns, config.LAT_COLUMNS, "latitude")
        alt_col = self._find_column(df.columns, config.ALT_COLUMNS, "altitude")

        truth = df[[ts_col, lon_col, lat_col, alt_col]].copy()
        truth[ts_col] = truth[ts_col].astype(str).str.strip()
        truth = truth.drop_duplicates(ts_col, keep="first").set_index(ts_col)

        image_map = self._collect_images()
        matched = sorted(set(image_map).intersection(truth.index), key=self._timestamp_key)
        if len(matched) < 2:
            raise ValueError("Need at least two exact timestamp matches between images and CSV")

        gt = truth.loc[matched]
        timestamps = list(matched)
        image_paths = [image_map[t] for t in timestamps]
        delta_t = self._delta_t_seconds(timestamps)
        return LoadedData(
            timestamps=timestamps,
            image_paths=image_paths,
            gt_lon=gt[lon_col].astype(float).to_numpy(),
            gt_lat=gt[lat_col].astype(float).to_numpy(),
            gt_alt=gt[alt_col].astype(float).to_numpy(),
            delta_t=delta_t,
        )

    def slice_from(self, data: LoadedData, start_index: int, end_index: int | None = None) -> LoadedData:
        end = len(data.timestamps) if end_index is None else end_index
        return LoadedData(
            timestamps=data.timestamps[start_index:end],
            image_paths=data.image_paths[start_index:end],
            gt_lon=data.gt_lon[start_index:end],
            gt_lat=data.gt_lat[start_index:end],
            gt_alt=data.gt_alt[start_index:end],
            delta_t=data.delta_t[start_index:end],
            start_index=start_index,
        )

    def _read_csv(self) -> pd.DataFrame:
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(self.csv_path, encoding=encoding)
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
        raise UnicodeDecodeError("csv", b"", 0, 1, "; ".join(errors))

    def _collect_images(self) -> dict[str, Path]:
        images: dict[str, Path] = {}
        for path in self.image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS:
                images[path.stem.strip()] = path
        if not images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        return images

    @staticmethod
    def _find_column(columns: Iterable[str], aliases: Iterable[str], logical_name: str) -> str:
        normalized = {str(col).strip().lower(): col for col in columns}
        for alias in aliases:
            key = alias.strip().lower()
            if key in normalized:
                return normalized[key]
        if len(list(columns)) >= 4:
            fallback = list(columns)[{"timestamp": 0, "longitude": 1, "latitude": 2, "altitude": 3}[logical_name]]
            return fallback
        raise ValueError(f"Cannot find {logical_name} column in CSV")

    @staticmethod
    def _timestamp_key(timestamp: str) -> tuple[int, str]:
        try:
            return int(timestamp), timestamp
        except ValueError:
            return 0, timestamp

    @staticmethod
    def _delta_t_seconds(timestamps: list[str]) -> np.ndarray:
        values = np.array([float(t) for t in timestamps], dtype=float)
        diffs = np.diff(values, prepend=values[0]) / 1000.0
        if len(diffs) > 1:
            diffs[0] = diffs[1]
        diffs[diffs <= 0] = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.2
        return diffs
