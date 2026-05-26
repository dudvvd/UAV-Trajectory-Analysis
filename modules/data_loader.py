"""Timestamp-exact data loading for UAV image sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
    timestamps: list[str]
    image_paths: list[Path]
    gt_lon: list[float]
    gt_lat: list[float]
    gt_alt: list[float]
    delta_t: list[float]


class DataLoader:
    """Load image paths and validation GPS rows by exact timestamp string."""

    def __init__(self, image_dir: Path, csv_path: Path) -> None:
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)

    def load(self) -> LoadedData:
        gps = self._read_csv()
        gps_by_timestamp = gps.set_index("timestamp", drop=False)
        rows: list[dict[str, object]] = []
        skipped = 0

        for image_path in sorted(self.image_dir.iterdir(), key=lambda p: p.stem):
            if image_path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            timestamp = image_path.stem.strip()
            if timestamp not in gps_by_timestamp.index:
                skipped += 1
                LOGGER.warning("Timestamp %s has image but no CSV row; skipped", timestamp)
                continue
            gps_row = gps_by_timestamp.loc[timestamp]
            if isinstance(gps_row, pd.DataFrame):
                gps_row = gps_row.iloc[0]
            rows.append(
                {
                    "timestamp": timestamp,
                    "image_path": image_path,
                    "lon": float(gps_row["longitude"]),
                    "lat": float(gps_row["latitude"]),
                    "alt": float(gps_row["altitude"]),
                }
            )

        if len(rows) < 2:
            raise ValueError("Need at least two timestamp-matched frames.")

        rows.sort(key=lambda r: int(str(r["timestamp"])))
        ts = [str(r["timestamp"]) for r in rows]
        ts_float = np.asarray(ts, dtype=np.float64)
        delta_t = [0.0] + ((ts_float[1:] - ts_float[:-1]) / 1000.0).astype(float).tolist()
        if skipped:
            LOGGER.info("Skipped %d images because exact CSV timestamp match was missing", skipped)

        return LoadedData(
            timestamps=ts,
            image_paths=[Path(r["image_path"]) for r in rows],
            gt_lon=[float(r["lon"]) for r in rows],
            gt_lat=[float(r["lat"]) for r in rows],
            gt_alt=[float(r["alt"]) for r in rows],
            delta_t=delta_t,
        )

    def _read_csv(self) -> pd.DataFrame:
        encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                df = pd.read_csv(self.csv_path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"Cannot decode CSV {self.csv_path}: {last_error}")

        rename = {}
        for column in df.columns:
            clean = str(column).strip()
            if clean in ("时间戳", "timestamp", "time"):
                rename[column] = "timestamp"
            elif clean in ("经度", "longitude", "lon"):
                rename[column] = "longitude"
            elif clean in ("纬度", "latitude", "lat"):
                rename[column] = "latitude"
            elif clean in ("高度", "altitude", "alt"):
                rename[column] = "altitude"
        df = df.rename(columns=rename)
        required = {"timestamp", "longitude", "latitude", "altitude"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        df["timestamp"] = df["timestamp"].astype(str).str.strip()
        for col in ("longitude", "latitude", "altitude"):
            df[col] = pd.to_numeric(df[col], errors="raise")
        return df.drop_duplicates("timestamp").reset_index(drop=True)
