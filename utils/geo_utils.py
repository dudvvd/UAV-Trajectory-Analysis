"""Geographic conversion helpers for local UAV motion."""

from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

import numpy as np

EARTH_RADIUS_M = 6371008.8
METERS_PER_DEG_LAT = 111320.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in meters."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(radians, (lon1, lat1, lon2, lat2))
    d_lon = lon2_r - lon1_r
    d_lat = lat2_r - lat1_r
    a = sin(d_lat / 2.0) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(max(0.0, 1.0 - a)))


def lonlat_to_local_m(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    ref_lon: float,
    ref_lat: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert lon/lat to local east/north meters around a reference point."""
    east = (np.asarray(lon) - ref_lon) * METERS_PER_DEG_LAT * cos(radians(ref_lat))
    north = (np.asarray(lat) - ref_lat) * METERS_PER_DEG_LAT
    return east, north


def local_m_to_lonlat(
    east: float,
    north: float,
    ref_lon: float,
    ref_lat: float,
) -> tuple[float, float]:
    """Convert local east/north meters back to lon/lat."""
    lat = ref_lat + north / METERS_PER_DEG_LAT
    lon = ref_lon + east / (METERS_PER_DEG_LAT * max(cos(radians(ref_lat)), 1e-8))
    return lon, lat


def pixel_to_geo_delta(
    pixel_dx: float,
    pixel_dy: float,
    altitude_m: float,
    fx: float,
    latitude_deg: float,
    yaw_deg: float,
) -> tuple[float, float, float, float, float]:
    """Convert pixel displacement to lon/lat deltas with yaw correction."""
    mpp = altitude_m / max(float(fx), 1e-6)
    theta = radians(yaw_deg)
    east = cos(theta) * pixel_dx * mpp - sin(theta) * pixel_dy * mpp
    north = sin(theta) * pixel_dx * mpp + cos(theta) * pixel_dy * mpp
    delta_lat = north / METERS_PER_DEG_LAT
    delta_lon = east / (METERS_PER_DEG_LAT * max(cos(radians(latitude_deg)), 1e-8))
    return delta_lon, delta_lat, east, north, mpp
