"""Geographic helpers used by localization and evaluation."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import numpy as np

EARTH_RADIUS_M = 6_378_137.0
METERS_PER_DEG_LAT = 111_320.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle horizontal distance in meters."""

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * asin(sqrt(a))


def pixel_to_geo_delta(
    pixel_dx: float,
    pixel_dy: float,
    altitude_m: float,
    fx: float,
    current_latitude_deg: float,
) -> tuple[float, float]:
    """Convert image displacement to longitude and latitude increments."""

    meters_per_pixel = altitude_m / max(float(fx), 1e-6)
    delta_north = -pixel_dy * meters_per_pixel
    delta_east = pixel_dx * meters_per_pixel
    lat_rad = radians(current_latitude_deg)
    delta_lat = delta_north / METERS_PER_DEG_LAT
    delta_lon = delta_east / (METERS_PER_DEG_LAT * max(cos(lat_rad), 1e-6))
    return float(delta_lon), float(delta_lat)


def lonlat_to_local_m(
    lon: np.ndarray,
    lat: np.ndarray,
    ref_lon: float,
    ref_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat arrays to a local east/north plane for plotting."""

    east = np.deg2rad(np.asarray(lon, dtype=float) - ref_lon) * EARTH_RADIUS_M * cos(radians(ref_lat))
    north = np.deg2rad(np.asarray(lat, dtype=float) - ref_lat) * EARTH_RADIUS_M
    return east, north
