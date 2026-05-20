"""Geographic coordinate helpers.

The estimator works in a local East-North-Up (ENU) coordinate frame because
image motion is naturally relative. Results are converted back to lon/lat/alt
for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class GeoReference:
    """Reference point for local tangent-plane conversion."""

    longitude: float
    latitude: float
    altitude: float

    @property
    def latitude_rad(self) -> float:
        return radians(self.latitude)


def geodetic_to_enu(
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude: np.ndarray,
    reference: GeoReference,
) -> np.ndarray:
    """Convert lon/lat/alt arrays to a local ENU array in meters."""

    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    alt = np.asarray(altitude, dtype=float)
    east = np.deg2rad(lon - reference.longitude) * EARTH_RADIUS_M * cos(reference.latitude_rad)
    north = np.deg2rad(lat - reference.latitude) * EARTH_RADIUS_M
    up = alt - reference.altitude
    return np.column_stack([east, north, up])


def enu_to_geodetic(enu: np.ndarray, reference: GeoReference) -> np.ndarray:
    """Convert local ENU coordinates in meters to lon/lat/alt."""

    arr = np.asarray(enu, dtype=float)
    longitude = reference.longitude + np.degrees(arr[:, 0] / (EARTH_RADIUS_M * cos(reference.latitude_rad)))
    latitude = reference.latitude + np.degrees(arr[:, 1] / EARTH_RADIUS_M)
    altitude = reference.altitude + arr[:, 2]
    return np.column_stack([longitude, latitude, altitude])


def horizontal_errors_m(predicted_enu: np.ndarray, truth_enu: np.ndarray) -> np.ndarray:
    """Return horizontal Euclidean position errors in meters."""

    delta = np.asarray(predicted_enu)[:, :2] - np.asarray(truth_enu)[:, :2]
    return np.linalg.norm(delta, axis=1)
