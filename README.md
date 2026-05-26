# Pure Vision UAV Localization

This project estimates UAV longitude, latitude, and altitude from an infrared
image sequence using only the first frame pose as an initial condition. Ground
truth GPS rows in the CSV are kept out of the prediction path and are used only
for final error evaluation and visualization.

## Features

- Exact timestamp matching between image filenames and CSV rows.
- Automatic camera intrinsic estimation with cached `calibration.json`.
- Long-window Shi-Tomasi + Lucas-Kanade feature tracking.
- Forward-backward optical flow validation and RANSAC outlier rejection.
- ORB fallback for large motion or unstable optical flow.
- Automatic camera yaw calibration from early GPS direction and visual direction.
- Four-level scale and altitude estimation with conservative fallback.
- Selective EKF updates that skip unreliable position or altitude observations.
- Blur frame detection with prediction-only EKF handling.
- CSV and plot outputs for trajectory, error over time, and altitude.

## Project Layout

```text
.
+-- main.py
+-- config.py
+-- modules/
|   +-- data_loader.py
|   +-- camera_calibrator.py
|   +-- feature_tracker.py
|   +-- scale_estimator.py
|   +-- yaw_calibrator.py
|   +-- scale_level_validator.py
|   +-- ekf_localizer.py
|   +-- evaluator.py
+-- utils/
|   +-- geo_utils.py
|   +-- visualization.py
+-- data/
|   +-- IR-image/
|   +-- position CSV
+-- output/
```

## Input Format

Images must be named by their 13-digit millisecond Unix timestamp:

```text
data/IR-image/1734763664200.jpg
data/IR-image/1734763664400.jpg
```

The CSV must contain timestamp, longitude, latitude, and altitude columns:

```csv
timestamp,longitude,latitude,altitude
1734763664200,116.168975,38.8121207,483.54
1734763664400,116.1690312,38.8121209,484.06
```

The image timestamp is matched exactly against the CSV timestamp string. Images
without a matching CSV row are logged and skipped.

## Installation

The repository has been tested with the existing `uav` conda environment:

```powershell
conda activate uav
python -m pip install -r requirements.txt
```

Or run commands through conda without activating:

```powershell
conda run -n uav python main.py --max-frames 100
```

## Quick Start

Run a short smoke test:

```powershell
conda run -n uav python main.py --max-frames 100 --log-level INFO
```

Run the full sequence:

```powershell
conda run -n uav python main.py
```

Force camera recalibration instead of using cached intrinsics:

```powershell
conda run -n uav python main.py --recalibrate
```

Use an explicit first frame and initial pose:

```powershell
conda run -n uav python main.py `
  --first-image-path data/IR-image/1734763664200.jpg `
  --initial-longitude 116.168975 `
  --initial-latitude 38.8121207 `
  --initial-altitude 483.54
```

If no initial pose is provided, the program uses the first selected CSV row only
as the first-frame initial condition. Later CSV rows are not used for prediction.

## Main Options

| Option | Description |
| --- | --- |
| `--image-dir` | Infrared image directory. Default: `data/IR-image`. |
| `--csv` | Validation GPS CSV. Default path is defined in `config.py`. |
| `--output-dir` | Output directory. Default: `output`. |
| `--first-image-path` | Optional first frame path. Timestamp selects the start frame. |
| `--initial-longitude` | Initial longitude in degrees. |
| `--initial-latitude` | Initial latitude in degrees. |
| `--initial-altitude` | Initial altitude in meters. |
| `--recalibrate` | Ignore cached calibration and estimate intrinsics again. |
| `--max-frames` | Optional frame limit for development or testing. |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

All algorithm thresholds and hyperparameters are centralized in `config.py`.

## Outputs

The pipeline writes:

```text
output/
+-- calibration.json
+-- trajectory_comparison.png
+-- error_over_time.png
+-- altitude_comparison.png
+-- results.csv
```

`results.csv` contains predicted pose, ground truth pose, per-frame errors, and
tracking diagnostics such as pixel displacement, inlier ratio, scale confidence,
homography inliers, keyframe resets, blur score, and processing method.

Additional diagnostics are included for the repaired scale, yaw, and EKF paths:

- `scale_method`: active scale level, where 1 is divergence, 2 is affine scale, 3 is homography, and 4 is EKF altitude velocity fallback.
- `divergence_ratio`: raw feature-point divergence ratio.
- `affine_scale`: raw affine scale factor.
- `raw_alt_estimate`: altitude estimate before EKF fusion.
- `ekf_alt`: EKF-smoothed altitude.
- `yaw_deg`: calibrated yaw correction used for the run.
- `ekf_update_mode`: 1 updates position and altitude, 2 updates position only, 3 prediction only.

## Pipeline

1. `DataLoader` loads timestamp-sorted image paths and validation GPS rows.
2. `CameraCalibrator` estimates or loads camera intrinsics.
3. `YawCalibrator` aligns early visual directions to early GPS directions.
4. `ScaleLevelValidator` checks which scale levels are useful on the sequence.
5. `FeatureTracker` tracks image features with LK optical flow and ORB fallback.
6. `ScaleEstimator` estimates altitude with divergence, affine, homography, and EKF fallback levels.
7. `EKFLocalizer` selectively fuses visual deltas with a constant-velocity motion model.
8. `Evaluator` computes errors and generates CSV/plots.

## Important Notes

This is a pure monocular visual odometry system. Without IMU, camera extrinsics,
known yaw, map constraints, or loop closure, long-range drift is expected. The
implementation reduces local noise and rejects poor frames, but it cannot fully
remove accumulated drift on non-overlapping flight paths.

For production-grade accuracy, provide camera intrinsics/extrinsics, UAV attitude
or gimbal angles, and a reliable scale source such as IMU/GNSS velocity, laser
altimeter, DEM, or map constraints.
