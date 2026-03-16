"""
GOES-16 satellite cloud analysis for the shadow-velocity thesis.

Fetches Cloud Optical Depth (COD) from GOES-16 ABI L2 products on AWS,
extracts opacity at Texas solar farm locations, and computes cloud
velocity (rate of change of opacity between consecutive frames).

GOES-16 ABI COD (Cloud Optical Depth):
  - Product: ABI-L2-CODC (CONUS domain)
  - Resolution: ~2km at nadir
  - Cadence: every 5 minutes
  - Grid: 1500 x 2500 pixels (geostationary fixed grid)
  - Source: s3://noaa-goes16/ (public, no auth)

Solar farm clusters in ERCOT (West Texas):
  - Permian Basin area: 31.5N, 102.5W (HB_WEST settlement)
  - South Texas: 27.5N, 97.5W
  - Central Texas: 31.0N, 100.0W
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "goes16"

# Texas solar farm cluster coordinates (lat, lon)
SOLAR_CLUSTERS = {
    "permian_basin": (31.5, -102.5),    # West TX, maps to HB_WEST
    "south_texas": (27.5, -97.5),       # South TX
    "central_texas": (31.0, -100.0),    # Central TX
}

# Radius in pixels around each cluster center to average COD
CLUSTER_RADIUS_PX = 15  # ~30km radius at 2km resolution


def _latlon_to_goes_xy(lat: float, lon: float, proj_attrs: dict) -> tuple[float, float]:
    """Convert lat/lon to GOES-16 fixed grid x,y coordinates (radians)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0 = np.radians(proj_attrs["longitude_of_projection_origin"])

    req = 6378137.0
    rpol = 6356752.31414
    e2 = 1 - (rpol / req) ** 2

    phi_c = np.arctan((rpol / req) ** 2 * np.tan(lat_rad))
    rc = rpol / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)

    sx = proj_attrs["perspective_point_height"] - rc * np.cos(phi_c) * np.cos(lon_rad - lon0)
    sy = -rc * np.cos(phi_c) * np.sin(lon_rad - lon0)
    sz = rc * np.sin(phi_c)

    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    y = np.arctan(sz / sx)
    return float(x), float(y)


def list_cod_files(
    date: str | datetime,
    hours: list[int] | None = None,
) -> list[str]:
    """List GOES-16 COD CONUS files for a given date.

    Returns S3 paths sorted by scan start time.
    """
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)

    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    doy = date.timetuple().tm_yday
    year = date.year

    if hours is None:
        hours = list(range(12, 22))  # UTC 12-21 = ~7am-4pm CT (daylight)

    all_files = []
    for hour in hours:
        prefix = f"noaa-goes16/ABI-L2-CODC/{year}/{doy:03d}/{hour:02d}"
        try:
            files = fs.ls(prefix)
            all_files.extend([f for f in files if f.endswith(".nc")])
        except FileNotFoundError:
            continue

    return sorted(all_files)


def fetch_cod_frame(s3_path: str) -> dict:
    """Download and parse a single GOES-16 COD frame.

    Returns dict with:
      - cod: 2D numpy array of cloud optical depth
      - x: 1D array of x coordinates (radians)
      - y: 1D array of y coordinates (radians)
      - proj_attrs: projection parameters
      - scan_start: datetime of scan start
    """
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download to temp file (streaming from S3 causes file-closed errors)
    filename = s3_path.split("/")[-1]
    local_path = CACHE_DIR / filename
    if not local_path.exists():
        fs.get(s3_path, str(local_path))

    ds = xr.open_dataset(str(local_path), engine="h5netcdf")

    result = {
        "cod": ds["COD"].values.astype(float),
        "x": ds["x"].values,
        "y": ds["y"].values,
        "proj_attrs": dict(ds["goes_imager_projection"].attrs),
    }

    # Parse scan start time from filename
    # Format: OR_ABI-L2-CODC-M6_G16_s20250971501174_e...
    parts = filename.split("_")
    for p in parts:
        if p.startswith("s"):
            year = int(p[1:5])
            doy = int(p[5:8])
            hour = int(p[8:10])
            minute = int(p[10:12])
            result["scan_start"] = datetime(year, 1, 1) + timedelta(
                days=doy - 1, hours=hour, minutes=minute
            )
            break

    ds.close()
    return result


def extract_cluster_cod(
    frame: dict,
    clusters: dict[str, tuple[float, float]] | None = None,
    radius_px: int = CLUSTER_RADIUS_PX,
) -> dict[str, float]:
    """Extract mean COD at each solar farm cluster location.

    Returns {cluster_name: mean_cod} for each cluster.
    NaN if the cluster is outside the image or all pixels are NaN.
    """
    if clusters is None:
        clusters = SOLAR_CLUSTERS

    cod = frame["cod"]
    x_arr = frame["x"]
    y_arr = frame["y"]
    proj = frame["proj_attrs"]

    results = {}
    for name, (lat, lon) in clusters.items():
        x_goes, y_goes = _latlon_to_goes_xy(lat, lon, proj)

        xi = np.argmin(np.abs(x_arr - x_goes))
        yi = np.argmin(np.abs(y_arr - y_goes))

        # Check bounds
        if (yi - radius_px < 0 or yi + radius_px >= cod.shape[0] or
                xi - radius_px < 0 or xi + radius_px >= cod.shape[1]):
            results[name] = np.nan
            continue

        # Extract patch and compute mean (ignoring NaN)
        patch = cod[yi - radius_px:yi + radius_px + 1,
                    xi - radius_px:xi + radius_px + 1]
        valid = patch[~np.isnan(patch)]
        results[name] = float(np.mean(valid)) if len(valid) > 0 else np.nan

    return results


def build_cod_timeseries(
    date: str,
    clusters: dict[str, tuple[float, float]] | None = None,
    hours: list[int] | None = None,
) -> pd.DataFrame:
    """Build a time series of COD at solar farm clusters for one day.

    Returns DataFrame indexed by scan_start with columns per cluster.
    """
    files = list_cod_files(date, hours)
    if not files:
        raise ValueError(f"No COD files found for {date}")

    rows = []
    for s3_path in files:
        try:
            frame = fetch_cod_frame(s3_path)
            cluster_cod = extract_cluster_cod(frame, clusters)
            cluster_cod["scan_start"] = frame["scan_start"]
            rows.append(cluster_cod)
        except Exception as e:
            print(f"  Warning: failed to process {s3_path.split('/')[-1]}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("scan_start").sort_index()
    return df


def compute_cloud_velocity(
    cod_ts: pd.DataFrame,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Compute cloud velocity (rate of change of COD) at each cluster.

    Returns DataFrame with:
      - {cluster}_cod: raw cloud optical depth
      - {cluster}_ramp: COD change per interval (dCOD/dt)
      - {cluster}_abs_ramp: absolute ramp rate
      - {cluster}_velocity: ramp rate normalized by current COD level

    High |ramp| = fast-moving cloud front (the "shadow velocity").
    """
    result = cod_ts.copy()

    for col in cod_ts.columns:
        result[f"{col}_ramp"] = cod_ts[col].diff()
        result[f"{col}_abs_ramp"] = cod_ts[col].diff().abs()
        # Velocity = ramp normalized by level (proportional change)
        level = cod_ts[col].replace(0, np.nan)
        result[f"{col}_velocity"] = cod_ts[col].diff() / level

    return result


def detect_fast_shadow_events(
    velocity_df: pd.DataFrame,
    ramp_threshold: float = 5.0,
    cluster: str = "permian_basin",
) -> pd.DataFrame:
    """Detect fast-shadow events: moments when COD ramps sharply upward.

    A fast shadow = COD increasing rapidly (cloud arriving) at a solar
    farm cluster. This predicts an imminent solar generation drop.

    Parameters
    ----------
    ramp_threshold : float
        Minimum COD ramp per 5-min interval to qualify as a fast shadow.
        COD of ~5 reduces irradiance by ~50%. A ramp of 5 in 5 minutes
        is a significant cloud front.

    Returns DataFrame of detected events with columns:
      time, cluster, cod_before, cod_after, ramp, predicted_gen_drop
    """
    ramp_col = f"{cluster}_ramp"
    cod_col = cluster

    if ramp_col not in velocity_df.columns:
        return pd.DataFrame()

    events = velocity_df[velocity_df[ramp_col] > ramp_threshold].copy()
    if events.empty:
        return pd.DataFrame()

    result = pd.DataFrame({
        "time": events.index,
        "cluster": cluster,
        "cod_before": events[cod_col] - events[ramp_col],
        "cod_after": events[cod_col],
        "ramp": events[ramp_col],
    })

    # Estimate generation impact: COD increase of X reduces irradiance
    # approximately by 1 - exp(-0.1 * X) (Beer-Lambert approximation)
    result["predicted_irradiance_drop_pct"] = (
        1 - np.exp(-0.1 * result["ramp"])
    ) * 100

    return result.reset_index(drop=True)
