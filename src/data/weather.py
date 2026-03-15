"""
Meteorological data fetchers for weather-based trading signals.

Data sources (all free, public):
  1. ENSO (El Nino Southern Oscillation):
     - Oceanic Nino Index (ONI) from NOAA CPC
     - Monthly, 3-month running mean of SST anomalies in Nino 3.4 region
     - https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt

  2. Heating/Cooling Degree Days (HDD/CDD):
     - Population-weighted US degree days from NOAA CPC
     - Weekly, measures deviation from 65F baseline
     - https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/

  3. NYC weather (for cloud cover / sunshine signal):
     - NOAA ISD (Integrated Surface Database) for LaGuardia Airport
     - Daily cloud cover, temperature, precipitation
     - Via Meteostat Python library or direct NOAA ISD-Lite

Publication lag: all NOAA weather data is published with 1-7 day lag.
We apply conservative offsets. Temperature/cloud data is typically
available next-day; ENSO ONI is published monthly with ~2 week lag.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
HEADERS = {"User-Agent": "alt-strats-research research@example.com"}


# ---------------------------------------------------------------------------
# ENSO / ONI
# ---------------------------------------------------------------------------

ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
# Backup: the ERSSTv5 Nino 3.4 index
ONI_BACKUP_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

ENSO_PUB_LAG_DAYS = 14  # ONI published ~2 weeks after month-end


def fetch_oni(use_cache: bool = True) -> pd.DataFrame:
    """Fetch the Oceanic Nino Index (ONI) from NOAA CPC.

    ONI is a 3-month running mean of SST anomalies in the Nino 3.4
    region (5N-5S, 170W-120W). Values:
      > +0.5 for 5 consecutive months = El Nino
      < -0.5 for 5 consecutive months = La Nina

    Returns DataFrame with columns: date, oni, phase
    where phase is 'el_nino', 'la_nina', or 'neutral'.
    Index is shifted by ENSO_PUB_LAG_DAYS for point-in-time correctness.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "oni.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # Try the ONI ascii table
    df = None
    for url in [ONI_BACKUP_URL, ONI_URL]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            df = _parse_oni_ascii(resp.text)
            if df is not None and len(df) > 100:
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise ValueError("Could not fetch ONI data from any source")

    # Apply publication lag
    df.index = df.index + pd.Timedelta(days=ENSO_PUB_LAG_DAYS)

    if use_cache:
        df.to_parquet(cache_path)

    return df


def _parse_oni_ascii(text: str) -> pd.DataFrame:
    """Parse the NOAA ONI ASCII table.

    Handles the standard CPC format:
      SEAS  YR   TOTAL   ANOM
      DJF 1950  24.72  -1.53
      JFM 1950  25.17  -1.34
    """
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        season = parts[0].upper()
        if season not in season_to_month:
            continue
        try:
            year = int(parts[1])
            anom = float(parts[3])  # ANOM column is the ONI value
        except (ValueError, IndexError):
            continue
        if abs(anom) > 10:
            continue

        month = season_to_month[season]
        date = pd.Timestamp(year=year, month=month, day=15)
        phase = "el_nino" if anom > 0.5 else "la_nina" if anom < -0.5 else "neutral"
        rows.append({"date": date, "oni": anom, "phase": phase})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Heating / Cooling Degree Days
# ---------------------------------------------------------------------------

HDD_URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/Population." 
# Full URLs: HDD_URL + "Heating.txt" or .Cooling.txt  -- but these are tricky.
# Easier: use the monthly CPC data
HDD_MONTHLY_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/HEATING/fweek.dat"
CDD_MONTHLY_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/COOLING/fweek.dat"

DEGDAY_PUB_LAG_DAYS = 3  # Degree day data available within a few days


def fetch_degree_days(use_cache: bool = True) -> pd.DataFrame:
    """Fetch US population-weighted HDD and CDD from NOAA CPC.

    Returns DataFrame indexed by date (pub-lag adjusted) with columns:
      hdd: heating degree days (higher = colder)
      cdd: cooling degree days (higher = hotter)
      hdd_anomaly: deviation from 30-year normal
      cdd_anomaly: deviation from 30-year normal

    Falls back to computing degree days from temperature data if the
    direct CPC files aren't parseable.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "degree_days.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # Strategy: fetch daily US temperature from NOAA Climate-at-a-Glance
    # and compute HDD/CDD ourselves (more reliable than parsing CPC FTP)
    df = _fetch_temp_and_compute_degree_days()

    if df is not None and not df.empty:
        # Apply publication lag
        df.index = df.index + pd.Timedelta(days=DEGDAY_PUB_LAG_DAYS)
        if use_cache:
            df.to_parquet(cache_path)
        return df

    raise ValueError("Could not fetch degree day data")


def _fetch_temp_and_compute_degree_days() -> pd.DataFrame:
    """Fetch US average temperature and compute HDD/CDD.

    Uses NOAA Climate-at-a-Glance national monthly temperature data.
    HDD = max(0, 65 - avg_temp), CDD = max(0, avg_temp - 65)
    Anomaly = current - 30-year rolling mean for that month.
    """
    # NOAA Climate at a Glance: National monthly avg temp
    url = ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance"
           "/national/time-series/110/tavg/1/0/2000-2026.csv")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception:
        # Try alternative URL format
        url = ("https://www.ncei.noaa.gov/cag/national/time-series"
               "/110-tavg-1-0-2000-2026.csv")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

    # Parse: skip comment lines (start with "National")
    lines = resp.text.split("\n")
    data_lines = []
    header_found = False
    for line in lines:
        if "Date" in line and "Value" in line:
            header_found = True
            data_lines.append(line)
            continue
        if header_found and line.strip():
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    if "Date" not in df.columns or "Value" not in df.columns:
        return pd.DataFrame()

    # Date is YYYYMM format
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df["temp_f"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["temp_f"]).set_index("date").sort_index()

    # Compute degree days (monthly totals approximated from avg temp)
    df["hdd"] = np.maximum(0, 65 - df["temp_f"]) * 30  # ~30 days/month
    df["cdd"] = np.maximum(0, df["temp_f"] - 65) * 30

    # Compute anomalies: deviation from same-month 20-year trailing mean
    df["month"] = df.index.month
    for col in ["hdd", "cdd"]:
        normals = []
        for idx in df.index:
            same_month = df.loc[(df.index < idx) & (df["month"] == idx.month), col]
            # Use last 20 years
            same_month = same_month.iloc[-20:]
            if len(same_month) >= 10:
                normals.append(same_month.mean())
            else:
                normals.append(np.nan)
        df[f"{col}_anomaly"] = df[col].values - np.array(normals)

    return df[["hdd", "cdd", "hdd_anomaly", "cdd_anomaly"]].dropna()


# ---------------------------------------------------------------------------
# NYC Cloud Cover / Sunshine (for Hirshleifer-Shumway effect)
# ---------------------------------------------------------------------------

NYC_WEATHER_PUB_LAG_DAYS = 1  # Weather data available next day


def fetch_nyc_cloud_cover(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily cloud cover for NYC from Meteostat.

    Uses Meteostat library (wraps NOAA ISD + other sources) to get
    daily weather at LaGuardia Airport (USAF: 725030).

    Returns DataFrame indexed by date (pub-lag adjusted) with columns:
      cloud_cover: oktas (0=clear, 8=overcast), daily mean
      sunshine_hours: hours of sunshine per day
      temp_f: daily mean temperature
      precip_mm: daily precipitation
      is_clear: True if cloud_cover <= 2 (mostly clear)
      is_overcast: True if cloud_cover >= 6 (mostly overcast)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "nyc_weather.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    from meteostat import daily, stations, Point

    # Find LaGuardia Airport station
    loc = Point(40.7769, -73.8740, 3)
    nearby = stations.nearby(loc)
    if nearby.empty:
        raise ValueError("Could not find weather stations near NYC in Meteostat")
    station_id = nearby.index[0]  # 72503 = LaGuardia

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    ts = daily(station_id, start, end)
    data = ts.fetch()

    if data is None or data.empty:
        raise ValueError("No weather data returned for NYC")

    result = pd.DataFrame(index=data.index)

    # Meteostat columns: tsun (sunshine minutes), cldc (cloud cover)
    if "tsun" in data.columns:
        result["sunshine_hours"] = pd.to_numeric(data["tsun"], errors="coerce") / 60.0
    if "cldc" in data.columns:
        result["cloud_cover"] = pd.to_numeric(data["cldc"], errors="coerce")

    # Temperature (Meteostat uses Celsius)
    if "tavg" in data.columns:
        result["temp_f"] = data["tavg"] * 9 / 5 + 32
    elif "tmin" in data.columns and "tmax" in data.columns:
        result["temp_f"] = (data["tmin"] + data["tmax"]) / 2 * 9 / 5 + 32

    if "prcp" in data.columns:
        result["precip_mm"] = pd.to_numeric(data["prcp"], errors="coerce")
    if "rhum" in data.columns:
        result["humidity"] = pd.to_numeric(data["rhum"], errors="coerce")

    # Derive clear/overcast flags
    if "cloud_cover" in result.columns:
        result["is_clear"] = result["cloud_cover"] <= 2
        result["is_overcast"] = result["cloud_cover"] >= 6
    elif "sunshine_hours" in result.columns:
        result["is_clear"] = result["sunshine_hours"] > 8
        result["is_overcast"] = result["sunshine_hours"] < 3

    # Diurnal temperature range as sunshine proxy (wider = clearer sky)
    if "tmax" in data.columns and "tmin" in data.columns:
        tmax = pd.to_numeric(data["tmax"], errors="coerce")
        tmin = pd.to_numeric(data["tmin"], errors="coerce")
        result["temp_range_c"] = tmax - tmin

    # Apply publication lag
    result.index = result.index + pd.Timedelta(days=NYC_WEATHER_PUB_LAG_DAYS)

    if use_cache and not result.empty:
        result.to_parquet(cache_path)

    return result
