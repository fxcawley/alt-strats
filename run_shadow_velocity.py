"""
Shadow-Price of Cloud Velocity: ERCOT solar ramp vs DAM-RTM spread.

Thesis: When solar generation ramps fast (high-velocity cloud fronts),
the ISO day-ahead forecast lags, creating a supply mismatch that
spikes real-time prices above day-ahead prices.

The trade: Long RTM volatility when satellite/solar data predicts
a fast ramp event that the standard forecast misses.

Data sources (all real, public):
  - ERCOT DAM Settlement Point Prices (via gridstatus)
  - ERCOT RTM Settlement Point Prices (via gridstatus)
  - ERCOT Solar Actual & Forecast (via gridstatus)
  - Settlement point: HB_WEST (West Texas, near major solar farms)

Validation: correlation between |solar_ramp| and |DAM-RTM spread|,
quintile analysis for the "volatility smile" pattern.
"""

from __future__ import annotations

import warnings
import ssl
import sys

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# SSL bypass for ERCOT (corp proxy or cert issues)
ssl._create_default_https_context = ssl._create_unverified_context
import requests
from requests.adapters import HTTPAdapter
_old_send = HTTPAdapter.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _old_send(self, request, **kwargs)
HTTPAdapter.send = _patched_send


def main():
    import gridstatus

    ercot = gridstatus.Ercot()
    hub = "HB_WEST"

    # Date range: recent data available on ERCOT MIS (~30 day window)
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=28)

    print("=" * 70)
    print("  SHADOW-PRICE OF CLOUD VELOCITY")
    print("  ERCOT solar ramp rate vs DAM-RTM price spread")
    print(f"  {hub} hub, {start.date()} to {end.date()}")
    print("=" * 70)

    # Fetch price data
    print(f"\nFetching DAM + RTM SPP...")
    dam = ercot.get_spp(date=start, end=end, market="DAY_AHEAD_HOURLY")
    rtm = ercot.get_spp(date=start, end=end, market="REAL_TIME_15_MIN")
    print(f"  DAM: {dam.shape[0]:,} rows, RTM: {rtm.shape[0]:,} rows")

    # Aggregate to hourly at target hub
    dam_h = dam[dam["Location"] == hub][["Interval Start", "SPP"]].copy()
    dam_h["hour"] = dam_h["Interval Start"].dt.tz_localize(None).dt.floor("h")
    dam_h = dam_h.groupby("hour")["SPP"].mean().rename("DAM_Price")

    rtm_h = rtm[rtm["Location"] == hub][["Interval Start", "SPP"]].copy()
    rtm_h["hour"] = rtm_h["Interval Start"].dt.tz_localize(None).dt.floor("h")
    rtm_h = rtm_h.groupby("hour")["SPP"].mean().rename("RTM_Price")

    merged = pd.DataFrame({"DAM_Price": dam_h, "RTM_Price": rtm_h}).dropna()
    merged["spread"] = merged["RTM_Price"] - merged["DAM_Price"]
    merged["abs_spread"] = merged["spread"].abs()

    print(f"  Merged hourly intervals: {len(merged)}")
    print(f"  Spread: mean=${merged['spread'].mean():.2f}, "
          f"std=${merged['spread'].std():.2f}, "
          f"range=[${merged['spread'].min():.2f}, ${merged['spread'].max():.2f}]")

    # Fetch solar generation data (day by day to avoid DST issues)
    print(f"\nFetching solar generation...")
    solar_dfs = []
    for d in pd.date_range(start, end - pd.Timedelta(days=1)):
        try:
            s = ercot.get_solar_actual_and_forecast_hourly(date=d.strftime("%Y-%m-%d"))
            if s is not None and not s.empty:
                solar_dfs.append(s)
        except Exception:
            continue

    if not solar_dfs:
        print("  ERROR: No solar generation data retrieved")
        sys.exit(1)

    solar = pd.concat(solar_dfs, ignore_index=True)
    solar["hour"] = solar["Interval Start"].dt.tz_localize(None).dt.floor("h")
    solar_h = solar.groupby("hour")["GEN SYSTEM WIDE"].mean().rename("solar_gen_mw")

    # Compute ramp rate
    solar_df = solar_h.to_frame()
    solar_df["solar_ramp"] = solar_df["solar_gen_mw"].diff()
    solar_df["abs_ramp"] = solar_df["solar_ramp"].abs()

    print(f"  Solar data: {len(solar_df)} hours")
    print(f"  Gen range: {solar_df['solar_gen_mw'].min():.0f} to "
          f"{solar_df['solar_gen_mw'].max():.0f} MW")

    # Combine price + solar, filter to daylight hours
    combined = merged.join(solar_df, how="inner")
    combined["local_hour"] = combined.index.hour
    daylight = combined[(combined["local_hour"] >= 8) & (combined["local_hour"] <= 18)].copy()
    daylight = daylight.dropna(subset=["abs_ramp", "abs_spread"])

    print(f"\n  Daylight hours with aligned data: {len(daylight)}")

    if len(daylight) < 20:
        print("  Insufficient data for analysis")
        sys.exit(1)

    # THE TEST
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")

    # 1. Correlation
    rho, pval = stats.spearmanr(daylight["abs_ramp"], daylight["abs_spread"])
    print(f"\n  Spearman rho(|solar_ramp|, |spread|) = {rho:.4f}  (p={pval:.4f}, n={len(daylight)})")

    # 2. Quintile analysis (the volatility smile test)
    n_q = min(5, len(daylight) // 5)
    if n_q >= 3:
        daylight["ramp_q"] = pd.qcut(daylight["abs_ramp"], n_q, labels=False, duplicates="drop")
        qt = daylight.groupby("ramp_q").agg(
            mean_ramp=("abs_ramp", "mean"),
            mean_abs_spread=("abs_spread", "mean"),
            spread_std=("spread", "std"),
            n=("spread", "count"),
        )
        print(f"\n  Quintile analysis ({n_q} bins by |solar ramp|):")
        print(f"  {'Q':>3s}  {'Ramp(MW/hr)':>12s}  {'Mean|Spread|':>13s}  {'Spread Std':>11s}  {'n':>4s}")
        for q, row in qt.iterrows():
            print(f"  Q{int(q):1d}  {row['mean_ramp']:12.0f}  ${row['mean_abs_spread']:12.2f}  "
                  f"${row['spread_std']:10.2f}  {int(row['n']):4d}")

        # Monotonicity of spread std across ramp quintiles
        mono_corr, _ = stats.spearmanr(qt.index.astype(float), qt["spread_std"])
        print(f"\n  Monotonicity (ramp quintile vs spread std): rho={mono_corr:.2f}")

    # 3. Fast shadow events
    neg_threshold = daylight["solar_ramp"].quantile(0.1)  # bottom 10% = fast drops
    fast_shadows = daylight[daylight["solar_ramp"] < neg_threshold]
    baseline = daylight[daylight["abs_ramp"] < daylight["abs_ramp"].median()]

    print(f"\n  Fast shadow events (ramp < {neg_threshold:.0f} MW/hr): n={len(fast_shadows)}")
    if len(fast_shadows) > 0:
        print(f"    Mean RTM-DAM spread: ${fast_shadows['spread'].mean():.2f}")
        print(f"    Mean |spread|:       ${fast_shadows['abs_spread'].mean():.2f}")
        print(f"    Spread std:          ${fast_shadows['spread'].std():.2f}")
    print(f"  Baseline (below-median ramp): n={len(baseline)}")
    if len(baseline) > 0:
        print(f"    Mean |spread|:       ${baseline['abs_spread'].mean():.2f}")
        print(f"    Spread std:          ${baseline['spread'].std():.2f}")

    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")
    print("  The thesis predicts a 'volatility smile': as |solar ramp|")
    print("  increases, the DAM-RTM spread variance should increase")
    print("  (the ISO forecast breaks down during fast ramp events).")
    print(f"  With {len(daylight)} observations over ~{(end-start).days} days:")
    if rho > 0.15:
        print(f"  -> rho={rho:.3f} supports the thesis direction.")
    elif rho > 0:
        print(f"  -> rho={rho:.3f} is weakly positive, directionally correct.")
    else:
        print(f"  -> rho={rho:.3f} does not support the thesis.")
    print("  Needs 3+ months of data for statistical significance.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
