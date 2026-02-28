#!/usr/bin/env python3
"""
fetch_aam_latlon.py
-------------------
Fetches NCEP/NCAR Reanalysis 1 daily-average u-wind on pressure levels
from NOAA PSL's OPeNDAP server, computes the vertically and zonally
integrated relative atmospheric angular momentum (AAM) at each latitude
band, then writes a compact text file consumed by aam_hovmoller.html.

Physics
-------
  M(phi) = (2*pi*a^2 * cos^2(phi) / g) * sum_p [ u(phi,p) * dp ]

where a = 6.371e6 m, g = 9.81 m/s^2, phi = latitude (rad),
p = pressure level (Pa), dp = layer thickness (Pa).

Output format (data/aam_lat_latest.txt)
---------------------------------------
Header lines beginning with '#':
  # GLOBAL_AAM_LAT_ANOM
  # Base: 1980-01-01 to 2010-12-31
  # Units: kg m^2 s^-1 per latitude band (scaled by 1e24)
  # Lats: -88.75 -86.25 -83.75 ... 88.75   (space-separated, 73 values)
  # Cols: Date  lat[0] lat[1] ... lat[72]
Data rows (one per day, newest first or chronological):
  YYYY.MM.DD  val0 val1 ... val72
"""

import sys
import os
import json
from datetime import datetime, timedelta, timezone
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────
A  = 6.371e6          # Earth radius (m)
G  = 9.80665          # gravity (m/s^2)

BASE_START = datetime(1980, 1, 1, tzinfo=timezone.utc)
BASE_END   = datetime(2010, 12, 31, tzinfo=timezone.utc)

# Days of NCEP/NCAR reanalysis data to fetch for the "recent" window.
# We fetch the last RECENT_YEARS years for the anomaly plot, plus the
# entire base period for climatology computation.
RECENT_YEARS = 5

# PSL OPeNDAP base URL for daily-average uwnd on pressure levels
OPENDAP_BASE = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.{year}.nc"
)

OUTPUT_PATH = os.path.join("data", "aam_lat_latest.txt")

# ── helpers ────────────────────────────────────────────────────────────────

def open_nc(url):
    """Open a NetCDF dataset via OPeNDAP (requires netCDF4 or xarray+pydap)."""
    try:
        import netCDF4 as nc4
        return nc4.Dataset(url)
    except Exception as e:
        raise RuntimeError(f"Cannot open {url}: {e}")


def ncep_time_to_dates(time_var):
    """Convert NCEP time variable (hours since 1800-01-01) to list of datetimes."""
    import netCDF4 as nc4
    return nc4.num2date(time_var[:], time_var.units, time_var.calendar
                        if hasattr(time_var, 'calendar') else 'standard')


def compute_aam_lat(uwnd_3d, lats_rad, levels_pa):
    """
    Compute M(phi) integrated over pressure for each latitude.

    Parameters
    ----------
    uwnd_3d : np.ndarray shape (nlev, nlat, nlon)
        Zonal wind (m/s). Masked values replaced with 0.
    lats_rad : np.ndarray (nlat,)
        Latitude in radians.
    levels_pa : np.ndarray (nlev,)
        Pressure levels in Pa.

    Returns
    -------
    aam_lat : np.ndarray (nlat,)   units: kg m^2 s^-1 per latitude band
    """
    # Replace masked / NaN with 0
    u = np.where(np.ma.getmaskarray(uwnd_3d), 0.0, np.asarray(uwnd_3d, dtype=np.float64))

    # Zonal mean (average over longitude)
    u_zm = u.mean(axis=2)           # (nlev, nlat)

    # Pressure layer thickness: centre-difference for interior, one-sided at edges
    nlev = len(levels_pa)
    dp = np.empty(nlev)
    dp[0]    = levels_pa[1] - levels_pa[0]   # Pa  (may be negative if top→sfc order)
    dp[-1]   = levels_pa[-1] - levels_pa[-2]
    for k in range(1, nlev - 1):
        dp[k] = (levels_pa[k+1] - levels_pa[k-1]) / 2.0
    dp = np.abs(dp)   # always positive thickness

    # Vertical integral: sum_p u_zm * dp / g  (kg/m^2 * m/s = kg/(m*s))
    vert_int = np.sum(u_zm * dp[:, np.newaxis], axis=0) / G   # (nlat,)

    # Scale to full latitude band: 2*pi*a^2*cos^2(phi)
    cos2 = np.cos(lats_rad) ** 2
    aam_lat = 2.0 * np.pi * A**2 * cos2 * vert_int    # kg m^2 s^-1

    return aam_lat


def fetch_year(year):
    """Return (dates_list, aam_array shape(ndays, nlat), lats_deg, levels_pa)."""
    url = OPENDAP_BASE.format(year=year)
    print(f"  Fetching {url}", flush=True)
    ds = open_nc(url)

    lats_deg  = np.array(ds.variables['lat'][:])
    lons_deg  = np.array(ds.variables['lon'][:])
    levels_mb = np.array(ds.variables['level'][:])
    levels_pa = levels_mb * 100.0

    time_var  = ds.variables['time']
    dates     = ncep_time_to_dates(time_var)

    uwnd_var  = ds.variables['uwnd']
    # shape: (time, level, lat, lon)
    ndays = len(dates)
    nlat  = len(lats_deg)

    lats_rad = np.deg2rad(lats_deg)
    aam_all  = np.empty((ndays, nlat), dtype=np.float64)

    for t in range(ndays):
        u3d = uwnd_var[t, :, :, :]          # (nlev, nlat, nlon)
        aam_all[t] = compute_aam_lat(u3d, lats_rad, levels_pa)

    ds.close()
    return dates, aam_all, lats_deg, levels_pa


# ── main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    now = datetime.now(timezone.utc)
    current_year = now.year

    # Years to fetch for recent display
    recent_start_year = current_year - RECENT_YEARS
    recent_years = list(range(recent_start_year, current_year + 1))

    # Years needed for base-period climatology (1980–2010)
    base_years = list(range(1980, 2011))

    # Fetch base-period data (needed for climatology, not written to output in full)
    print("=== Fetching base-period data for climatology (1980–2010) ===")
    base_sums  = None   # will be (nlat,)
    base_sum2  = None
    base_count = None
    lats_deg   = None
    levels_pa  = None

    for year in base_years:
        try:
            dates, aam, ld, lp = fetch_year(year)
            if lats_deg is None:
                lats_deg  = ld
                levels_pa = lp
                nlat = len(lats_deg)
                base_sums  = np.zeros(nlat)
                base_sum2  = np.zeros(nlat)
                base_count = np.zeros(nlat, dtype=int)

            for i, d in enumerate(dates):
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                if BASE_START <= dt <= BASE_END:
                    base_sums  += aam[i]
                    base_sum2  += aam[i] ** 2
                    base_count += 1
        except Exception as e:
            print(f"  WARNING: could not fetch {year}: {e}", file=sys.stderr)

    if base_count is None or base_count.min() == 0:
        print("ERROR: base period data insufficient.", file=sys.stderr)
        sys.exit(1)

    base_mean = base_sums / base_count
    base_var  = base_sum2 / base_count - base_mean ** 2
    base_std  = np.sqrt(np.maximum(base_var, 1e-30))

    print(f"Base climatology computed from {base_count.mean():.0f} days per lat.")

    # Fetch recent years for output
    print(f"\n=== Fetching recent data ({recent_start_year}–{current_year}) ===")
    all_dates = []
    all_aam   = []

    for year in recent_years:
        try:
            dates, aam, ld, lp = fetch_year(year)
            # Trim to only past dates (no future)
            for i, d in enumerate(dates):
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                if dt <= now:
                    all_dates.append(dt)
                    all_aam.append(aam[i])
        except Exception as e:
            print(f"  WARNING: could not fetch {year}: {e}", file=sys.stderr)

    if not all_dates:
        print("ERROR: no recent data fetched.", file=sys.stderr)
        sys.exit(1)

    # Sort chronologically
    order = np.argsort(all_dates)
    all_dates = [all_dates[i] for i in order]
    all_aam   = np.array([all_aam[i] for i in order])   # (ndays, nlat)

    # Standardise: anomaly in sigma units
    anom = (all_aam - base_mean[np.newaxis, :]) / base_std[np.newaxis, :]

    # Scale for output (keep as σ, but scale AAM to 1e24 for readability)
    # We write raw anomalies in σ units directly.

    # ── write output ───────────────────────────────────────────────────────
    lat_str = " ".join(f"{v:.2f}" for v in lats_deg)

    with open(OUTPUT_PATH, "w") as f:
        f.write("# GLOBAL_AAM_LAT_ANOM\n")
        f.write(f"# Base: {BASE_START.strftime('%Y-%m-%d')} to {BASE_END.strftime('%Y-%m-%d')}\n")
        f.write("# Units: sigma anomaly (standardised departure from 1980-2010 climatology)\n")
        f.write(f"# Source: NCEP/NCAR Reanalysis 1, NOAA PSL OPeNDAP\n")
        f.write(f"# Generated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"# Lats: {lat_str}\n")
        f.write("# Cols: Date  lat[0..N-1]\n")

        for i, dt in enumerate(all_dates):
            row_vals = " ".join(f"{v:.4f}" for v in anom[i])
            f.write(f"{dt.strftime('%Y.%m.%d')}  {row_vals}\n")

    print(f"\nWrote {len(all_dates)} rows × {len(lats_deg)} lats → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()