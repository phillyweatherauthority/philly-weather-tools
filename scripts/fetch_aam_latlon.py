#!/usr/bin/env python3
"""
fetch_aam_latlon.py
-------------------
Fetches NCEP/NCAR Reanalysis 1 daily-average u-wind on pressure levels
from NOAA PSL's OPeNDAP server, computes the vertically and zonally
integrated relative atmospheric angular momentum (AAM) at each latitude
band, then writes a compact text file consumed by aam_hovmoller.html.

Climatology caching
-------------------
On the FIRST run, the script fetches 1980-2010 base period data (~45 min),
computes the per-latitude mean and std, and saves them to:
    data/aam_climo.npz

On every SUBSEQUENT run, it loads that cached file instead — skipping the
entire base period fetch. Daily runs therefore take ~2 minutes.

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
  # Units: sigma anomaly
  # Lats: -88.75 -86.25 ... 88.75
  # Cols: Date  lat[0] lat[1] ... lat[N-1]
Data rows (one per day, chronological):
  YYYY.MM.DD  val0 val1 ... valN
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────
A  = 6.371e6   # Earth radius (m)
G  = 9.80665   # gravity (m/s^2)

BASE_START = datetime(1980, 1, 1, tzinfo=timezone.utc)
BASE_END   = datetime(2010, 12, 31, tzinfo=timezone.utc)

# How many days of recent data to write to the output file
RECENT_DAYS = 60

# PSL OPeNDAP URL template
OPENDAP_BASE = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.{year}.nc"
)

OUTPUT_PATH = os.path.join("data", "aam_lat_latest.txt")
CLIMO_PATH  = os.path.join("data", "aam_climo.npz")

# ── helpers ────────────────────────────────────────────────────────────────

def open_nc(url):
    try:
        import netCDF4 as nc4
        return nc4.Dataset(url)
    except Exception as e:
        raise RuntimeError(f"Cannot open {url}: {e}")


def ncep_time_to_dates(time_var):
    import netCDF4 as nc4
    cal = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
    return nc4.num2date(time_var[:], time_var.units, cal)


def compute_aam_lat(uwnd_3d, lats_rad, levels_pa):
    u = np.where(np.ma.getmaskarray(uwnd_3d), 0.0,
                 np.asarray(uwnd_3d, dtype=np.float64))
    u_zm = u.mean(axis=2)  # zonal mean → (nlev, nlat)

    nlev = len(levels_pa)
    dp = np.empty(nlev)
    dp[0]  = levels_pa[1] - levels_pa[0]
    dp[-1] = levels_pa[-1] - levels_pa[-2]
    for k in range(1, nlev - 1):
        dp[k] = (levels_pa[k+1] - levels_pa[k-1]) / 2.0
    dp = np.abs(dp)

    vert_int = np.sum(u_zm * dp[:, np.newaxis], axis=0) / G  # (nlat,)
    cos2     = np.cos(lats_rad) ** 2
    return 2.0 * np.pi * A**2 * cos2 * vert_int  # kg m^2 s^-1


def fetch_year(year):
    """Fetch one year of daily uwnd and return AAM by latitude."""
    url = OPENDAP_BASE.format(year=year)
    print(f"  Fetching {url}", flush=True)
    ds = open_nc(url)

    lats_deg  = np.array(ds.variables['lat'][:])
    levels_pa = np.array(ds.variables['level'][:]) * 100.0
    dates     = ncep_time_to_dates(ds.variables['time'])
    uwnd_var  = ds.variables['uwnd']

    lats_rad = np.deg2rad(lats_deg)
    nlat     = len(lats_deg)
    ndays    = len(dates)
    aam_all  = np.empty((ndays, nlat), dtype=np.float64)

    for t in range(ndays):
        aam_all[t] = compute_aam_lat(uwnd_var[t, :, :, :], lats_rad, levels_pa)

    ds.close()
    return dates, aam_all, lats_deg, levels_pa


# ── climatology: load cache or compute from scratch ────────────────────────

def load_climo():
    """Load cached climatology. Returns (base_mean, base_std, lats_deg) or None."""
    if not os.path.exists(CLIMO_PATH):
        return None
    print(f"Loading cached climatology from {CLIMO_PATH}", flush=True)
    d = np.load(CLIMO_PATH)
    return d['base_mean'], d['base_std'], d['lats_deg']


def build_climo():
    """Fetch 1980-2010, compute per-latitude mean & std, save to cache."""
    print("=== No climatology cache found — fetching 1980–2010 base period ===")
    print("    (This only happens once. Future runs will load the cache.)\n")

    base_sums  = None
    base_sum2  = None
    base_count = None
    lats_deg   = None
    levels_pa  = None

    for year in range(1980, 2011):
        try:
            dates, aam, ld, lp = fetch_year(year)
            if lats_deg is None:
                lats_deg  = ld
                levels_pa = lp
                nlat       = len(lats_deg)
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

    # Save cache
    np.savez(CLIMO_PATH, base_mean=base_mean, base_std=base_std, lats_deg=lats_deg)
    print(f"\nClimatology cached to {CLIMO_PATH} "
          f"({base_count.mean():.0f} days/lat)")

    return base_mean, base_std, lats_deg


# ── main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    now          = datetime.now(timezone.utc)
    current_year = now.year
    cutoff_dt    = now - timedelta(days=RECENT_DAYS)

    # ── Step 1: get climatology (from cache or full fetch) ──────────────────
    climo = load_climo()
    if climo is None:
        base_mean, base_std, lats_deg = build_climo()
    else:
        base_mean, base_std, lats_deg = climo

    # ── Step 2: fetch only the recent window (60 days) ─────────────────────
    # We need at most two years: the current year and possibly the previous
    # (e.g., if today is Jan 15, our 60-day window starts in mid-November)
    recent_years = sorted({cutoff_dt.year, current_year})
    print(f"\n=== Fetching recent data (last {RECENT_DAYS} days) ===", flush=True)

    all_dates = []
    all_aam   = []

    for year in recent_years:
        try:
            dates, aam, ld, lp = fetch_year(year)
            for i, d in enumerate(dates):
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                if cutoff_dt <= dt <= now:
                    all_dates.append(dt)
                    all_aam.append(aam[i])
        except Exception as e:
            print(f"  WARNING: could not fetch {year}: {e}", file=sys.stderr)

    if not all_dates:
        print("ERROR: no recent data fetched.", file=sys.stderr)
        sys.exit(1)

    # Sort chronologically
    order     = np.argsort(all_dates)
    all_dates = [all_dates[i] for i in order]
    all_aam   = np.array([all_aam[i] for i in order])  # (ndays, nlat)

    # Standardise to sigma units
    anom = (all_aam - base_mean[np.newaxis, :]) / base_std[np.newaxis, :]

    # ── Step 3: write output ────────────────────────────────────────────────
    lat_str = " ".join(f"{v:.2f}" for v in lats_deg)

    with open(OUTPUT_PATH, "w") as f:
        f.write("# GLOBAL_AAM_LAT_ANOM\n")
        f.write(f"# Base: {BASE_START.strftime('%Y-%m-%d')} to "
                f"{BASE_END.strftime('%Y-%m-%d')}\n")
        f.write("# Units: sigma anomaly (standardised departure from "
                "1980-2010 climatology)\n")
        f.write("# Source: NCEP/NCAR Reanalysis 1, NOAA PSL OPeNDAP\n")
        f.write(f"# Generated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"# Lats: {lat_str}\n")
        f.write("# Cols: Date  lat[0..N-1]\n")

        for i, dt in enumerate(all_dates):
            row_vals = " ".join(f"{v:.4f}" for v in anom[i])
            f.write(f"{dt.strftime('%Y.%m.%d')}  {row_vals}\n")

    print(f"\nWrote {len(all_dates)} rows × {len(lats_deg)} lats → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
