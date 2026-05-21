#!/usr/bin/env python3
"""
fetch_aam_latlon.py
────────────────────────────────────────────────────────────────────────────────
Computes relative atmospheric angular momentum (AAM) anomaly by latitude band
from ERA5 reanalysis and writes data/aam_lat_latest.txt for the
PhillyWeatherAuthority Hovmöller diagram.

SOURCE  : ERA5 hourly data on pressure levels (reanalysis-era5-pressure-levels)
          via ECMWF Copernicus Climate Data Store (CDS) API.
          Coverage: 1940–present, updated daily with ~5-day lag.
          https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

METHOD  : Relative (wind-term) AAM per latitude band:
            M(φ) = (2π/g) · cos²φ · a³ · Ω · Σ_p [ ū(φ,p) · Δp ]
          where ū is the zonal-mean daily-average u-wind summed over
          all pressure levels.  Anomalies are standardised σ departures
          from the 1980–2010 daily climatology (31-day centred smoothing).

OUTPUT  : data/aam_lat_latest.txt   — rolling 90-day window
          data/aam_climo.npz        — cached climatology (rebuilt if missing)

AUTH    : Reads the CDS personal access token from the environment variable
          CDS_API_KEY.  In GitHub Actions add this as a repository secret.
          The script writes ~/.cdsapirc at runtime — nothing is committed.

DEPENDENCIES: cdsapi, netCDF4, numpy
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import calendar
import os
import sys
import time
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import cdsapi
from netCDF4 import Dataset  # noqa: N813

# ── constants ────────────────────────────────────────────────────────────────
OMEGA = 7.292115e-5   # Earth rotation rate (rad/s)
A     = 6.371e6       # Earth radius (m)
G     = 9.80665       # standard gravity (m/s²)

# Full ERA5 pressure level set (hPa)
PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300','350','400',
    '450','500','550','600','650','700','750','775','800','825',
    '850','875','900','925','950','975','1000'
]

# Climatology base period (matches original AER/R1 baseline)
CLIMO_START = 1980
CLIMO_END   = 2010

# Rolling output window in days
WINDOW_DAYS = 90

# Smoothing half-width for daily climatology (31-day centred window)
SMOOTH_HALF = 15

# ERA5 grid resolution for download (degrees)
GRID = [1.0, 1.0]

# Paths (script lives in scripts/, repo root is one level up)
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_TXT = DATA_DIR / "aam_lat_latest.txt"
CLIMO_NPZ  = DATA_DIR / "aam_climo.npz"


# ── auth ─────────────────────────────────────────────────────────────────────

def setup_cds_auth() -> None:
    """Write ~/.cdsapirc from the CDS_API_KEY environment variable."""
    key = os.environ.get("CDS_API_KEY", "").strip()
    if not key:
        print("ERROR: CDS_API_KEY environment variable is not set.", file=sys.stderr)
        print("  GitHub Actions: add CDS_API_KEY as a repository secret.", file=sys.stderr)
        print("  Local: export CDS_API_KEY=<your-personal-access-token>", file=sys.stderr)
        sys.exit(1)
    rc = Path.home() / ".cdsapirc"
    rc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
    log(f"CDS credentials written to {rc}")


# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def pressure_layer_dp(levels_pa: np.ndarray) -> np.ndarray:
    """
    Layer thicknesses (Pa) for vertically integrated AAM.
    Expects levels_pa sorted descending (surface → top).
    Edge levels get half the adjacent spacing.
    """
    n  = len(levels_pa)
    dp = np.empty(n)
    for i in range(n):
        lo = levels_pa[i + 1] if i + 1 < n else levels_pa[i]
        hi = levels_pa[i - 1] if i > 0      else levels_pa[i]
        dp[i] = abs(hi - lo) / 2.0
    return dp


def doy365(d: date) -> int:
    """1-based day-of-year, capped at 365 (leap days map to 365)."""
    return min(d.timetuple().tm_yday, 365)


def smooth_circular(arr: np.ndarray, half: int) -> np.ndarray:
    """Centred running mean over axis-0, wrapping at year boundaries."""
    n   = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(n):
        idx    = [(i + k) % n for k in range(-half, half + 1)]
        out[i] = arr[idx].mean(axis=0)
    return out


# ── ERA5 monthly fetch ────────────────────────────────────────────────────────

def fetch_month(year: int, month: int) -> tuple[np.ndarray, list[date], np.ndarray]:
    """
    Download ERA5 u-wind on all pressure levels for one calendar month
    (4× daily, then averaged to daily means).

    Returns
    -------
    lats      (nlat,)        degrees, 90 N → 90 S
    days      list[date]     dates in this month
    aam       (ndays, nlat)  relative AAM per latitude (kg m s⁻¹ per unit width)
    """
    ndays     = calendar.monthrange(year, month)[1]
    days_list = [date(year, month, d) for d in range(1, ndays + 1)]
    days_str  = [f"{d:02d}" for d in range(1, ndays + 1)]

    client  = cdsapi.Client(quiet=True)
    request = {
        "product_type":    ["reanalysis"],
        "variable":        ["u_component_of_wind"],
        "pressure_level":  PRESSURE_LEVELS,
        "year":            [str(year)],
        "month":           [f"{month:02d}"],
        "day":             days_str,
        "time":            ["00:00", "06:00", "12:00", "18:00"],
        "data_format":     "netcdf",
        "download_format": "unarchived",
        "grid":            GRID,
    }

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    for attempt in range(1, 4):
        try:
            client.retrieve("reanalysis-era5-pressure-levels", request, tmp_path)
            break
        except Exception as exc:
            log(f"  [attempt {attempt}/3] CDS error: {exc}")
            if attempt < 3:
                time.sleep(60)
            else:
                raise

    try:
        ds    = Dataset(tmp_path)
        u_var = ds.variables["u"]           # (time, level, lat, lon)

        lats_deg = np.array(ds.variables["latitude"][:])
        # level variable name varies slightly across ERA5 files
        lev_key  = "pressure_level" if "pressure_level" in ds.variables else "level"
        levs_hpa = np.array(ds.variables[lev_key][:])

        # sort surface → top (descending hPa)
        lev_order = np.argsort(levs_hpa)[::-1]
        levs_pa   = levs_hpa[lev_order] * 100.0
        dp        = pressure_layer_dp(levs_pa)                # (nlev,)

        lats_rad  = np.deg2rad(lats_deg)
        prefactor = (2.0 * np.pi / G) * np.cos(lats_rad) ** 2 * A**3 * OMEGA  # (nlat,)

        ntimes         = u_var.shape[0]          # ndays × 4
        steps_per_day  = ntimes // ndays
        nlat           = len(lats_deg)
        aam            = np.zeros((ndays, nlat), dtype=np.float64)

        for t in range(ntimes):
            day_idx = t // steps_per_day
            if day_idx >= ndays:
                break
            u_raw = np.array(u_var[t, :, :, :])             # (level, lat, lon)
            if hasattr(u_raw, "filled"):
                u_raw = u_raw.filled(np.nan)
            u_raw    = u_raw[lev_order, :, :]                # sort levels
            u_zonal  = np.nanmean(u_raw, axis=2)             # (level, lat)
            vert_int = np.nansum(u_zonal * dp[:, np.newaxis], axis=0)  # (lat,)
            aam[day_idx] += prefactor * vert_int

        aam /= steps_per_day      # time-mean → daily average
        ds.close()

    finally:
        os.unlink(tmp_path)

    return lats_deg, days_list, aam


# ── climatology ───────────────────────────────────────────────────────────────

def build_climatology(lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Download CLIMO_START–CLIMO_END and compute smoothed daily mean/std."""
    log(f"Building ERA5 climatology {CLIMO_START}–{CLIMO_END} …")
    nlat      = len(lats)
    day_sum   = np.zeros((365, nlat), dtype=np.float64)
    day_sum2  = np.zeros((365, nlat), dtype=np.float64)
    day_count = np.zeros(365,         dtype=np.int32)

    for yr in range(CLIMO_START, CLIMO_END + 1):
        for mo in range(1, 13):
            log(f"  {yr}-{mo:02d} …")
            try:
                _, days_list, aam_mo = fetch_month(yr, mo)
            except Exception as exc:
                log(f"  WARNING: skipping {yr}-{mo:02d}: {exc}")
                continue
            for i, d in enumerate(days_list):
                doy = doy365(d) - 1
                day_sum[doy]   += aam_mo[i]
                day_sum2[doy]  += aam_mo[i] ** 2
                day_count[doy] += 1

    safe_n        = np.where(day_count > 0, day_count, 1)
    mean_raw      = day_sum  / safe_n[:, np.newaxis]
    var_raw       = day_sum2 / safe_n[:, np.newaxis] - mean_raw ** 2
    std_raw       = np.sqrt(np.maximum(var_raw, 0.0))

    climo_mean = smooth_circular(mean_raw, SMOOTH_HALF)
    climo_std  = smooth_circular(std_raw,  SMOOTH_HALF)
    climo_std  = np.where(climo_std > 0, climo_std, 1.0)

    return climo_mean, climo_std


def load_or_build_climatology(
    lats: np.ndarray, rebuild: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if not rebuild and CLIMO_NPZ.exists():
        log(f"Loading cached climatology from {CLIMO_NPZ} …")
        data = np.load(CLIMO_NPZ)
        cm, cs, cl = data["climo_mean"], data["climo_std"], data["lats"]
        if cm.shape[1] == len(lats) and np.allclose(cl, lats):
            return cm, cs
        log("  Grid mismatch — rebuilding.")

    cm, cs = build_climatology(lats)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIMO_NPZ, climo_mean=cm, climo_std=cs, lats=lats)
    log(f"Climatology saved → {CLIMO_NPZ}")
    return cm, cs


# ── rolling window fetch ──────────────────────────────────────────────────────

def fetch_recent(window_days: int) -> tuple[np.ndarray, list[date], np.ndarray]:
    """
    Fetch the most recent `window_days` of ERA5 data.
    Requests an extra 10 days to absorb ERA5's ~5-day production lag.
    """
    today  = date.today()
    start  = today - timedelta(days=window_days + 10)

    # collect unique (year, month) pairs
    months: set[tuple[int, int]] = set()
    d = start.replace(day=1)
    while d <= today:
        months.add((d.year, d.month))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)

    log(f"Fetching recent ERA5 data window ({window_days}d, up to ~{today}) …")

    lats: np.ndarray | None = None
    all_dates: list[date]   = []
    all_aam: list[np.ndarray] = []
    cutoff = today - timedelta(days=window_days)

    for yr, mo in sorted(months):
        log(f"  {yr}-{mo:02d} …")
        try:
            yr_lats, days_list, aam_mo = fetch_month(yr, mo)
        except Exception as exc:
            log(f"  WARNING: could not fetch {yr}-{mo:02d}: {exc}")
            continue
        if lats is None:
            lats = yr_lats
        for i, d in enumerate(days_list):
            if cutoff <= d <= today:
                all_dates.append(d)
                all_aam.append(aam_mo[i])

    if not all_dates:
        print("ERROR: no recent ERA5 data retrieved.", file=sys.stderr)
        sys.exit(1)

    return lats, all_dates, np.array(all_aam)


# ── output ────────────────────────────────────────────────────────────────────

def write_output(dates: list[date], lats: np.ndarray, anom: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relative AAM anomaly by latitude — ERA5 reanalysis",
        "# Source: ECMWF Copernicus CDS — reanalysis-era5-pressure-levels",
        f"# Climatology base: {CLIMO_START}–{CLIMO_END} (31-day centred smoothing)",
        "# Units: standardised sigma departures",
        f"# Generated: {date.today().isoformat()}",
        "# Lats: " + " ".join(f"{lat:.2f}" for lat in lats),
    ]
    for d, row in zip(dates, anom):
        vals = " ".join(f"{v:8.4f}" for v in row)
        lines.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {vals}")
    OUTPUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"Wrote {len(dates)} rows → {OUTPUT_TXT}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ERA5 relative AAM by latitude → Hovmöller data file."
    )
    parser.add_argument(
        "--rebuild-climo", action="store_true",
        help="Force rebuild of the 1980–2010 climatology even if cache exists."
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_DAYS,
        help=f"Rolling window in days (default: {WINDOW_DAYS})."
    )
    args = parser.parse_args()

    setup_cds_auth()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    lats, dates, aam_recent         = fetch_recent(args.window)
    climo_mean, climo_std           = load_or_build_climatology(lats, args.rebuild_climo)

    log("Computing standardised anomalies …")
    anom = np.empty_like(aam_recent)
    for i, d in enumerate(dates):
        doy      = doy365(d) - 1
        anom[i]  = (aam_recent[i] - climo_mean[doy]) / climo_std[doy]

    write_output(dates, lats, anom)
    log("Done.")


if __name__ == "__main__":
    main()
