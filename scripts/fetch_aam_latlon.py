#!/usr/bin/env python3
"""
fetch_aam_latlon.py
────────────────────────────────────────────────────────────────────────────────
Computes relative AAM anomaly by latitude from ERA5 and writes
data/aam_lat_latest.txt for the PhillyWeatherAuthority Hovmöller.

TWO MODES
  --mode daily     (default) Fetch last 60 days, compute anomalies using
                   the cached climatology, write output. Fast (~5–10 min).
  --mode climo     Build the 1980–2010 climatology from scratch and save
                   data/aam_climo.npz. Run once via build_aam_climo.yml.
                   Takes ~60–90 min with bulk annual CDS requests.

SOURCE  : ERA5 reanalysis-era5-pressure-levels via Copernicus CDS API.
GRID    : 2.5° × 2.5° (matches original R1/R2 resolution, smaller files)
LEVELS  : 15 key pressure levels spanning surface to stratosphere
AUTH    : CDS_API_KEY environment variable → written to ~/.cdsapirc at runtime
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

# 15 pressure levels — span surface to lower stratosphere.
# AAM is dominated by 1000–100 hPa; finer sampling above adds noise, not signal.
PRESSURE_LEVELS = [
    '1000','925','850','700','600','500','400','300',
    '250','200','150','100','50','20','10'
]

CLIMO_START  = 1980
CLIMO_END    = 2010
WINDOW_DAYS  = 60
SMOOTH_HALF  = 15      # 31-day centred smoothing for climatology
GRID         = [2.5, 2.5]

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_TXT         = DATA_DIR / "aam_lat_latest.txt"
TENDENCY_TXT       = DATA_DIR / "aam_tendency_latest.txt"
GLOBAL_TEND_TXT    = DATA_DIR / "aam_global_tendency_latest.txt"
GLOBAL_AAM_TXT     = DATA_DIR / "aam_global_latest.txt"
CLIMO_NPZ          = DATA_DIR / "aam_climo.npz"


# ── auth ──────────────────────────────────────────────────────────────────────
def setup_cds_auth() -> None:
    key = os.environ.get("CDS_API_KEY", "").strip()
    if not key:
        print("ERROR: CDS_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    (Path.home() / ".cdsapirc").write_text(
        f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n"
    )
    log("CDS auth configured.")


# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def pressure_dp(levels_pa: np.ndarray) -> np.ndarray:
    """Layer thicknesses (Pa), sorted descending (surface→top)."""
    n  = len(levels_pa)
    dp = np.empty(n)
    for i in range(n):
        lo = levels_pa[i + 1] if i + 1 < n else levels_pa[i]
        hi = levels_pa[i - 1] if i > 0      else levels_pa[i]
        dp[i] = abs(hi - lo) / 2.0
    return dp


def doy365(d: date) -> int:
    return min(d.timetuple().tm_yday, 365)


def smooth_circular(arr: np.ndarray, half: int) -> np.ndarray:
    n   = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(n):
        idx    = [(i + k) % n for k in range(-half, half + 1)]
        out[i] = arr[idx].mean(axis=0)
    return out


def iter_months(start_ym: tuple, end_ym: tuple):
    """Yield (year, month) tuples from start_ym to end_ym inclusive."""
    yr, mo = start_ym
    while (yr, mo) <= end_ym:
        yield yr, mo
        mo += 1
        if mo > 12:
            mo, yr = 1, yr + 1


# ── ERA5 fetch: single month ──────────────────────────────────────────────────
def fetch_month(year: int, month: int) -> tuple[np.ndarray, list[date], np.ndarray]:
    """
    Download ERA5 u-wind for one month (4× daily → daily mean).
    Returns (lats, days_list, aam shape (ndays, nlat)).
    """
    ndays     = calendar.monthrange(year, month)[1]
    days_list = [date(year, month, d) for d in range(1, ndays + 1)]

    client  = cdsapi.Client(quiet=True)
    request = {
        "product_type":    ["reanalysis"],
        "variable":        ["u_component_of_wind"],
        "pressure_level":  PRESSURE_LEVELS,
        "year":            [str(year)],
        "month":           [f"{month:02d}"],
        "day":             [f"{d:02d}" for d in range(1, ndays + 1)],
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
        aam, lats_deg = _nc_to_aam(tmp_path, ndays)
    finally:
        os.unlink(tmp_path)

    return lats_deg, days_list, aam


# ── ERA5 fetch: full year (bulk) ──────────────────────────────────────────────
def fetch_year(year: int) -> tuple[np.ndarray, list[date], np.ndarray]:
    """
    Download ERA5 u-wind for an entire year in one CDS request.
    Used for climatology building — far fewer API calls than monthly.
    Returns (lats, days_list, aam shape (ndays, nlat)).
    """
    import calendar as cal_mod
    is_leap   = cal_mod.isleap(year)
    ndays     = 366 if is_leap else 365
    days_list = [date(year, 1, 1) + timedelta(days=i) for i in range(ndays)]

    client  = cdsapi.Client(quiet=True)
    request = {
        "product_type":    ["reanalysis"],
        "variable":        ["u_component_of_wind"],
        "pressure_level":  PRESSURE_LEVELS,
        "year":            [str(year)],
        "month":           [f"{m:02d}" for m in range(1, 13)],
        "day":             [f"{d:02d}" for d in range(1, 32)],
        "time":            ["00:00", "06:00", "12:00", "18:00"],
        "data_format":     "netcdf",
        "download_format": "unarchived",
        "grid":            GRID,
    }

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    log(f"  Submitting CDS request for {year} …")
    for attempt in range(1, 4):
        try:
            client.retrieve("reanalysis-era5-pressure-levels", request, tmp_path)
            break
        except Exception as exc:
            log(f"  [attempt {attempt}/3] CDS error: {exc}")
            if attempt < 3:
                time.sleep(120)
            else:
                raise

    try:
        aam, lats_deg = _nc_to_aam(tmp_path, ndays)
    finally:
        os.unlink(tmp_path)

    return lats_deg, days_list, aam


# ── shared NetCDF → AAM processor ────────────────────────────────────────────
def _nc_to_aam(nc_path: str, expected_days: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a downloaded ERA5 NetCDF, return (aam shape (ndays, nlat), lats).
    """
    ds    = Dataset(nc_path)
    u_var = ds.variables["u"]                    # (time, level, lat, lon)

    lats_deg = np.array(ds.variables["latitude"][:])
    lev_key  = "pressure_level" if "pressure_level" in ds.variables else "level"
    levs_hpa = np.array(ds.variables[lev_key][:])

    # sort surface → top (descending hPa)
    lev_order = np.argsort(levs_hpa)[::-1]
    levs_pa   = levs_hpa[lev_order] * 100.0
    dp        = pressure_dp(levs_pa)              # (nlev,)

    lats_rad  = np.deg2rad(lats_deg)
    prefactor = (2.0 * np.pi / G) * np.cos(lats_rad)**2 * A**3 * OMEGA  # (nlat,)

    ntimes        = u_var.shape[0]
    nlat          = len(lats_deg)
    steps_per_day = ntimes // expected_days
    aam           = np.zeros((expected_days, nlat), dtype=np.float64)

    for t in range(ntimes):
        day_idx = t // steps_per_day
        if day_idx >= expected_days:
            break
        u_raw = np.array(u_var[t, :, :, :])
        if hasattr(u_raw, "filled"):
            u_raw = u_raw.filled(np.nan)
        u_raw    = u_raw[lev_order, :, :]
        u_zonal  = np.nanmean(u_raw, axis=2)     # (level, lat)
        vert_int = np.nansum(u_zonal * dp[:, np.newaxis], axis=0)
        aam[day_idx] += prefactor * vert_int

    aam /= steps_per_day
    ds.close()
    return aam, lats_deg


# ── climatology build ─────────────────────────────────────────────────────────
def build_climatology() -> None:
    """
    Download 1980–2010 year by year (one CDS request per year),
    compute daily mean/std AAM per latitude, smooth, save npz.
    Resumes automatically from checkpoint if a previous run was interrupted.
    """
    log(f"Building ERA5 climatology {CLIMO_START}–{CLIMO_END} …")
    lats         = None
    day_sum      = None
    day_sum2     = None
    day_count    = None
    resume_from  = CLIMO_START
    resume_month = 1

    # resume from checkpoint if one exists
    ckpt = DATA_DIR / "aam_climo_checkpoint.npz"
    if ckpt.exists():
        try:
            c         = np.load(ckpt)
            lats      = c["lats"]
            day_sum   = c["day_sum"]
            day_sum2  = c["day_sum2"]
            day_count = c["day_count"]
            if "last_year" in c.files and "last_month" in c.files:
                last_yr  = int(c["last_year"])
                last_mo  = int(c["last_month"])
                # advance to next month
                if last_mo == 12:
                    resume_from = last_yr + 1
                    resume_month = 1
                else:
                    resume_from  = last_yr
                    resume_month = last_mo + 1
                log(f"Checkpoint found — resuming from {resume_from}-{resume_month:02d}.")
            elif "last_year" in c.files:
                resume_from  = int(c["last_year"]) + 1
                resume_month = 1
                log(f"Checkpoint found — resuming from year {resume_from}.")
            else:
                log("Checkpoint found but incomplete — starting from scratch.")
                lats = day_sum = day_sum2 = day_count = None
                resume_month = 1
        except Exception as exc:
            log(f"Checkpoint unreadable ({exc}) — starting from scratch.")

    for yr in range(resume_from, CLIMO_END + 1):
        start_mo = resume_month if yr == resume_from else 1
        for mo in range(start_mo, 13):
            log(f"  {yr}-{mo:02d} …")
            try:
                mo_lats, days_list, aam_mo = fetch_month(yr, mo)
            except Exception as exc:
                log(f"  WARNING: skipping {yr}-{mo:02d}: {exc}")
                continue

            if lats is None:
                nlat      = len(mo_lats)
                lats      = mo_lats
                day_sum   = np.zeros((365, nlat), dtype=np.float64)
                day_sum2  = np.zeros((365, nlat), dtype=np.float64)
                day_count = np.zeros(365, dtype=np.int32)

            for i, d in enumerate(days_list):
                doy = doy365(d) - 1
                day_sum[doy]   += aam_mo[i]
                day_sum2[doy]  += aam_mo[i] ** 2
                day_count[doy] += 1

            # checkpoint after every month — safe against timeout at any point
            _save_climo_checkpoint(lats, day_sum, day_sum2, day_count,
                                   last_year=yr, last_month=mo)
            log(f"  Checkpoint saved after {yr}-{mo:02d}.")

    if day_count is None:
        log("ERROR: no data was collected — all months failed. Check CDS quota.", file=sys.stderr)
        sys.exit(1)

    _finalise_climatology(lats, day_sum, day_sum2, day_count)


def _save_climo_checkpoint(lats, day_sum, day_sum2, day_count,
                           last_year: int = None, last_month: int = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(lats=lats, day_sum=day_sum, day_sum2=day_sum2, day_count=day_count)
    if last_year  is not None: kwargs["last_year"]  = np.array(last_year)
    if last_month is not None: kwargs["last_month"] = np.array(last_month)
    np.savez_compressed(DATA_DIR / "aam_climo_checkpoint.npz", **kwargs)


def _finalise_climatology(lats, day_sum, day_sum2, day_count) -> None:
    safe_n     = np.where(day_count > 0, day_count, 1)
    mean_raw   = day_sum  / safe_n[:, np.newaxis]
    var_raw    = day_sum2 / safe_n[:, np.newaxis] - mean_raw**2
    std_raw    = np.sqrt(np.maximum(var_raw, 0.0))
    climo_mean = smooth_circular(mean_raw, SMOOTH_HALF)
    climo_std  = smooth_circular(std_raw,  SMOOTH_HALF)
    # Polar artifact fix: near 90N/90S, cos²φ → 0 so AAM → 0 and std is
    # near-zero, causing division to blow up. Apply a latitude-weighted floor:
    # minimum std = 2% of the global mean std across all latitudes.
    # This preserves real polar signal while suppressing division artifacts.
    global_mean_std = float(np.nanmean(climo_std))
    std_floor       = 0.02 * global_mean_std
    climo_std       = np.where(climo_std > std_floor, climo_std, std_floor)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CLIMO_NPZ,
        climo_mean=climo_mean, climo_std=climo_std, lats=lats
    )
    log(f"Climatology saved → {CLIMO_NPZ}")

    # clean up checkpoint
    ckpt = DATA_DIR / "aam_climo_checkpoint.npz"
    if ckpt.exists():
        ckpt.unlink()


def load_climatology() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CLIMO_NPZ.exists():
        log("ERROR: aam_climo.npz not found. Run the build-climo workflow first.", file=sys.stderr)
        sys.exit(1)
    try:
        data = np.load(CLIMO_NPZ)
        return data["climo_mean"], data["climo_std"], data["lats"]
    except Exception as exc:
        log(f"ERROR: could not load climatology: {exc}", file=sys.stderr)
        sys.exit(1)


# ── daily fetch ───────────────────────────────────────────────────────────────
def fetch_recent(window_days: int) -> tuple[np.ndarray, list[date], np.ndarray]:
    """Fetch the last `window_days` of ERA5, returning (lats, dates, aam)."""
    today  = date.today()
    # request extra days to absorb ERA5 ~5-day lag
    start  = today - timedelta(days=window_days + 7)

    # unique (year, month) pairs — safe iteration
    months = list(iter_months((start.year, start.month), (today.year, today.month)))

    log(f"Fetching ERA5 for last {window_days} days …")
    lats       = None
    all_dates  : list[date]       = []
    all_aam    : list[np.ndarray] = []
    cutoff     = today - timedelta(days=window_days)

    for yr, mo in months:
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
        log("ERROR: no recent ERA5 data retrieved.", file=sys.stderr)
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


# ── tendency output ──────────────────────────────────────────────────────────
def compute_tendency(dates: list[date], anom: np.ndarray) -> tuple[list[date], np.ndarray]:
    """
    Compute day-to-day tendency of the standardised AAM anomaly field.
    Tendency[i] = anom[i] - anom[i-1]  (centred on day i-0.5, assigned to day i).
    Returns (tend_dates, tend) dropping the first day which has no predecessor.
    Units: σ/day
    """
    tend       = np.diff(anom, axis=0)           # (ndays-1, nlat)
    tend_dates = dates[1:]                        # aligned to the later day
    # Clamp tendency — genuine values stay within ±2σ/day
    tend = np.clip(tend, -2.0, 2.0)
    return tend_dates, tend


def write_tendency(dates: list[date], lats: np.ndarray, tend: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relative AAM tendency by latitude — ERA5 reanalysis",
        "# Source: ECMWF Copernicus CDS — reanalysis-era5-pressure-levels",
        f"# Climatology base: {CLIMO_START}–{CLIMO_END}",
        "# Units: sigma per day (day-to-day difference of standardised AAM anomaly)",
        f"# Generated: {date.today().isoformat()}",
        "# Lats: " + " ".join(f"{lat:.2f}" for lat in lats),
    ]
    for d, row in zip(dates, tend):
        vals = " ".join(f"{v:8.4f}" for v in row)
        lines.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {vals}")
    TENDENCY_TXT.write_text("\n".join(lines) + "\n")
    log(f"Wrote {len(dates)} tendency rows → {TENDENCY_TXT}")


# ── global tendency ──────────────────────────────────────────────────────────
def compute_global_tendency(
    dates: list[date], lats: np.ndarray, tend: np.ndarray
) -> tuple[list[date], np.ndarray]:
    """
    Integrate tendency across all latitudes weighted by cos²φ·Δφ to get
    a single global tendency time series (σ/day).
    Matches the "GLOBAL TEND" panel on the AER/COREe diagnostic plots.
    """
    lats_rad = np.deg2rad(lats)
    cos2     = np.cos(lats_rad) ** 2          # (nlat,)
    dlat     = np.abs(np.gradient(lats_rad))  # (nlat,) — uniform at 2.5°
    weights  = cos2 * dlat                    # (nlat,)
    weights /= weights.sum()                  # normalise so sum = 1

    global_tend = tend @ weights              # (ndays,)
    return dates, global_tend


def write_global_tendency(dates: list[date], global_tend: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Global relative AAM tendency — ERA5 reanalysis",
        "# Source: ECMWF Copernicus CDS — reanalysis-era5-pressure-levels",
        f"# Climatology base: {CLIMO_START}–{CLIMO_END}",
        "# Units: cos²φ-weighted mean sigma/day across all latitudes",
        f"# Generated: {date.today().isoformat()}",
    ]
    for d, v in zip(dates, global_tend):
        lines.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {v:10.6f}")
    GLOBAL_TEND_TXT.write_text("\n".join(lines) + "\n")
    log(f"Wrote {len(dates)} global tendency rows → {GLOBAL_TEND_TXT}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["daily", "climo"], default="daily",
        help="'daily' fetches recent data; 'climo' builds the climatology."
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_DAYS,
        help=f"Rolling window in days for daily mode (default: {WINDOW_DAYS})."
    )
    args = parser.parse_args()

    setup_cds_auth()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "climo":
        build_climatology()
        return

    # daily mode
    climo_mean, climo_std, climo_lats = load_climatology()
    lats, dates, aam_recent           = fetch_recent(args.window)

    # verify grid matches
    if not np.allclose(climo_lats, lats):
        log("ERROR: climatology lat grid does not match ERA5 data grid.", file=sys.stderr)
        log("Re-run the build-climo workflow to rebuild aam_climo.npz.", file=sys.stderr)
        sys.exit(1)

    log("Computing standardised anomalies …")
    anom = np.empty_like(aam_recent)
    for i, d in enumerate(dates):
        doy     = doy365(d) - 1
        anom[i] = (aam_recent[i] - climo_mean[doy]) / climo_std[doy]
    # Clamp to ±4σ — genuine AAM anomalies never exceed this;
    # values beyond it are polar grid artifacts from near-zero std
    anom = np.clip(anom, -4.0, 4.0)

    write_output(dates, lats, anom)

    # Global observed AAM (cos²φ-weighted)
    lats_rad   = np.deg2rad(lats)
    cos2_w     = np.cos(lats_rad) ** 2
    dlat_w     = np.abs(np.gradient(lats_rad))
    weights_g  = cos2_w * dlat_w
    weights_g /= weights_g.sum()
    global_aam = anom @ weights_g
    g_lines    = [
        "# Global relative AAM anomaly — ERA5 reanalysis",
        "# Units: cos²φ-weighted mean sigma",
        f"# Generated: {date.today().isoformat()}",
        "# Cols: date  sigma",
    ]
    for d, v in zip(dates, global_aam):
        g_lines.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {v:10.6f}")
    GLOBAL_AAM_TXT.write_text("\n".join(g_lines) + "\n")
    log(f"Wrote global AAM → {GLOBAL_AAM_TXT}")

    log("Computing AAM tendency …")
    tend_dates, tend = compute_tendency(dates, anom)
    write_tendency(tend_dates, lats, tend)

    log("Computing global tendency …")
    gtend_dates, global_tend = compute_global_tendency(tend_dates, lats, tend)
    write_global_tendency(gtend_dates, global_tend)

    log("Done.")


if __name__ == "__main__":
    main()
