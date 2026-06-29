#!/usr/bin/env python3
"""
fetch_gfs_aam.py
────────────────────────────────────────────────────────────────────────────────
Fetches GFS Ensemble (GEFS) u-wind forecasts and computes relative AAM anomaly
by latitude for the 7-day forecast period.

SOURCE  : NOAA GEFS via AWS S3 open data bucket (no auth required)
          s3://noaa-gefs-pds/gefs.YYYYMMDD/HH/atmos/pgrb2ap5/
          Fallback: NOMADS NCEP if AWS unavailable

MEMBERS : gep01–gep30 (30 perturbed members) + gec00 (control) = 31 total
LEVELS  : Same 15 pressure levels as ERA5 pipeline (1000–10 hPa)
HOURS   : 4 six-hourly snapshots per forecast day averaged to daily mean.
          Day 1 = f006+f012+f018+f024, Day 2 = f030+…+f048, … Day 7 = f150+…+f168

BIAS CORRECTION:
          After computing σ anomalies, a linear ramp correction is applied
          to eliminate the seam discontinuity between ERA5 and GEFS.
          The offset at day 1 is (last ERA5 anomaly − GEFS day-1 anomaly)
          per latitude, decaying linearly to zero by day 7.
          This anchors the forecast to the observed state without distorting
          the forecast evolution signal.

OUTPUT  : data/aam_fcst_mean_latest.txt  — ensemble mean AAM anomaly (σ)
          data/aam_fcst_std_latest.txt   — ensemble std dev (σ)

CLIMO   : Loads data/aam_climo.npz (built by build_aam_climo workflow)
          to express forecasts as σ anomalies on the same baseline as ERA5.

DEPENDENCIES: requests, numpy, netCDF4, cfgrib, eccodes (system)
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import tempfile
import requests
import numpy as np
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
OMEGA = 7.292115e-5
A     = 6.371e6
G     = 9.80665

# Same 15 pressure levels as ERA5 pipeline (hPa)
LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 20, 10]

# 7 forecast days; each day uses 4 six-hourly snapshots averaged to daily mean.
# Day N covers hours (N-1)*24+6 through N*24 in 6-hr steps.
N_FCST_DAYS   = 7
HOURS_PER_DAY = [6, 12, 18, 24]

def day_hours(day: int) -> list[int]:
    """Return the 4 forecast hours for day N (1-based)."""
    base = (day - 1) * 24
    return [base + h for h in HOURS_PER_DAY]

# Number of ensemble members (control + 30 perturbed)
N_MEMBERS = 31

# AWS S3 base — primary source, no auth
AWS_BASE    = "https://noaa-gefs-pds.s3.amazonaws.com"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"

# Paths
REPO_ROOT            = Path(__file__).resolve().parent.parent
DATA_DIR             = REPO_ROOT / "data"
CLIMO_NPZ            = DATA_DIR / "aam_climo.npz"
ERA5_TXT             = DATA_DIR / "aam_lat_latest.txt"
FCST_MEAN_TXT        = DATA_DIR / "aam_fcst_mean_latest.txt"
FCST_STD_TXT         = DATA_DIR / "aam_fcst_std_latest.txt"
FCST_GLOBAL_TXT      = DATA_DIR / "aam_fcst_global_latest.txt"
FCST_TEND_MEAN_TXT   = DATA_DIR / "aam_fcst_tend_mean_latest.txt"
FCST_TEND_STD_TXT    = DATA_DIR / "aam_fcst_tend_std_latest.txt"
FCST_GLOBAL_TEND_TXT = DATA_DIR / "aam_fcst_global_tend_latest.txt"

GRID_RES = 0.5   # GEFS 0.5° grid


# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def doy365(d: date) -> int:
    return min(d.timetuple().tm_yday, 365)


def pressure_dp(levels_pa: np.ndarray) -> np.ndarray:
    n  = len(levels_pa)
    dp = np.empty(n)
    for i in range(n):
        lo = levels_pa[i + 1] if i + 1 < n else levels_pa[i]
        hi = levels_pa[i - 1] if i > 0      else levels_pa[i]
        dp[i] = abs(hi - lo) / 2.0
    return dp


# ── ERA5 anchor — read last row of aam_lat_latest.txt ────────────────────────
def load_era5_last_row() -> np.ndarray | None:
    """
    Read the most recent ERA5 anomaly row from aam_lat_latest.txt.
    Returns (nlat,) array or None if file unavailable.
    """
    if not ERA5_TXT.exists():
        log("WARNING: ERA5 file not found — bias correction disabled.")
        return None
    try:
        last_row = None
        with open(ERA5_TXT) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                last_row = np.array([float(v) for v in parts[1:]])
        if last_row is None:
            log("WARNING: ERA5 file empty — bias correction disabled.")
        return last_row
    except Exception as exc:
        log(f"WARNING: could not read ERA5 file ({exc}) — bias correction disabled.")
        return None


# ── find latest GEFS cycle ────────────────────────────────────────────────────
def latest_gefs_cycle() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    candidates = []
    for delta in range(3):
        d = now - timedelta(days=delta)
        for hh in [18, 12, 6, 0]:
            run_time = d.replace(hour=hh, minute=0, second=0, microsecond=0)
            lag = (now - run_time).total_seconds() / 3600
            if lag >= 6:
                candidates.append((d.strftime("%Y%m%d"), f"{hh:02d}"))

    for date_str, cycle in candidates:
        url = (f"{AWS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/"
               f"gec00.t{cycle}z.pgrb2a.0p50.f006.idx")
        try:
            r = requests.head(url, timeout=8)
            if r.status_code == 200:
                log(f"Using GEFS cycle: {date_str} {cycle}Z")
                return date_str, cycle
        except Exception:
            continue

    raise RuntimeError("Could not find a recent GEFS cycle on AWS.")


# ── GRIB2 byte-range fetch ────────────────────────────────────────────────────
def parse_idx(idx_text: str, levels_hpa: list[int]) -> list[tuple[int, int]]:
    ranges = []
    lines  = idx_text.strip().split("\n")
    target_levels = {f"{l} mb" for l in levels_hpa}
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        var   = parts[3].strip()
        level = parts[4].strip()
        if var == "UGRD" and level in target_levels:
            start = int(parts[1])
            end   = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            ranges.append((start, end))
    return ranges


def fetch_grib_bytes(url: str, byte_start: int, byte_end) -> bytes:
    range_hdr = f"bytes={byte_start}-{byte_end}" if byte_end != "" else f"bytes={byte_start}-"
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers={"Range": range_hdr}, timeout=30)
            if r.status_code in (200, 206):
                return r.content
        except Exception as exc:
            log(f"    [attempt {attempt}/3] fetch error: {exc}")
            if attempt < 3:
                time.sleep(10)
    raise RuntimeError(f"Failed to fetch {url} bytes {byte_start}-{byte_end}")


# ── GRIB2 u-wind extraction ───────────────────────────────────────────────────
def extract_ugrd_from_grib(grib_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp = f.name

    try:
        import cfgrib as _cfgrib
        ds_list = _cfgrib.open_datasets(tmp, backend_kwargs={"indexpath": ""})
        ds = None
        for d in ds_list:
            if "u" in d and "isobaricInhPa" in d.coords:
                ds = d
                break
        if ds is None:
            ds = xr.open_dataset(
                tmp,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
                filter_by_keys={"typeOfLevel": "isobaricInhPa", "shortName": "u"},
            )

        u    = ds["u"].values
        lats = ds.coords["latitude"].values
        lons = ds.coords["longitude"].values
        if u.ndim == 2:
            u = u[np.newaxis, :, :]

        if "isobaricInhPa" in ds.coords:
            actual_levels_hpa = ds.coords["isobaricInhPa"].values
        elif "pressure_level" in ds.coords:
            actual_levels_hpa = ds.coords["pressure_level"].values
        else:
            actual_levels_hpa = np.array(LEVELS_HPA[:u.shape[0]])

        return lats, lons, u, actual_levels_hpa
    finally:
        os.unlink(tmp)
        for ext in [".idx", ".923a8.idx"]:
            p = tmp + ext
            if os.path.exists(p):
                os.unlink(p)


# ── AAM computation ───────────────────────────────────────────────────────────
def aam_from_ugrd(lats: np.ndarray, u_array: np.ndarray,
                  levels_pa: np.ndarray) -> np.ndarray:
    lev_order = np.argsort(levels_pa)[::-1]
    levs_sort = levels_pa[lev_order]
    u_sort    = u_array[lev_order, :, :]
    dp        = pressure_dp(levs_sort)
    lats_rad  = np.deg2rad(lats)
    prefactor = (2.0 * np.pi / G) * np.cos(lats_rad)**2 * A**3 * OMEGA
    u_zonal   = np.nanmean(u_sort, axis=2)
    vert_int  = np.nansum(u_zonal * dp[:, np.newaxis], axis=0)
    return prefactor * vert_int


# ── fetch one member × one forecast hour ─────────────────────────────────────
def fetch_member_fxx(date_str: str, cycle: str,
                     member: int, fxx: int) -> tuple[np.ndarray, np.ndarray] | None:
    mem_str  = "gec00" if member == 0 else f"gep{member:02d}"
    fxx_str  = f"f{fxx:03d}"
    fname    = f"{mem_str}.t{cycle}z.pgrb2a.0p50.{fxx_str}"
    base_url = f"{AWS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/{fname}"
    idx_url  = base_url + ".idx"

    try:
        idx_resp = requests.get(idx_url, timeout=10)
        if idx_resp.status_code != 200:
            base_url = (f"{NOMADS_BASE}/gefs.{date_str}/{cycle}/atmos/"
                        f"pgrb2ap5/{fname}")
            idx_url  = base_url + ".idx"
            idx_resp = requests.get(idx_url, timeout=10)
            if idx_resp.status_code != 200:
                log(f"    Cannot find index for {mem_str} f{fxx:03d}")
                return None

        byte_ranges = parse_idx(idx_resp.text, LEVELS_HPA)
        if not byte_ranges:
            log(f"    No UGRD levels found in index for {mem_str} f{fxx:03d}")
            return None

        grib_chunks = []
        for start, end in byte_ranges:
            chunk = fetch_grib_bytes(base_url, start, end)
            grib_chunks.append(chunk)

        grib_bytes = b"".join(grib_chunks)
        lats, _, u_array, actual_levels_hpa = extract_ugrd_from_grib(grib_bytes)
        levels_pa = actual_levels_hpa.astype(np.float64) * 100.0
        aam = aam_from_ugrd(lats, u_array, levels_pa)
        return lats, aam

    except Exception as exc:
        log(f"    WARNING: {mem_str} f{fxx:03d} failed: {exc}")
        return None


# ── climatology ───────────────────────────────────────────────────────────────
def load_climatology() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CLIMO_NPZ.exists():
        print("ERROR: aam_climo.npz not found. Run build_aam_climo workflow first.",
              file=sys.stderr)
        sys.exit(1)
    data = np.load(CLIMO_NPZ)
    return data["climo_mean"], data["climo_std"], data["lats"]


# ── grid interpolation ────────────────────────────────────────────────────────
def interp_to_climo_grid(lats_gefs: np.ndarray, aam: np.ndarray,
                         climo_lats: np.ndarray) -> np.ndarray:
    sort_idx      = np.argsort(lats_gefs)
    lats_asc      = lats_gefs[sort_idx]
    aam_asc       = aam[sort_idx]
    climo_asc_idx = np.argsort(climo_lats)
    climo_asc     = climo_lats[climo_asc_idx]
    interp_asc    = np.interp(climo_asc, lats_asc, aam_asc)
    result        = np.empty_like(interp_asc)
    result[climo_asc_idx] = interp_asc
    return result


# ── bias correction ───────────────────────────────────────────────────────────
def compute_bias_ramp(era5_last: np.ndarray, gefs_day1: np.ndarray,
                      n_days: int) -> np.ndarray:
    """
    Compute a per-latitude linear ramp correction.

    At day 1: correction = era5_last - gefs_day1  (closes the seam)
    At day 7: correction = 0                       (forecast free to evolve)

    Returns shape (n_days, nlat).
    """
    offset = era5_last - gefs_day1          # (nlat,) — seam gap at day 1
    ramp   = np.linspace(1.0, 0.0, n_days)  # 1.0 → 0.0 over forecast period
    return ramp[:, np.newaxis] * offset[np.newaxis, :]  # (n_days, nlat)


# ── output ────────────────────────────────────────────────────────────────────
def cos2_weights(lats: np.ndarray) -> np.ndarray:
    lats_rad = np.deg2rad(lats)
    cos2     = np.cos(lats_rad) ** 2
    dlat     = np.abs(np.gradient(lats_rad))
    w        = cos2 * dlat
    return w / w.sum()


def global_integral(anom: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return anom @ weights


def fcst_tendency(mean_anom: np.ndarray) -> np.ndarray:
    return np.diff(mean_anom, axis=0)


def write_fcst_files(fcst_dates: list[date], lats: np.ndarray,
                     mean_anom: np.ndarray, std_anom: np.ndarray,
                     bias_corrected: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bc_note = "linear ramp bias correction applied (day1=ERA5-anchored, day7=free)" \
              if bias_corrected else "no bias correction (ERA5 anchor unavailable)"
    header = [
        "# GFS Ensemble (GEFS) relative AAM forecast anomaly",
        "# Source: NOAA GEFS via AWS S3 open data",
        "# Averaging: 4 six-hourly snapshots per day (f006/f012/f018/f024 pattern)",
        f"# Bias correction: {bc_note}",
        "# Units: standardised sigma departures from ERA5 1980–2010 climatology",
        f"# Generated: {date.today().isoformat()}",
        "# Lats: " + " ".join(f"{lat:.2f}" for lat in lats),
    ]

    for txt_path, arr, label in [
        (FCST_MEAN_TXT, mean_anom, "Ensemble mean"),
        (FCST_STD_TXT,  std_anom,  "Ensemble std dev"),
    ]:
        lines = header[:] + [f"# Field: {label}"]
        for d, row in zip(fcst_dates, arr):
            vals = " ".join(f"{v:8.4f}" for v in row)
            lines.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {vals}")
        txt_path.write_text("\n".join(lines) + "\n")
        log(f"Wrote {len(fcst_dates)} rows → {txt_path}")

    # Global AAM forecast
    weights     = cos2_weights(lats)
    global_mean = global_integral(mean_anom, weights)
    global_std  = global_integral(std_anom,  weights)

    lines_g = [
        "# GEFS global relative AAM forecast",
        "# Units: cos²φ-weighted mean sigma / std sigma",
        f"# Generated: {date.today().isoformat()}",
        "# Cols: date  mean_sigma  std_sigma",
    ]
    for d, m, s in zip(fcst_dates, global_mean, global_std):
        lines_g.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {m:10.6f}  {s:10.6f}")
    FCST_GLOBAL_TXT.write_text("\n".join(lines_g) + "\n")
    log(f"Wrote global AAM forecast → {FCST_GLOBAL_TXT}")

    # Tendency forecast
    if len(fcst_dates) > 1:
        tend_mean  = fcst_tendency(mean_anom)
        tend_std   = fcst_tendency(std_anom)
        tend_dates = fcst_dates[1:]
        tend_mean  = np.clip(tend_mean, -2.0, 2.0)
        tend_std   = np.clip(np.abs(tend_std), 0.0, 1.0)

        header_t = [
            "# GEFS relative AAM tendency forecast by latitude",
            "# Units: sigma/day",
            f"# Generated: {date.today().isoformat()}",
            "# Lats: " + " ".join(f"{lat:.2f}" for lat in lats),
        ]
        for txt_path, arr in [(FCST_TEND_MEAN_TXT, tend_mean),
                              (FCST_TEND_STD_TXT,  tend_std)]:
            lines_t = header_t[:]
            for d, row in zip(tend_dates, arr):
                vals = " ".join(f"{v:8.4f}" for v in row)
                lines_t.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {vals}")
            txt_path.write_text("\n".join(lines_t) + "\n")
            log(f"Wrote tendency forecast → {txt_path}")

        gtend_mean = global_integral(tend_mean, weights)
        gtend_std  = global_integral(tend_std,  weights)
        lines_gt = [
            "# GEFS global relative AAM tendency forecast",
            "# Units: sigma/day",
            f"# Generated: {date.today().isoformat()}",
            "# Cols: date  mean_sigma  std_sigma",
        ]
        for d, m, s in zip(tend_dates, gtend_mean, gtend_std):
            lines_gt.append(f"{d.year:04d}.{d.month:02d}.{d.day:02d}  {m:10.6f}  {s:10.6f}")
        FCST_GLOBAL_TEND_TXT.write_text("\n".join(lines_gt) + "\n")
        log(f"Wrote global tendency forecast → {FCST_GLOBAL_TEND_TXT}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    climo_mean, climo_std, climo_lats = load_climatology()

    # Load ERA5 last row now — used for bias correction after anomalies computed
    era5_last = load_era5_last_row()

    date_str, cycle = latest_gefs_cycle()
    cycle_dt = datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")

    log(f"Fetching GEFS {date_str} {cycle}Z — {N_MEMBERS} members × {N_FCST_DAYS} days "
        f"× {len(HOURS_PER_DAY)} snapshots = {N_MEMBERS * N_FCST_DAYS * len(HOURS_PER_DAY)} fetches")

    lats_ref = None

    day_member_snaps: dict[int, dict[int, list[np.ndarray]]] = {
        d: {m: [] for m in range(N_MEMBERS)} for d in range(N_FCST_DAYS)
    }

    for day_idx in range(N_FCST_DAYS):
        hours = day_hours(day_idx + 1)
        log(f"\nForecast day {day_idx + 1} — hours {hours} …")
        for fxx in hours:
            log(f"  f{fxx:03d} …")
            for mem in range(N_MEMBERS):
                result = fetch_member_fxx(date_str, cycle, mem, fxx)
                if result is None:
                    continue
                lats, aam = result
                if lats_ref is None:
                    lats_ref = lats
                aam_interp = interp_to_climo_grid(lats, aam, climo_lats)
                day_member_snaps[day_idx][mem].append(aam_interp)

        snap_counts = [len(day_member_snaps[day_idx][m]) for m in range(N_MEMBERS)]
        log(f"  Day {day_idx + 1}: members with ≥1 snapshot: "
            f"{sum(1 for c in snap_counts if c > 0)}/{N_MEMBERS}, "
            f"avg snapshots/member: {np.mean(snap_counts):.1f}")

    if lats_ref is None:
        print("ERROR: no forecast data retrieved.", file=sys.stderr)
        sys.exit(1)

    lats_out = climo_lats

    # Compute daily mean per member, then ensemble stats, then σ anomaly
    fcst_dates = []
    mean_anom  = []
    std_anom   = []

    for day_idx in range(N_FCST_DAYS):
        fxx_mid   = day_idx * 24 + 24
        fcst_dt   = cycle_dt + timedelta(hours=fxx_mid)
        fcst_date = fcst_dt.date()
        doy       = doy365(fcst_date) - 1

        member_daily = []
        for mem in range(N_MEMBERS):
            snaps = day_member_snaps[day_idx][mem]
            if not snaps:
                continue
            member_daily.append(np.mean(snaps, axis=0))

        if not member_daily:
            log(f"  Day {day_idx + 1}: no members — skipping")
            continue

        stack    = np.array(member_daily)
        ens_mean = stack.mean(axis=0)
        ens_std  = stack.std(axis=0, ddof=1)

        mean_sig = (ens_mean - climo_mean[doy]) / climo_std[doy]
        std_sig  = ens_std / climo_std[doy]

        mean_sig = np.clip(mean_sig, -4.0, 4.0)
        std_sig  = np.clip(std_sig,   0.0,  2.0)

        fcst_dates.append(fcst_date)
        mean_anom.append(mean_sig)
        std_anom.append(std_sig)
        log(f"  Day {day_idx + 1} → {fcst_date} (DOY {doy+1}) | "
            f"n_members={len(member_daily)} | "
            f"mean σ range: {mean_sig.min():.2f} to {mean_sig.max():.2f}")

    mean_anom = np.array(mean_anom)   # (n_days, nlat)
    std_anom  = np.array(std_anom)

    # ── bias correction ───────────────────────────────────────────────────────
    bias_corrected = False
    if era5_last is not None and len(mean_anom) > 0:
        if len(era5_last) == mean_anom.shape[1]:
            log("\nApplying ERA5-anchor bias correction …")
            ramp = compute_bias_ramp(era5_last, mean_anom[0], len(fcst_dates))
            log(f"  Seam offset — min: {(era5_last - mean_anom[0]).min():.3f}σ  "
                f"max: {(era5_last - mean_anom[0]).max():.3f}σ  "
                f"global mean: {(era5_last - mean_anom[0]).mean():.3f}σ")
            mean_anom     = mean_anom + ramp
            mean_anom     = np.clip(mean_anom, -4.0, 4.0)
            bias_corrected = True
            log("  Bias correction applied.")
        else:
            log(f"WARNING: ERA5 lat count ({len(era5_last)}) != GEFS lat count "
                f"({mean_anom.shape[1]}) — bias correction skipped.")

    write_fcst_files(fcst_dates, lats_out, mean_anom, std_anom, bias_corrected)
    log("\nDone.")


if __name__ == "__main__":
    main()
