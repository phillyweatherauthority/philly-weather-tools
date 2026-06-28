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
HOURS   : f024, f048, f072, f096, f120, f144, f168  (day 1–7, 00Z each day)

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
import struct
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

# GEFS forecast hours — one per day for 7 days
FCST_HOURS = [24, 48, 72, 96, 120, 144, 168]

# Number of ensemble members (control + 30 perturbed)
N_MEMBERS = 31

# AWS S3 base — primary source, no auth
AWS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# NOMADS fallback
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"

# Paths
REPO_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR        = REPO_ROOT / "data"
CLIMO_NPZ       = DATA_DIR / "aam_climo.npz"
FCST_MEAN_TXT   = DATA_DIR / "aam_fcst_mean_latest.txt"
FCST_STD_TXT    = DATA_DIR / "aam_fcst_std_latest.txt"

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


# ── find latest GEFS cycle ────────────────────────────────────────────────────
def latest_gefs_cycle() -> tuple[str, str]:
    """
    Return (date_str YYYYMMDD, cycle_str HH) for the most recent
    GEFS run available on AWS (~6hr lag after init time).
    Checks 00Z of today then yesterday as fallback.
    """
    now = datetime.now(timezone.utc)
    candidates = []
    for delta in range(3):
        d = now - timedelta(days=delta)
        for hh in [18, 12, 6, 0]:
            run_time = d.replace(hour=hh, minute=0, second=0, microsecond=0)
            lag = (now - run_time).total_seconds() / 3600
            if lag >= 6:   # need at least 6hr after init for f168 to exist
                candidates.append((d.strftime("%Y%m%d"), f"{hh:02d}"))

    for date_str, cycle in candidates:
        # verify the control member f024 exists
        url = (f"{AWS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/"
               f"gec00.t{cycle}z.pgrb2a.0p50.f024.idx")
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
    """
    Parse a GEFS .idx file and return byte ranges for UGRD at requested levels.
    Index line format:  recnum:byte_offset:date:var:level:fcst_type:...
    """
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
            # end byte = start of next record - 1
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            ranges.append((start, end))

    return ranges


def fetch_grib_bytes(url: str, byte_start: int, byte_end) -> bytes:
    """Fetch a byte range from a GRIB2 file."""
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
def extract_ugrd_from_grib(grib_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse concatenated GRIB2 messages containing UGRD at multiple levels.
    Returns (lats, lons, u_array shape (nlev, nlat, nlon)).
    Uses cfgrib via a temp file.
    """
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp = f.name

    try:
        # cfgrib.open_datasets (plural) handles multiple GRIB message types
        import cfgrib as _cfgrib
        ds_list = _cfgrib.open_datasets(tmp, backend_kwargs={"indexpath": ""})
        # Find the dataset containing u-wind on pressure levels
        ds = None
        for d in ds_list:
            if "u" in d and "isobaricInhPa" in d.coords:
                ds = d
                break
        if ds is None:
            # Last resort: open single dataset directly
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

        # Get actual pressure levels present in this file
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
    """
    Compute relative AAM per latitude band from u-wind array.
    u_array shape: (nlev, nlat, nlon)
    Returns aam shape: (nlat,)
    """
    lev_order = np.argsort(levels_pa)[::-1]   # surface → top
    levs_sort = levels_pa[lev_order]
    u_sort    = u_array[lev_order, :, :]
    dp        = pressure_dp(levs_sort)

    lats_rad  = np.deg2rad(lats)
    prefactor = (2.0 * np.pi / G) * np.cos(lats_rad)**2 * A**3 * OMEGA

    u_zonal  = np.nanmean(u_sort, axis=2)                          # (nlev, nlat)
    vert_int = np.nansum(u_zonal * dp[:, np.newaxis], axis=0)      # (nlat,)
    return prefactor * vert_int


# ── fetch one member × one forecast hour ─────────────────────────────────────
def fetch_member_fxx(date_str: str, cycle: str,
                     member: int, fxx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fetch u-wind GRIB2 for one ensemble member at one forecast hour.
    member 0 = control (gec00), 1-30 = perturbed (gep01-gep30).
    Returns (lats, aam shape (nlat,)) or None on failure.
    """
    mem_str  = "gec00" if member == 0 else f"gep{member:02d}"
    fxx_str  = f"f{fxx:03d}"
    fname    = f"{mem_str}.t{cycle}z.pgrb2a.0p50.{fxx_str}"
    base_url = f"{AWS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/{fname}"
    idx_url  = base_url + ".idx"

    try:
        idx_resp = requests.get(idx_url, timeout=10)
        if idx_resp.status_code != 200:
            # try NOMADS fallback
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

        # Fetch all level byte ranges and concatenate
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


# ── output ────────────────────────────────────────────────────────────────────
def write_fcst_files(fcst_dates: list[date], lats: np.ndarray,
                     mean_anom: np.ndarray, std_anom: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "# GFS Ensemble (GEFS) relative AAM forecast anomaly",
        "# Source: NOAA GEFS via AWS S3 open data",
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


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    climo_mean, climo_std, climo_lats = load_climatology()

    date_str, cycle = latest_gefs_cycle()
    cycle_dt = datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")

    log(f"Fetching GEFS {date_str} {cycle}Z — {N_MEMBERS} members × {len(FCST_HOURS)} days")

    lats_ref    = None
    # all_aam[fxx_idx][member] = aam array (nlat,)
    all_aam = {fxx: [] for fxx in FCST_HOURS}

    for fxx in FCST_HOURS:
        log(f"\nForecast hour f{fxx:03d} …")
        for mem in range(N_MEMBERS):
            result = fetch_member_fxx(date_str, cycle, mem, fxx)
            if result is None:
                continue
            lats, aam = result
            if lats_ref is None:
                lats_ref = lats
            all_aam[fxx].append(aam)
        log(f"  Got {len(all_aam[fxx])}/{N_MEMBERS} members")

    if lats_ref is None:
        print("ERROR: no forecast data retrieved.", file=sys.stderr)
        sys.exit(1)

    # Verify lat grid matches climatology
    if not np.allclose(climo_lats, lats_ref, atol=0.1):
        # interpolate if grids differ slightly
        log("WARNING: lat grid mismatch — interpolating to climo grid")
        lats_ref_new = climo_lats
        for fxx in FCST_HOURS:
            all_aam[fxx] = [np.interp(climo_lats, lats_ref, a) for a in all_aam[fxx]]
        lats_ref = lats_ref_new

    # Compute ensemble mean and std, convert to σ anomalies
    fcst_dates  = []
    mean_anom   = []
    std_anom    = []

    for fxx in FCST_HOURS:
        members = all_aam[fxx]
        if not members:
            log(f"  Skipping f{fxx:03d} — no members available")
            continue

        fcst_dt   = cycle_dt + timedelta(hours=fxx)
        fcst_date = fcst_dt.date()
        doy       = doy365(fcst_date) - 1

        stack     = np.array(members)                  # (n_mem, nlat)
        ens_mean  = stack.mean(axis=0)                 # (nlat,)
        ens_std   = stack.std(axis=0, ddof=1)          # (nlat,)

        mean_sig  = (ens_mean - climo_mean[doy]) / climo_std[doy]
        # std stays in physical units relative to climo std
        std_sig   = ens_std / climo_std[doy]

        mean_sig  = np.clip(mean_sig, -4.0, 4.0)
        std_sig   = np.clip(std_sig,  0.0,  2.0)

        fcst_dates.append(fcst_date)
        mean_anom.append(mean_sig)
        std_anom.append(std_sig)
        log(f"  f{fxx:03d} → {fcst_date} | mean σ range: "
            f"{mean_sig.min():.2f} to {mean_sig.max():.2f}")

    write_fcst_files(fcst_dates, lats_ref,
                     np.array(mean_anom), np.array(std_anom))
    log("\nDone.")


if __name__ == "__main__":
    main()
