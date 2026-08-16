"""
racechrono_data.py — RaceChrono CSV data loader
================================================
Parses RaceChrono Pro CSV exports and returns Session/Lap/DataPoint objects.

RaceChrono CSV structure:
  - Lines 1-8: Metadata (title, format, session type, track, driver, created date, notes)
  - Line 9:    Column headers
  - Line 10:   Units (s, m, deg, %, Hz, etc.)
  - Line 11:   Device sources (100: gps, 200: canbus, 101: acc, 102: gyro)
  - Line 12+:  Data rows

Key advantage: explicit 'lap_number' column eliminates lap detection heuristics.

Channel mapping (substring matching, case-insensitive):
  timestamp        ← unix timestamp (seconds)
  lap_number       ← lap number
  elapsed_time     ← seconds
  latitude         ← decimal degrees
  longitude        ← decimal degrees
  speed            ← m/s (converted to km/h)
  gforce_x         ← longitudinal_acc (G units)
  gforce_y         ← lateral_acc (G units)
  gforce_z         ← z_acc (G units)
  rpm              ← rpm (CAN)
  steering_angle   ← steering_angle (CAN)
  alt              ← altitude (m)
  distance         ← distance_traveled (m)
  accelerator_pos  ← accelerator_pos (%)
  brake_pos        ← brake_pos (%)
  coolant_temp     ← coolant_temp (°C)
  engine_oil_temp  ← engine_oil_temp (°C)
  lean_angle       ← lean_angle (degrees)

Every other column becomes a DataPoint.extra entry.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_model import DataPoint, Lap, Session
from exceptions import MissingHeaderError, NoDataRowsError, CSVParseError

logger = logging.getLogger(__name__)

_INLAP_SLOWNESS_THRESHOLD = 1.5


# ── Public detection ──────────────────────────────────────────────────────────

def is_racechrono_csv(path: str) -> bool:
    """Return True if *path* is a RaceChrono CSV file."""
    if Path(path).suffix.lower() != '.csv':
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        return 'RaceChrono' in first_line
    except Exception:
        return False


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _find_col(headers: List[str], *names: str) -> Optional[int]:
    """Find the first column matching any of the given names (case-insensitive substring)."""
    for name in names:
        norm_name = name.lower().strip()
        for i, header in enumerate(headers):
            if norm_name in header.lower():
                return i
    return None


def _parse_metadata(path: str) -> Dict[str, str]:
    """Extract metadata from the first 8 lines of the RaceChrono CSV."""
    metadata = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(8):
                line = f.readline().strip()
                if ',' in line:
                    key, val = line.split(',', 1)
                    metadata[key.lower().strip()] = val.strip().strip('"')
    except Exception:
        pass
    return metadata


# ── Main loader ───────────────────────────────────────────────────────────────

def load_racechrono_csv(path: str) -> Session:
    """Load a RaceChrono CSV file and return a Session object."""
    
    # Parse metadata
    metadata = _parse_metadata(path)
    track_name = metadata.get('track name', '')
    session_type = metadata.get('session type', '')
    driver_name = metadata.get('driver name', '')
    
    # Read the full file to extract headers and data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = f.readlines()
    
    # Headers are on line 9 (0-indexed), so skip lines 0-8
    # Line 9 is the header, lines 10-11 are units/device info
    # Data starts at line 12
    if len(all_lines) < 12:
        raise NoDataRowsError(f"File too short, expected at least 12 lines: {path}")
    
    header_line = all_lines[9].strip()
    headers = [h.strip() for h in header_line.split(',')]
    
    # Parse data rows (starting from line 12, 0-indexed)
    all_rows = []
    reader = csv.reader(all_lines[12:])
    for row in reader:
        all_rows.append(row)
    
    if not all_rows:
        raise NoDataRowsError(f"No data rows in {path}")
    
    # Find column indices
    idx_time = _find_col(headers, 'timestamp')
    idx_lat = _find_col(headers, 'latitude')
    idx_lon = _find_col(headers, 'longitude')
    idx_lap = _find_col(headers, 'lap_number')
    idx_speed = _find_col(headers, 'speed')
    idx_elapsed = _find_col(headers, 'elapsed_time')
    idx_distance = _find_col(headers, 'distance_traveled')
    idx_alt = _find_col(headers, 'altitude')
    idx_lat_g = _find_col(headers, 'lateral_acc')
    idx_lon_g = _find_col(headers, 'longitudinal_acc')
    idx_vert_g = _find_col(headers, 'z_acc')
    idx_rpm = _find_col(headers, 'rpm')
    idx_steering = _find_col(headers, 'steering_angle')
    idx_accel = _find_col(headers, 'accelerator_pos')
    idx_brake = _find_col(headers, 'brake_pos')
    idx_coolant = _find_col(headers, 'coolant_temp')
    idx_oil = _find_col(headers, 'engine_oil_temp')
    idx_lean = _find_col(headers, 'lean_angle')
    
    if idx_time is None or idx_lat is None or idx_lon is None:
        raise MissingHeaderError(f"Missing required columns (timestamp/latitude/longitude) in {path}")
    
    if idx_lap is None:
        raise MissingHeaderError(f"Missing lap_number column in {path}")
    
    # Extra channels
    consumed_idx = {
        idx_time, idx_lat, idx_lon, idx_lap, idx_speed, idx_elapsed, idx_distance,
        idx_alt, idx_lat_g, idx_lon_g, idx_vert_g, idx_rpm, idx_steering, idx_accel,
        idx_brake, idx_coolant, idx_oil, idx_lean
    }
    consumed_idx = {i for i in consumed_idx if i is not None}
    
    extra_names: List[str] = []
    extra_channel_meta: Dict[str, dict] = {}
    for i, header in enumerate(headers):
        if i in consumed_idx or not header.strip():
            continue
        name = header.strip()
        extra_names.append(name)
        extra_channel_meta[name] = {'label': name, 'unit': ''}
    
    # Parse data rows
    all_pts: List[DataPoint] = []
    
    for record_idx, row in enumerate(all_rows):
        if not row or len(row) <= max(c for c in [idx_time, idx_lat, idx_lon] if c is not None):
            continue
        
        def _col(idx: Optional[int], default: float = 0.0) -> float:
            if idx is None or idx >= len(row):
                return default
            try:
                return float(row[idx])
            except (ValueError, IndexError):
                return default
        
        try:
            timestamp = _col(idx_time)
            if timestamp == 0.0:
                continue
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            logger.warning(f"Skipping row {record_idx} with invalid timestamp {row[idx_time] if idx_time and idx_time < len(row) else 'N/A'}")
            continue
        
        lat = _col(idx_lat)
        lon = _col(idx_lon)
        speed_ms = _col(idx_speed)
        speed_kmh = speed_ms * 3.6  # m/s → km/h
        elapsed = _col(idx_elapsed)
        distance = _col(idx_distance)
        alt = _col(idx_alt)
        lat_g = _col(idx_lat_g)
        lon_g = _col(idx_lon_g)
        vert_g = _col(idx_vert_g)
        rpm = _col(idx_rpm)
        steering = _col(idx_steering)
        accel_pos = _col(idx_accel)
        brake_pos = _col(idx_brake)
        coolant = _col(idx_coolant)
        oil = _col(idx_oil)
        lean = _col(idx_lean)
        # Empty lap_number means pre-lap data (warmup/calibration); default to 0
        lap_num = int(_col(idx_lap, 0.0))
        
        extra = {}
        for name in extra_names:
            idx = next((i for i, h in enumerate(headers) if h.strip() == name), None)
            if idx is None or idx >= len(row):
                extra[name] = 0.0
            else:
                try:
                    extra[name] = float(row[idx])
                except ValueError:
                    extra[name] = 0.0
        
        all_pts.append(DataPoint(
            record=record_idx,
            time=dt,
            lat=lat,
            lon=lon,
            alt=alt,
            speed=speed_kmh,
            gforce_x=lon_g,
            gforce_y=lat_g,
            gforce_z=vert_g,
            lap=lap_num,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            rpm=rpm,
            extra=extra,
        ))
    
    if not all_pts:
        raise NoDataRowsError(f"No valid data rows parsed from {path}")
    
    # ── Elapsed times ─────────────────────────────────────────────────────────
    
    if all_pts:
        t0 = all_pts[0].time
        for pt in all_pts:
            pt.elapsed = (pt.time - t0).total_seconds()
    
    # ── Build laps ────────────────────────────────────────────────────────────
    
    buckets: Dict[int, List[DataPoint]] = defaultdict(list)
    for pt in all_pts:
        buckets[pt.lap].append(pt)
    
    laps: List[Lap] = []
    for lap_num in sorted(buckets.keys()):
        pts = buckets[lap_num]
        if not pts:
            continue
        lap_t0 = pts[0].time
        for pt in pts:
            pt.lap_elapsed = (pt.time - lap_t0).total_seconds()
        dur = (pts[-1].time - pts[0].time).total_seconds()
        laps.append(Lap(lap_num=lap_num, points=pts, duration=dur, is_outlap=(lap_num == 0)))
    
    timed = [l for l in laps if l.lap_num > 0]
    if len(timed) >= 3:
        med = sorted(l.duration for l in timed)[len(timed) // 2]
        if timed[-1].duration > med * _INLAP_SLOWNESS_THRESHOLD:
            timed[-1].is_inlap = True
    
    best_lap_time = min((l.duration for l in timed), default=0.0)
    date_str = all_pts[0].time.strftime('%Y-%m-%dT%H:%M:%SZ') if all_pts else ''
    
    return Session(
        source='RaceChrono',
        date_utc=date_str,
        track=track_name,
        configuration='',
        session_type=session_type,
        best_lap_time=best_lap_time,
        all_points=all_pts,
        laps=laps,
        is_bike=False,
        csv_path=path,
        source_speed_unit='kmh',
        extra_channel_meta=extra_channel_meta,
    )
