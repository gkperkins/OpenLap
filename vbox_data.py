"""
vbox_data.py — Racelogic VBOX .vbo loader
==========================================
Parses the text-based VBOX format.  Files are divided into named sections
([header], [channel units], [channel names], [comments], [data]) and data
rows are whitespace-delimited.

Coordinate format : DDMM.MMMMM (degrees + decimal minutes) → decimal degrees
Time format       : HHMMSS.SS combined with date from [comments]
Speed             : 'velocity kmh' in km/h; bare 'velocity' assumed knots
G-forces          : G units — lateral-acc → gforce_y, longitudinal-acc → gforce_x
Lap detection     : 'lap trigger' channel counter if present, else single lap (1)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_model import DataPoint, Lap, Session
from exceptions import MissingHeaderError, NoDataRowsError, CSVParseError

logger = logging.getLogger(__name__)

_INLAP_SLOWNESS_THRESHOLD = 1.5
_VBOX_LAP_CROSSING_MIN_GAP_SECONDS = 80.0


# ── Public detection ──────────────────────────────────────────────────────────

def is_vbox(path: str) -> bool:
    """Return True if *path* is a Racelogic VBOX text file."""
    if Path(path).suffix.lower() != '.vbo':
        return False
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            head = f.read(512)
        return '[header]' in head.lower()
    except Exception:
        return False


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_sections(path: str) -> Dict[str, List[str]]:
    """Split a .vbo file into named sections; skip blank lines within sections."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1].strip().lower()
                sections[current] = []
            elif current is not None and line.strip():
                sections[current].append(line)
    return sections


def _parse_date_from_comments(comments: str) -> Optional[datetime]:
    """Extract the session date from the [comments] section text."""
    # "File created on DD/MM/YYYY at HH:MM:SS by VBOX …"
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', comments)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(year, month, day, tzinfo=timezone.utc)
    return None


def _parse_vbox_start_line(line: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Parse a [laptiming] "Start" line into two GPS points.

    The VBOX file records the start/finish line as two coordinates, typically
    longitude/latitude pairs in the logger's minute convention. We only need the
    endpoint pair to detect crossings of that line over time.
    """
    m = re.search(
        r'^\s*Start\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+'
        r'([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)',
        line,
        re.IGNORECASE,
    )
    if not m:
        return None
    lon1, lat1, lon2, lat2 = [float(v) for v in m.groups()]
    return (
        (-_vbox_minutes_to_decimal(lon1), _vbox_minutes_to_decimal(lat1)),
        (-_vbox_minutes_to_decimal(lon2), _vbox_minutes_to_decimal(lat2)),
    )


def _vbox_minutes_to_decimal(raw: float) -> float:
    """Convert VBOX minute values to decimal degrees.

    The reference data shows VBOX latitude/longitude are stored as minute values,
    not as packed degree+minute integers. In the example .vbo file, values like
    +04958.156830 correspond to 4958.156830 minutes = 82.635947... degrees,
    but the longitude sign is inverted relative to standard GPS for this logger
    format. We keep the minute conversion generic and flip longitude below.
    """
    value = float(raw)
    if value == 0.0:
        return 0.0
    if abs(value) > 100000.0:
        value /= 100000.0
    return value / 60.0


def _parse_hhmmss(raw: float) -> Tuple[int, int, float]:
    """Decompose HHMMSS.SS float into (hours, minutes, seconds).

    Raises CSVParseError if the decomposed value is out of sane bounds
    (e.g. a garbled hour/minute field, or seconds rounding up to 60.0)
    instead of letting an invalid value crash later inside datetime(...)
    with an uninformative, uncaught ValueError.
    """
    h = int(raw) // 10000
    m = (int(raw) // 100) % 100
    s = round(raw - h * 10000 - m * 100, 6)
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s < 60):
        raise CSVParseError(
            f"Invalid HHMMSS time value {raw!r} decoded to h={h} m={m} s={s}")
    return h, m, s


# ── Main loader ───────────────────────────────────────────────────────────────

def load_vbo(path: str) -> Session:
    sections = _parse_sections(path)

    header_lines = sections.get('header', [])
    if not header_lines:
        raise MissingHeaderError(f"No [header] section in {path}")

    channels = [c.strip() for c in header_lines]
    norm_channels = [c.lower() for c in channels]

    unit_lines = [u.strip() for u in sections.get('channel units', [])]
    units: Dict[str, str] = dict(zip(channels, unit_lines)) if unit_lines else {}
    unit_lookup = {ch.lower(): unit for ch, unit in units.items()}

    def _unit_for(ch: str) -> str:
        return unit_lookup.get(ch.strip().lower(), units.get(ch, ''))

    # ── Channel index lookup ──────────────────────────────────────────────────

    def _find(*names: str) -> Optional[int]:
        for name in names:
            norm_name = name.strip().lower()
            for i, ch in enumerate(norm_channels):
                norm_ch = ch.strip().lower()
                if (norm_ch == norm_name or norm_ch.startswith(norm_name)
                        or norm_ch.endswith(norm_name) or norm_name in norm_ch):
                    return i
        return None

    idx_time    = _find('time')
    idx_lat     = _find('latitude north', 'latitude south', 'latitude')
    idx_lon     = _find('longitude east', 'longitude west', 'longitude')
    idx_speed   = _find('velocity kmh', 'velocity mph', 'velocity', 'speed')
    idx_height  = _find('height', 'altitude')
    idx_lat_g   = _find('lateral-acc', 'lateral acc', 'ay')
    idx_lon_g   = _find('longitudinal-acc', 'longitudinal acc', 'ax')
    idx_vert_g  = _find('az', 'vertical-acc', 'vertical acc')
    idx_lap     = _find('lap trigger', 'lap-trigger', 'lapctr', 'lap beacon', 'lap count')
    idx_rpm     = _find('rpm', 'engine rpm', 'engine_rpm', 'engine rpm ', 'engine_rpm ')
    idx_yaw     = _find('yaw rate', 'yaw-rate')

    if idx_time is None or idx_lat is None or idx_lon is None:
        raise MissingHeaderError(f"Missing required channels (time/lat/lon) in {path}")

    # VBOX coordinates are minute values; do not infer a hemisphere from the
    # channel name. The value's sign (when present) is the authoritative source.
    # Speed conversion factor
    speed_ch = channels[idx_speed] if idx_speed is not None else ''
    speed_unit = _unit_for(speed_ch).lower()
    if 'kmh' in speed_ch.lower() or 'km/h' in speed_unit or 'kph' in speed_unit:
        speed_factor = 1.0
        source_speed_unit = 'kmh'
    elif 'mph' in speed_ch or 'mph' in speed_unit:
        speed_factor = 1.60934
        source_speed_unit = 'mph'
    elif 'm/s' in speed_unit:
        speed_factor = 3.6
        source_speed_unit = 'ms'
    else:
        speed_factor = 1.852  # bare 'velocity' → knots
        source_speed_unit = 'kmh'  # knots isn't a selectable display unit; default to kmh

    # Session date from [comments]
    comments_text = '\n'.join(sections.get('comments', []))
    session_date = _parse_date_from_comments(comments_text)

    # ── Data rows ─────────────────────────────────────────────────────────────

    data_lines = sections.get('data', [])
    if not data_lines:
        raise NoDataRowsError(f"No [data] section in {path}")

    lap_line = None
    if idx_lap is None:
        for raw_line in sections.get('laptiming', []):
            if not raw_line.lstrip().lower().startswith('start'):
                continue
            lap_line = _parse_vbox_start_line(raw_line)
            if lap_line is not None:
                break

    all_pts: List[DataPoint] = []
    prev_dt: Optional[datetime] = None
    day_offset = 0

    consumed_idx = {
        idx_time,
        idx_lat,
        idx_lon,
        idx_speed,
        idx_height,
        idx_lat_g,
        idx_lon_g,
        idx_vert_g,
        idx_lap,
        idx_rpm,
        idx_yaw,
    }
    consumed_idx = {i for i in consumed_idx if i is not None}
    extra_names: List[str] = []
    extra_channel_meta: Dict[str, dict] = {}
    for i, ch in enumerate(channels):
        if i in consumed_idx or not ch.strip():
            continue
        name = ch.strip()
        extra_names.append(name)
        extra_channel_meta[name] = {'label': name, 'unit': _unit_for(name)}

    for record_idx, line in enumerate(data_lines):
        cols = line.split()
        min_idx = max(c for c in [idx_time, idx_lat, idx_lon] if c is not None)
        if len(cols) <= min_idx:
            continue

        def _col(idx: Optional[int], default: float = 0.0) -> float:
            if idx is None or idx >= len(cols):
                return default
            try:
                return float(cols[idx])
            except ValueError:
                return default

        try:
            h, m, s = _parse_hhmmss(_col(idx_time))
        except CSVParseError as exc:
            logger.warning('Skipping row %d with invalid time value in %s: %s',
                           record_idx, path, exc)
            continue
        if session_date is not None:
            dt = session_date + timedelta(hours=h, minutes=m, seconds=s, days=day_offset)
            if prev_dt is not None and (dt - prev_dt).total_seconds() < -3600:
                day_offset += 1
                dt += timedelta(days=1)
        else:
            dt = datetime(1970, 1, 1, h, m, int(s),
                          microsecond=int((s % 1) * 1_000_000),
                          tzinfo=timezone.utc)
        prev_dt = dt

        lat = _vbox_minutes_to_decimal(_col(idx_lat))
        lon = -_vbox_minutes_to_decimal(_col(idx_lon))

        speed  = _col(idx_speed) * speed_factor
        lat_g  = _col(idx_lat_g)   # → gforce_y (lateral)
        lon_g  = _col(idx_lon_g)   # → gforce_x (longitudinal)
        vert_g = _col(idx_vert_g)
        height = _col(idx_height)
        rpm    = _col(idx_rpm)
        yaw    = _col(idx_yaw)     # deg/s; stored in gyro_z slot

        # lap trigger increments at each beacon crossing (0 = outlap)
        lap_num = int(_col(idx_lap)) if idx_lap is not None else 1

        extra = {}
        for name in extra_names:
            idx = next((i for i, ch in enumerate(norm_channels)
                        if ch.strip() == name.strip().lower()), None)
            if idx is None or idx >= len(cols):
                extra[name] = 0.0
                continue
            try:
                extra[name] = float(cols[idx])
            except ValueError:
                extra[name] = 0.0

        all_pts.append(DataPoint(
            record     = record_idx,
            time       = dt,
            lat        = lat,
            lon        = lon,
            alt        = height,
            speed      = speed,
            gforce_x   = lon_g,
            gforce_y   = lat_g,
            gforce_z   = vert_g,
            lap        = lap_num,
            gyro_x     = 0.0,
            gyro_y     = 0.0,
            gyro_z     = yaw,
            rpm        = rpm,
            extra      = extra,
        ))

    if not all_pts:
        raise NoDataRowsError(f"No valid data rows parsed from {path}")

    if idx_lap is None and lap_line is not None:
        a, b = lap_line
        v_x = b[0] - a[0]
        v_y = b[1] - a[1]
        cross_vals: List[float] = []
        for pt in all_pts:
            w_x = pt.lon - a[0]
            w_y = pt.lat - a[1]
            cross_vals.append(v_x * w_y - v_y * w_x)

        # Collect all pos->neg crossings (positive to negative), which occur once per lap
        # on a closed loop track. Filter them by gap to eliminate jitter.
        pos_neg_crossings: List[Tuple[int, datetime]] = []
        for i in range(1, len(all_pts)):
            prev, curr = cross_vals[i - 1], cross_vals[i]
            if prev == 0.0:
                prev = 1e-12
            if curr == 0.0:
                curr = 1e-12
            # Only count positive-to-negative direction crossings
            if prev > 0.0 and curr < 0.0:
                pos_neg_crossings.append((i, all_pts[i].time))

        # Filter crossings: keep only those with sufficient gap from the previous one
        valid_crossings: List[Tuple[int, datetime]] = []
        for i, (idx, t) in enumerate(pos_neg_crossings):
            if i == 0:
                # Always accept first crossing
                valid_crossings.append((idx, t))
            else:
                prev_time = pos_neg_crossings[i - 1][1]
                gap = (t - prev_time).total_seconds()
                if gap >= _VBOX_LAP_CROSSING_MIN_GAP_SECONDS:
                    valid_crossings.append((idx, t))

        # Assign lap numbers based on valid crossing boundaries
        # Points from the session start up to the first valid crossing are lap 1
        # Points between crossing N and crossing N+1 are lap N+1
        lap_nums = [0] * len(all_pts)
        if valid_crossings:
            for i, pt_cross_val in enumerate(cross_vals):
                lap_num = 1
                for j, (cross_idx, _) in enumerate(valid_crossings):
                    if i >= cross_idx:
                        lap_num = j + 2
                lap_nums[i] = lap_num
        else:
            lap_nums = [1] * len(all_pts)

        for pt, lap_num in zip(all_pts, lap_nums):
            pt.lap = lap_num

    # ── Elapsed times ─────────────────────────────────────────────────────────

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
        laps.append(Lap(lap_num=lap_num, points=pts, duration=dur,
                        is_outlap=(lap_num == 0)))

    timed = [l for l in laps if l.lap_num > 0]
    if len(timed) >= 3:
        med = sorted(l.duration for l in timed)[len(timed) // 2]
        if timed[-1].duration > med * _INLAP_SLOWNESS_THRESHOLD:
            timed[-1].is_inlap = True

    best_lap_time = min((l.duration for l in timed), default=0.0)
    date_str = session_date.strftime('%Y-%m-%dT%H:%M:%SZ') if session_date else ''

    return Session(
        source        = 'VBOX',
        date_utc      = date_str,
        track         = '',
        configuration = '',
        session_type  = '',
        best_lap_time = best_lap_time,
        all_points    = all_pts,
        laps          = laps,
        is_bike       = False,
        csv_path      = path,
        source_speed_unit = source_speed_unit,
        extra_channel_meta = extra_channel_meta,
    )
