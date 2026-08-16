import pytest
from racechrono_data import load_racechrono_csv, is_racechrono_csv
from exceptions import MissingHeaderError, NoDataRowsError


def test_is_racechrono_csv():
    """RaceChrono CSV files should be detected by header."""
    assert is_racechrono_csv('examples/session_20260811_132911_mid-ohio_v3.csv')
    assert not is_racechrono_csv('examples/session_20260811_132911_mid-ohio.vbo')


def test_load_racechrono_csv_real_session():
    """Load the real RaceChrono session and verify lap times."""
    session = load_racechrono_csv('examples/session_20260811_132911_mid-ohio_v3.csv')
    
    # Check basic session properties
    assert session.source == 'RaceChrono'
    assert session.track == 'Mid-Ohio'
    assert len(session.all_points) > 0
    assert len(session.laps) > 0
    
    # Check that we have 12 timed laps (plus outlap and possibly an incomplete lap)
    timed_laps = [l for l in session.laps if l.lap_num > 0]
    assert len(timed_laps) >= 12
    
    # Verify exact lap times match RaceChrono
    assert session.laps[1].lap_num == 1
    assert session.laps[1].duration == pytest.approx(123.21, abs=0.05)
    
    assert session.laps[6].lap_num == 6
    assert session.laps[6].duration == pytest.approx(115.08, abs=0.05)
    
    assert session.laps[11].lap_num == 11
    assert session.laps[11].duration == pytest.approx(110.93, abs=0.05)
    
    assert session.laps[12].lap_num == 12
    assert session.laps[12].duration == pytest.approx(128.02, abs=0.05)


def test_racechrono_csv_channels():
    """Verify that all expected channels are parsed correctly."""
    session = load_racechrono_csv('examples/session_20260811_132911_mid-ohio_v3.csv')
    
    # Pick a point from lap 1
    lap1_pt = session.laps[1].points[0]
    
    assert lap1_pt.lat != 0.0  # GPS latitude
    assert lap1_pt.lon != 0.0  # GPS longitude
    assert lap1_pt.speed > 0.0  # Speed should be positive during lap
    assert lap1_pt.rpm > 0.0  # Engine running
    assert lap1_pt.lap == 1  # Correct lap number


def test_racechrono_csv_outlap():
    """Pre-lap warm-up data should be assigned to lap 0 (outlap)."""
    session = load_racechrono_csv('examples/session_20260811_132911_mid-ohio_v3.csv')
    
    # First lap should be lap 0 (outlap/pre-lap data)
    assert session.laps[0].lap_num == 0
    assert session.laps[0].duration > 200.0  # Pre-lap warmup is long
