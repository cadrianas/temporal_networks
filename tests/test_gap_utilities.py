import pytest
from datetime import datetime
from temporal_networks._gap_utilities import calculate_time_difference

def test_calculate_time_difference_days():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 3, 15)
    assert calculate_time_difference(d1, d2, unit="days") == 14.0

def test_calculate_time_difference_weeks():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 3, 15)
    assert calculate_time_difference(d1, d2, unit="weeks") == 2.0

def test_calculate_time_difference_months():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 5, 1)
    assert calculate_time_difference(d1, d2, unit="months") == 2.0

def test_calculate_time_difference_years():
    d1 = datetime(2023, 3, 1)
    d2 = datetime(2024, 9, 1)
    assert calculate_time_difference(d1, d2, unit="years") == 1.5

def test_calculate_time_difference_default_unit():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 5, 1)
    assert calculate_time_difference(d1, d2) == 2.0

def test_calculate_time_difference_swapped_dates():
    d1 = datetime(2024, 5, 1)
    d2 = datetime(2024, 3, 1)
    assert calculate_time_difference(d1, d2, unit="months") == 2.0

def test_calculate_time_difference_same_date():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 3, 1)
    assert calculate_time_difference(d1, d2, unit="days") == 0.0

def test_calculate_time_difference_invalid_unit():
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 5, 1)
def test_calculate_time_difference_invalid_unit():
    """Test that calculating time difference with an invalid unit raises a ValueError."""
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 5, 1)

    with pytest.raises(ValueError, match="Unknown unit: invalid. Use 'days', 'weeks', 'months', or 'years'"):
        calculate_time_difference(d1, d2, unit="invalid")
