import pytest
from datetime import datetime
from temporal_networks._gap_utilities import calculate_time_difference

def test_calculate_time_difference_invalid_unit():
    """Test that calculating time difference with an invalid unit raises a ValueError."""
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 5, 1)

    with pytest.raises(ValueError, match="Unknown unit: invalid. Use 'days', 'weeks', 'months', or 'years'"):
        calculate_time_difference(d1, d2, unit="invalid")
