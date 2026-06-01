import unittest

from temporal_networks._gap_utilities import parse_flexible_datetime

class TestParseFlexibleDatetime(unittest.TestCase):
    def test_parse_flexible_datetime_invalid_fallback(self):
        # Invalid format
        self.assertIsNone(parse_flexible_datetime("invalid"))

        # Valid format but invalid date values
        self.assertIsNone(parse_flexible_datetime("2024-13-45"))

        # Another invalid string
        self.assertIsNone(parse_flexible_datetime("not-a-date"))

        # Empty string
        self.assertIsNone(parse_flexible_datetime(""))

if __name__ == '__main__':
    unittest.main()
