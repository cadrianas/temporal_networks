import unittest

from temporal_networks._gap_utilities import parse_flexible_datetime

from datetime import datetime

class TestParseFlexibleDatetime(unittest.TestCase):
    def test_parse_flexible_datetime_valid_formats(self):
        # YYYY-MM
        self.assertEqual(parse_flexible_datetime("2024-03"), datetime(2024, 3, 1))

        # YYYY-MM-DD
        self.assertEqual(parse_flexible_datetime("2024-03-15"), datetime(2024, 3, 15))

        # YYYY-W## (First day of ISO week 12 in 2024 is Monday, March 18)
        self.assertEqual(parse_flexible_datetime("2024-W12"), datetime(2024, 3, 18))

        # YYYY-Q# (First day of quarter 2 is April 1)
        self.assertEqual(parse_flexible_datetime("2024-Q2"), datetime(2024, 4, 1))

        # YYYY
        self.assertEqual(parse_flexible_datetime("2024"), datetime(2024, 1, 1))

        # Leading and trailing spaces
        self.assertEqual(parse_flexible_datetime("  2024-03  "), datetime(2024, 3, 1))
        self.assertEqual(parse_flexible_datetime("2024-03-15 "), datetime(2024, 3, 15))
        self.assertEqual(parse_flexible_datetime(" 2024-Q2"), datetime(2024, 4, 1))
        self.assertEqual(parse_flexible_datetime("2024 "), datetime(2024, 1, 1))

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
