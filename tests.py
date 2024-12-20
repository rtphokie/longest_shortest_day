import unittest
import os
from longest_shortest_day import risesetextremes, chart_rise_set, chart_solar_days


class MyTestCase(unittest.TestCase):
    def test_Raleigh_extremes(self):
        result, data = risesetextremes(35.78, -78.64, 'America/New_York', startYear=2024, years=1, verbose=False)
        self.assertEqual(result['2024']['Winter Solstice'].month, 12)
        self.assertEqual(result['2024']['longest_date'].month, 6)
        self.assertEqual(result['2024']['max_meridian_day'].month, 12)

    def test_solar_days(self):
        chart_solar_days('2024-06-20', '2025-06-20', 35.78, -78.64, 'America/New_York', 'Raleigh, NC')
        self.assertTrue(os.path.exists('solardays_Raleigh,_NC_2024-06-20_2025-06-20.png'))
    def test_chart(self):
        chart_rise_set('2024-06-20', '2025-06-20', 35.78, -78.64, 'America/New_York', 'Raleigh, NC')
        self.assertTrue(os.path.exists('sunriseset_Raleigh,_NC_2024-06-20_2025-06-20.png'))



if __name__ == '__main__':
    unittest.main()
