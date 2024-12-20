import unittest
from pprint import pprint
from longest_shortest_day import risesetextremes, chart_rise_set, chart_solar_days


class MyTestCase(unittest.TestCase):
    def test_extremes(self):
        # 35.7796° N, 78.6382° W
        result, data = risesetextremes(0.0, 0.0, 'America/New_York', startYear=2024, years=1, verbose=False)
        result, data = risesetextremes(51.56, -55.73, 'America/New_York', startYear=2024, years=1, verbose=False)
        result, data = risesetextremes(46.36, -101.31, 'America/New_York', startYear=2024, years=1, verbose=False)
        result, data = risesetextremes(35.78, -78.64, 'America/New_York', startYear=2024, years=1, verbose=False)
        pprint(result)
        for attr in ['rise_latest', 'set_earliest']:
            print(attr, result['2024'][attr])

    def test_solar_days(self):
        chart_solar_days('2024-06-20', '2025-06-20', 35.78, -78.64, 'America/New_York', 'Raleigh, NC')
    def test_chart(self):
        chart_rise_set('2024-06-20', '2025-06-20', 35.78, -78.64, 'America/New_York', 'Raleigh, NC')
        # chart_rise_set('2024-06-20', '2025-06-20', 29.44, -82.46, 'America/New_York', 'Raleigh, FL')
        # chart_rise_set('2024-06-20', '2025-06-20', 46.36, -101.31, 'America/Chicago', 'Raleigh, ND')
        # chart_rise_set('2024-06-20', '2025-06-20', 51.56, -55.73, 'America/St_Johns', 'Raleigh, Newfoundland')


if __name__ == '__main__':
    unittest.main()
