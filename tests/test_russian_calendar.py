"""Reviewed calendar dates affect coverage, training and future targets together."""
import unittest

import pandas as pd

from thesaurus.market_analysis import forecast, grid
from thesaurus.rate_forecasts import working_dates
from thesaurus.reporting import ReportBuilder


class RussianCalendarTests(unittest.TestCase):
    def test_cbr_2026_transferred_days_are_not_missing_or_forecast_targets(self):
        days_off = ['2026-01-09', '2026-03-09', '2026-05-11', '2026-12-31']
        for day in days_off:
            self.assertEqual(working_dates(day, day), [])
            self.assertEqual(len(grid(day, day, {'frequency': 'daily', 'calendar': 'russia_working'})), 0)
        self.assertEqual(len(grid('2026-05-11', '2026-05-11', {'frequency': 'daily', 'calendar': 'calendar'})), 1)
        self.assertEqual(len(working_dates('2026-05-12', '2026-05-12')), 1)
        self.assertEqual(len(grid('2026-05-12', '2026-05-12', {'frequency': 'daily', 'calendar': 'cbr_fx'})), 0)
        quality = {}
        ReportBuilder._business_coverage(quality, [{'date': '2026-05-12', 'metric': 'RUONIA', 'value': 10}],
                                        '2026-05-09', '2026-05-12', ['RUONIA'])
        self.assertEqual(quality['expected_points'], 1)
        self.assertEqual(quality['missing_points'], 0)

    def test_complete_miacr_history_is_not_trimmed_at_transferred_holidays(self):
        meta = {'frequency': 'daily', 'calendar': 'russia_working', 'unit': '%'}
        days = grid('2025-01-01', '2026-10-06', meta)
        rows = [{'date': d.date().isoformat(), 'value': 10+i*.01} for i, d in enumerate(days)]
        result = forecast(rows, meta, created_at='2026-10-07', horizon={'value': 4, 'unit': 'weeks'})
        self.assertEqual(len(result), 4)
        self.assertTrue(all(p['validation_n'] >= 10 and p['lower'] is not None for p in result))
        # Real missing observations are still not filled or bridged in training.
        missing = days[-10].date().isoformat()
        self.assertEqual(forecast([r for r in rows if r['date'] != missing], meta, created_at='2026-10-07'), [])

    def test_unavailable_prehistory_does_not_disable_forecast_but_missing_tail_does(self):
        meta = {'frequency': 'daily', 'calendar': 'calendar', 'unit': '%'}
        rows = [{'date': d.date().isoformat(), 'value': None if i < 200 else 10+i*.01}
                for i, d in enumerate(pd.date_range('2025-01-01', periods=400))]
        result = forecast(rows, meta, created_at=rows[-1]['date'])
        self.assertEqual(len(result), 4)
        self.assertTrue(all(p['validation_n'] >= 10 for p in result))
        rows.append({'date': (pd.Timestamp(rows[-1]['date'])+pd.Timedelta(days=1)).date().isoformat(), 'value': None})
        self.assertEqual(forecast(rows, meta, created_at=rows[-1]['date']), [])


if __name__ == '__main__':
    unittest.main()
