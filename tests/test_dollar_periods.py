"""Analysis windows must preserve gaps and respect publication availability."""
import unittest
from datetime import date

from thesaurus import dollar_periods as periods


class DollarPeriodsTests(unittest.TestCase):
    NOW = '2026-10-11T18:00:00+00:00'

    def test_personal_dates_normalize_without_mutating_settings(self):
        original = {'amount': 1500, 'history_start': None, 'history_end': None}
        updated = periods.change(original, 'history_start', '29.02.2024', now=self.NOW)
        self.assertEqual(updated['history_start'], '2024-02-29')
        self.assertIsNone(original['history_start'])
        self.assertEqual(updated['amount'], 1500)
        self.assertEqual(periods.history_label(updated), 'С 29.02.2024 до сегодня')
        self.assertIsNone(periods.change(updated, 'history_start', 'Вся история', now=self.NOW)['history_start'])

    def test_dynamic_today_tracks_moscow_midnight_but_fixed_end_stays_fixed(self):
        before = periods.resolve({}, now='2026-10-11T20:59:00+00:00')
        after = periods.resolve({}, now='2026-10-11T21:01:00+00:00')
        self.assertEqual(before['end'], '2026-10-11')
        self.assertEqual(after['end'], '2026-10-12')
        self.assertEqual(after['cutoff'], '2026-10-12T00:01:00+03:00')
        self.assertFalse(after['retrospective'])
        fixed = periods.resolve({'history_end': '2026-10-11'}, now='2026-10-11T21:01:00+00:00')
        self.assertTrue(fixed['retrospective'])
        self.assertEqual(fixed['cutoff'], '2026-10-11T23:59:59.999999+03:00')
        cleared = periods.change({'history_end': '2024-02-29'}, 'history_end', 'Сегодня', now=self.NOW)
        self.assertIsNone(cleared['history_end'])

    def test_bad_dates_and_reversed_range_are_rejected(self):
        for value in ('2023-02-29', '31.04.2024', '2026-10-12', '1992-06-30', '2024-1-1', 'tomorrow', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                periods.change({}, 'history_start', value, now=self.NOW)
        with self.assertRaisesRegex(ValueError, 'позже'):
            periods.validate({'history_start': '2025-01-02', 'history_end': '2025-01-01'}, now=self.NOW)
        periods.validate({'history_start': '1992-07-01', 'history_end': '2026-10-11'}, now=self.NOW)

    def test_persisted_values_and_naive_cutoff_are_validated(self):
        for value in (True, False, 0, 91, 7.0, '7', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                periods.validate({'forecast_days': value}, now=self.NOW)
        for value in (True, -1, 61, 12.0):
            with self.subTest(months=value), self.assertRaises(ValueError):
                periods.validate({'analysis_months': value}, now=self.NOW)
        with self.assertRaises(ValueError):
            periods.resolve({}, now='2026-10-11T18:00:00')

    def test_clip_preserves_none_and_source_values_with_inclusive_dates(self):
        rows = [{'date': '2024-02-28', 'value': 10},
                {'date': '2024-02-29', 'value': None},
                {'date': '2024-03-01', 'value': -2},
                {'date': '2024-03-02', 'value': 9}]
        window = periods.resolve({'history_start': '2024-02-29', 'history_end': '2024-03-01'}, now=self.NOW)
        selected = periods.clip(rows, window)
        self.assertEqual(selected, rows[1:3])
        self.assertIs(selected[0], rows[1])
        self.assertIsNone(selected[0]['value'])

    def test_clip_excludes_unfinished_period_and_future_publications(self):
        rows = [{'date': '2024-02-01', 'period_end': '2024-02-29', 'value': 1},
                {'date': '2024-03-01', 'period_end': '2024-03-31', 'value': 2},
                {'date': '2024-03-01', 'published_at': '2024-03-02T00:00:00+03:00', 'value': 3},
                {'date': '2024-03-01', 'published_at': '2024-03-01T20:59:59+00:00', 'value': 4},
                {'date': '2024-03-01', 'published_at': '2024-03-01', 'value': 5},
                {'date': '2024-03-01', 'retrieved_at': self.NOW, 'value': None}]
        window = periods.resolve({'history_end': '2024-03-01'}, now=self.NOW)
        self.assertEqual(periods.clip(rows, window), [rows[i] for i in (0, 3, 4, 5)])

    def test_current_day_filters_publication_after_calculation_time(self):
        rows = [{'date': '2026-10-11', 'value': 1, 'published_at': '2026-10-11T21:00:00+03:00'},
                {'date': '2026-10-11', 'value': 2, 'published_at': '2026-10-11T21:00:01+03:00'}]
        self.assertEqual(periods.clip(rows, periods.resolve({}, now=self.NOW)), rows[:1])
        with self.assertRaisesRegex(ValueError, 'публикации'):
            periods.clip([{'date': '2026-10-11', 'published_at': 'broken'}], periods.resolve({}, now=self.NOW))

    def test_daily_targets_include_exact_endpoint_and_do_not_go_past_it(self):
        config = periods.change({'analysis_months': 60}, 'forecast_days', '45', now=self.NOW)
        self.assertEqual(config['analysis_months'], 0)
        anchor = date(2024, 2, 28)
        dates = periods.targets(config, anchor)
        self.assertEqual([days for days, _ in dates], [1, 2, 3, 7, 14, 30, 45])
        self.assertTrue(all((point-anchor).days == days for days, point in dates))
        self.assertEqual(dates[0], (1, date(2024, 2, 29)))
        self.assertEqual(periods.horizon_label(config), '45 дн.')
        self.assertEqual(periods.targets({'forecast_days': 1}, '2024-02-28'), [(1, date(2024, 2, 29))])
        for value in ('0', '91', '7.5', '-1', '7 days', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                periods.change({}, 'forecast_days', value, now=self.NOW)

    def test_monthly_mode_keeps_short_overview_alongside_month_horizon(self):
        config = {'analysis_months': 24, 'forecast_days': 90}
        anchor = date(2026, 10, 11)
        self.assertEqual([days for days, _ in periods.targets(config, anchor)], [1, 2, 3, 7])
        self.assertEqual(periods.horizon_label(config), '2 года')
        self.assertEqual(periods.horizon_label({'analysis_months': 18}), '18 мес.')


if __name__ == '__main__':
    unittest.main()
