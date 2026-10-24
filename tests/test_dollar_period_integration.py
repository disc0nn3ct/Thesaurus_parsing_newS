"""A personal USD window reaches every numerical path and saved forecast input."""
import json
import math
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from tests import test_dollar as fixtures
from thesaurus import dollar, market, market_store
from thesaurus.dollar_jobs import run_job
from thesaurus.market_analysis import grid

NOW = fixtures.NOW


class DollarPeriodIntegrationTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp
    load = fixtures.DollarBotTests.load

    def load_series(self, identity=dollar.USD, start='2023-01-03', end='2026-10-08'):
        record = self.store.one('SELECT metadata FROM market_series WHERE id=?', (identity,))
        metadata = json.loads(record['metadata'])
        rows = [{'date': day.date().isoformat(), 'value': 80+.01*i+math.sin(i/13),
                 'retrieved_at': '2026-10-08T08:00:00+00:00'}
                for i, day in enumerate(grid(start, end, metadata))]
        market_store.save_rows(self.store, identity, rows)
        market.enable(self.store, identity, True)
        return rows

    def test_window_bounds_short_long_context_factors_and_revision_ids(self):
        self.load_series()
        self.load_series('cbr.fx.eur')
        # A value dated inside the window but published after it must not leak.
        market_store.save_rows(self.store, 'cbr.fx.eur', [
            {'date': '2023-12-29', 'value': 99999., 'published_at': '2023-12-30T00:00:00+03:00',
             'retrieved_at': NOW}])
        cfg = {**dollar.DEFAULTS, 'history_start': '2023-03-01', 'history_end': '2023-12-29',
               'history_mode': 'recent', 'analysis_months': 12}
        with patch('thesaurus.dollar_models.forecast', return_value=[]) as short, \
                patch('thesaurus.dollar_long.forecast', return_value=[]) as long:
            result, series = dollar.build(self.store, cfg, now=NOW)
        self.assertEqual(series.index.min(), pd.Timestamp('2023-03-01'))
        self.assertEqual(series.index.max(), pd.Timestamp('2023-12-29'))
        self.assertTrue(result['period']['retrospective'])
        self.assertTrue(result['model_data_fresh'])
        self.assertFalse(result['data_fresh'])
        self.assertEqual(result['period']['effective_history_mode'], 'full')
        for call in (short.call_args, long.call_args):
            rows, metadata = call.args
            self.assertTrue(rows)
            self.assertTrue(all('2023-03-01' <= row['date'] <= '2023-12-29' for row in rows))
            self.assertNotIn('break_date', metadata)
            self.assertEqual(call.kwargs['history'], 'full')
            self.assertEqual(call.kwargs['today'], date(2023, 12, 29))
            self.assertEqual(call.kwargs['now'], result['period']['cutoff'])
            self.assertEqual(result['input_revisions'], [row['id'] for row in rows])
        short_targets = short.call_args.kwargs['targets']
        self.assertEqual(short_targets[0], (1, date(2023, 12, 30)))
        self.assertEqual(short_targets[-1], (7, date(2024, 1, 5)))
        context = next(row for row in result['context'] if row['id'] == 'cbr.fx.eur')
        self.assertEqual(context['date'], '2023-12-28')
        self.assertLess(context['value'], 100.)
        factor = short.call_args.kwargs['inputs'][0]
        self.assertTrue(all('2023-03-01' <= row['date'] <= '2023-12-28' for row in factor['rows']))
        self.assertEqual(context['revisions'], [row['id'] for row in factor['rows']])
        self.assertEqual(result['model_inputs'][0]['revisions'], context['revisions'])

    def test_actual_retrospective_forecast_ignores_shocks_before_and_after_window(self):
        self.load_series(start='2025-01-02')
        self.load_series('cbr.fx.eur', start='2025-01-02')
        cfg = {**dollar.DEFAULTS, 'history_start': '2025-02-01', 'history_end': '2026-09-30',
               'forecast_days': 1, 'history_mode': 'recent'}
        first, first_series = dollar.build(self.store, cfg, now=NOW)
        self.assertTrue(first['forecast'])
        self.assertEqual(first['forecast'][-1]['target'], '2026-10-01')
        for identity in (dollar.USD, 'cbr.fx.eur'):
            market_store.save_rows(self.store, identity, [
                {'date': '2025-01-30', 'value': 1., 'retrieved_at': NOW},
                {'date': '2026-10-01', 'value': 99999., 'retrieved_at': NOW}])
        second, second_series = dollar.build(self.store, cfg, now=NOW)
        pd.testing.assert_series_equal(first_series, second_series)
        for key in ('forecast', 'stats', 'context', 'input_revisions', 'model_inputs'):
            self.assertEqual(first[key], second[key], key)
        self.assertTrue(all(point['retrospective_period'] for point in second['forecast']))
        self.assertTrue(all(point['analysis_as_of'] == '2026-09-30' for point in second['forecast']))
        self.assertIn('исторический анализ', second['decision'])

    def test_explicit_start_overrides_recent_mode_for_purchase_deadline_too(self):
        self.load_series()
        cfg = {**dollar.DEFAULTS, 'history_start': '2023-03-01', 'history_mode': 'recent',
               'forecast_days': 45, 'trip': '2026-12-20', 'quote': 95., 'quote_at': NOW, 'mode': 'purchase'}
        with patch('thesaurus.dollar_models.forecast', return_value=[]) as model, \
                patch('thesaurus.dollar.analysis.forecast') as legacy:
            result, _ = dollar.build(self.store, cfg, now=NOW)
        legacy.assert_not_called()
        self.assertTrue(result['data_fresh'])
        self.assertEqual(model.call_count, 2)
        for call in model.call_args_list:
            rows, metadata = call.args
            self.assertEqual(rows[0]['date'], '2023-03-01')
            self.assertEqual(rows[-1]['date'], NOW[:10])
            self.assertNotIn('break_date', metadata)
            self.assertEqual(call.kwargs['history'], 'full')
        self.assertEqual(model.call_args_list[0].kwargs['targets'][-1], (45, date(2026, 11, 22)))
        self.assertEqual(model.call_args_list[1].kwargs['targets'], [(73, date(2026, 12, 20))])

    def test_empty_and_short_windows_do_not_borrow_history_or_make_up_points(self):
        self.load()
        empty_cfg = {**dollar.DEFAULTS, 'history_start': '1992-07-01', 'history_end': '1993-01-01',
                     'analysis_months': 12}
        with patch('thesaurus.dollar_models.forecast') as short, \
                patch('thesaurus.dollar_long.forecast') as long:
            empty, series = dollar.build(self.store, empty_cfg, now=NOW)
        self.assertIsNone(empty['stats'])
        self.assertTrue(series.empty)
        self.assertEqual(empty['forecast'], [])
        self.assertEqual(empty['long_term'], [])
        short.assert_not_called()
        long.assert_not_called()
        short_cfg = {**dollar.DEFAULTS, 'history_start': '2026-10-07', 'analysis_months': 12}
        limited, series = dollar.build(self.store, short_cfg, now=NOW)
        self.assertLessEqual(len(series), 2)
        self.assertEqual(limited['forecast'], [])
        self.assertEqual(limited['long_term'], [])
        self.assertEqual(limited['stats']['first'], '2026-10-07')

    def test_current_job_saves_custom_horizon_and_period_without_collecting(self):
        self.load()
        cfg = dollar.change_settings({}, 'forecast_days', '45', now=NOW)
        cfg = dollar.change_settings(cfg, 'history_start', '01.06.2026', now=NOW)
        with patch('thesaurus.dollar_jobs.utcnow', return_value=NOW), \
                patch('thesaurus.market.collect') as collect:
            report_id = run_job(self.store, self.settings, {'id': 1401, 'kind': 'dollar_analysis'},
                                {'actor_id': 2, 'config': cfg}, dry_run=True, charts=False)
        collect.assert_not_called()
        report = self.store.one('SELECT * FROM reports WHERE id=?', (report_id,))
        result = json.loads(report['metrics'])
        self.assertEqual(result['config']['forecast_days'], 45)
        self.assertEqual(result['period']['start'], '2026-06-01')
        self.assertEqual(result['period']['end'], NOW[:10])
        self.assertFalse(result['period']['retrospective'])
        self.assertTrue(result['forecast'])
        self.assertEqual(result['forecast'][-1]['target'], '2026-11-22')
        self.assertTrue(all(point['target'] <= '2026-11-22' for point in result['forecast']))
        run = self.store.one('SELECT * FROM forecast_runs')
        self.assertEqual(json.loads(run['duration']), {'unit': 'days', 'value': 45})
        saved = json.loads(run['inputs'])
        self.assertEqual(saved['period'], result['period'])
        self.assertEqual(saved['revisions'], result['input_revisions'])
        self.assertEqual(len(self.store.query('SELECT * FROM forecast_points')), len(result['forecast']))
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))

    def test_retrospective_job_keeps_report_but_never_freezes_forecasts_as_prospective(self):
        self.load_series()
        cfg = {**dollar.DEFAULTS, 'history_start': '2023-02-01', 'history_end': '2026-09-30',
               'analysis_months': 1, 'long_method': 'last', 'distribution': False,
               'mode': 'purchase', 'quote': 95., 'quote_at': NOW, 'trip': '2026-12-20'}
        with patch('thesaurus.dollar_jobs.utcnow', return_value=NOW), \
                patch('thesaurus.market.collect') as collect:
            report_id = run_job(self.store, self.settings, {'id': 1402, 'kind': 'dollar_analysis'},
                                {'actor_id': 2, 'config': cfg}, dry_run=True, charts=False)
        collect.assert_not_called()
        result = json.loads(self.store.one('SELECT metrics FROM reports WHERE id=?', (report_id,))['metrics'])
        self.assertTrue(result['period']['retrospective'])
        self.assertTrue(result['forecast'])
        self.assertTrue(result['long_term'])
        self.assertFalse(result['data_fresh'])
        self.assertEqual(result['purchase_guide']['status'], 'retrospective')
        self.assertEqual(result['scenarios'], [])
        self.assertTrue(all(p['cost'] is None for p in result['purchase_candidates']))
        self.assertNotIn('Покупка сегодня:', dollar.summary(result))
        self.assertIsNone(result['premium'])
        self.assertFalse(self.store.query('SELECT * FROM forecast_runs'))
        self.assertFalse(self.store.query('SELECT * FROM forecast_points'))
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))
        origin = date.fromisoformat(result['period']['end'])
        self.assertEqual(result['forecast'][-1]['target'], (origin+timedelta(days=7)).isoformat())


if __name__ == '__main__':
    unittest.main()
