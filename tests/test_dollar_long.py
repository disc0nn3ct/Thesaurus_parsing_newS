"""Offline long FX horizons: calendar, leakage, validation, personal UI and recovery."""
import csv
import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_dollar as fixtures

from thesaurus import dollar, dollar_long, market, market_store
from thesaurus.dollar_jobs import run_job

NOW = '2026-10-10T09:00:00+00:00'
META = {'frequency': 'daily', 'calendar': 'weekdays'}


def sample(start='1992-07-01'):
    dates = pd.bdate_range(start, '2026-10-09')
    return [{'date': d.date().isoformat(), 'value': float(30*np.exp(.002*(d.year*12+d.month-1992*12-7))),
             'retrieved_at': NOW} for d in dates]


class LongModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = sample()

    def forecast(self, **kwargs):
        return dollar_long.forecast(self.rows, META, now=NOW, today=date(2026, 10, 10), **kwargs)

    def test_completed_month_closes_not_average_and_gaps_are_not_filled(self):
        rows = sample('2026-07-01')
        rows = [r for r in rows if not r['date'].startswith('2026-08')]
        rows[-1]['value'] = 99999.
        series = dollar_long.monthly_closes(rows, META, now=NOW, today=date(2026, 10, 10))
        self.assertEqual(series.index[-1], pd.Timestamp('2026-09-30'))
        self.assertTrue(pd.isna(series.loc['2026-08-31']))
        september = [r['value'] for r in rows if r['date'].startswith('2026-09')]
        self.assertEqual(series.iloc[-1], september[-1])
        self.assertLess(series.iloc[-1], 100.)

    def test_low_coverage_month_and_future_published_observations_excluded(self):
        rows = sample('2026-07-01')
        for row in rows:
            if '2026-08-01' <= row['date'] < '2026-08-20':
                row['value'] = None
            if row['date'] == '2026-09-30':
                row.update(value=99999., published_at='2027-01-01T00:00:00+00:00')
        series = dollar_long.monthly_closes(rows, META, now=NOW, today=date(2026, 10, 10))
        self.assertTrue(pd.isna(series.loc['2026-08-31']))
        self.assertLess(series.loc['2026-09-30'], 100.)

    def test_one_year_and_five_year_targets_validate_at_their_own_horizon(self):
        points = self.forecast(months=60, history='full')
        self.assertEqual([p['horizon'] for p in points], [1, 3, 6, 12, 24, 36, 60])
        year = next(p for p in points if p['horizon'] == 12)
        self.assertEqual(year['target'], '2027-09-30')
        self.assertEqual(points[-1]['target'], '2031-09-30')
        self.assertEqual(year['validation_n'], 12)
        self.assertIsNotNone(year['lower'])
        self.assertLess(points[-1]['validation_n'], 10)
        self.assertIsNone(points[-1]['lower'])
        self.assertEqual(points[-1]['model'], 'last')
        for point in points:
            for i, fold in enumerate(point['folds']):
                self.assertLessEqual(fold['training_end'], fold['origin'])
                self.assertLess(fold['origin'], fold['target'])
                self.assertEqual(pd.Period(fold['target'], 'M')-pd.Period(fold['origin'], 'M'), pd.offsets.MonthEnd(point['horizon']))
                if i:
                    self.assertLessEqual(point['folds'][i-1]['target'], fold['origin'])

    def test_recent_history_does_not_borrow_old_errors_for_year_forecast(self):
        points = self.forecast(months=12, history='recent')
        self.assertTrue(points)
        self.assertTrue(all(p['training_start'] >= '2024-07-01' for p in points))
        self.assertEqual(points[-1]['validation_n'], 0)
        self.assertIsNone(points[-1]['lower'])
        self.assertIn('условный сценарий', '\n'.join(dollar_long.describe(points, 12)))

    def test_selection_fit_and_scaling_only_use_prefix(self):
        series = dollar_long.monthly_closes(self.rows, META, now=NOW, today=date(2026, 10, 10))
        x = dollar_long.design(series)
        first = dollar_long.fit(series.iloc[:220], x.iloc[:220], 12, ('ridge', 'full', 10.))
        series.iloc[220:] = 99999.
        x.iloc[220:] = -99999.
        second = dollar_long.fit(series.iloc[:220], x.iloc[:220], 12, ('ridge', 'full', 10.))
        self.assertEqual(first, second)
        self.assertLessEqual(first['training_end'], series.index[219].date().isoformat())

    def test_complex_model_requires_improvement_and_manual_baseline_is_honored(self):
        points = self.forecast(months=12, history='full', method='ridge')
        self.assertEqual(points[-1]['model'], 'ridge')
        self.assertLess(points[-1]['validation_mae'], .95*points[-1]['baseline_mae'])
        baseline = self.forecast(months=12, method='last')
        self.assertTrue(all(p['model'] == 'last' and p['prediction'] == p['baseline'] for p in baseline))
        self.assertEqual({p['model'] for p in points[-1]['alternatives']}, {'last', 'damped', 'ridge'})
        self.assertTrue(all(p['research_only'] for p in points[-1]['alternatives']))

    def test_ridge_lags_do_not_bridge_a_missing_month(self):
        series = pd.Series(np.arange(30)+50., index=pd.date_range('2020-01-31', periods=30, freq='ME'))
        series.iloc[15] = np.nan
        x = dollar_long.design(series)
        self.assertTrue(x.iloc[16:28]['USD:изменение_12м'].isna().all())
        self.assertTrue(np.isfinite(x.iloc[29]['USD:изменение_12м']))

    def test_custom_months_and_validation_rejects_invalid_payloads(self):
        points = self.forecast(months=7)
        self.assertEqual([p['horizon'] for p in points], [1, 3, 6, 7])
        for months, method in [(61, 'auto'), (-1, 'auto'), (True, 'auto'), (12., 'auto'), (12, 'unknown')]:
            with self.subTest(months=months, method=method), self.assertRaises(ValueError):
                self.forecast(months=months, method=method)


class LongChartTests(unittest.TestCase):
    def plot(self, points):
        series = pd.Series([84., 85.], index=pd.to_datetime(['2026-09-29', '2026-09-30']))
        with patch('matplotlib.figure.Figure.savefig', autospec=True) as save:
            part = dollar_long.chart(series, points, 'unused.png')
        return save.call_args.args[0].axes[0], part

    def point(self, **overrides):
        return {'origin': '2026-09-30', 'target': '2027-09-30', 'horizon': 12, 'prediction': 85.,
                'lower': 70., 'upper': 100., 'model': 'last', 'validation_n': 12, **overrides}

    def test_baseline_is_not_presented_as_a_directional_model(self):
        points = [self.point(), self.point(target='2028-09-30', horizon=24)]
        ax, part = self.plot(points)
        labels = ax.get_legend_handles_labels()[1]
        self.assertIn('Базовый ориентир · последний курс', labels)
        self.assertNotIn('Модель · проверка от 10 периодов', labels)
        self.assertIn('Ошибки базового метода · диапазон 80%', labels)
        self.assertIn('Направление не определено', ax.get_title())
        self.assertEqual(len(ax.collections[0].get_facecolors()), 0)
        self.assertIn('не предсказание неизменности', part['caption'])
        self.assertEqual([p['prediction'] for p in points], [85., 85.])

    def test_mixed_models_keep_distinct_encodings_and_hide_unchecked_bounds(self):
        ax, _ = self.plot([self.point(model='ridge'), self.point(target='2031-09-30', horizon=60, validation_n=5)])
        labels = ax.get_legend_handles_labels()[1]
        self.assertIn('Модель · проверка от 10 периодов', labels)
        self.assertIn('Базовый сценарий · мало проверок', labels)
        self.assertEqual(len(ax.collections), 3)  # Two estimates, only one checked interval.
        self.assertGreater(len(ax.collections[0].get_facecolors()), 0)
        self.assertEqual(len(ax.collections[2].get_facecolors()), 0)


class LongBotTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp

    def test_controls_persist_per_user_checked_and_do_not_require_purchase_inputs(self):
        c = self.bot.dollar_controls
        c.callback(self.client, 2, 'usd:horizon:12')
        self.assertEqual(c.config(2)['analysis_months'], 12)
        self.assertEqual(c.config(3)['analysis_months'], 0)
        buttons = self.client.send_message.call_args.kwargs['reply_markup'].inline_keyboard
        self.assertTrue(any(b.text.startswith('✓') and b.callback_data == 'usd:horizon:12' for row in buttons for b in row))
        c.callback(self.client, 2, 'usd:longmethod:ridge')
        self.assertEqual(c.config(2)['long_method'], 'ridge')
        c.callback(self.client, 2, 'usd:mode:purchase')
        c.callback(self.client, 2, 'usd:longrun')
        payload = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        self.assertEqual(payload['config']['analysis_months'], 12)
        self.assertEqual(payload['config']['mode'], 'overview')
        self.assertIsNone(payload['config']['trip'])
        self.assertIsNone(payload['config']['quote'])

    def test_invalid_callbacks_custom_form_and_revoked_access(self):
        c = self.bot.dollar_controls
        for route in ('usd:horizon:61', 'usd:horizon:7', 'usd:longmethod:wrong'):
            with self.assertRaises(ValueError):
                c.callback(self.client, 2, route)
        c.callback(self.client, 2, 'usd:set:analysis_months')
        with self.assertRaises(ValueError):
            c.input(self.client, 2, '12.5')
        self.assertTrue(c.input(self.client, 2, '18'))
        self.assertEqual(c.config(2)['analysis_months'], 18)
        self.assertIn('Срок анализа USD/RUB', self.client.send_message.call_args.args[1])
        c.callback(self.client, 2, 'usd:set:analysis_months')
        self.store.set_setting('ui_screen:2', {'back': 'usd:horizon'})
        event = SimpleNamespace(from_user=SimpleNamespace(id=2), chat=SimpleNamespace(id=2, type='private'), text='/cancel', id=7)
        self.bot.handle_message(self.client, event)
        self.assertIn('Срок анализа USD/RUB', self.client.send_message.call_args.args[1])
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        with self.assertRaises(PermissionError):
            c.callback(self.client, 2, 'usd:horizon:24')

    def test_offline_job_saves_long_forecasts_input_versions_and_recovers_without_duplication(self):
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (dollar.USD,))['metadata'])
        dates = market.analysis.grid('2010-01-01', '2026-10-09', meta)
        market_store.save_rows(self.store, dollar.USD, [{'date': d.date().isoformat(), 'value': 80+i*.005, 'retrieved_at': NOW} for i, d in enumerate(dates)])
        market.enable(self.store, dollar.USD, True)
        cfg = {**dollar.DEFAULTS, 'analysis_months': 12, 'history_mode': 'full', 'forecast_engine': 'basic'}
        job = {'id': 986, 'kind': 'dollar_analysis'}
        with patch('thesaurus.dollar_jobs.utcnow', return_value=NOW), patch('thesaurus.market.collect') as collect:
            report_id = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg}, dry_run=True, charts=True)
            again = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg}, dry_run=True)
        collect.assert_not_called()
        self.assertEqual(again, report_id)
        report = self.store.one('SELECT * FROM reports WHERE id=?', (report_id,))
        metrics = json.loads(report['metrics'])
        self.assertEqual(len(metrics['long_term']), 4)
        self.assertIn('Долгосрочный анализ · до 12 мес.', report['text'])
        self.assertTrue(any(p['path'].endswith('dollar-long-term.png') for p in json.loads(report['parts'])))
        runs = self.store.query('SELECT * FROM forecast_runs')
        self.assertEqual(len(runs), 2)
        long_run = next(r for r in runs if json.loads(r['inputs'])['model_version'] == 'dollar-monthly-3-periods')
        self.assertEqual(len(json.loads(long_run['inputs'])['revisions']), len(dates))
        self.assertEqual(len(self.store.query('SELECT * FROM forecast_points WHERE run_id=?', (long_run['id'],))), 4)
        saved_points = self.store.query('SELECT payload FROM forecast_points WHERE run_id=?', (long_run['id'],))
        self.assertTrue(all('distribution' in json.loads(p['payload']) for p in saved_points))
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))

    def test_saved_monthly_forecast_evaluates_last_fixing_when_month_ends_on_weekend(self):
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (dollar.USD,))['metadata'])
        days = market.analysis.grid('2026-05-01', '2026-05-31', meta)
        self.assertNotEqual(days[-1].date().isoformat(), '2026-05-31')
        market_store.save_rows(self.store, dollar.USD, [{'date': d.date().isoformat(), 'value': 80.+i,
                          'retrieved_at': NOW} for i, d in enumerate(days)])
        self.store.execute('INSERT INTO forecast_runs(id,series_id,fingerprint,created_at,duration,inputs) VALUES(?,?,?,?,?,?)',
                           ('test-month', dollar.USD, 'test-month', '2026-01-01T00:00:00+00:00', '{}', '{}'))
        payload = {'target': '2026-05-31', 'created_at': '2026-01-01T00:00:00+00:00', 'model_family': 'monthly_usd'}
        self.store.execute('INSERT INTO forecast_points(run_id,target,payload) VALUES(?,?,?)',
                           ('test-month', payload['target'], json.dumps(payload)))
        with patch('thesaurus.market_store.utcnow', return_value=NOW):
            points = market_store.forecast_history(self.store, dollar.USD)
        self.assertEqual(points[0]['actual'], 80.+len(days)-1)

    def test_export_keeps_monthly_frequency_and_scenario_quality(self):
        from thesaurus.market_jobs import export_market
        points = [{'origin': '2026-09-30', 'target': '2027-09-30', 'prediction': 85., 'baseline': 85.,
                   'lower': None, 'upper': None, 'created_at': NOW, 'model': 'last', 'frequency': 'monthly',
                   'model_family': 'monthly_usd', 'validation_n': 2}]
        with patch('thesaurus.market_jobs.repository.forecast_history', return_value=[{**points[0], 'run_id': 'r'}]):
            path = self.root/'long.csv'
            export_market(self.store, path, [dollar.USD], include_forecasts=True)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(exported[0]['frequency'], 'monthly')
        self.assertEqual(exported[0]['record_type'], 'scenario')
        self.assertEqual(exported[0]['quality'], 'conditional_scenario')


if __name__ == '__main__':
    unittest.main()
