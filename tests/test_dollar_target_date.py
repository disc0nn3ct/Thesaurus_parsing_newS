"""Fixed date forecasts: no rolling endpoint, month interpolation or future leakage."""
import json
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import test_dollar as fixtures
import test_dollar_period_ui as ui_fixtures
from matplotlib.figure import Figure

from thesaurus import dollar, market_store
from thesaurus import dollar_periods as periods
from thesaurus.dollar_charts import near_term
from thesaurus.dollar_jobs import run_job
from thesaurus.telegram_dollar import form_parent

NOW = fixtures.NOW


class TargetDateMathTests(unittest.TestCase):
    def test_fixed_endpoint_accepts_both_formats_and_survives_next_day(self):
        for text in ('15.12.2026', '2026-12-15'):
            cfg = dollar.change_settings({'analysis_months': 60}, 'forecast_date', text, now=NOW)
            self.assertEqual(cfg['forecast_date'], '2026-12-15')
            self.assertEqual(cfg['analysis_months'], 0)
            self.assertEqual(periods.horizon_label(cfg), 'на 15.12.2026')
            self.assertEqual(periods.targets(cfg, '2026-10-08'), [(68, date(2026, 12, 15))])
            self.assertEqual(periods.targets(cfg, '2026-10-09'), [(67, date(2026, 12, 15))])

    def test_invalid_expired_and_more_than_five_years_rejected(self):
        for value in ('31.02.2027', '2026-10-08', '2026-10-07', '2031-10-09', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dollar.change_settings({}, 'forecast_date', value, now=NOW)
        cfg = dollar.change_settings({}, 'forecast_date', '2031-10-08', now=NOW)
        self.assertEqual(periods.targets(cfg, NOW[:10])[-1][1], date(2031, 10, 8))
        with self.assertRaisesRegex(ValueError, 'Курс на дату'):
            periods.validate({'forecast_date': '2026-10-09'}, now='2026-10-09T09:00:00+00:00')
        with self.assertRaises(ValueError):
            periods.validate({'forecast_date': '2027-10-08', 'analysis_months': 12}, now=NOW)

    def test_leap_year_bound_and_retrospective_anchor(self):
        cfg = dollar.change_settings({}, 'forecast_date', '28.02.2029', now='2024-02-29T09:00:00+00:00')
        self.assertEqual(periods.targets(cfg, '2024-02-29')[0][1], date(2029, 2, 28))
        with self.assertRaises(ValueError):
            dollar.change_settings({}, 'forecast_date', '01.03.2029', now='2024-02-29T09:00:00+00:00')
        cfg = dollar.change_settings({'history_end': '2024-01-01'}, 'forecast_date', '01.07.2024', now=NOW)
        self.assertEqual(periods.targets(cfg, '2024-01-01'), [(182, date(2024, 7, 1))])

    def test_day_month_and_reset_clear_fixed_endpoint_including_expired_dates(self):
        cfg = {**dollar.DEFAULTS, 'forecast_date': '2026-10-01'}
        for field, text in (('forecast_days', '30'), ('analysis_months', '12'), ('forecast_date', 'Сбросить')):
            self.assertIsNone(dollar.change_settings(cfg, field, text, now=NOW)['forecast_date'])

    def test_single_point_chart_names_exact_date_and_does_not_invent_trajectory(self):
        series = pd.Series([84., 85.], index=pd.to_datetime(['2026-10-07', '2026-10-08']))
        point = {'target': '2027-04-15', 'days_ahead': 189, 'prediction': 85.,
                 'model': 'last', 'validation_n': 3, 'lower': None, 'upper': None}
        with patch.object(Figure, 'savefig', autospec=True) as save:
            part = near_term(series, [point], Path('date-preview.png'))
        fig = save.call_args.args[0]
        self.assertIn('курс на 15.04.2027', fig._suptitle.get_text())
        self.assertEqual(len(fig.axes[1].get_yticklabels()), 1)
        self.assertEqual(fig.axes[1].get_lines(), [])
        self.assertNotIn('Полоса:', part['caption'])


class TargetDateUITests(unittest.TestCase):
    setUp = ui_fixtures.DollarPeriodUITests.setUp
    view = ui_fixtures.DollarPeriodUITests.view
    click = ui_fixtures.DollarPeriodUITests.click

    def test_personal_form_selection_navigation_and_frozen_queue(self):
        self.click('usd:menu')
        self.assertIn('usd:set:forecast_date', self.view()[1])
        self.click('usd:set:forecast_date')
        self.assertEqual(form_parent('forecast_date'), 'usd:horizon')
        before = self.controls.config(2)
        with self.assertRaises(ValueError):
            self.controls.input(self.client, 2, 'invalid')
        self.assertEqual(self.controls.config(2), before)
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'dollar:forecast_date')
        self.controls.input(self.client, 2, '15.12.2026')
        self.assertTrue(self.view()[1]['usd:set:forecast_date'].startswith('✓'))
        self.assertFalse(self.view()[1]['usd:days:7'].startswith('✓'))
        self.assertIn('usd:model', self.view()[1])
        self.assertNotIn('usd:longmethods', self.view()[1])
        self.assertNotIn('usd:distribution', self.view()[1])
        self.assertIsNone(self.controls.config(3)['forecast_date'])
        self.click('usd:longrun')
        payload = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        self.assertEqual(payload['config']['forecast_date'], '2026-12-15')
        self.click('usd:horizon')
        self.click('usd:days:30')
        self.assertIsNone(self.controls.config(2)['forecast_date'])
        self.assertEqual(json.loads(self.store.one('SELECT payload FROM jobs')['payload']), payload)
        self.click('usd:set:forecast_date')
        self.controls.input(self.client, 2, '15.12.2026')
        self.click('usd:horizon:12')
        self.assertIsNone(self.controls.config(2)['forecast_date'])

    def test_form_is_private_and_ungranted_actor_cannot_set_date(self):
        self.click('usd:set:forecast_date', kind='group').answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2'))
        self.store.add_user(4, subscriptions=['vivino'])
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 4, 'usd:set:forecast_date')
        self.assertIsNone(self.store.get_setting('dollar:4'))


class TargetDateJobTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp
    load = fixtures.DollarBotTests.load

    def test_exact_far_date_is_frozen_without_network_or_interpolation(self):
        self.load()
        cfg = dollar.change_settings({}, 'forecast_date', '2027-10-15', now=NOW)
        with patch('thesaurus.dollar_jobs.utcnow', return_value=NOW), \
                patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
            report_id = run_job(self.store, self.settings, {'id': 1501, 'kind': 'dollar_analysis'},
                                {'actor_id': 2, 'config': cfg}, dry_run=True, charts=False)
        report = self.store.one('SELECT * FROM reports WHERE id=?', (report_id,))
        result = json.loads(report['metrics'])
        self.assertEqual([p['target'] for p in result['timing']], ['2027-10-15'])
        self.assertEqual([p['target'] for p in result['forecast']], ['2027-10-15'])
        self.assertEqual(result['forecast'][0]['model'], 'last')
        self.assertIsNone(result['forecast'][0]['lower'])
        self.assertIn('Курс на 15.10.2027', report['text'])
        self.assertIn('Метод: последнее известное значение', report['text'])
        self.assertIn('недостаточно проверочных ошибок', report['text'])
        self.assertEqual(json.loads(self.store.one('SELECT duration FROM forecast_runs')['duration']), {'unit': 'days', 'value': 372})
        self.assertEqual(self.store.one('SELECT target FROM forecast_points')['target'], '2027-10-15')
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))

    def test_retrospective_target_ignores_observations_after_origin(self):
        self.load()
        cfg = dollar.change_settings({'history_end': '2026-10-01'}, 'forecast_date', '2026-10-07', now=NOW)
        first, _ = dollar.build(self.store, cfg, now=NOW)
        market_store.save_rows(self.store, dollar.USD, [{'date': '2026-10-06', 'value': 9999., 'retrieved_at': NOW}])
        second, _ = dollar.build(self.store, cfg, now=NOW)
        self.assertEqual(first['forecast'], second['forecast'])
        self.assertEqual([p['target'] for p in second['timing']], ['2026-10-07'])
        self.assertTrue(second['period']['retrospective'])


if __name__ == '__main__':
    unittest.main()
