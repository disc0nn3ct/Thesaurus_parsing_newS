"""Full-history FX regression contracts and temporal isolation, without live APIs."""
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import test_dollar as fixtures

from thesaurus import dollar, dollar_models, dollar_sources, market, market_store
from thesaurus.dollar_jobs import run_job
from thesaurus.market_sources import catalogue, fetch, parse_currency


class HistoryModelsTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp

    def sample(self, n=360):
        days = pd.bdate_range(end='2026-10-08', periods=n)
        rng = np.random.default_rng(25)
        signal = rng.normal(0, .004, n)
        price = 80*np.exp(np.r_[0, np.cumsum(signal[:-1])])
        s = pd.Series(price, index=days)
        x = dollar_models.design(s, pd.DataFrame({'factor:lag1': signal}, index=days))
        return s, x

    def test_ridge_selects_lagged_signal_on_training_and_exposes_coefficients_window(self):
        s, x = self.sample()
        r = dollar_models.fit_predict(s, x, 1, 'full', 1.)
        self.assertEqual(r['training_start'], s.index[5].date().isoformat())
        self.assertLess(r['training_end'], s.index[-1].date().isoformat())
        selected = next(f for f in r['features'] if f['name'] == 'factor:lag1')
        self.assertGreater(selected['training_correlation'], .99)
        self.assertAlmostEqual(r['prediction'], s.iloc[-1]*np.exp(x['factor:lag1'].iloc[-1]), delta=.03)

    def test_feature_screening_never_reads_future_labels_or_feature_rows(self):
        s, x = self.sample()
        before = dollar_models.fit_predict(s.iloc[:250], x.iloc[:250], 1, 'full', 10.)
        s.iloc[250:] = 99999
        x.iloc[250:] = -99999
        after = dollar_models.fit_predict(s.iloc[:250], x.iloc[:250], 1, 'full', 10.)
        self.assertEqual(before, after)

    def test_missing_target_does_not_become_zero_or_joined_return(self):
        s, x = self.sample()
        complete = dollar_models.fit_predict(s, x, 3, 'full', 10.)
        s.iloc[150] = np.nan
        missing = dollar_models.fit_predict(s, x, 3, 'full', 10.)
        self.assertEqual(complete['training_n']-missing['training_n'], 4)
        self.assertGreater(missing['prediction'], 0)

    def test_release_delay_and_known_publication_are_respected(self):
        index = pd.bdate_range('2026-09-28', '2026-10-08')
        rows = [{'date': '2026-09-28', 'value': 1.}, {'date': '2026-09-29', 'value': 2.},
                {'date': '2026-09-30', 'value': 8., 'published_at': '2026-10-06T12:00:00+00:00'}]
        item = {'id': 'signal', 'rows': rows, 'metadata': {'frequency': 'daily', 'calendar': 'weekdays'}, 'assumed_lag_days': 2}
        x = dollar_models.align_features(index, [item], '2026-10-08T15:00:00+00:00')
        self.assertTrue(pd.isna(x.loc['2026-09-30', 'signal:lag1']))
        self.assertEqual(x.loc['2026-10-01', 'signal:lag1'], 1.)
        self.assertEqual(x.loc['2026-10-06', 'signal:lag1'], 1.)
        self.assertEqual(x.loc['2026-10-07', 'signal:lag1'], 6.)

    def test_outer_rejection_reports_baseline_error_and_baseline_interval(self):
        s, _ = self.sample()
        rows = [{'date': stamp.date().isoformat(), 'value': v} for stamp, v in s.items()]
        def bad(series, x, step, history, cache=None):
            return {'model': 'lagged_ridge', 'prediction': float(series.iloc[-1]+20), 'features': []}
        with patch('thesaurus.dollar_models.select', side_effect=bad):
            points = dollar_models.forecast(rows, {'frequency': 'daily', 'calendar': 'weekdays'}, now=fixtures.NOW, today=date(2026, 10, 8))
        self.assertTrue(all(p['model'] == 'last' for p in points))
        self.assertTrue(all(p['validation_mae'] == p['baseline_mae'] for p in points))
        self.assertTrue(all(p['validation_n'] == 10 for p in points))

    def test_multivariate_model_is_used_when_lagged_signal_improves_outer_mae(self):
        s, x = self.sample()
        rows = [{'date': stamp.date().isoformat(), 'value': v} for stamp, v in s.items()]
        with patch('thesaurus.dollar_models.align_features', return_value=x[['factor:lag1']]):
            points = dollar_models.forecast(rows, {'frequency': 'daily', 'calendar': 'weekdays'},
                                           now=fixtures.NOW, today=date(2026, 10, 8))
        self.assertEqual(points[0]['model'], 'lagged_ridge')
        self.assertLess(points[0]['validation_mae'], .95*points[0]['baseline_mae'])
        self.assertTrue(any(f['name'] == 'factor:lag1' for f in points[0]['features']))

    def test_calendar_carry_does_not_invent_validation_or_a_zero_width_interval(self):
        s, _ = self.sample()
        s.index = pd.bdate_range(end='2026-10-09', periods=len(s))
        rows = [{'date': stamp.date().isoformat(), 'value': v} for stamp, v in s.items()]
        points = dollar_models.forecast(rows, {'frequency': 'daily', 'calendar': 'weekdays'},
                                       now='2026-10-10T09:00:00+00:00', today=date(2026, 10, 10))
        self.assertEqual(points[0]['model'], 'calendar_carry')
        self.assertEqual(points[0]['validation_n'], 0)
        self.assertIsNone(points[0]['lower'])

    def test_moscow_midnight_uses_current_effective_date_but_not_future_training(self):
        now = '2026-10-08T21:10:00+00:00'
        market_store.save_rows(self.store, dollar.USD, [
            {'date': '2026-10-09', 'value': 85., 'retrieved_at': '2026-10-08T21:05:00+00:00'},
            {'date': '2026-10-10', 'value': 9999., 'retrieved_at': '2026-10-08T21:05:00+00:00'}])
        r, _ = dollar.build(self.store, dollar.DEFAULTS, now=now)
        self.assertEqual(r['stats']['date'], '2026-10-09')
        self.assertEqual(r['stats']['last'], 85.)
        self.assertNotIn(r['published_revision'], r['input_revisions'])

    def test_full_history_keeps_old_training_and_recent_mode_filters_it(self):
        s = pd.Series(np.linspace(50, 85, 800), index=pd.bdate_range(end='2026-10-08', periods=800))
        x = dollar_models.design(s, pd.DataFrame(index=s.index))
        full = dollar_models.fit_predict(s, x, 1, 'full', 10.)
        recent = dollar_models.fit_predict(s, x, 1, 'recent', 10.)
        self.assertLess(full['training_start'], '2024-06-13')
        self.assertGreaterEqual(recent['training_start'], '2024-06-13')

    def test_full_history_supports_an_explicit_long_deadline_without_tail_truncation(self):
        s, _ = self.sample()
        rows = [{'date': stamp.date().isoformat(), 'value': v} for stamp, v in s.items()]
        points = dollar_models.forecast(rows, {'frequency': 'daily', 'calendar': 'weekdays'}, now=fixtures.NOW,
                                       today=date(2026, 10, 8), history='full', targets=[(90, date(2027, 1, 6))])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['target'], '2027-01-06')
        self.assertEqual(points[0]['days_ahead'], 90)
        self.assertEqual(points[0]['history_mode'], 'full')

    def test_monthly_features_remain_context_and_are_not_multiplied_by_day(self):
        item = next(r for r in catalogue() if r['id'] == 'cbr.fx_context.balance')
        self.store.execute('UPDATE market_series SET enabled=1 WHERE id=?', (item['id'],))
        chosen, excluded = dollar_models.factors(self.store, [item['id']], now=fixtures.NOW)
        self.assertEqual(chosen, [])
        self.assertEqual(len(excluded), 1)

    def test_full_history_controls_are_private_and_download_admin_only(self):
        controls = self.bot.dollar_controls
        controls.callback(self.client, 2, 'usd:model:history:full')
        controls.callback(self.client, 2, 'usd:model:engine:basic')
        self.assertEqual(controls.config(2)['history_mode'], 'full')
        self.assertEqual(controls.config(3)['history_mode'], 'auto')
        with self.assertRaises(PermissionError):
            controls.callback(self.client, 2, 'usd:update_history')
        with self.assertRaises(ValueError):
            controls.callback(self.client, 2, 'usd:model:history:arbitrary')

    def test_official_empty_csv_is_unavailable_not_network_error(self):
        item = dollar_sources.catalogue()[0]
        t = Mock()
        t.get.return_value = SimpleNamespace(content=b'\n', headers={'Content-Type': 'application/csv'})
        self.assertEqual(dollar_sources.fetch(item, t, date(2021, 12, 27), date(2021, 12, 28)), [(item, [])])
        t.get.return_value = SimpleNamespace(content=b'\n', headers={'Content-Type': 'text/html'})
        with self.assertRaises(ValueError):
            dollar_sources.fetch(item, t, date(2021, 12, 27), date(2021, 12, 28))

    def test_old_absences_are_cached_as_null_recent_absences_retry(self):
        identity = 'fred.brent'
        market.enable(self.store, identity, True)
        self.store.set_setting('market_history_start:'+identity, '2026-08-01')
        t = Mock()
        t.get.return_value = SimpleNamespace(content=b'\n', headers={'Content-Type': 'application/csv'})
        r = market.collect(self.store, selection=[identity], transport=t, today=date(2026, 10, 8))
        self.assertFalse(r['errors'])
        saved = self.store.query('SELECT * FROM market_observations WHERE series_id=?', (identity,))
        self.assertTrue(saved)
        self.assertTrue(all(v['value'] is None and v['quality']=='source_no_observation' for v in saved))
        calls = t.get.call_count
        market.collect(self.store, selection=[identity], transport=t, today=date(2026, 10, 8))
        self.assertEqual(calls, t.get.call_count)

    def test_usd_old_rubles_are_normalized_and_bad_xml_cannot_cache_missing(self):
        xml = b'<ValCurs><Record Date="31.12.1997"><Nominal>1</Nominal><Value>5960</Value></Record><Record Date="01.01.1998"><Nominal>1</Nominal><Value>5.96</Value></Record></ValCurs>'
        item = next(r for r in catalogue() if r['id'] == dollar.USD)
        t = Mock()
        t.get.return_value = SimpleNamespace(content=xml)
        values = fetch(item, t, date(1997, 12, 31), date(1998, 1, 1))[0][1]
        self.assertEqual(values[0]['value'], values[1]['value'])
        self.assertIn('redenominated', values[0]['quality'])
        with self.assertRaises(ValueError):
            parse_currency(b'<Error>broken</Error>')

    def test_full_download_persists_source_boundary_without_sending(self):
        with patch('thesaurus.market.collect', return_value={'errors': []}):
            report = run_job(self.store, self.settings, {'id': 1001, 'kind': 'dollar_collect'},
                             {'actor_id': 1, 'scope': 'context', 'selection': [dollar.USD, 'fred.brent'], 'history': 'full', 'silent': True})
        self.assertEqual(self.store.get_setting('market_history_start:'+dollar.USD), '1992-07-01')
        self.assertEqual(self.store.get_setting('market_history_start:fred.brent'), '1992-07-01')
        self.assertFalse(self.store.query('SELECT * FROM deliveries WHERE report_id=?', (report,)))


if __name__ == '__main__':
    unittest.main()
