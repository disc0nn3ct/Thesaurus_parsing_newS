"""Official empty intervals, partial tables, gap caching and readable diagnostics."""
import json
import shutil
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from requests import HTTPError

from thesaurus import (
    market,
    market_analysis,
    market_incremental,
    market_sources,
    market_store,
)
from thesaurus.source_errors import diagnostic
from thesaurus.store import Store


class CbrSourceReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('cbr-reliability-' + uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.meta = {'id': 'test.rate', 'dataset': 'test.rate', 'source': 'cbr', 'kind': 'series',
                     'title': 'Ставка', 'unit': '%', 'frequency': 'daily', 'calendar': 'calendar',
                     'transform': 'difference', 'url': 'https://cbr.ru/hd_base/keyrate/', 'contracts': [], 'params': {}}
        market_store.register(self.store, [self.meta])
        market.enable(self.store, self.meta['id'], True)

    def test_header_only_yield_curve_is_valid_empty_not_schema_drift(self):
        meta = next(m for m in market_sources.catalogue() if m['id'] == 'cbr.hd_base.zcyc_params')
        maturities = ['0,25', '0,5', '0,75', '1', '2', '3', '5', '7', '10', '15', '20', '30']
        html = '<table><tr><th rowspan="2">Дата</th><th colspan="12">Срок до погашения, лет</th></tr><tr>'
        html += ''.join('<th>' + value + '</th>' for value in maturities) + '</tr></table>'
        contract = meta['contracts'][0]
        self.assertEqual(market_sources.parse_table(html, contract), [])
        with self.assertRaises(ValueError):
            market_sources.parse_table(html.replace('0,25', '0,2'), contract)
        with self.assertRaises(ValueError):
            market_sources.parse_table('<table></table>', contract)

    def test_monthly_miacr_keeps_present_table_when_other_three_are_empty(self):
        meta = next(m for m in market_sources.catalogue() if m['id'] == 'cbr.hd_base.mkr.mkr_monthes')
        header = '<tr>' + ''.join('<th>'+h+'</th>' for h in meta['contracts'][0]['headers']) + '</tr>'
        values = ['октябрь 2008 г.', '7,78', '10,83', '12,38', '13,23', '8,98', '—']
        html = '<table>'+header+'<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr></table>'
        html += ('<table>'+header+'</table>')*3
        transport = Mock()
        transport.get.return_value.text = html
        result = market_sources.fetch(meta, transport, date(2008, 10, 1), date(2008, 10, 31))
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0][1][0]['value'], 7.78)
        self.assertIsNone(result[-1][1][0]['value'])
        self.assertTrue(all(m['table_identity'] == 0 for m, _ in result))

    def test_official_empty_notice_is_accepted_for_all_reviewed_html_sources(self):
        transport = Mock()
        transport.get.return_value.text = '<input name="UniDbQuery.From"><p>За выбранный вами период нет информации.</p>'
        identities = {'cbr.hd_base.impliedratecny', 'cbr.hd_base.ruonia.dynamics', 'cbr.hd_base.metall.metall_base_old',
                      'cbr.hd_base.mkr.retro.mkr_retro_mibid'}
        for meta in market_sources.catalogue():
            if meta['id'] in identities:
                self.assertEqual(market_sources.fetch(meta, transport, date(2025, 1, 1), date(2025, 1, 2)), [])
        transport.get.return_value.text = '<p>За выбранный вами период нет информации.</p>'
        with self.assertRaises(ValueError):
            market_sources.fetch(next(m for m in market_sources.catalogue() if m['id'] == 'cbr.hd_base.impliedratecny'),
                                 transport, date(2025, 1, 1), date(2025, 1, 2))

    def test_confirmed_historical_absence_is_not_redownloaded_after_ttl(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': 1}, {'date': '2025-01-03', 'value': 3}])
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        market_incremental.remember(self.store, self.meta, date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 1), now)
        self.assertEqual(market_incremental.plan(self.store, self.meta, date(2025, 1, 1), date(2025, 1, 4), now+timedelta(days=30)),
                         [(date(2025, 1, 4), date(2025, 1, 4))])
        self.assertEqual(len(market_store.history(self.store, 'test.rate')), 2)
        quality = market_analysis.diagnostics(market_store.history(self.store, 'test.rate'), self.meta)
        self.assertEqual(quality['missing'], 1)
        self.assertAlmostEqual(quality['coverage'], 2/3)

    def test_recent_unavailable_date_remains_retryable(self):
        first, last = date(2026, 10, 7), date(2026, 10, 9)
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        market_store.save_rows(self.store, 'test.rate', [{'date': first.isoformat(), 'value': 1}])
        market_incremental.remember(self.store, self.meta, first, last, first, now)
        self.assertEqual(market_incremental.plan(self.store, self.meta, first, last, now), [])
        self.assertEqual(market_incremental.plan(self.store, self.meta, first, last, now+timedelta(hours=2)),
                         [(date(2026, 10, 8), date(2026, 10, 9))])

    def test_confirmed_absence_is_scoped_to_source_parameters(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': 1}])
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        market_incremental.remember(self.store, self.meta, date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 1), now)
        changed = {**self.meta, 'params': {'UniDbQuery.ValID': 'other'}}
        self.assertEqual(market_incremental.plan(self.store, changed, date(2025, 1, 1), date(2025, 1, 3), now),
                         [(date(2025, 1, 1), date(2025, 1, 3))])

    def test_volatility_does_not_connect_across_missing_observation(self):
        observations = [{'date': f'2025-01-0{i}', 'value': value} for i, value in enumerate([100, 101, None, 200, 201, 200], 1)]
        meta = {**self.meta, 'transform': 'returns'}
        result = market_analysis.diagnostics(observations, meta)
        expected = np.log(pd.Series([100, 101, np.nan, 200, 201, 200])).diff().std()
        self.assertAlmostEqual(result['volatility'], expected)
        self.assertEqual(result['missing'], 1)
        self.assertEqual(result['previous_date'], '2025-01-05')

    def test_off_calendar_values_do_not_inflate_coverage(self):
        observations = [{'date': day, 'value': value} for day, value in [('2025-01-03', 1), ('2025-01-04', 2), ('2025-01-07', 3)]]
        result = market_analysis.diagnostics(observations, {**self.meta, 'calendar': 'weekdays'})
        self.assertEqual(result['expected'], 3)
        self.assertEqual(result['off_calendar'], 1)
        self.assertEqual(result['missing'], 1)
        self.assertAlmostEqual(result['coverage'], 2/3)

    def test_sparse_prices_do_not_emit_nan_volatility(self):
        observations = [{'date': day, 'value': 100} for day in ('2025-01-01', '2025-01-03', '2025-01-05')]
        result = market_analysis.diagnostics(observations, {**self.meta, 'transform': 'returns'})
        self.assertIsNone(result['volatility'])
        json.dumps(result, allow_nan=False)

    def test_safe_errors_distinguish_schema_http_and_private_exception(self):
        self.assertEqual(diagnostic(ValueError('CBR table headers changed'))[0], 'schema_changed')
        exc = HTTPError('https://user:password@proxy.invalid')
        exc.response = Mock(status_code=403)
        self.assertEqual(diagnostic(exc), ('http_error', 'Источник вернул HTTP 403.'))
        self.assertNotIn('password', str(diagnostic(ValueError('password'))))

    def test_conflicting_source_dates_are_quarantined_without_losing_other_history(self):
        observations = [{'date': '2025-01-01', 'value': 7}, {'date': '2025-01-02', 'value': 8},
                        {'date': '2025-01-02', 'value': 9}, {'date': '2025-01-03', 'value': 10},
                        {'date': '2025-01-03', 'value': 10}]
        self.store.set_setting('market_history_start:test.rate', '2025-01-01')
        meta = {**self.meta, 'available_end': '2025-01-03'}
        market_store.register(self.store, [meta])
        with patch('thesaurus.market.sources.fetch', return_value=[(meta, observations)]) as fetch:
            result = market.collect(self.store, today=date(2026, 10, 9), selection=['test.rate'])
            self.assertEqual(result['errors'][0]['code'], 'source_conflict')
            saved = market_store.history(self.store, 'test.rate')
            self.assertEqual([row['value'] for row in saved], [7, None, 10])
            self.assertEqual(saved[1]['quality'], 'source_conflicting_values')
            self.assertEqual(self.store.one("SELECT COUNT(*) n FROM integrity_incidents WHERE kind='source_conflict'")['n'], 1)
            details = json.loads(self.store.one("SELECT new_payload FROM integrity_incidents WHERE kind='source_conflict'")['new_payload'])
            self.assertEqual([row['value'] for row in details['candidates']], [8, 9])
            fetch.reset_mock()
            again = market.collect(self.store, today=date(2026, 10, 9), selection=['test.rate'])
            fetch.assert_not_called()
            self.assertEqual(again['errors'][0]['code'], 'source_conflict')

    def test_conflict_detection_preserves_previously_saved_fact(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-02', 'value': 8}])
        observations, conflicts = market_store.partition_rows([{'date': '2025-01-02', 'value': 9},
                                                               {'date': '2025-01-02', 'value': 10}])
        self.assertIn('2025-01-02', conflicts)
        market_store.save_rows(self.store, 'test.rate', market_incremental.only_missing(self.store, 'test.rate', observations))
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], 8)

    def test_report_explains_period_and_changes_without_raw_json(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': 10}, {'date': '2025-01-03', 'value': 11}])
        self.store.set_setting('market_last_collection', {'quality': {'errors': [{'source': 'test.rate', 'error': 'Источник вернул HTTP 403.'}]}})
        pair = {'left': 'test.rate', 'right': 'test.rate', 'status': 'ok', 'pearson': .96, 'spearman': .95,
                'n': 60, 'window': 60, 'frequency': 'daily', 'coverage': 1, 'lag': 0, 'levels': False}
        with patch('thesaurus.market.pair_results', return_value=[pair]):
            report_id = market.report(self.store, 1, charts=False, forecast_enabled=False)
        full = self.store.one('SELECT full_text FROM reports WHERE id=?', (report_id,))['full_text']
        self.assertIn('2025-01-01 — 2025-01-03', full)
        self.assertIn('+1 п.п. с 2025-01-01', full)
        self.assertIn('Pearson 0.96', full)
        self.assertIn('• Ставка: Источник вернул HTTP 403.', full)
        self.assertNotIn('"pearson"', full)
        self.assertNotIn('"error"', full)
        self.assertEqual(json.loads(self.store.one('SELECT metrics FROM reports WHERE id=?', (report_id,))['metrics'])['test.rate']['missing'], 1)


if __name__ == '__main__':
    unittest.main()
