"""Expanded FX acceptance: official data contracts, caching, short decisions and UX."""
import json
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_dollar as fixtures

from thesaurus import dollar, dollar_sources, dollar_timing, market, market_store
from thesaurus.delivery import prepare_delivery
from thesaurus.dollar_jobs import run_job
from thesaurus.market_analysis import grid
from thesaurus.market_sources import catalogue, fetch
from thesaurus.store import dumps

NOW = fixtures.NOW


class ExpandedDollarTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp
    load = fixtures.DollarBotTests.load
    config = fixtures.DollarBotTests.config

    def balance_html(self, exports='100,9'):
        fields = ''.join(f'<tr><td>{name}</td><td>{exports if name == "Exports of goods" else "16,9"}</td><td>99</td><td>-8,1</td></tr>' for name in dollar_sources.TRADE)
        return '<table><tr><th></th><th>2025</th><th>2025</th><th>2026</th></tr><tr><th>Key Aggregates</th><th>Q4</th><th>2025</th><th>Q1</th></tr>'+fields+'</table>'

    def test_fred_contract_missing_duplicates_and_nonfinite(self):
        rows = dollar_sources.parse_fred('observation_date,DFF\n2026-10-01,3.88\n2026-10-02,.\n', 'DFF')
        self.assertEqual(rows[0]['value'], 3.88)
        self.assertIsNone(rows[1]['value'])
        for text in ('date,wrong\n2026-10-01,1\n', 'DATE,DFF\n2026-10-01,nan\n',
                     'DATE,DFF\n2026-10-01,1\n2026-10-01,2\n', 'DATE,DFF\n2026-10-01,1,2\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                dollar_sources.parse_fred(text, 'DFF')

    def test_balance_excludes_yearly_totals_and_preserves_signed_flows(self):
        item = dollar_sources.catalogue()[-1]
        pairs = dollar_sources.parse_balance(item, self.balance_html())
        self.assertEqual(len(pairs), 5)
        self.assertEqual([r['date'] for r in pairs[0][1]], ['2025-10-01', '2026-01-01'])
        self.assertEqual(pairs[0][1][1]['value'], -8.1)
        self.assertEqual(pairs[0][1][1]['period_end'], '2026-03-31')
        with self.assertRaises(ValueError):
            dollar_sources.parse_balance(item, self.balance_html().replace('Trade balance', 'Changed row'))

    def test_unpublished_inflation_is_empty_but_structure_change_fails(self):
        item = next(r for r in catalogue() if r['id'] == 'cbr.hd_base.infl')
        transport = Mock()
        transport.get.return_value = SimpleNamespace(text='<input name="UniDbQuery.From"><div>За выбранный вами период нет информации.</div>')
        self.assertEqual(fetch(item, transport, date(2026, 9, 1), date(2026, 10, 8)), [])
        transport.get.return_value = SimpleNamespace(text='<div>Structure changed</div>')
        with self.assertRaises(ValueError):
            fetch(item, transport, date(2026, 9, 1), date(2026, 10, 8))

    def test_fred_calendar_and_incremental_download_do_not_repeat_history(self):
        item = next(r for r in catalogue() if r['id'] == 'fred.usd_broad')
        market.enable(self.store, item['id'], True)
        self.store.set_setting('market_history_years', {'daily': 1})
        first = date(2026, 10, 8)
        transport = Mock()
        def response(url, params):
            days = grid(params['cosd'], params['coed'], item)
            content = ('observation_date,DTWEXBGS\n'+''.join(f'{d.date().isoformat()},120\n' for d in days)).encode()
            return SimpleNamespace(content=content)
        transport.get.side_effect = response
        self.assertFalse(market.collect(self.store, today=first, selection=[item['id']], transport=transport)['errors'])
        count = transport.get.call_count
        self.assertFalse(market.collect(self.store, today=first, selection=[item['id']], transport=transport)['errors'])
        self.assertEqual(transport.get.call_count, count)
        market.collect(self.store, today=first+timedelta(days=1), selection=[item['id']], transport=transport)
        params = transport.get.call_args.args[1]
        self.assertEqual(params['cosd'], '2026-10-09')
        self.assertEqual(params['coed'], '2026-10-09')
        self.assertEqual(len(grid('2026-10-10', '2026-10-11', item)), 0)

    def test_snapshot_caches_day_detects_revision_and_keeps_original(self):
        identity = 'cbr.fx_context.balance'
        market.enable(self.store, identity, True)
        transport = Mock()
        transport.get.return_value = SimpleNamespace(text=self.balance_html())
        first = market.collect(self.store, today=date(2026, 10, 8), selection=[identity], transport=transport)
        self.assertFalse(first['errors'])
        self.assertEqual(len(self.store.query('SELECT * FROM market_observations')), 10)
        market.collect(self.store, today=date(2026, 10, 8), selection=[identity], transport=transport)
        self.assertEqual(transport.get.call_count, 1)
        transport.get.return_value = SimpleNamespace(text=self.balance_html(exports='101,9'))
        result = market.collect(self.store, today=date(2026, 10, 9), selection=[identity], transport=transport)
        self.assertFalse(result['errors'])
        incidents = self.store.query("SELECT * FROM integrity_incidents WHERE kind='source_revision'")
        self.assertEqual(len(incidents), 1)
        old = self.store.one("SELECT value FROM market_observations WHERE series_id=? AND date='2025-10-01'", (identity+'.exports',))
        self.assertEqual(old['value'], 100.9)

    def test_purchase_wizard_blocks_incomplete_job_overview_does_not(self):
        controls = self.bot.dollar_controls
        controls.callback(self.client, 2, 'usd:mode:purchase')
        controls.callback(self.client, 2, 'usd:run', 'incomplete')
        self.assertFalse(self.store.query('SELECT * FROM jobs'))
        controls.callback(self.client, 2, 'usd:date:7')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'dollar:quote')
        controls.input(self.client, 2, '90,5')
        controls.callback(self.client, 2, 'usd:run', 'complete')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        controls.callback(self.client, 3, 'usd:run', 'overview')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 2)

    def test_pack_selection_is_admin_only_and_does_not_disable_shared_sources(self):
        identity = 'fred.brent'
        market.enable(self.store, identity, True)
        token = self.store.one('SELECT rowid token FROM market_series WHERE id=?', (identity,))['token']
        with self.assertRaises(PermissionError):
            self.bot.dollar_controls.callback(self.client, 2, f'usd:source:{token}')
        with self.assertRaises(PermissionError):
            self.bot.dollar_controls.callback(self.client, 2, 'usd:update_all')
        self.bot.dollar_controls.callback(self.client, 1, f'usd:source:{token}')
        self.assertNotIn(identity, self.store.get_setting('dollar_source_pack'))
        self.assertTrue(self.store.one('SELECT enabled FROM market_series WHERE id=?', (identity,))['enabled'])

    def test_small_savings_are_not_a_wait_signal_and_deadline_limits_comparison(self):
        config = {**self.config(), 'trip': '2026-10-10', 'mode': 'purchase'}
        point = {'target': '2026-10-09', 'prediction': 89.9, 'validation_mae': 1.,
                 'validation_n': 16, 'days_ahead': 1, 'native_step': 1}
        result = dollar_timing.comparison([point], config, spot=90., days=2, ready=True)[0]
        self.assertLess(result['saving'], result['error_rub'])
        self.assertIn('меньше типичной ошибки', result['assessment'])
        point['days_ahead'] = 7
        result = dollar_timing.comparison([point], config, spot=90., days=2, ready=True)[0]
        self.assertIn('Позже вашего срока', result['assessment'])

    def test_short_models_use_past_only_and_minimum_sample(self):
        self.load()
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (dollar.USD,))['metadata'])
        rows = market_store.history(self.store, dollar.USD)
        rows.append({'date': '2026-10-09', 'value': 9999., 'published_at': '2026-10-10T00:00:00+00:00'})
        with patch('thesaurus.dollar_timing.analysis.model_prediction', wraps=market.analysis.model_prediction) as model:
            points = dollar_timing.forecast(rows, meta, now=NOW, today=date(2026, 10, 8))
        self.assertEqual([p['days_ahead'] for p in points], [1, 2, 3, 7])
        self.assertTrue(all(np.max(c.args[0]) < 9999 for c in model.call_args_list))
        self.assertEqual(dollar_timing.forecast(rows[:20], meta, now=NOW, today=date(2026, 10, 8)), [])

    def test_published_next_day_is_not_model_training_data(self):
        self.load()
        market_store.save_rows(self.store, dollar.USD, [{'date': '2026-10-09', 'value': 98., 'retrieved_at': NOW}])
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]):
            result, _ = dollar.build(self.store, {**dollar.DEFAULTS, 'mode': 'overview'}, now=NOW)
        self.assertLess(result['stats']['last'], 98.)
        first = result['timing'][0]
        self.assertEqual(first['prediction'], 98.)
        self.assertTrue(first['known_published'])
        self.assertIsNone(first['validation_mae'])
        self.assertNotIn(first['known_published_revision'], result['input_revisions'])
        self.assertIn('уже опубликован', dollar.summary(result))

    def test_dollar_delivery_is_short_and_does_not_compare_with_market(self):
        self.load()
        self.store.execute("INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,metrics) VALUES('old','market',?,1,0,'old','old',?)", (NOW, dumps({'old': {'unit': '%', 'value': 5}})))
        prepare_delivery(self.store, 'old', '2', detail='full')
        self.store.execute("UPDATE deliveries SET state='sent',message_id=10")
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]):
            report = run_job(self.store, self.settings, {'id': 991, 'kind': 'dollar_analysis'}, {'actor_id': 2, 'config': dollar.DEFAULTS}, charts=False)
        record = self.store.one('SELECT * FROM reports WHERE id=?', (report,))
        delivered = self.store.query('SELECT payload FROM deliveries WHERE report_id=?', (report,))
        text = '\n'.join(json.loads(r['payload']).get('text', '') for r in delivered)
        self.assertNotIn('изменился набор данных', text)
        self.assertNotIn('Pearson', text)
        self.assertLess(len(record['text']), len(record['full_text']))
        self.assertTrue(any(json.loads(r['payload']).get('buttons') for r in delivered))

    def test_context_collection_partial_and_dry_run_does_not_enable(self):
        selection = ['cbr.fx.usd', 'fred.brent']
        with patch('thesaurus.market.collect', side_effect=[{'errors': []}, {'errors': [{'source': 'fred.brent', 'error': 'safe'}]}]) as collect:
            report = run_job(self.store, self.settings, {'id': 992, 'kind': 'dollar_collect'},
                             {'actor_id': 1, 'scope': 'context', 'selection': selection}, dry_run=True)
        self.assertEqual(collect.call_count, 2)
        self.assertFalse(self.store.one("SELECT enabled FROM market_series WHERE id='fred.brent'")['enabled'])
        self.assertIn('неполное', self.store.one('SELECT text FROM reports WHERE id=?', (report,))['text'])
        self.assertEqual(self.store.get_setting('dollar_context_collection')['status'], 'partial')


if __name__ == '__main__':
    unittest.main()
