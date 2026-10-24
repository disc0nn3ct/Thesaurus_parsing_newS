"""CBR ingestion and Telegram discovery acceptance, without live services."""
import json
import shutil
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from thesaurus import market, market_analysis, market_sources, market_store
from thesaurus.market_jobs import run_market_job
from thesaurus.source_catalog import GROUPS
from thesaurus.store import Store
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_cbr import descendants
from thesaurus.telegram_navigation import CURRENT


class CbrCatalogueTests(unittest.TestCase):
    def test_full_policy_rate_notes_are_text_and_numeric_columns_remain_strict(self):
        identity = 'cbr.hd_base.procstav.ir_chg_mpo.full'
        item = next(i for i in market_sources.catalogue() if i['id'] == identity)
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('cbr_sources.json').read_text(encoding='utf-8'))
        html = fixtures[identity]['html']
        transport = Mock()
        transport.get.return_value.text = html
        rows = market_sources.fetch(item, transport, date(2000, 1, 1), date(2099, 1, 1))
        self.assertTrue(rows)
        self.assertTrue(all(meta['column'] != 'Примечания' for meta, _ in rows))
        self.assertNotIn(64, item['contracts'][0]['numeric'])
        transport.get.return_value.text = html.replace('<td>14,00</td>', '<td>не число</td>', 1)
        with self.assertRaises(ValueError):
            market_sources.fetch(item, transport, date(2000, 1, 1), date(2099, 1, 1))

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('cbr-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, role='admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        market.initialize(self.store)
        self.bot = BotControls(self.store)
        self.reply = Mock()
        self.bot.cbr_controls.reply = self.reply
        self.addCleanup(CURRENT.set, None)

    def token(self, identity):
        return str(self.store.one('SELECT rowid token FROM market_series WHERE id=?', (identity,))['token'])

    def click(self, data, actor=1):
        self.bot.cbr_controls.callback(Mock(), actor, data)

    def test_all_29_roots_discoverable_before_loading(self):
        roots = [identity for _, identities in GROUPS.values() for identity in identities]
        self.assertEqual(len(roots), 29)
        self.assertEqual(len(set(roots)), 29)
        for key, (_, identities) in GROUPS.items():
            if not identities:
                continue
            self.click(f'cb:group:{key}:0', actor=2)
            text = self.reply.call_args.args[2]
            for identity in identities:
                title = self.store.one('SELECT title FROM market_series WHERE id=?', (identity,))['title']
                self.assertIn(title, text)
                children = descendants(self.store, identity)
                self.assertTrue(any(json.loads(row['metadata'])['kind'] in {'dataset', 'series'} for row in children), identity)

    def test_admin_connects_only_chosen_base_and_deduplicates_pending_job(self):
        token = self.token('cbr.hd_base.mb_nd')
        self.click('cb:load:' + token)
        self.click('cb:load:' + token)
        jobs = self.store.query('SELECT * FROM jobs')
        self.assertEqual(len(jobs), 1)
        payload = json.loads(jobs[0]['payload'])
        self.assertTrue(payload['notify_actor'])
        self.assertEqual(len(payload['selection']), 2)
        self.assertFalse(self.store.get_setting('market_schedule', {}).get('enabled'))
        self.assertFalse(self.store.one("SELECT enabled FROM market_series WHERE id='cbr.hd_base.metall.metall_base_new'")['enabled'])
        self.assertTrue(self.store.one("SELECT enabled FROM market_series WHERE id='cbr.hd_base.mb_nd'")['enabled'])

    def test_subscriber_cannot_load_or_change_source_parameters(self):
        token = self.token('cbr.hd_base.seldomc.sc_dynamic')
        for data in ['cb:load:' + token, 'cb:params:' + token + ':0', 'cb:param:' + token + ':8']:
            with self.assertRaises(PermissionError):
                self.click(data, actor=2)
        self.assertFalse(self.store.query('SELECT * FROM jobs'))
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=?', (2,))
        with self.assertRaises(PermissionError):
            self.click('cb:menu', actor=2)

    def test_date_bound_only_for_archives_and_svs_both_directions(self):
        items = market_sources.catalogue()
        for item in items:
            if item.get('data_type') == 'actual':
                self.assertNotIn('available_end', item, item['id'])
        directions = {item.get('default_params', {}).get('UniDbQuery.flow') for item in items if item['id'].startswith('cbr.hd_base.svs_db') and item['kind'] == 'dataset'}
        self.assertEqual(directions, {'1', '2'})
        self.assertEqual(len([item for item in items if item.get('adapter') == 'currency']), 54)

    def test_official_tables_parse_save_and_reach_analytics_without_zero_fill(self):
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('cbr_sources.json').read_text(encoding='utf-8'))
        entries = {item['id']: item for item in market_sources.catalogue()}
        populated = 0
        for identity, fixture in fixtures.items():
            item = entries[identity]
            transport = Mock()
            transport.get.return_value.text = fixture['html']
            output = market_sources.fetch(item, transport, date(1900, 1, 1), date(2026, 10, 7))
            self.assertTrue(output, identity)
            for meta, rows in output:
                meta.pop('_snapshot', None)
                market_store.register(self.store, [meta])
                market_store.save_rows(self.store, meta['id'], rows)
                history = market_store.history(self.store, meta['id'])
                self.assertEqual(len(history), len({row['date'] for row in rows}), identity)
                if history:
                    values = market_analysis.values(history, meta)
                    self.assertEqual(values.notna().sum(), sum(row['value'] is not None for row in history), identity)
                if any(row['value'] is not None for row in rows):
                    self.assertNotEqual(meta['unit'], 'source_units', (identity, meta.get('column')))
                    populated += 1
        self.assertGreater(populated, 200)

    def test_ruonia_publication_date_is_not_a_series_dimension(self):
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('cbr_sources.json').read_text(encoding='utf-8'))
        identity = 'cbr.hd_base.ruonia.dynamics'
        item = next(meta for meta in market_sources.catalogue() if meta['id'] == identity)
        transport = Mock()
        transport.get.return_value.text = fixtures[identity]['html']
        output = market_sources.fetch(item, transport, date(1900, 1, 1), date(2026, 10, 7))
        self.assertEqual(len(output), 8)
        self.assertTrue(all(not meta['dimensions'] for meta, _ in output))
        self.assertTrue(all(row['published_at'] for _, rows in output for row in rows))

    def test_html_spans_stop_at_tbody_and_cancelled_auction_is_not_zero(self):
        html = '<table><thead><tr><th rowspan="3">Дата</th><th>Объем</th></tr></thead><tbody><tr><td>07.10.2026</td><td>-4</td></tr></tbody></table>'
        rows = market_sources.parse_table(html, {'table': 0, 'headers': ['Дата', 'Объем'], 'numeric': [1], 'dimensions': []})
        self.assertEqual(rows[0]['value'], -4)
        event = 'Аукцион не состоялся в связи с отсутствием заявок.'
        html = f'<table><tr><td>Дата</td><td>Объем</td></tr><tr><td>07.10.2026</td><td colspan="8">{event}</td></tr></table>'
        self.assertEqual(market_sources.parse_table(html, {'table': 0, 'headers': ['Дата', 'Объем'], 'numeric': [1], 'dimensions': []}), [])

    def test_targeted_collect_enables_dimensioned_children_and_reuses_history(self):
        identity = 'cbr.hd_base.metall.metall_base_new'
        self.click('cb:load:' + self.token(identity))
        payload = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        meta = {'id': identity + '.gold', 'source': 'cbr', 'dataset': identity, 'title': 'Золото',
                'kind': 'series', 'adapter': 'html', 'frequency': 'daily', 'calendar': 'calendar',
                'unit': 'руб за грамм', 'dimensions': 'Металл=золото', 'url': 'https://cbr.ru/hd_base/metall/'}
        observations = [{'date': '2026-10-07', 'value': 10}]
        with patch('thesaurus.market_sources.fetch', return_value=[(meta, observations)]) as fetch:
            result = market.collect(self.store, selection=payload['selection'], today=date(2026, 10, 7), transport=Mock())
            self.assertFalse(result['errors'])
            first_count = fetch.call_count
            market.collect(self.store, selection=payload['selection'], today=date(2026, 10, 7), transport=Mock())
            self.assertEqual(fetch.call_count, first_count)
        self.assertTrue(self.store.one('SELECT enabled FROM market_series WHERE id=?', (meta['id'],))['enabled'])
        self.assertIn(meta['id'], {row['id'] for row in market.selected(self.store, 2)})

    def test_chart_card_clears_previous_multi_selection(self):
        market.enable(self.store, 'cbr.fx.usd', True)
        self.store.set_setting('chart:2:market', {'period': 'year', 'horizon': 4, 'identity': None,
            'series_ids': ['cbr.fx.eur'], 'selection_mode': 'multiple'})
        self.bot.chart_controls.menu = Mock()
        self.click('cb:chart:' + self.token('cbr.fx.usd'), actor=2)
        options = self.store.get_setting('chart:2:market')
        self.assertEqual(options['identity'], 'cbr.fx.usd')
        self.assertFalse(options.get('series_ids'))
        self.assertFalse(options.get('selection_mode'))

    def test_load_completion_is_private_durable_and_recovery_does_not_recollect(self):
        identity = 'cbr.hd_base.metall.metall_base_new'
        self.click('cb:load:' + self.token(identity))
        job = self.store.one('SELECT * FROM jobs')
        payload = json.loads(job['payload'])
        settings = SimpleNamespace(reports_dir=self.root / 'reports')
        with patch('thesaurus.market.collect', return_value={'errors': [], 'status': 'ok'}) as collect, patch('thesaurus.market_alerts.process'):
            report = run_market_job(self.store, settings, job, payload, charts=False)
            recovered = run_market_job(self.store, settings, job, payload, charts=False)
        self.assertEqual(report, recovered)
        self.assertEqual(collect.call_count, 1)
        self.assertEqual({row['recipient'] for row in self.store.query('SELECT recipient FROM deliveries')}, {'1'})

    def test_fx_grid_follows_effective_dates(self):
        days = market_analysis.grid('2026-10-05', '2026-10-11', {'frequency': 'daily', 'calendar': 'cbr_fx'})
        self.assertEqual([day.dayofweek for day in days], [1, 2, 3, 4, 5])

    def test_correlations_use_common_publication_days_without_inventing_values(self):
        left_meta = {'frequency': 'daily', 'calendar': 'cbr_fx', 'transform': 'difference'}
        right_meta = {**left_meta, 'calendar': 'russia_working'}
        def history(meta):
            return [{'date': stamp.date().isoformat(), 'value': float((stamp-pd.Timestamp('2026-01-01')).days**2)}
                    for stamp in market_analysis.grid('2026-01-01', '2026-09-30', meta)]
        result = market_analysis.correlations(history(left_meta), history(right_meta), left_meta, right_meta, window=60)
        self.assertEqual(result['status'], 'ok')
        self.assertGreaterEqual(result['coverage'], .9)
        self.assertGreater(result['pearson'], .6)

    def test_miacr_rates_and_volumes_have_distinct_units_and_aggregation(self):
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('cbr_sources.json').read_text(encoding='utf-8'))
        identity = 'cbr.hd_base.mkr.mkr_base'
        item = next(meta for meta in market_sources.catalogue() if meta['id'] == identity)
        transport = Mock()
        transport.get.return_value.text = fixtures[identity]['html']
        output = market_sources.fetch(item, transport, date(1900, 1, 1), date(2026, 10, 7))
        for meta, _ in output:
            index = meta['table_identity']
            self.assertEqual(meta['unit'], '%' if index < 4 else 'млн долларов США' if index == 5 else 'млн руб')
            self.assertEqual(meta['aggregation'], 'mean' if index < 4 else 'sum')
            self.assertIn('ставка' if index < 4 else 'объём', meta['title'])
            self.assertEqual(meta['id'], market_sources.metric_id(identity, f'Таблица {index+1}: ' + meta['column']))

    def test_existing_miacr_metadata_is_repaired_without_history_requests(self):
        dataset = 'cbr.hd_base.mkr.mkr_base'
        item = next(meta for meta in market_sources.catalogue() if meta['id'] == dataset)
        identity = market_sources.metric_id(dataset, 'Таблица 7: 1 день')
        old = {**item, 'id': identity, 'dataset': dataset, 'kind': 'series', 'table_identity': 6,
               'column': '1 день', 'title': 'Старое название', 'unit': '%', 'aggregation': 'mean'}
        market_store.register(self.store, [old])
        market.enable(self.store, identity, True)
        market_store.save_rows(self.store, identity, [{'date': '2026-10-06', 'value': 645313.6}])
        before = market_store.history(self.store, identity)
        revisions = self.store.query('SELECT * FROM market_revisions WHERE series_id=?', (identity,))
        with patch('thesaurus.market_sources.fetch', side_effect=AssertionError('No download')):
            market.initialize(self.store)
        row = self.store.one('SELECT * FROM market_series WHERE id=?', (identity,))
        updated = json.loads(row['metadata'])
        self.assertEqual(updated['unit'], 'млн руб')
        self.assertEqual(updated['aggregation'], 'sum')
        self.assertEqual(market_store.history(self.store, identity), before)
        self.assertEqual(self.store.query('SELECT * FROM market_revisions WHERE series_id=?', (identity,)), revisions)
        self.assertTrue(row['enabled'])


if __name__ == '__main__':
    unittest.main()
