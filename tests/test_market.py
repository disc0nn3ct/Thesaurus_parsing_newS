"""Offline acceptance of market adapters, vintages, models, permissions and jobs."""
import csv
import json
import shutil
import sqlite3
import unittest
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd

from thesaurus import market, market_analysis, market_sources, market_store
from thesaurus.config import Settings
from thesaurus.market_alerts import add_rule, evaluate_rule
from thesaurus.market_jobs import export_market
from thesaurus.service import Worker
from thesaurus.store import Store
from thesaurus.telegram_bot import BotControls


def series(identity='test.rate', frequency='daily', **extra):
    return {'id': identity, 'source': 'cbr', 'dataset': identity, 'title': identity, 'kind': 'series',
            'frequency': frequency, 'calendar': 'calendar', 'unit': '%', 'transform': 'difference',
            'aggregation': 'last', 'url': 'https://cbr.ru/hd_base/keyrate/', **extra}


def rows(n=160, frequency='daily', values=None):
    freq = {'daily': 'D', 'weekly': 'W-FRI', 'monthly': 'MS'}[frequency]
    dates = pd.date_range('2023-01-01', periods=n, freq=freq)
    return [{'date': d.date().isoformat(), 'value': float(values[i] if values is not None else i + 10),
             'definition': '1', 'retrieved_at': '2026-01-01T00:00:00+00:00'} for i, d in enumerate(dates)]


class MarketTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ('market_test_' + uuid4().hex)
        self.root.mkdir()
        self.store = Store(self.root / 'db.sqlite3')
        self.store.add_user(1, role='admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        market_store.register(self.store, [series()])
        market.enable(self.store, 'test.rate', True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def message(self, text, user=1):
        return SimpleNamespace(text=text, id=uuid4().hex, from_user=SimpleNamespace(id=user),
            chat=SimpleNamespace(id=user, type='private'))

    def test_transactional_upgrade_has_backup_and_preserves_users(self):
        self.assertEqual(self.store.one('PRAGMA user_version')['user_version'], 3)
        self.assertEqual(len(list(self.root.glob('*.market-backup-*.sqlite3'))), 1)
        Store(self.store.path)
        self.assertEqual(len(list(self.root.glob('*.market-backup-*.sqlite3'))), 1)
        self.assertTrue(self.store.authorize(1, admin=True))

    def test_v1_upgrade(self):
        path = self.root / 'legacy.db'
        from thesaurus.store import SCHEMA
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(SCHEMA)
            conn.execute('PRAGMA user_version=1')
            conn.execute("INSERT INTO users VALUES(7,'subscriber',1,'7','2020-01-01')")
            conn.commit()
        upgraded = Store(path)
        self.assertTrue(upgraded.authorize(7))
        self.assertEqual(upgraded.one('PRAGMA user_version')['user_version'], 3)

    def test_vintages_are_asof_and_unchanged_load_is_idempotent(self):
        base = {'date': '2025-01-01', 'value': 10, 'retrieved_at': '2025-01-02T00:00:00+00:00'}
        market_store.save_rows(self.store, 'test.rate', [base, base])
        market_store.save_rows(self.store, 'test.rate', [{**base, 'value': 12, 'retrieved_at': '2025-02-01T00:00:00+00:00'}])
        self.assertEqual(len(self.store.query('SELECT * FROM market_revisions')), 2)
        self.assertEqual(market_store.history(self.store, 'test.rate', as_of='2025-01-03T00:00:00+00:00')[0]['value'], 10)
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], 12)

    def test_unknown_and_missing_are_not_zero(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': None}])
        self.assertIsNone(market_store.history(self.store, 'test.rate')[0]['value'])
        self.assertEqual(market_analysis.diagnostics(market_store.history(self.store, 'test.rate'), series())['status'], 'missing')

    def test_disabled_sources_make_no_network_calls(self):
        market.enable(self.store, 'test.rate', False)
        transport = Mock()
        market.collect(self.store, transport=transport)
        transport.get.assert_not_called()

    def test_dry_run_never_uses_transport(self):
        transport = Mock()
        self.assertTrue(market.collect(self.store, dry_run=True, transport=transport)['cached'])
        transport.get.assert_not_called()

    def test_partial_collection_redacts_errors(self):
        item = series('broken', adapter='currency', code='R01235')
        market_store.register(self.store, [item])
        market.enable(self.store, 'broken', True)
        transport = Mock()
        transport.get.side_effect = RuntimeError('secret-provider-value')
        result = market.collect(self.store, transport=transport)
        self.assertEqual(result['status'], 'partial')
        self.assertNotIn('secret-provider-value', json.dumps(result))

    def test_all_source_roots_registered_disabled(self):
        catalog = market_sources.catalogue()
        roots = [item for item in catalog if item.get('parent') is None and item['source'] == 'cbr' and item['id'].startswith(('cbr.hd_base.', 'cbr.currency_base', 'cbr.pr', 'cbr.oper_br.', 'cbr.statistics.'))]
        self.assertEqual(len(roots), 29)
        market.initialize(self.store)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE enabled=1")['n'], 1)

    def test_html_contract_and_schema_drift(self):
        html = '<table><tr><th>Дата</th><th>Ставка</th></tr><tr><td>01.01.2025</td><td>−1,5</td></tr></table>'
        contract = market_sources.contracts(html)[0]
        self.assertEqual(market_sources.parse_table(html, contract)[0]['value'], -1.5)
        with self.assertRaises(ValueError):
            market_sources.parse_table(html.replace('Ставка', 'Другое'), contract)

    def test_html_spans(self):
        html = '<table><tr><th rowspan="2">Дата</th><th colspan="2">Ставки</th></tr><tr><th>А</th><th>Б</th></tr><tr><td>01.01.2025</td><td>1</td><td>2</td></tr></table>'
        contract = market_sources.contracts(html)[0]
        self.assertEqual(contract['headers'], ['Дата', 'Ставки / А', 'Ставки / Б'])
        self.assertEqual(len(market_sources.parse_table(html, contract)), 2)

    def test_currency_nominal_and_xml_entities(self):
        xml = b'<ValCurs><Record Date="01.01.2025"><Nominal>100</Nominal><Value>150,5</Value></Record></ValCurs>'
        self.assertAlmostEqual(market_sources.parse_currency(xml)[0]['value'], 1.505)
        with self.assertRaises(ValueError):
            market_sources.parse_currency(b'<!DOCTYPE x><ValCurs/>')

    def test_iss_columns_and_missing_prices(self):
        data = {'history': {'columns': ['TRADEDATE', 'CLOSE'], 'data': [['2025-01-01', None]]}}
        self.assertIsNone(market_sources.moex_rows(data, 'CLOSE')[0]['value'])
        with self.assertRaises(ValueError):
            market_sources.moex_rows(data, 'NOT_A_FIELD')

    def test_rate_limit_and_host_validation(self):
        session = Mock()
        session.get.return_value = SimpleNamespace(status_code=200, content=b'ok', raise_for_status=lambda: None)
        sleep = Mock()
        transport = market_sources.Transport(session=session, sleep=sleep, clock=lambda: 10.)
        transport.get('https://cbr.ru/hd_base/')
        transport.get('https://cbr.ru/hd_base/infl/')
        sleep.assert_called_once_with(1.)
        with self.assertRaises(ValueError):
            transport.get('https://localhost/private')

    def test_correlation_lag_and_fdr(self):
        rng = np.random.default_rng(7)
        increments = rng.normal(size=200)
        a = rows(200, values=np.cumsum(increments))
        b = rows(200, values=np.cumsum(np.r_[0, increments[:-1]]))
        result = market_analysis.correlations(a, b, series(), series(), window=120, lag=1)
        self.assertEqual(result['status'], 'ok')
        self.assertAlmostEqual(result['pearson'], 1.)
        self.assertEqual(result['direction'], 'left precedes right')
        self.assertLess(market_analysis.fdr([result])[0]['spearman_q'], .05)

    def test_insufficient_pairs_and_constant_series(self):
        self.assertEqual(market_analysis.correlations(rows(20), rows(20), series(), series())['status'], 'insufficient')
        self.assertEqual(market_analysis.correlations(rows(100, values=np.ones(100)), rows(100), series(), series())['status'], 'insufficient')

    def test_monthly_alignment_does_not_create_daily_observations(self):
        a = rows(1500)
        b = rows(48, 'monthly', values=np.arange(48.)**2)
        result = market_analysis.correlations(a, b, series(), series(frequency='monthly'), window=24)
        self.assertEqual(result['frequency'], 'monthly')
        self.assertLessEqual(result['n'], 24)

    def test_signed_liquidity_remains_negative(self):
        data = rows(160, values=-np.arange(160.)-100)
        output = market_analysis.forecast(data, series(), created_at='2023-06-09T23:00:00+00:00')
        self.assertTrue(output)
        self.assertTrue(all(p['prediction'] < 0 for p in output))

    def test_forecast_future_values_do_not_change_past_prediction(self):
        data = rows(180)
        cutoff = '2023-05-20T23:00:00+00:00'
        expected = market_analysis.forecast(data, series(), created_at=cutoff)
        self.assertTrue(expected)
        changed = [{**r, 'value': 1e9 if r['date'] > '2023-05-20' else r['value']} for r in data]
        self.assertEqual(expected, market_analysis.forecast(changed, series(), created_at=cutoff))

    def test_administrative_series_uses_last_value(self):
        output = market_analysis.forecast(rows(), series(administrative=True), created_at='2023-06-09T23:00:00+00:00')
        self.assertTrue(all(p['model'] == 'last' for p in output))

    def test_forecast_rejects_gaps_and_insufficient_history(self):
        self.assertEqual(market_analysis.forecast(rows(20), series(), created_at='2025-01-01'), [])
        data = rows(100)
        for i in range(0, 100, 4):
            data[i]['value'] = None
        self.assertEqual(market_analysis.forecast(data, series(), created_at='2025-01-01'), [])

    def test_event_study_requires_independent_events(self):
        events = [r['date'] for r in rows(100)]
        result = market_analysis.event_study(rows(100), series(), events)
        self.assertLess(result['n'], 10)
        self.assertEqual(result['status'], 'insufficient')

    def test_access_revoke_cancels_pending_personal_jobs(self):
        self.store.enqueue('market_report', {'actor_id': 2})
        self.store.enqueue('report', {'recipient': '2'})
        self.store.enqueue('report', {'section': 'ruonia'})
        market_store.change_access(self.store, 1, 2, False)
        self.assertEqual([r['state'] for r in self.store.query('SELECT state FROM jobs ORDER BY id')], ['cancelled', 'cancelled', 'pending'])
        self.assertFalse(self.store.authorize(2))
        self.assertEqual(self.store.one('SELECT action FROM access_audit')['action'], 'deny')

    def test_cannot_disable_admin_or_escalate_subscriber(self):
        self.store.add_user(3, role='admin', subscriptions='all')
        with self.assertRaises(ValueError):
            market_store.change_access(self.store, 1, 3, False)
        with self.assertRaises(PermissionError):
            market_store.change_access(self.store, 2, 4, True)
        market_store.change_access(self.store, 1, 2, True)
        self.assertFalse(self.store.authorize(2, admin=True))

    def test_add_user_button_input_and_start(self):
        bot, client = BotControls(self.store), Mock()
        bot.market_controls.callback(client, 1, 'user:add:new')
        bot.handle_message(client, self.message('99'))
        self.assertTrue(self.store.authorize(99))
        self.assertFalse(self.store.authorize(99, admin=True))
        self.assertFalse(self.store.one('SELECT started_at FROM bot_sessions WHERE user_id=99'))
        bot.handle_message(client, self.message('/start', 99))
        self.assertTrue(self.store.one('SELECT started_at FROM bot_sessions WHERE user_id=99')['started_at'])
        self.assertEqual(self.store.one('SELECT SUM(enabled) n FROM subscriptions WHERE user_id=99')['n'], 0)

    def test_invalid_user_input_is_not_added(self):
        bot, client = BotControls(self.store), Mock()
        bot.market_controls.callback(client, 1, 'user:add:new')
        for value in ('@someone', '-1', '0', '1.1'):
            bot.handle_message(client, self.message(value))
        self.assertEqual(len(self.store.query('SELECT * FROM users')), 2)

    def test_repeated_add_preserves_settings_restore_disables_schedule(self):
        from thesaurus.scheduling import seed_subscriptions
        seed_subscriptions(self.store, '2', 2)
        self.store.execute("UPDATE subscriptions SET detail='full',enabled=1 WHERE user_id=2")
        market_store.change_access(self.store, 1, 2, True)
        self.assertEqual(self.store.one('SELECT SUM(enabled) n FROM subscriptions WHERE user_id=2')['n'], 5)
        market_store.change_access(self.store, 1, 2, False)
        market_store.change_access(self.store, 1, 2, True)
        self.assertEqual(self.store.one('SELECT SUM(enabled) n FROM subscriptions WHERE user_id=2')['n'], 0)
        self.assertEqual(self.store.one('SELECT detail FROM subscriptions WHERE user_id=2')['detail'], 'full')

    def test_export_formula_units_and_quality(self):
        market_store.register(self.store, [series(title='=DANGEROUS()')])
        market_store.save_rows(self.store, 'test.rate', rows(3))
        output = export_market(self.store, self.root / 'data.csv', ['test.rate'], date_from='2023-01-02')
        with output.open(encoding='utf-8-sig') as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), 2)
        self.assertTrue(exported[0]['title'].startswith("'="))
        self.assertEqual(exported[0]['unit'], '%')
        self.assertTrue(exported[0]['quality'])
        export_market(self.store, self.root / 'data.xlsx', ['test.rate'], format='xlsx')

    def test_market_worker_uses_queue_and_dry_run(self):
        market_store.save_rows(self.store, 'test.rate', rows(100))
        job = self.store.enqueue('market_report', {'actor_id': 2, 'recipient': '2', 'profile': 'all'})
        worker = Worker(self.store, Settings(root=self.root), dry_run=True, charts=False)
        with patch('thesaurus.market_sources.Transport.get', side_effect=AssertionError('network forbidden')):
            worker.process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (job,))['state'], 'done')
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])
        self.assertEqual(self.store.one("SELECT section FROM reports")['section'], 'market')

    def test_revoked_market_job_never_collects(self):
        self.store.enqueue('market_report', {'actor_id': 2})
        self.store.add_user(2, enabled=False, subscriptions='all')
        with patch('thesaurus.market.collect') as collect:
            Worker(self.store, Settings(root=self.root), dry_run=True, charts=False).process_next()
        collect.assert_not_called()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')

    def test_market_rules_owned_disabled_and_coverage_guard(self):
        identifier = add_rule(self.store, 2, 'test.rate', 'deviation')
        rule = self.store.one('SELECT * FROM market_rules WHERE id=?', (identifier,))
        self.assertEqual(rule['enabled'], 0)
        self.assertEqual(rule['mode'], 'off')
        data = rows(100)
        data[-1]['value'] = 10000
        self.assertIsNotNone(evaluate_rule(data, series(), rule))
        for row in data[-50:]:
            row['value'] = None
        self.assertIsNone(evaluate_rule(data, series(), rule))

    def test_known_publication_after_cutoff_is_not_used(self):
        data = rows(180)
        normal = market_analysis.forecast(data, series(), created_at='2023-05-20T12:00:00+00:00')
        data[139]['published_at'] = '2023-05-21T12:00:00+00:00'
        data[139]['value'] = 1e8
        delayed = market_analysis.forecast(data, series(), created_at='2023-05-20T12:00:00+00:00')
        self.assertTrue(delayed)
        self.assertNotEqual(normal[0]['origin'], delayed[0]['origin'])
        self.assertLess(delayed[0]['prediction'], 1000)

    def test_monthly_forecast_has_three_monthly_targets(self):
        data = rows(70, 'monthly')
        prediction = market_analysis.forecast(data, series(frequency='monthly'), created_at=data[-1]['date'])
        self.assertEqual(len(prediction), 3)
        self.assertTrue(all(p['target'].endswith('-01') for p in prediction))

    def test_calibrated_interval_and_model_selection_threshold(self):
        output = market_analysis.forecast(rows(252), series(), created_at='2023-09-09T23:00:00+00:00')
        self.assertEqual(len(output), 4)
        self.assertTrue(all(p['lower'] is not None for p in output))
        for p in output:
            if p['model'] != 'last':
                self.assertLessEqual(p['validation_mae'], p['baseline_mae']*.95)

    def test_all_reviewed_table_contracts_reject_changed_headers(self):
        for item in market_sources.catalogue():
            for contract in item.get('contracts', []):
                with self.subTest(source=item['id'], table=contract['table']):
                    # Structural fixture for every reviewed source, including all nested datasets.
                    headers = contract['headers']
                    self.assertTrue(headers)
                    modified = {'table': 0, 'headers': ['changed', *headers[1:]]}
                    html = '<table><tr>' + ''.join('<th>'+h+'</th>' for h in headers) + '</tr><tr><td>01.01.2025</td>' + '<td>1</td>'*(len(headers)-1) + '</tr></table>'
                    with self.assertRaises(ValueError):
                        market_sources.parse_table(html, modified)

    def test_signal_cooldown_duplicate_and_crash_recovery(self):
        from thesaurus.market_alerts import dispatch, process
        data = rows(100)
        data[-1]['value'] = 10000
        market_store.save_rows(self.store, 'test.rate', data)
        identifier = add_rule(self.store, 2, 'test.rate', 'above', 1000)
        self.store.execute("UPDATE market_rules SET enabled=1,mode='immediate' WHERE id=?", (identifier,))
        process(self.store)
        process(self.store)
        self.assertEqual(len(self.store.query('SELECT * FROM market_signals')), 1)
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), 1)
        self.store.execute('DELETE FROM deliveries')
        dispatch(self.store)
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), 1)

    def test_signal_details_include_local_graph(self):
        from thesaurus.market_alerts import process
        data = rows(100)
        data[-1]['value'] = 10000
        market_store.save_rows(self.store, 'test.rate', data)
        identifier = add_rule(self.store, 2, 'test.rate', 'above', 1000)
        self.store.execute("UPDATE market_rules SET enabled=1,mode='immediate' WHERE id=?", (identifier,))
        process(self.store, Settings(root=self.root))
        parts = json.loads(self.store.one('SELECT parts FROM reports')['parts'])
        self.assertEqual(parts[0]['kind'], 'photo')
        self.assertTrue(Path(parts[0]['path']).is_file())

    def test_forecasts_and_reports_reused_after_worker_crash(self):
        self.store.enqueue('market_report', {'actor_id': 2})
        worker = Worker(self.store, Settings(root=self.root), dry_run=True, charts=False)
        worker.process_next()
        self.store.execute("UPDATE jobs SET state='pending'")
        worker.process_next()
        self.assertEqual(len(self.store.query('SELECT * FROM reports')), 1)

    def test_subscriber_cannot_change_sources_or_other_users_rule(self):
        bot, client = BotControls(self.store), Mock()
        for action in ('user:add:new', 'user:deny:1', 'ms:toggle:1'):
            with self.assertRaises(PermissionError):
                bot.market_controls.callback(client, 2, action)
        identifier = add_rule(self.store, 1, 'test.rate', 'above', 1000)
        with self.assertRaises(ValueError):
            bot.market_controls.alerts(client, 2, 'on '+str(identifier))

    def test_disabled_numeric_child_is_not_reenabled_on_collection(self):
        dataset = series('test.dataset', kind='dataset', contracts=[{'table': 0, 'headers': ['Дата', 'Цена']}])
        market_store.register(self.store, [dataset])
        market.enable(self.store, 'test.dataset', True)
        transport = Mock()
        transport.get.return_value.text = '<table><tr><th>Дата</th><th>Цена</th></tr><tr><td>01.01.2025</td><td>100</td></tr></table>'
        market.collect(self.store, transport=transport, today=date(2025, 1, 2))
        child = market_sources.metric_id('test.dataset', 'Цена')
        market.enable(self.store, child, False)
        market.collect(self.store, transport=transport, today=date(2025, 1, 2))
        self.assertEqual(self.store.one('SELECT enabled FROM market_series WHERE id=?', (child,))['enabled'], 0)

    def test_global_collection_sleep_coalesces_and_has_no_delivery(self):
        config = {'enabled': True, 'times': ['10:00', '18:30'], 'actor_id': 1,
                  'created_at': '2025-01-01T00:00:00+00:00', 'last_scheduled_at': None}
        self.store.set_setting('market_schedule', config)
        market.enqueue_collection(self.store, datetime(2025, 1, 2, 20, tzinfo=timezone.utc))
        market.enqueue_collection(self.store, datetime(2025, 1, 6, 20, tzinfo=timezone.utc))
        jobs = self.store.query('SELECT * FROM jobs')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['kind'], 'market_collect')
        self.assertTrue(json.loads(jobs[0]['payload'])['scheduled_for'].startswith('2025-01-06'))
        with patch('thesaurus.market.collect') as collect:
            Worker(self.store, Settings(root=self.root), dry_run=True, charts=False).process_next()
            collect.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])

    def test_global_collection_rechecks_admin_and_disabled_schedule(self):
        self.store.enqueue('market_collect', {'actor_id': 2})
        with patch('thesaurus.market.collect') as collect:
            Worker(self.store, Settings(root=self.root), dry_run=True, charts=False).process_next()
            collect.assert_not_called()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')

    def test_ambiguous_duplicate_observations_rejected_without_partial_write(self):
        with self.assertRaises(ValueError):
            market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': 1}, {'date': '2025-01-01', 'value': 2}])
        self.assertEqual(market_store.history(self.store, 'test.rate'), [])

    def test_isolated_cli_preview_does_not_compete_with_live_worker(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from thesaurus.cli import run
        from thesaurus.locking import worker_lock
        settings = Settings(root=self.root, admin_ids=(1,))
        with worker_lock(settings.data_dir / 'worker.lock'), redirect_stdout(StringIO()):
            result = run(['once', '--section', 'market', '--dry-run', '--no-charts', '--db', str(self.root / 'preview.db')], settings=settings)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
