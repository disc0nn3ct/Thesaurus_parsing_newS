"""Offline travel FX acceptance: realistic costs, freshness, privacy and worker recovery."""
import json
import shutil
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from thesaurus import dollar, market, market_store
from thesaurus.config import Settings
from thesaurus.dollar_jobs import run_job
from thesaurus.service import Worker
from thesaurus.store import Store, utcnow
from thesaurus.telegram_bot import BotControls

NOW = '2026-10-08T09:00:00+00:00'


class DollarMathTests(unittest.TestCase):
    def test_costs_include_owned_percent_and_fixed_fees_per_operation(self):
        cfg = {**dollar.DEFAULTS, 'amount': 1500., 'owned': 500., 'quote': 100.,
               'fee_pct': 2., 'fee_fixed': 100., 'rub_yield': 10.}
        rows = dollar.scenarios(cfg, days=365)
        flat = rows[2]
        self.assertEqual(flat['now'], 102100.)
        self.assertEqual(flat['wait'], 91900.)
        self.assertEqual(flat['staged'], 97100.)
        self.assertAlmostEqual(rows[-1]['wait'], 102100.)

    def test_no_purchase_has_no_fees(self):
        cfg = {**dollar.DEFAULTS, 'owned': 2000., 'quote': 90., 'fee_fixed': 200.}
        self.assertTrue(all(r['now'] == r['wait'] == r['staged'] == 0 for r in dollar.scenarios(cfg, days=30)))

    def test_input_rejects_nonfinite_and_bounds(self):
        for field, text in [('quote', 'nan'), ('quote', '0'), ('amount', '-1'), ('rub_yield', '101'),
                            ('trip', '2020-01-01'), ('trip', 'not a date'), ('owned', 'inf')]:
            with self.subTest(field=field, text=text), self.assertRaises(ValueError):
                dollar.change_settings({}, field, text, now=NOW)
        self.assertEqual(dollar.change_settings({}, 'quote', '95,5', now=NOW)['quote_at'], NOW)
        self.assertEqual(dollar.change_settings({}, 'trip', '2026-10-08', now=NOW)['trip'], '2026-10-08')

    def test_missing_observations_are_not_zero_and_future_not_seen(self):
        rows = [{'date': '2026-10-06', 'value': 90}, {'date': '2026-10-07', 'value': None},
                {'date': '2026-10-08', 'value': 91}, {'date': '2026-10-09', 'value': 1}]
        stats, series = dollar.history_stats(rows, {'frequency': 'daily', 'calendar': 'cbr_fx'}, now=NOW)
        self.assertEqual(stats['last'], 91)
        self.assertEqual(stats['missing'], 1)
        self.assertEqual(stats['n'], 2)
        self.assertNotIn(pd.Timestamp('2026-10-09'), series.index)


class DollarBotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]/'.test-tmp'/('dollar-'+uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root/'state.sqlite3')
        self.store.add_user(1, role='admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.bot = BotControls(self.store)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=42)
        self.settings = Settings(root=self.root)

    def load(self, now=NOW):
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (dollar.USD,))['metadata'])
        from thesaurus.market_analysis import grid
        dates = grid(pd.Timestamp(now[:10])-pd.Timedelta(days=200), pd.Timestamp(now[:10]), meta)
        market_store.save_rows(self.store, dollar.USD, [{'date': d.date().isoformat(), 'value': 90+i*.01,
                      'retrieved_at': now} for i, d in enumerate(dates)])
        market.enable(self.store, dollar.USD, True)

    def config(self):
        return {**dollar.DEFAULTS, 'mode': 'purchase', 'trip': '2026-10-10', 'quote': 95., 'quote_at': NOW}

    def test_menu_form_is_personal_and_retry_keeps_input(self):
        self.bot.dollar_controls.callback(self.client, 2, 'usd:set:quote')
        with self.assertRaises(ValueError):
            self.bot.dollar_controls.input(self.client, 2, 'nan')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'dollar:quote')
        self.bot.dollar_controls.input(self.client, 2, '98,2')
        self.assertEqual(self.bot.dollar_controls.config(2)['quote'], 98.2)
        self.assertIsNone(self.bot.dollar_controls.config(3)['quote'])
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])

    def test_commands_private_callback_and_authorization(self):
        event = SimpleNamespace(from_user=SimpleNamespace(id=2), chat=SimpleNamespace(id=2, type='private'), text='/dollar', id=5)
        self.bot.handle_message(self.client, event)
        self.assertIn('Доллар для поездки', self.client.send_message.call_args.args[1])
        with self.assertRaises(PermissionError):
            self.bot.dollar_controls.callback(self.client, 2, 'usd:update')
        self.assertFalse(self.store.query('SELECT * FROM jobs'))
        event.chat = SimpleNamespace(id=2, type='group')
        self.client.reset_mock()
        self.bot.handle_message(self.client, event)
        self.client.send_message.assert_not_called()
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        with self.assertRaises(PermissionError):
            self.bot.dollar_controls.callback(self.client, 2, 'usd:run')

    def test_repeated_callback_deduplicates_immutable_config(self):
        self.bot.dollar_controls.callback(self.client, 2, 'usd:run', 'same')
        self.bot.dollar_controls.callback(self.client, 2, 'usd:run', 'same')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        saved = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        self.assertIsNone(saved['config']['quote'])

    def test_fresh_urgent_trip_and_stale_quote(self):
        self.load()
        cfg = self.config()
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]):
            result, _ = dollar.build(self.store, cfg, now=NOW)
            self.assertIn('не больше недели', result['decision'])
            cfg['quote_at'] = '2026-10-06T00:00:00+00:00'
            stale, _ = dollar.build(self.store, cfg, now=NOW)
            self.assertEqual(stale['purchase_guide']['status'], 'insufficient')
            self.assertIn('свежая цена банка', stale['decision'])
        self.assertAlmostEqual(result['premium'], (95/result['stats']['last']-1)*100)

    def test_no_data_and_expired_deadline_do_not_generate_directional_advice(self):
        result, _ = dollar.build(self.store, self.config(), now=NOW)
        self.assertFalse(result['data_fresh'])
        self.assertEqual(result['purchase_guide']['status'], 'insufficient')
        cfg = {**self.config(), 'trip': '2026-10-01'}
        result, _ = dollar.build(self.store, cfg, now=NOW)
        self.assertIn('прошёл', result['decision'])
        self.assertEqual(result['scenarios'][2]['interest'], 0)

    def test_models_receive_only_known_revisions_and_methodology_cutoff(self):
        self.load()
        market_store.save_rows(self.store, dollar.USD, [{'date': '2026-10-08', 'value': 1.,
                    'retrieved_at': '2026-10-09T00:00:00+00:00'}])
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]) as forecast:
            result, _ = dollar.build(self.store, self.config(), now=NOW)
        args, kwargs = forecast.call_args
        self.assertEqual(args[1]['break_date'], '2024-06-13')
        self.assertEqual(kwargs['created_at'], NOW)
        self.assertGreater(result['stats']['last'], 90.)
        self.assertTrue(all(r['retrieved_at'] <= NOW for r in args[0]))

    def test_local_worker_writes_chart_and_report_without_api_and_reuses_delivery(self):
        now = utcnow()
        self.load(now)
        today = dollar.today_moscow(now)
        cfg = {**self.config(), 'trip': (today+timedelta(days=30)).isoformat(), 'quote_at': now}
        job = {'id': 900, 'kind': 'dollar_analysis'}
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]), patch('thesaurus.market.collect') as collect:
            report_id = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg}, dry_run=True)
        collect.assert_not_called()
        report = self.store.one('SELECT * FROM reports WHERE id=?', (report_id,))
        paths = [Path(p['path']) for p in json.loads(report['parts'])]
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertEqual(paths[0].suffix, '.png')
        self.assertFalse(self.store.query('SELECT * FROM deliveries'))
        replay = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg})
        self.assertEqual(replay, report_id)
        deliveries = self.store.query('SELECT * FROM deliveries')
        run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg})
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), len(deliveries))

    def test_worker_rechecks_revocation_and_dry_collection_no_network(self):
        self.store.enqueue('dollar_analysis', {'actor_id': 2, 'config': self.config()})
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        worker = Worker(self.store, self.settings, dry_run=True)
        worker.process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')
        with patch('thesaurus.market.collect', return_value={'errors': []}) as collect:
            run_job(self.store, self.settings, {'id': 91, 'kind': 'dollar_collect'}, {'actor_id': 1}, dry_run=True)
        collect.assert_called_once_with(self.store, selection=[dollar.USD], dry_run=True)

    def test_real_local_forecasts_commit_with_report_and_resume_once(self):
        now = utcnow()
        self.load(now)
        cfg = {**self.config(), 'trip': (dollar.today_moscow(now)+timedelta(days=28)).isoformat(), 'quote_at': now}
        job = {'id': 980, 'kind': 'dollar_analysis'}
        report_id = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg}, dry_run=True, charts=False)
        report = self.store.one('SELECT metrics FROM reports WHERE id=?', (report_id,))
        points = json.loads(report['metrics'])['forecast']
        self.assertTrue(points)
        self.assertEqual(len(self.store.query('SELECT * FROM forecast_points')), len(points))
        self.assertTrue(any(p['validation_n'] >= 10 for p in points))
        replay = run_job(self.store, self.settings, job, {'actor_id': 2, 'config': cfg}, dry_run=True)
        self.assertEqual(replay, report_id)
        self.assertEqual(len(self.store.query('SELECT * FROM forecast_runs')), 1)

    def test_cancel_returns_to_dollar_and_missing_data_does_not_refetch(self):
        event = SimpleNamespace(from_user=SimpleNamespace(id=2), chat=SimpleNamespace(id=2, type='private'), text='/dollar', id=6)
        self.bot.handle_message(self.client, event)
        self.bot.dollar_controls.callback(self.client, 2, 'usd:set:trip')
        self.store.set_setting('ui_screen:2', {'back': 'usd:menu'})
        event.text = '/cancel'
        self.bot.handle_message(self.client, event)
        self.assertIn('Ввод отменён', self.client.send_message.call_args.args[1])
        with patch('thesaurus.market.collect') as collect:
            report = run_job(self.store, self.settings, {'id': 999, 'kind': 'dollar_analysis'},
                             {'actor_id': 2, 'config': self.config()}, dry_run=True)
        collect.assert_not_called()
        self.assertIn('Нет сохранённой истории', self.store.one('SELECT text FROM reports WHERE id=?', (report,))['text'])

    def test_context_distinguishes_key_rate_actual_inflation_and_target(self):
        self.load()
        prefix = 'Инфляция и ключевая ставка Банка России: '
        for identity, title, value in [('infl.1', 'Цель по инфляции, %', 4),
                                      ('infl.2', 'Инфляция, % г/г', 7.5),
                                      ('infl.3', 'Ключевая ставка, % годовых', 12)]:
            metadata = {'id': identity, 'title': prefix+title, 'source': 'cbr', 'dataset': 'cbr.hd_base.infl',
                        'unit': '%', 'frequency': 'monthly', 'kind': 'series', 'aggregation': 'last'}
            market_store.register(self.store, [metadata])
            market.enable(self.store, identity, True)
            market_store.save_rows(self.store, identity, [{'date': '2026-08-01', 'value': value-1, 'retrieved_at': NOW},
                                                        {'date': '2026-09-01', 'value': value, 'retrieved_at': NOW}])
        with patch('thesaurus.dollar.analysis.forecast', return_value=[]):
            result, _ = dollar.build(self.store, self.config(), now=NOW)
        self.assertEqual({r['id'] for r in result['context']}, {'infl.2', 'infl.3'})
        self.assertTrue(all(r['change_unit'] == 'п.п.' for r in result['context']))
        self.assertNotIn('Цель по инфляции', dollar.render(result))


if __name__ == '__main__':
    unittest.main()
