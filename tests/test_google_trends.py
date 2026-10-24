"""Google website CSV snapshots, access checks and fully offline processing."""
import base64
import io
import json
import shutil
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import google_trends as trends
from thesaurus import market, market_analysis, market_store
from thesaurus.charts import chart_series, selections
from thesaurus.collectors import Collector
from thesaurus.config import Settings
from thesaurus.market_jobs import export_market
from thesaurus.scheduling import seed_subscriptions, update_subscription
from thesaurus.service import Worker
from thesaurus.store import Store, utcnow
from thesaurus.telegram_bot import BotControls


class GoogleTrendsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('trends-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.settings = Settings(root=self.root)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=99)
        self.bot = BotControls(self.store)
        self.raw = 'Категория: Все категории\n\nНеделя,рост цен: (Россия)\n2025-01-05,20\n2025-01-12,30\n'.encode()

    def event(self, actor=1, text='', private=True):
        return SimpleNamespace(text=text, id=12, from_user=SimpleNamespace(id=actor),
                               chat=SimpleNamespace(id=actor, type='private' if private else 'group'))

    def test_russian_weekly_and_english_daily(self):
        parsed = trends.parse_csv(self.raw, ['рост цен'])
        self.assertEqual(parsed['frequency'], 'weekly')
        self.assertEqual(parsed['columns'][0]['rows'][0]['period_end'], '2025-01-11')
        raw = b'Category: All categories\n\nDay,price: (Worldwide)\n2025-01-01,100\n'
        self.assertEqual(trends.parse_csv(raw, ['price'])['frequency'], 'daily')

    def test_bom_semicolon_monthly_and_missing(self):
        raw = '\ufeffМесяц;рост цен: (Россия)\n2025-01;0\n2025-02;<1\n2025-03;\n'.encode()
        parsed = trends.parse_csv(raw, ['рост цен'])
        rows = parsed['columns'][0]['rows']
        self.assertEqual(rows[1]['period_end'], '2025-02-28')
        self.assertTrue(all(row['value'] is None for row in rows))
        self.assertEqual(rows[0]['quality'], 'low_volume_or_zero')

    def test_current_week_is_not_training_data(self):
        raw = b'Week,price: (Worldwide)\n2025-01-05,100\n'
        parsed = trends.parse_csv(raw, ['price'], today=date(2025, 1, 8))
        self.assertIsNone(parsed['columns'][0]['rows'][0]['value'])
        self.assertEqual(parsed['columns'][0]['rows'][0]['quality'], 'incomplete_period')

    def test_reject_malformed_future_duplicate_and_wrong_columns(self):
        for raw in [b'Day,price: (Russia)\n2025-01-01,101\n',
                    b'Day,price: (Russia)\n2025-01-01,nan\n',
                    b'Day,price: (Russia)\n2025-01-01,10\n2025-01-01,20\n',
                    b'Day,price: (Russia)\n2099-01-01,10\n',
                    b'Country,price\nRussia,100\n',
                    b'Day,other: (Russia)\n2025-01-01,10\n',
                    b'Day,price: (Russia)\n2025-01-01,10,20\n',
                    b'x' * (trends.MAX_BYTES + 1)]:
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                trends.parse_csv(raw, ['price'])

    def test_atomic_validation(self):
        bad = self.raw + b'2025-01-19,999\n'
        with self.assertRaises(ValueError):
            trends.import_csv(self.store, bad, 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source='google_trends'")['n'], 0)

    def test_snapshot_replacement_preserves_old_history_and_settings(self):
        first = trends.import_csv(self.store, self.raw, 1)[0]
        self.store.set_setting('market_selection:2', [first])
        self.store.set_setting('chart:2:trends', {'identity': first, 'horizon': 1, 'period': 'all'})
        self.store.execute('INSERT INTO market_rules(user_id,series_id,kind,config,enabled) VALUES(2,?,?,?,1)', (first, 'threshold', '{}'))
        second = trends.import_csv(self.store, self.raw.replace(b',20', b',60'), 1)[0]
        self.assertNotEqual(first, second)
        self.assertEqual(market_store.history(self.store, first)[0]['value'], 20)
        self.assertEqual(market_store.history(self.store, second)[0]['value'], 60)
        self.assertEqual(self.store.get_setting('market_selection:2'), [second])
        self.assertEqual(self.store.get_setting('chart:2:trends')['identity'], second)
        self.assertEqual(self.store.one('SELECT enabled FROM market_rules')['enabled'], 0)
        self.assertEqual([r['id'] for r in market.selected(self.store)], [second])
        trends.import_csv(self.store, self.raw, 1)
        self.assertEqual([r['id'] for r in market.selected(self.store)], [second])
        self.assertEqual(len(market_store.history(self.store, first)), 2)

    def test_separate_regions_and_frequency(self):
        first = trends.import_csv(self.store, self.raw, 1)[0]
        second = trends.import_csv(self.store, self.raw.decode().replace('Россия', 'Москва').encode(), 1)[0]
        self.assertNotEqual(first, second)
        self.assertEqual(len(market.selected(self.store)), 2)

    def test_subscriber_cannot_import_and_revoked_job_cannot_run(self):
        with self.assertRaises(PermissionError):
            trends.import_csv(self.store, self.raw, 2)
        self.store.enqueue('trends_import', {'actor_id': 1, 'csv': base64.b64encode(self.raw).decode()})
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=1')
        Worker(self.store, self.settings).drain()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source='google_trends'")['n'], 0)

    def test_switch_backs_up_and_disables_old_search_without_network(self):
        seed_subscriptions(self.store, '1', 1)
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE section='wordstat'")
        self.store.enqueue('report', {'section': 'wordstat'})
        trends.activate(self.store)
        self.assertTrue(list(self.root.glob('*.trends-backup-*.sqlite3')))
        self.assertEqual(self.store.one("SELECT enabled FROM subscriptions WHERE section='wordstat'")['enabled'], 0)
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'cancelled')
        provider = Mock()
        result = Collector(self.store, self.settings, provider).collect('wordstat', force=True)
        self.assertEqual(result['status'], 'cached')
        provider.assert_not_called()
        self.assertEqual(provider.mock_calls, [])
        with self.assertRaises(ValueError):
            update_subscription(self.store, 1, 'wordstat', enabled=True)
        backups = len(list(self.root.glob('*.trends-backup-*.sqlite3')))
        trends.activate(self.store)
        self.assertEqual(len(list(self.root.glob('*.trends-backup-*.sqlite3'))), backups)

    def test_document_preview_requires_admin_private_chat_and_confirmation(self):
        self.bot.trends_controls.callback(self.client, 1, 'gt:upload')
        event = self.event()
        event.document = SimpleNamespace(file_name='multiTimeline.csv', file_size=len(self.raw))
        self.client.download_media.side_effect = lambda *a, **kw: io.BytesIO(self.raw)
        self.bot.handle_document(self.client, event)
        preview = self.store.get_setting('trends_preview:1')
        self.assertIsNotNone(preview)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source='google_trends'")['n'], 0)
        for invalid in (self.event(actor=2), self.event(private=False)):
            invalid.document = event.document
            self.bot.handle_document(self.client, invalid)
        self.assertEqual(self.client.download_media.call_count, 1)
        self.bot.trends_controls.callback(self.client, 1, 'gt:confirm:' + preview['token'])
        Worker(self.store, self.settings).drain()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'done')
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source='google_trends'")['n'], 1)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM deliveries')['n'], 1)
        with self.assertRaises(ValueError):
            self.bot.trends_controls.callback(self.client, 1, 'gt:confirm:' + preview['token'])

    def test_dry_run_import_does_not_write_data_or_delivery(self):
        self.store.enqueue('trends_import', {'actor_id': 1, 'csv': base64.b64encode(self.raw).decode()})
        Worker(self.store, self.settings, dry_run=True).drain()
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_series WHERE source='google_trends'")['n'], 0)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM deliveries')['n'], 0)

    def test_cancel_and_expired_preview_cannot_be_confirmed(self):
        preview = {'token': 'test', 'csv': base64.b64encode(self.raw).decode(), 'expires': '2000-01-01T00:00:00+00:00'}
        self.store.set_setting('trends_preview:1', preview)
        with self.assertRaises(ValueError):
            self.bot.trends_controls.callback(self.client, 1, 'gt:confirm:test')
        self.bot.trends_controls.menu(self.client, 1)
        self.assertIsNone(self.store.get_setting('trends_preview:1'))
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM jobs')['n'], 0)

    def test_csv_series_collection_never_calls_network(self):
        trends.import_csv(self.store, self.raw, 1)
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            result = market.collect(self.store)
        self.assertEqual(result['errors'], [])

    def test_chart_worker_builds_png_without_network_or_real_delivery(self):
        identity = trends.import_csv(self.store, self.raw, 1)[0]
        report_id = uuid4().hex
        self.store.enqueue('telegram_chart', {'actor_id': 2, 'section': 'trends', 'identity': identity,
                           'selection': [identity], 'options': {'period': 'all', 'horizon': 0}, 'report_id': report_id})
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            Worker(self.store, self.settings).drain()
        report = self.store.one('SELECT * FROM reports WHERE id=?', (report_id,))
        self.assertIn('Google Trends', report['text'])
        self.assertTrue(Path(json.loads(report['parts'])[0]['path']).is_file())
        self.assertEqual(self.store.one('SELECT state FROM deliveries')['state'], 'pending')

    def test_google_menu_replaces_wordstat_and_hides_old_schedules(self):
        trends.activate(self.store)
        self.bot.handle_message(self.client, self.event(actor=2, text='/report'))
        data = [b.callback_data for row in self.client.send_message.call_args.kwargs['reply_markup'].inline_keyboard for b in row]
        self.assertIn('ch:menu:trends', data)
        self.assertNotIn('request:wordstat', data)
        self.bot.handle_message(self.client, self.event(actor=2, text='/subscriptions'))
        data = [b.callback_data for row in self.client.send_message.call_args.kwargs['reply_markup'].inline_keyboard for b in row]
        self.assertNotIn('ui:subscription:wordstat', data)
        self.assertNotIn('ui:subscription:regions', data)

    def test_selection_and_export_keep_units_source_and_missing(self):
        identity = trends.import_csv(self.store, self.raw, 1)[0]
        self.assertEqual(selections(self.store, 2, 'trends')[0][0], identity)
        output = export_market(self.store, self.root / 'trends.csv', [identity])
        text = output.read_text(encoding='utf-8-sig')
        self.assertIn('google_trends', text)
        self.assertIn('индекс интереса (0–100)', text)
        self.assertIn('publication_unknown', text)

    def test_offline_charts_and_forecasts_bounded_and_immutable(self):
        last = date.fromisoformat(utcnow()[:10]) - timedelta(days=1)
        content = 'Day,рост цен: (Россия)\n' + '\n'.join(f'{last-timedelta(days=89-i)},{10+i}' for i in range(90))
        identity = trends.import_csv(self.store, content.encode(), 1)[0]
        options = {'period': 'month', 'horizon': 4, 'identity': identity}
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            result, _, _ = chart_series(self.store, self.settings, 2, 'trends', identity, options)
        self.assertLessEqual(len(result[0]['history']), 30)
        forecasts = self.store.query('SELECT * FROM market_forecasts')
        self.assertTrue(forecasts)
        self.assertTrue(all(0 <= p['prediction'] <= 100 for p in forecasts))
        self.assertTrue(all(p['upper'] is None or p['upper'] <= 100 for p in forecasts))
        before = json.dumps(forecasts, sort_keys=True)
        chart_series(self.store, self.settings, 2, 'trends', identity, options)
        self.assertEqual(json.dumps(self.store.query('SELECT * FROM market_forecasts'), sort_keys=True), before)

    def test_future_rows_do_not_change_forecast(self):
        end = date(2025, 3, 1)
        rows = [{'date': (end-timedelta(days=79-i)).isoformat(), 'value': 20+i/10} for i in range(80)]
        meta = {'frequency': 'daily', 'calendar': 'calendar', 'unit': 'index', 'value_bounds': [0, 100]}
        expected = market_analysis.forecast(rows, meta, created_at=end.isoformat())
        actual = market_analysis.forecast(rows + [{'date': '2025-04-01', 'value': 100}], meta, created_at=end.isoformat())
        self.assertEqual(actual, expected)
