"""Unpaid/offline tests for the experimental Google website JSON adapter."""
import json
import shutil
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from thesaurus import google_trends, market_analysis, market_store, trends_api
from thesaurus.config import Settings
from thesaurus.service import Worker
from thesaurus.store import Store, utcnow
from thesaurus.telegram_bot import BotControls


def response(body=None, status=200, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.iter_content.return_value = [(")]}'\n" + json.dumps(body)).encode()]
    return result


def timeline(count=40):
    first = date(2024, 1, 7)
    return {'default': {'timelineData': [
        {'time': str(int(datetime.combine(first + timedelta(days=i*7), datetime.min.time(), timezone.utc).timestamp())),
         'value': [10 + i % 70], 'hasData': [True], 'isPartial': False} for i in range(count)]}}


class TrendsAPITests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('trends-api-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.set_setting('search_provider', 'google_trends')
        self.settings = Settings(root=self.root)
        self.session, self.sleep = Mock(), Mock()
        self.client = trends_api.TrendsClient(self.store, session=self.session, sleep=self.sleep, clock=lambda: 1.)
        self.store.set_setting('trends_schedule', {'enabled': True, 'actor_id': 1, 'geo': 'RU', 'window': 'today 5-y',
                                                'times': ['08:30'], 'created_at': '2026-10-01T00:00:00+00:00', 'last_scheduled_at': None,
                                                'phrases': ['рост цен', 'обвал']})

    def test_two_json_requests_no_cookie_page_and_safe_params(self):
        self.session.get.side_effect = [response({'widgets': [{'id': 'TIMESERIES', 'request': {'time': 'today 5-y'}, 'token': 'widget-token'}]}), response(timeline())]
        raw = self.client.fetch('рост цен')
        parsed = google_trends.parse_csv(raw, ['рост цен'])
        self.assertEqual(parsed['frequency'], 'weekly')
        self.assertEqual(len(parsed['columns'][0]['rows']), 40)
        self.assertEqual(self.store.get_setting('trends_usage')['calls'], 2)
        self.sleep.assert_called_once_with(1.)
        self.assertTrue(all(call.kwargs['allow_redirects'] is False for call in self.session.get.call_args_list))
        self.assertTrue(all('verify' not in call.kwargs for call in self.session.get.call_args_list))

    def test_429_stops_collection_pauses_and_preserves_cache(self):
        self.session.get.return_value = response(status=429, headers={'Retry-After': '2'})
        raw = trends_api.to_csv(timeline(), 'рост цен', 'RU', 'today 5-y')
        old = google_trends.import_csv(self.store, raw, 1)[0]
        result = trends_api.collect(self.store, 1, client_factory=lambda _: self.client)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(self.session.get.call_count, 1)
        self.assertEqual(len(market_store.history(self.store, old)), 40)
        self.assertGreater(self.store.get_setting('trends_pause_until'), utcnow())
        result = trends_api.collect(self.store, 1, client_factory=lambda _: self.client)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(self.session.get.call_count, 1)

    def test_partial_success_keeps_first_dataset(self):
        client = Mock()
        client.fetch.side_effect = [trends_api.to_csv(timeline(), 'рост цен', 'RU', 'today 5-y'), trends_api.TrendsError('network', 900)]
        result = trends_api.collect(self.store, 1, client_factory=lambda _: client)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['series']), 1)
        metadata = json.loads(self.store.one("SELECT metadata FROM market_series WHERE source='google_trends'")['metadata'])
        self.assertEqual(metadata['acquisition'], 'unofficial_api')
        client.close.assert_called_once()

    def test_schema_changes_do_not_include_provider_payload(self):
        self.session.get.return_value = response({'widgets': [], 'secret': 'should-not-appear'})
        with self.assertRaises(trends_api.TrendsError) as caught:
            self.client.fetch('рост цен')
        self.assertEqual(caught.exception.kind, 'schema')
        self.assertNotIn('should-not-appear', str(caught.exception))
        self.assertEqual(self.session.get.call_count, 1)

    def test_budget_blocks_request_and_counts_attempts(self):
        day = datetime.now(trends_api.ZoneInfo('Europe/Moscow')).date().isoformat()
        self.store.set_setting('trends_usage', {'day': day, 'calls': trends_api.LIMIT})
        with self.assertRaises(trends_api.TrendsError) as caught:
            self.client.fetch('рост цен')
        self.assertEqual(caught.exception.kind, 'budget')
        self.session.get.assert_not_called()

    def test_incomplete_and_missing_flags_never_become_values(self):
        data = timeline(3)
        data['default']['timelineData'][1]['isPartial'] = True
        data['default']['timelineData'][2]['hasData'] = [False]
        parsed = google_trends.parse_csv(trends_api.to_csv(data, 'рост цен', 'RU', 'today 5-y'), ['рост цен'])
        self.assertTrue(all(row['value'] is None for row in parsed['columns'][0]['rows'][1:]))

    def test_invalid_numeric_and_schema_and_region_fail_safely(self):
        for value in [True, -1, 101, float('nan'), '10']:
            data = timeline(3)
            data['default']['timelineData'][0]['value'] = [value]
            with self.subTest(value=value), self.assertRaises(trends_api.TrendsError):
                trends_api.to_csv(data, 'рост цен', 'RU', 'today 5-y')
        with self.assertRaises(ValueError):
            self.client.fetch('рост цен', geo='http://other')
        self.session.get.assert_not_called()

    def test_dry_run_never_creates_client_or_calls_api(self):
        factory = Mock()
        result = trends_api.collect(self.store, 1, dry_run=True, client_factory=factory)
        factory.assert_not_called()
        self.assertTrue(result['dry_run'])

    def test_only_admin_can_collect_or_configure(self):
        with self.assertRaises(PermissionError):
            trends_api.collect(self.store, 2, client_factory=Mock())
        controls = BotControls(self.store)
        with self.assertRaises(PermissionError):
            controls.trends_controls.callback(Mock(), 2, 'gt:on')

    def test_schedule_coalesces_sleep_and_does_not_duplicate(self):
        now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        first = trends_api.enqueue_due(self.store, now)
        self.assertIsNotNone(first)
        self.assertIsNone(trends_api.enqueue_due(self.store, now))
        next_job = trends_api.enqueue_due(self.store, now + timedelta(days=2))
        self.assertEqual(first, next_job)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM jobs')['n'], 1)
        config = self.store.get_setting('trends_schedule')
        config['enabled'] = False
        self.store.set_setting('trends_schedule', config)
        self.assertIsNone(trends_api.enqueue_due(self.store, now + timedelta(days=3)))

    def test_disabled_or_revoked_schedule_performs_no_requests(self):
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=1')
        self.assertIsNone(trends_api.enqueue_due(self.store, datetime(2026, 10, 5, 12, tzinfo=timezone.utc)))
        self.store.enqueue('trends_collect', {'actor_id': 1, 'scheduled': True})
        Worker(self.store, self.settings).drain()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')

    def test_expired_pause_resumes_once_without_waiting_until_tomorrow(self):
        config = self.store.get_setting('trends_schedule')
        config.update(created_at='2026-10-05T06:00:00+00:00', last_scheduled_at=None)
        self.store.set_setting('trends_schedule', config)
        self.store.set_setting('trends_pause_until', '2026-10-05T07:00:00+00:00')
        self.assertIsNone(trends_api.enqueue_due(self.store, datetime(2026, 10, 5, 6, 30, tzinfo=timezone.utc)))
        job = trends_api.enqueue_due(self.store, datetime(2026, 10, 5, 7, 1, tzinfo=timezone.utc))
        self.assertIsNotNone(job)
        self.assertIsNone(trends_api.enqueue_due(self.store, datetime(2026, 10, 5, 7, 2, tzinfo=timezone.utc)))
        self.assertEqual(self.store.get_setting('trends_pause_until'), '')

    def test_expired_pause_does_not_replace_manual_recipient(self):
        manual = self.store.enqueue('trends_collect', {'actor_id': 1})
        self.store.set_setting('trends_pause_until', '2026-10-05T07:00:00+00:00')
        trends_api.enqueue_due(self.store, datetime(2026, 10, 5, 8, tzinfo=timezone.utc))
        payload = json.loads(self.store.one('SELECT payload FROM jobs WHERE id=?', (manual,))['payload'])
        self.assertNotIn('scheduled', payload)

    def test_bot_time_region_and_window_settings_persist(self):
        controls, client = BotControls(self.store), Mock()
        controls.trends_controls.callback(client, 1, 'gt:geo')
        controls.trends_controls.input(client, 1, 'WORLD')
        self.assertEqual(self.store.get_setting('trends_schedule')['geo'], '')
        controls.trends_controls.callback(client, 1, 'gt:time')
        controls.trends_controls.input(client, 1, '09:00,19:30')
        self.assertEqual(self.store.get_setting('trends_schedule')['times'], ['09:00', '19:30'])
        controls.trends_controls.callback(client, 1, 'gt:window:1')
        self.assertEqual(self.store.get_setting('trends_schedule')['window'], 'today 12-m')
        controls.trends_controls.callback(client, 1, 'gt:off')
        self.assertFalse(self.store.get_setting('trends_schedule')['enabled'])

    def test_weekly_google_dates_use_period_end_not_future_value_at_start(self):
        raw = trends_api.to_csv(timeline(), 'рост цен', 'RU', 'today 5-y')
        identity = google_trends.import_csv(self.store, raw, 1)[0]
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (identity,))['metadata'])
        rows = market_store.history(self.store, identity)
        values = market_analysis.values(rows, meta, cutoff='2024-01-10')
        self.assertTrue(values.empty)
        values = market_analysis.values(rows, meta)
        self.assertEqual(values.index[0].date(), date(2024, 1, 13))
        self.assertEqual(meta['week_anchor'], 'SAT')

    def test_weekly_google_and_daily_market_correlation_align_completed_weeks(self):
        data = timeline(60)
        for i, row in enumerate(data['default']['timelineData']):
            row['value'] = [10 + (i*i) % 60]
        identity = google_trends.import_csv(self.store, trends_api.to_csv(data, 'рост цен', 'RU', 'today 5-y'), 1)[0]
        meta = json.loads(self.store.one('SELECT metadata FROM market_series WHERE id=?', (identity,))['metadata'])
        weekly = market_store.history(self.store, identity)
        daily = []
        for row in weekly:
            for day in range(7):
                daily.append({'date': (date.fromisoformat(row['date']) + timedelta(days=day)).isoformat(), 'value': row['value']})
        result = market_analysis.correlations(weekly, daily, meta, {'frequency': 'daily', 'calendar': 'calendar', 'aggregation': 'last'}, window=52)
        self.assertEqual(result['status'], 'ok')
        self.assertAlmostEqual(result['pearson'], 1)

    def test_oversized_or_non_json_response_never_imported(self):
        result = response()
        result.iter_content.return_value = [b'x' * (google_trends.MAX_BYTES + 1)]
        self.session.get.return_value = result
        with self.assertRaises(trends_api.TrendsError):
            self.client.fetch('рост цен')
        result.close.assert_called_once()

    def test_pair_and_model_choices_follow_new_snapshot(self):
        raw = trends_api.to_csv(timeline(), 'рост цен', 'RU', 'today 5-y')
        old = google_trends.import_csv(self.store, raw, 1)[0]
        self.store.set_setting('market_pairs:2', [{'left': old, 'right': 'cbr.test', 'lag': 1}])
        self.store.set_setting('market_features:' + old, ['cbr.test'])
        new = google_trends.import_csv(self.store, raw.replace(b',10\r', b',11\r'), 1)[0]
        self.assertNotEqual(old, new)
        self.assertEqual(self.store.get_setting('market_pairs:2')[0]['left'], new)
        self.assertEqual(self.store.get_setting('market_features:' + new), ['cbr.test'])
