"""Search shutdown and CBR entry paths through offline Telegram handlers."""
import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import Mock, patch

import test_telegram_ux as fixtures

from thesaurus import (
    autocollect,
    chart_schedules,
    google_trends,
    market,
    market_store,
    search_provider,
    trends_api,
)
from thesaurus.collectors import Collector
from thesaurus.config import Settings
from thesaurus.scheduling import seed_subscriptions, update_subscription
from thesaurus.service import Worker
from thesaurus.store import utcnow


class SearchDisabledTests(unittest.TestCase):
    message = fixtures.TelegramUXTests.message
    click = fixtures.TelegramUXTests.click
    view = fixtures.TelegramUXTests.view

    def setUp(self):
        fixtures.TelegramUXTests.setUp(self)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('Offline acceptance')))

    def disable(self):
        search_provider.deactivate(self.store)

    def test_shutdown_retains_history_and_settings_but_cancels_only_search_work(self):
        meta = {'id': 'test.google', 'source': 'google_trends', 'dataset': 'test.google',
                'kind': 'series', 'title': 'Search', 'frequency': 'weekly', 'unit': 'index'}
        market_store.register(self.store, [meta])
        self.store.execute('UPDATE market_series SET enabled=1 WHERE id=?', (meta['id'],))
        market_store.save_rows(self.store, meta['id'], [{'date': '2025-01-01', 'value': 42}])
        self.store.set_setting('search_provider', 'google_trends')
        self.store.set_setting('trends_schedule', {'enabled': True, 'geo': 'RU', 'times': ['08:30']})
        self.store.set_setting('market_schedule', {'enabled': True, 'times': ['18:30']})
        self.store.set_setting('chart:2:trends', {'period': 'all', 'identity': meta['id']})
        self.store.set_setting('chart_schedule:2:trends', {'enabled': True, 'section': 'trends', 'actor_id': 2})
        seed_subscriptions(self.store, '2', 2)
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE section IN ('wordstat','regions','ruonia')")
        for kind, payload in [('trends_collect', {}), ('trends_import', {}), ('report', {'section': 'wordstat'}),
                              ('telegram_chart', {'section': 'trends'}), ('market_collect', {}), ('report', {'section': 'ruonia'})]:
            self.store.enqueue(kind, payload)
        self.disable()
        backups = list(self.root.glob('*.search-backup-*.sqlite3'))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as conn:
            self.assertEqual(json.loads(conn.execute("SELECT value FROM settings WHERE key='search_provider'").fetchone()[0]), 'google_trends')
        self.disable()
        self.assertEqual(len(list(self.root.glob('*.search-backup-*.sqlite3'))), 1)
        self.assertEqual([row['state'] for row in self.store.query('SELECT state FROM jobs ORDER BY id')],
                         ['cancelled']*4 + ['pending']*2)
        self.assertFalse(self.store.get_setting('trends_schedule')['enabled'])
        self.assertFalse(self.store.get_setting('chart_schedule:2:trends')['enabled'])
        self.assertTrue(self.store.get_setting('market_schedule')['enabled'])
        self.assertEqual(market_store.history(self.store, meta['id'])[0]['value'], 42)
        self.assertEqual(self.store.get_setting('chart:2:trends')['period'], 'all')
        self.assertEqual(self.store.one("SELECT enabled FROM subscriptions WHERE section='ruonia' AND user_id=2")['enabled'], 1)
        with self.assertRaises(ValueError):
            market.enable(self.store, meta['id'], True)

    def test_worker_rejects_stale_jobs_without_any_provider_call(self):
        self.disable()
        providers = Mock()
        worker = Worker(self.store, Settings(root=self.root), providers, charts=False)
        for kind, payload in [('trends_collect', {}), ('trends_import', {}), ('report', {'section': 'regions'})]:
            self.store.enqueue(kind, payload)
            self.assertTrue(worker.process_next())
        self.assertTrue(all(row['state'] == 'cancelled' for row in self.store.query('SELECT state FROM jobs')))
        self.assertEqual(providers.mock_calls, [])

    def test_shutdown_cancels_pending_parts_but_preserves_sent_and_uncertain(self):
        stamp = utcnow()
        for section in ('trends', 'ruonia'):
            self.store.execute('INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text) VALUES(?,?,?,1,0,?,?)',
                               (section, section, stamp, 'report', 'report'))
            for index, state in enumerate(('pending', 'sent', 'uncertain')):
                self.store.execute('INSERT INTO deliveries(report_id,recipient,part_index,payload,state,message_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
                                   (section, '2', index, '{}', state, 123 if state == 'sent' else None, stamp, stamp))
        self.disable()
        self.assertEqual([r['state'] for r in self.store.query("SELECT state FROM deliveries WHERE report_id='trends' ORDER BY part_index")],
                         ['cancelled', 'sent', 'uncertain'])
        self.assertEqual([r['state'] for r in self.store.query("SELECT state FROM deliveries WHERE report_id='ruonia' ORDER BY part_index")],
                         ['pending', 'sent', 'uncertain'])

    def test_failed_shutdown_rolls_back_provider_and_schedules(self):
        self.store.set_setting('search_provider', 'google_trends')
        self.store.set_setting('trends_schedule', {'enabled': True})
        self.store.execute("CREATE TRIGGER reject_shutdown BEFORE UPDATE ON market_series BEGIN SELECT RAISE(ABORT,'test rollback'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.disable()
        self.assertEqual(self.store.get_setting('search_provider'), 'google_trends')
        self.assertTrue(self.store.get_setting('trends_schedule')['enabled'])

    def test_collectors_import_and_enable_cannot_bypass_shutdown(self):
        self.disable()
        with self.assertRaises(ValueError):
            autocollect.update_schedule(self.store, 'trends', 1, enabled=True)
        with self.assertRaises(ValueError):
            update_subscription(self.store, 2, 'wordstat', enabled=True)
        with self.assertRaises(ValueError):
            trends_api.collect(self.store, 1)
        with self.assertRaises(ValueError):
            google_trends.import_csv(self.store, b'invalid', 1)
        providers = Mock()
        with self.assertRaises(ValueError):
            Collector(self.store, Settings(root=self.root), providers).collect('wordstat')
        self.assertEqual(providers.mock_calls, [])

    def test_main_screens_hide_search_without_wordstat_fallback(self):
        self.disable()
        for actor in (1, 2):
            self.message('/start', actor)
            self.click('pj:market', actor)
            self.assertIn('cb:menu', self.view()[1])
            self.assertIn('cb:guide', self.view()[1])
            for command in ('/chart', '/graphs', '/subscriptions', '/settings'):
                self.message(command, actor)
                text, buttons = self.view()
                self.assertFalse(any(search_provider.hidden_route(route) for route in buttons))
                self.assertNotIn('Google Trends', text + ' '.join(buttons.values()))
                self.assertNotIn('Wordstat', text + ' '.join(buttons.values()))
        self.message('/autocollect', 1)
        self.assertNotIn('Google', self.view()[0])
        self.assertIn('ui:autocollect:market:on', self.view()[1])

    def test_old_commands_and_buttons_cannot_queue_search_work(self):
        self.disable()
        for command in ('/trends', '/report wordstat', '/chart trends', '/subscriptions wordstat on', '/backtest'):
            self.message(command, 1)
            self.assertIn('временно отключены', self.view()[0])
            self.assertIn('cb:menu', self.view()[1])
        for route in ('gt:on', 'gt:collect', 'request:wordstat', 'cs:trends:on', 'ui:autocollect:trends:on'):
            event = self.click(route, 1, message_id=999)
            event.answer.assert_called_once()
            self.assertIn('временно отключены', self.view()[0])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertEqual(trends_api.enqueue_due(self.store), None)

    def test_help_keeps_cbr_paths_without_search_commands(self):
        self.disable()
        self.message('/start')
        self.click('pj:market')
        self.click('menu:help')
        self.click('ui:help:commands')
        self.assertIn('Показатели ЦБ', self.view()[0])
        self.assertNotIn('/trends', self.view()[0])
        self.assertNotIn('wordstat', self.view()[0])

    def test_search_shutdown_does_not_hide_regions_in_vivino(self):
        self.disable()
        self.message('/vivino')
        self.assertIn('va:list:2:regions:0:all', self.view()[1])
        self.click('va:list:2:regions:0:all')
        self.assertNotIn('Поисковые индикаторы', self.view()[0])
        self.assertFalse(search_provider.hidden_route('va:list:2:regions:0:all'))

    def test_search_chart_schedule_cannot_be_saved_or_activated(self):
        self.disable()
        config = chart_schedules.defaults(2, 'trends', {'period': 'all'})
        with self.assertRaises(ValueError):
            chart_schedules.save(self.store, 2, 'trends', config, enabled=True)
        self.assertFalse(chart_schedules.active(self.store, {'actor_id': 2, 'section': 'trends', 'generation': 'old'}))

    def test_cbr_starter_and_loaded_cards_preserve_back_and_do_not_collect(self):
        market.initialize(self.store)
        self.disable()
        market_store.save_rows(self.store, 'ux.0', [{'date': '2025-01-01', 'value': 10}])
        market_store.save_rows(self.store, 'ux.1', [{'date': '2025-01-01', 'value': None}])
        self.message('/start')
        self.click('pj:market')
        self.click('cb:guide')
        self.assertIn('три разные настройки', self.view()[0])
        self.click('cb:basics')
        cards = [route for route in self.view()[1] if route.startswith('cb:card:')]
        self.assertEqual(len(cards), 6)
        self.click(cards[0])
        self.assertEqual(self.view()[1]['cb:basics'], '← Назад')
        self.assertFalse(any(route.startswith('cb:load:') for route in self.view()[1]))
        self.click('cb:basics')
        self.click('cb:menu')
        self.click('cb:loaded:0')
        text, buttons = self.view()
        self.assertIn('Уже загруженные', text)
        self.assertIn('Показатель 0', text)
        self.assertNotIn('Показатель 1', text)
        card = next(route for route in buttons if route.startswith('cb:card:'))
        self.click(card)
        self.assertEqual(self.view()[1]['cb:loaded:0'], '← Назад')
        self.assertTrue(any(route.startswith('cb:chart:') for route in self.view()[1]))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
