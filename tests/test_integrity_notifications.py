"""Integrity notifications stay concise, inspectable and idempotent."""
import json
import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from thesaurus import integrity, market_store
from thesaurus.integrity_text import digest
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_integrity import IntegrityControls


class IntegrityNotificationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('notices-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        parent = {'id': 'archive.miacr', 'dataset': 'archive.miacr', 'source': 'cbr',
                  'title': 'Архив MIACR 1996–2000', 'frequency': 'daily', 'kind': 'dataset', 'unit': '%'}
        market_store.register(self.store, [parent, *[{**parent, 'id': f'archive.miacr.hash{i}', 'kind': 'series',
                                                      'title': f'MIBID · {i+1} день'} for i in range(24)]])

    def test_hundreds_of_conflicts_use_one_delivery_without_losing_journal(self):
        with self.store.connection() as conn:
            conn.executemany('INSERT INTO integrity_incidents(fingerprint,series_id,date,kind,severity,new_payload,created_at) VALUES(?,?,?,?,?,?,?)',
                [(str(i), f'archive.miacr.hash{i%24}', f'1997-{i%12+1:02}-01', 'source_conflict', 'warning',
                  dumps({'candidates': [{'value': 7.63}, {'value': 16.4}]}), utcnow()) for i in range(757)])
        now = datetime(2026, 10, 9, 16, tzinfo=timezone.utc)
        integrity.dispatch(self.store, now=now)
        report = self.store.one('SELECT * FROM reports')
        self.assertIn('Архив MIACR 1996–2000', report['text'])
        self.assertIn('Противоречивые значения источника: 757; показателей: 24', report['text'])
        self.assertIn('исключены из расчётов', report['text'])
        self.assertLess(len(report['text']), 1500)
        self.assertNotIn('source_conflict', report['text'])
        self.assertNotIn('hash', report['text'])
        deliveries = self.store.query('SELECT * FROM deliveries')
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(deliveries[0]['recipient'], '1')
        payload = json.loads(deliveries[0]['payload'])
        self.assertEqual(payload['buttons'][0][0]['callback_data'], 'ig:menu')
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_incidents WHERE notified_at IS NOT NULL')['n'], 757)
        integrity.dispatch(self.store, now=now)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM reports')['n'], 1)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_incidents')['n'], 757)

    def test_closed_incidents_are_not_announced(self):
        identity = integrity.incident(self.store, 'source_conflict', series_id='archive.miacr.hash0')
        integrity.resolve(self.store, 1, identity, 'reject')
        integrity.dispatch(self.store, now=datetime(2026, 10, 9, 16, tzinfo=timezone.utc))
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM reports')['n'], 0)

    def test_critical_problem_stays_visible_when_many_sources_have_warnings(self):
        rows = [{'id': i, 'severity': 'warning', 'kind': 'source_schema', 'series_id': f'unknown{i}', 'date': '2020-01-01'}
                for i in range(20)]
        rows.append({'id': 30, 'severity': 'critical', 'kind': 'sqlite_integrity', 'series_id': None, 'date': None})
        text = digest(self.store, rows)
        self.assertIn('Критическая · Локальная база SQLite', text)
        self.assertIn('Сбор остановлен', text)
        self.assertIn('Ещё групп: 15', text)
        self.assertLess(len(text), 3900)

    def test_conflict_card_explains_values_and_does_not_offer_arbitrary_acceptance(self):
        identity = integrity.incident(self.store, 'source_conflict', series_id='archive.miacr.hash0', day='1999-11-30',
            new={'candidates': [{'value': 7.63}, {'value': 16.4}], 'action': 'excluded_from_analysis'})
        reply = Mock()
        controls = IntegrityControls(SimpleNamespace(store=self.store, _reply=reply))
        controls.callback(Mock(), 1, f'ig:show:{identity}')
        text = reply.call_args.args[2]
        self.assertIn('MIBID · 1 день (%)', text)
        self.assertIn('7.63, 16.4', text)
        self.assertNotIn('hash0', text)
        markup = reply.call_args.args[3]
        self.assertFalse(any(button.callback_data.startswith('ig:accept:') for row in markup.inline_keyboard for button in row))
        with self.assertRaises(PermissionError):
            controls.callback(Mock(), 2, f'ig:show:{identity}')


if __name__ == '__main__':
    unittest.main()
