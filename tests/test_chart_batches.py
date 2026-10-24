"""Automatic pages: permissions, confirmation, interruption and recovery, offline."""
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import chart_batches, market, market_store
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending, prepare_delivery, resolve_uncertain
from thesaurus.service import Worker
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_charts import ChartControls


class ChartBatchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('chart-batch-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        identities = [f'test.{i}' for i in range(25)]
        for identity in identities:
            market_store.register(self.store, [{'id': identity, 'title': identity, 'source': 'cbr',
                'dataset': 'test', 'kind': 'series', 'unit': '%', 'frequency': 'daily'}])
            market.enable(self.store, identity, True)
        self.report_id = uuid4().hex
        self.quality = {'actor_id': 1, 'chart_page': 0, 'chart_pages': 3, 'chart_total': 25,
                        'chart_batch_id': self.report_id, 'chart_request': {
                            'section': 'market', 'identity': None, 'options': {'period': 'all', 'horizon': 0},
                            'selection': identities, 'order': identities}}
        self.make_report(self.report_id, self.quality)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=42)

    def make_report(self, identifier, quality):
        self.store.execute('INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,quality) VALUES(?,?,?,?,?,?,?,?)',
                           (identifier, 'market', utcnow(), 1, 1, 'Графики', 'Графики', dumps(quality)))

    def config(self):
        return self.store.get_setting('chart_batch:' + self.report_id, {})

    def rendered(self):
        config = self.config()
        job = self.store.one('SELECT * FROM jobs WHERE id=?', (config['job_id'],))
        payload = json.loads(job['payload'])
        quality = {**self.quality, 'chart_page': payload['page']}
        self.make_report(payload['report_id'], quality)
        self.store.execute("UPDATE jobs SET state='done' WHERE id=?", (job['id'],))
        parts = prepare_delivery(self.store, payload['report_id'], '1', detail='full', chart_auto=payload['chart_auto'])
        return job, payload, parts

    def test_checkbox_starts_remaining_pages_only_and_waits_for_confirmation(self):
        original = chart_batches.navigation(self.store, self.report_id, self.quality)
        self.assertTrue(any(b['text'].startswith('☐') for row in original['buttons'] for b in row))
        with self.assertRaises(PermissionError):
            chart_batches.toggle(self.store, 2, self.report_id)
        response = chart_batches.toggle(self.store, 1, self.report_id)
        self.assertTrue(any(b['text'].startswith('✓') for row in response['buttons'] for b in row))
        self.assertTrue(all(len(b['callback_data'].encode()) <= 64 for row in response['buttons'] for b in row))
        with self.assertRaises(ValueError):
            ChartControls(self.store, Mock()).queue_page(self.client, 1, self.report_id, 1, 'manual')
        chart_batches.pump(self.store)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        _, payload, _ = self.rendered()
        self.assertEqual(payload['page'], 1)
        self.assertEqual(payload['options'], self.quality['chart_request']['options'])
        chart_batches.pump(self.store)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        deliver_pending(self.store, self.client)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 2)
        _, last_payload, _ = self.rendered()
        self.assertEqual(last_payload['page'], 2)
        deliver_pending(self.store, self.client)
        self.assertFalse(self.config()['enabled'])
        chart_batches.pump(self.store)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 2)

    def test_stop_and_resume_preserve_sent_parts_and_invalidate_old_generation(self):
        chart_batches.toggle(self.store, 1, self.report_id)
        job, payload, parts = self.rendered()
        self.store.execute("UPDATE deliveries SET state='sent',message_id=105 WHERE id=?", (parts[0],))
        chart_batches.toggle(self.store, 1, self.report_id)
        self.assertEqual(self.store.one('SELECT state FROM deliveries WHERE id=?', (parts[1],))['state'], 'cancelled')
        deliver_pending(self.store, self.client)
        self.client.send_message.assert_not_called()
        chart_batches.toggle(self.store, 1, self.report_id)
        self.assertFalse(chart_batches.active(self.store, payload['chart_auto']))
        self.assertEqual(self.config()['job_id'], job['id'])
        self.assertEqual(self.store.one('SELECT message_id FROM deliveries WHERE id=?', (parts[0],))['message_id'], 105)
        deliver_pending(self.store, self.client)
        self.client.send_message.assert_called_once()
        self.assertEqual(self.config()['cursor'], 2)

    def test_uncertain_delivery_survives_restart_and_never_advances_until_resolved(self):
        chart_batches.toggle(self.store, 1, self.report_id)
        _, _, parts = self.rendered()
        self.client.send_message.side_effect = TimeoutError
        deliver_pending(self.store, self.client)
        self.store = Store(self.store.path)
        chart_batches.pump(self.store)
        self.assertEqual(self.config()['cursor'], 1)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        self.assertEqual(self.store.one('SELECT state FROM deliveries WHERE id=?', (parts[0],))['state'], 'uncertain')
        resolve_uncertain(self.store, parts[0], 'sent')
        self.client.send_message.side_effect = None
        deliver_pending(self.store, self.client)
        self.assertEqual(self.config()['cursor'], 2)

    def test_disabled_user_cancels_auto_queue_and_cannot_restore_it(self):
        chart_batches.toggle(self.store, 1, self.report_id)
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=1')
        chart_batches.pump(self.store)
        self.assertFalse(self.config()['enabled'])
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'cancelled')
        with self.assertRaises(PermissionError):
            chart_batches.toggle(self.store, 1, self.report_id)

    def test_stop_before_render_can_resume_same_queued_job(self):
        chart_batches.toggle(self.store, 1, self.report_id)
        identifier = self.config()['job_id']
        chart_batches.toggle(self.store, 1, self.report_id)
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'cancelled')
        chart_batches.toggle(self.store, 1, self.report_id)
        self.assertEqual(self.config()['job_id'], identifier)
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'pending')
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)

    def test_old_checkbox_buttons_set_explicit_state_instead_of_inverting_it(self):
        controls, reply = ChartControls(self.store, Mock()), Mock()
        controls.reply = reply
        controls.callback(self.client, 1, f'ch:auto:{self.report_id}:on', 'on')
        generation = self.config()['generation']
        controls.callback(self.client, 1, f'ch:auto:{self.report_id}:on', 'old-on')
        self.assertTrue(self.config()['enabled'])
        self.assertEqual(self.config()['generation'], generation)
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)
        controls.callback(self.client, 1, f'ch:auto:{self.report_id}:off', 'off')
        controls.callback(self.client, 1, f'ch:auto:{self.report_id}:off', 'old-off')
        self.assertFalse(self.config()['enabled'])
        labels = [b.text for row in reply.call_args.args[3].inline_keyboard for b in row]
        self.assertTrue(any(label.startswith('☐') for label in labels))

    def test_real_worker_builds_remaining_pages_without_api_or_duplicate_charts(self):
        for identity in self.quality['chart_request']['selection']:
            market_store.save_rows(self.store, identity, [{'date': '2025-01-01', 'value': 10},
                                                         {'date': '2025-01-02', 'value': 11}])
        chart_batches.toggle(self.store, 1, self.report_id)
        worker = Worker(self.store, Settings(root=self.root))
        self.client.send_photo.return_value = SimpleNamespace(id=43)
        with patch('thesaurus.market_sources.Transport.get', side_effect=AssertionError('No network')):
            self.assertTrue(worker.process_next())
            job = self.store.one('SELECT * FROM jobs WHERE id=?', (self.config()['job_id'],))
            self.assertEqual(job['state'], 'done')
            first_id = json.loads(job['payload'])['report_id']
            first = self.store.one('SELECT * FROM reports WHERE id=?', (first_id,))
            self.assertEqual(len(json.loads(first['parts'])), 12)
            self.assertEqual(json.loads(first['quality'])['chart_batch_id'], self.report_id)
            deliver_pending(self.store, self.client)
            self.assertTrue(worker.process_next())
            job = self.store.one('SELECT * FROM jobs WHERE id=?', (self.config()['job_id'],))
            last = self.store.one('SELECT * FROM reports WHERE id=?', (json.loads(job['payload'])['report_id'],))
            self.assertEqual(len(json.loads(last['parts'])), 1)
            deliver_pending(self.store, self.client)
        self.assertFalse(self.config()['enabled'])
        self.assertEqual(self.client.send_photo.call_count, 13)
        final_navigation = chart_batches.navigation(self.store, last['id'], json.loads(last['quality']))
        self.assertIn('последняя страница', final_navigation['text'])
        self.assertFalse(any(b['callback_data'].startswith('ch:auto:') for row in final_navigation['buttons'] for b in row))
        first_titles = {p['caption'] for p in json.loads(first['parts'])}
        self.assertFalse(first_titles.intersection(p['caption'] for p in json.loads(last['parts'])))


if __name__ == '__main__':
    unittest.main()
