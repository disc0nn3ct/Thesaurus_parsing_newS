"""Offline ZIP import, owner isolation, local maps and revocable delivery."""
import csv
import io
import json
import shutil
import stat
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import vivino
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending
from thesaurus.service import Worker
from thesaurus.store import Store
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_projects import setting_key
from thesaurus.vivino_html import render


def csv_bytes(rows=None):
    target = io.StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=list(vivino.FIELDS) + ['Personal Note'])
    writer.writeheader()
    for row in rows if rows is not None else [
        {'Wine name': 'Вино A', 'Winery': 'Дом A', 'Country': 'France', 'Wine type': 'Red',
         'Your rating': '4.5', 'Your review': 'Ягоды, насыщенное, мягкое', 'Personal Note': 'DO-NOT-IMPORT'},
        {'Wine name': 'Вино B', 'Winery': 'Дом B', 'Country': 'Italy', 'Wine type': 'White',
         'Your rating': '', 'Your review': 'Цитрусы, свежее, лёгкое'},
        {'Wine name': 'Вино C', 'Winery': 'Дом C', 'Country': 'France', 'Wine type': 'Red',
         'Your rating': '3.5', 'Your review': ''},
    ]:
        writer.writerow(row)
    return target.getvalue().encode('utf-8-sig')


def archive_bytes(csv_data=None, entries=None):
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in (entries or {'vivino_data/full_wine_list.csv': csv_data or csv_bytes(),
                                      'vivino_data/user_profile.csv': b'DO-NOT-IMPORT'}).items():
            archive.writestr(name, value)
    return target.getvalue()


class VivinoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('vivino-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.settings = Settings(root=self.root)
        self.bot = BotControls(self.store)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=100)
        self.client.send_document.return_value = SimpleNamespace(id=101)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=1,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def click(self, data, actor=2):
        query = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=900, chat=SimpleNamespace(id=actor, type='private')), answer=Mock())
        self.bot.handle_callback(self.client, query)
        return query

    def upload(self, raw=None, actor=2, filename='export.zip', private=True):
        raw = raw or archive_bytes()
        self.client.download_media.return_value = io.BytesIO(raw)
        event = SimpleNamespace(id=200, from_user=SimpleNamespace(id=actor),
            chat=SimpleNamespace(id=actor, type='private' if private else 'group'),
            document=SimpleNamespace(file_name=filename, file_size=len(raw)))
        self.bot.handle_document(self.client, event)
        return self.store.get_setting(setting_key(actor, 'vivino', 'preview'))

    def confirmed(self, actor=2):
        self.message('/vivino', actor)
        self.click('vi:upload', actor)
        preview = self.upload(actor=actor)
        self.click('vi:confirm:' + preview['token'], actor)
        return self.store.one("SELECT * FROM jobs WHERE kind='vivino_map' ORDER BY id DESC LIMIT 1")

    def test_nested_zip_only_wine_list_and_missing_scores_are_not_zero(self):
        dataset = vivino.parse_upload(archive_bytes(), 'export.zip')
        self.assertEqual(len(dataset['rows']), 3)
        self.assertEqual(dataset['quality']['reviewed'], 2)
        self.assertIsNone(dataset['rows'][1]['rating'])
        self.assertNotIn('DO-NOT-IMPORT', json.dumps(dataset))
        self.assertEqual(dataset['rows'][0]['name'], 'Вино A')

    def test_duplicates_invalid_ratings_and_missing_names_are_accounted(self):
        row = {'Wine name': 'Wine', 'Winery': 'Winery', 'Country': 'France', 'Your rating': 'NaN'}
        dataset = vivino.parse_upload(csv_bytes([row, row, {'Country': 'France'}]), 'full_wine_list.csv')
        self.assertEqual(len(dataset['rows']), 1)
        self.assertEqual(dataset['quality']['duplicates'], 1)
        self.assertEqual(dataset['quality']['missing_names'], 1)
        self.assertIsNone(dataset['rows'][0]['rating'])

    def test_rejects_traversal_absolute_paths_symlinks_and_multiple_targets(self):
        for name in ('../file.txt', '/file.txt', 'C:/file.txt', 'x\\..\\file.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                vivino.parse_upload(archive_bytes(entries={'full_wine_list.csv': csv_bytes(), name: b'x'}), 'export.zip')
        with self.assertRaises(ValueError):
            vivino.parse_upload(archive_bytes(entries={'a/full_wine_list.csv': csv_bytes(), 'b/full_wine_list.csv': csv_bytes()}), 'export.zip')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            member = zipfile.ZipInfo('full_wine_list.csv')
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, csv_bytes())
        with self.assertRaises(ValueError):
            vivino.parse_upload(stream.getvalue(), 'export.zip')

    def test_rejects_broken_empty_oversized_and_wrong_exports(self):
        for raw, name in [(b'bad', 'file.zip'), (b'', 'x.csv'), (b'\n', 'x.csv'),
                          (b'Username,Email\nx,y', 'profile.csv'), (b'data', 'file.rar')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                vivino.parse_upload(raw, name)
        with patch.object(vivino, 'MAX_CSV', 10), self.assertRaises(ValueError):
            vivino.parse_upload(archive_bytes(), 'export.zip')
        with patch.object(vivino, 'MAX_ROWS', 1), self.assertRaises(ValueError):
            vivino.parse_upload(csv_bytes(), 'x.csv')

    def test_subscription_user_can_preview_without_committing_or_starting_jobs(self):
        self.message('/vivino')
        preview = self.upload()
        self.assertEqual(len(preview['dataset']['rows']), 3)
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'dataset')))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertIsNone(self.store.get_setting(setting_key(3, 'vivino', 'preview')))

    def test_confirmation_queues_once_and_does_not_grant_anyone_access(self):
        job = self.confirmed()
        payload = json.loads(job['payload'])
        self.click('vi:confirm:' + payload['generation'])
        self.click('vi:build')
        self.assertEqual(len(self.store.query("SELECT * FROM jobs WHERE kind='vivino_map'")), 1)
        self.assertFalse(self.store.authorize(99))
        self.assertFalse(self.store.authorize(2, admin=True))
        self.assertNotIn('review', job['payload'])

    def test_new_review_import_proposes_stock_change_only_after_confirmation(self):
        from thesaurus import wine_cellar as cellar
        lot = cellar.add_bottle(self.store, 2, {'name': 'Вино A', 'winery': 'Дом A',
                                'vintage': 2020, 'quantity': 2, 'purchase_price': 1000})
        row = {'Wine name': 'Вино A', 'Winery': 'Дом A', 'Vintage': '2020', 'Country': 'France'}
        self.message('/vivino')
        initial = self.upload(archive_bytes(csv_bytes([row])))
        self.click('vi:confirm:' + initial['token'])
        self.assertEqual(cellar.proposals(self.store, 2), [])
        self.click('vi:upload')
        preview = self.upload(archive_bytes(csv_bytes([{**row, 'Your review': 'Пробовал сегодня'}])))
        self.assertEqual(cellar.proposals(self.store, 2), [])
        self.click('vi:confirm:' + preview['token'])
        suggestions = cellar.proposals(self.store, 2)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]['lot_ids'], [lot['id']])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        self.click('vi:upload')
        again = self.upload(archive_bytes(csv_bytes([{**row, 'Your review': 'Исправление отзыва'}])))
        self.click('vi:confirm:' + again['token'])
        self.assertEqual(len(cellar.proposals(self.store, 2)), 1)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)

    def test_stolen_or_expired_preview_cannot_be_confirmed(self):
        self.message('/vivino')
        preview = self.upload()
        self.click('vi:confirm:' + preview['token'], 3).answer.assert_called_once()
        preview['expires'] = '2000-01-01T00:00:00+00:00'
        self.store.set_setting(setting_key(2, 'vivino', 'preview'), preview)
        self.click('vi:confirm:' + preview['token']).answer.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_offline_worker_renders_file_and_delivery_sends_only_to_owner(self):
        job = self.confirmed()
        worker = Worker(self.store, self.settings)
        worker.drain()
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?', (job['id'],))['state'], 'done')
        report = self.store.one("SELECT * FROM reports WHERE section='vivino'")
        file = Path(json.loads(report['parts'])[0]['path'])
        self.assertTrue(file.is_file())
        self.assertIn('Мой отзыв', file.read_text(encoding='utf-8'))
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_called_once()
        self.assertEqual(self.client.send_document.call_args.args[0], 2)
        vivino.run_job(self.store, self.settings, job, json.loads(job['payload']))
        deliver_pending(self.store, self.client)
        self.assertEqual(self.client.send_document.call_count, 1)

    def test_revoked_user_cannot_upload_or_receive_queued_result(self):
        self.message('/vivino')
        self.store.add_user(2, enabled=False, subscriptions='all')
        self.client.download_media.reset_mock()
        self.upload()
        self.client.download_media.assert_not_called()
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'preview')))

    def test_private_chat_and_explicit_project_required_before_download(self):
        self.message('/start')
        self.upload()
        self.client.download_media.assert_not_called()
        self.message('/vivino')
        self.upload(private=False)
        self.client.download_media.assert_not_called()

    def test_delete_cancels_pending_jobs_erases_reviews_and_keeps_other_owner(self):
        self.confirmed()
        self.confirmed(3)
        self.store.set_setting('chart:2:ruonia', {'period': 'month'})
        self.bot.vivino_controls.delete(2)
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'dataset')))
        self.assertIsNotNone(self.store.get_setting(setting_key(3, 'vivino', 'dataset')))
        states = [row['state'] for row in self.store.query("SELECT * FROM jobs WHERE kind='vivino_map' ORDER BY id")]
        self.assertEqual(states, ['cancelled', 'pending'])
        self.assertEqual(self.store.get_setting('chart:2:ruonia'), {'period': 'month'})

    def test_delete_after_calculation_blocks_all_delivery_parts_and_removes_html(self):
        self.confirmed()
        Worker(self.store, self.settings).drain()
        self.bot.vivino_controls.delete(2)
        self.client.reset_mock()
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_not_called()
        self.assertEqual(list((self.root / 'vivino' / '2').glob('*.html')), [])

    def test_dataset_changed_during_render_cannot_publish_old_map(self):
        job = self.confirmed()
        from thesaurus.vivino_geo_html import render as actual_render
        def erase(*args):
            self.bot.vivino_controls.delete(2)
            return actual_render(*args)
        with patch('thesaurus.vivino_geo_html.render', side_effect=erase), self.assertRaises(PermissionError):
            vivino.run_job(self.store, self.settings, job, json.loads(job['payload']))
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])
        self.assertEqual(list((self.root / 'vivino' / '2').glob('*.html')), [])

    def test_cannot_reuse_another_owners_report(self):
        own = self.confirmed()
        Worker(self.store, self.settings).drain()
        other = self.confirmed(3)
        payload = json.loads(other['payload'])
        payload['report_id'] = json.loads(own['payload'])['report_id']
        with self.assertRaises(PermissionError):
            vivino.run_job(self.store, self.settings, other, payload)

    def test_dry_run_never_prepares_delivery(self):
        self.confirmed()
        Worker(self.store, self.settings, dry_run=True).drain()
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])
        self.client.send_document.assert_not_called()

    def test_html_escapes_script_terminators_and_has_no_external_assets(self):
        dataset = vivino.parse_upload(csv_bytes(), 'x.csv')
        dataset['rows'][0]['review'] = '</script><img src=x onerror=alert(1)>'
        page = render(dataset['rows'], [[0, 0], [1, 1], [2, 2]], dataset['quality'])
        self.assertNotIn('</script><img', page)
        self.assertIn('\\u003c/script', page)
        self.assertNotIn('src="https:', page)
        self.assertNotIn('innerHTML', page)
        self.assertIn("default-src 'none'", page)

    def test_coordinates_are_finite_and_repeatable_with_missing_reviews(self):
        rows = vivino.parse_upload(csv_bytes(), 'x.csv')['rows']
        self.assertEqual(vivino.coordinates(rows), vivino.coordinates(rows))
        self.assertEqual(vivino.coordinates(rows[:1]), [[0.0, 0.0]])
        self.assertEqual(len(vivino.coordinates(rows)), 3)
