"""Project navigation, isolation and access, with an offline Telegram client."""
import shutil
import unittest
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus.store import Store
from thesaurus.telegram_bot import COMMANDS, BotControls
from thesaurus.telegram_commands import PUBLIC
from thesaurus.telegram_projects import PROJECTS, setting_key


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('projects-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        for actor, role in [(1, 'admin'), (2, 'subscriber'), (3, 'subscriber')]:
            self.store.add_user(actor, role, subscriptions='all')
        self.bot = BotControls(self.store)
        self.client = Mock()
        sequence = count(100)
        self.client.send_message.side_effect = lambda *args, **kwargs: SimpleNamespace(id=next(sequence))
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=1,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def click(self, data, actor=2, kind='private', chat=None, identifier=None):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=identifier or record.get('message_id', 100),
                                    chat=SimpleNamespace(id=chat or actor, type=kind)), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self):
        rpc = next(call for call in reversed(self.client.mock_calls) if call[0] in {'send_message', 'edit_message_text'})
        text = rpc.args[2] if rpc[0] == 'edit_message_text' else rpc.args[1]
        markup = rpc.kwargs.get('reply_markup')
        return text, {b.callback_data: b.text for row in markup.inline_keyboard for b in row}

    def test_home_is_project_picker_without_market_actions(self):
        self.message('/start')
        text, buttons = self.view()
        self.assertIn('Главное меню', text)
        self.assertTrue({'pj:market', 'pj:life', 'menu:guide', 'menu:status'} <= buttons.keys())
        self.assertFalse({'ch:menu', 'menu:settings', 'menu:report', 'menu:admin'} & buttons.keys())
        self.assertEqual(buttons['pj:life'], 'Жизнь')
        self.assertNotIn('pj:vivino', buttons)
        self.assertLessEqual(len(buttons), 5)

    def test_market_graph_back_returns_to_project_then_projects(self):
        self.message('/projects')
        self.click('pj:market')
        self.assertIn('Построить график', self.view()[1].values())
        self.click('ch:menu')
        self.assertEqual(self.view()[1]['pj:market'], '← Назад')
        self.click('pj:market')
        self.assertIn('Рынок и экономика', self.view()[0])
        self.click('menu:home')
        self.assertIn('Жизнь', self.view()[0])
        self.assertEqual(self.client.send_message.call_count, 1)
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_vivino_has_upload_and_help_with_own_return(self):
        self.message('/vivino')
        self.assertIn('географический HTML-атлас', self.view()[0])
        self.assertNotIn('ch:menu', self.view()[1])
        self.click('vi:upload')
        self.assertIn('full_wine_list.csv', self.view()[0])
        self.assertEqual(self.view()[1]['vi:menu'], '← Назад')
        self.click('vi:menu')
        self.click('vi:help')
        self.assertIn('Скачайте присланный HTML', self.view()[0])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_project_switch_preserves_filters_chart_and_active_schedules(self):
        settings = {'chart:2:ruonia': {'period': 'month', 'horizon': 2},
                    'chart_schedule:2:ruonia': {'enabled': True, 'weekdays': [4], 'times': ['15:30']},
                    'market_selection:2': ['test.series'], 'trends_schedule': {'enabled': True}}
        for key, value in settings.items():
            self.store.set_setting(key, value)
        self.message('/projects')
        self.click('pj:life')
        self.click('pj:vivino')
        self.click('menu:home')
        self.click('pj:market')
        for key, value in settings.items():
            self.assertEqual(self.store.get_setting(key), value)
        self.assertIsNone(self.store.get_setting(setting_key(3, 'vivino', 'visited')))

    def test_switch_cancels_unfinished_input_and_keeps_stale_market_buttons_out(self):
        self.message('/chart ruonia')
        self.click('ch:view:ruonia:period')
        self.click('ch:custom:ruonia')
        self.assertTrue(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.message('/vivino')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.click('ch:build:ruonia').answer.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.message('2020-01-01 2025-01-01')
        self.assertNotIn('Начало периода', self.view()[0])

    def test_global_admin_and_market_admin_are_separate(self):
        self.message('/admin', 1)
        _, buttons = self.view()
        self.assertTrue({'menu:users', 'menu:jobs', 'menu:budget', 'pj:market:manage'} <= buttons.keys())
        self.assertNotIn('menu:sources', buttons)
        self.click('pj:market:manage', 1)
        self.assertIn('menu:autocollect', self.view()[1])
        self.click('menu:autocollect', 1)
        self.assertIn('pj:market:manage', self.view()[1])
        self.assertNotIn('menu:users', self.view()[1])

    def test_authorization_private_chat_and_revocation_on_every_project_page(self):
        for actor, kind, chat in [(99, 'private', 99), (2, 'group', -1), (2, 'private', 3)]:
            event = self.click('pj:vivino', actor=actor, kind=kind, chat=chat)
            event.answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.message('/start')
        self.click('pj:market:manage', identifier=900).answer.assert_called_once_with(
            'Действие доступно администратору', show_alert=True)
        self.store.add_user(2, enabled=False, subscriptions='all')
        self.click('pj:vivino').answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'visited')))

    def test_registry_and_commands_are_unique_and_unknown_routes_do_not_create_state(self):
        self.assertEqual(len(PROJECTS), len({item.id for item in PROJECTS}))
        self.assertTrue({'projects', 'vivino'} <= set(COMMANDS))
        self.assertTrue({'projects', 'vivino'} <= {name for name, _ in PUBLIC})
        for item in PROJECTS:
            self.assertLessEqual(len(item.route.encode('utf-8')), 64)
        self.message('/start')
        before = self.store.query('SELECT * FROM settings')
        self.click('pj:unknown', identifier=900).answer.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM settings'), before)
        with self.assertRaises(ValueError):
            setting_key(True, 'market', 'visited')
        with self.assertRaises(ValueError):
            setting_key(2, '../market', 'visited')
        with self.assertRaises(ValueError):
            setting_key(2, 'market', 'a:b')

    def test_vivino_owner_settings_cannot_collide_with_market_or_other_users(self):
        for actor, identifier, expected in [(2, 'market', 'month'), (2, 'vivino', 'rating'), (3, 'vivino', 'region')]:
            self.store.set_setting(setting_key(actor, identifier, 'filter'), expected)
        self.assertEqual(self.store.get_setting(setting_key(2, 'market', 'filter')), 'month')
        self.assertEqual(self.store.get_setting(setting_key(2, 'vivino', 'filter')), 'rating')
        self.assertEqual(self.store.get_setting(setting_key(3, 'vivino', 'filter')), 'region')
