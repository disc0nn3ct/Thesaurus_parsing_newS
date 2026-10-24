"""Shared footer rules and cross-project journeys, without Telegram/network."""
import shutil
import unittest
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus.store import Store
from thesaurus.telegram_navigation import normalize_buttons


def _button(text, data):
    from pyrogram.types import InlineKeyboardButton
    return InlineKeyboardButton(text, callback_data=data)


def _flat(rows):
    return [button for row in rows for button in row]


class NavigationButtonTests(unittest.TestCase):
    def test_context_cancel_wins_over_generic_cancel(self):
        rows = [[_button('Добавить вручную', 'wc:add')], [_button('Отмена', 'ui:cancel')],
                [_button('Отменить добавление', 'wc:cancel')]]
        result = _flat(normalize_buttons(rows, pending='wine:photo', back='wc:menu', route='wc:photo'))
        self.assertEqual([button.callback_data for button in result if 'отмен' in button.text.casefold()], ['wc:cancel'])
        self.assertEqual(sum(button.callback_data == 'menu:home' for button in result), 1)
        self.assertEqual(sum(button.callback_data == 'wc:menu' for button in result), 1)

    def test_cancel_returning_to_bottle_is_not_repeated_as_back(self):
        rows = [[_button('Отмена', 'wc:lot:2:10')]]
        result = _flat(normalize_buttons(rows, pending='wine:price:10', back='wc:lot:2:10', route='wc:price:10'))
        self.assertEqual([button.callback_data for button in result], ['wc:lot:2:10', 'menu:home'])

    def test_cancel_job_remains_distinct_from_cancel_form(self):
        rows = [[_button('Отменить #10', 'ig:cancel:10')], [_button('Отменить #11', 'ig:cancel:11')]]
        result = _flat(normalize_buttons(rows, pending='input', back='menu:admin', route='ig:input'))
        self.assertTrue({'ig:cancel:10', 'ig:cancel:11', 'ui:cancel'} <= {button.callback_data for button in result})

    def test_distinct_actions_and_pagination_are_preserved(self):
        rows = [[_button('CSV', 'export:csv'), _button('Excel', 'export:xlsx')],
                [_button('Следующая ›', 'page:2'), _button('Предыдущая ‹', 'page:0')],
                [_button('Одинаковое название', 'item:1'), _button('Одинаковое название', 'item:2')],
                [_button('CSV повтор', 'export:csv')]]
        result = _flat(normalize_buttons(rows))
        self.assertEqual(len(result), 6)
        self.assertEqual({button.callback_data for button in result},
                         {'export:csv', 'export:xlsx', 'page:2', 'page:0', 'item:1', 'item:2'})

    def test_duplicate_back_prefers_actual_context_and_keeps_one_home(self):
        rows = [[_button('← Назад', 'old:back')], [_button('← Назад', 'correct:back')],
                [_button('⌂ Проекты', 'menu:home')], [_button('Главное меню', 'menu:home')]]
        result = _flat(normalize_buttons(rows, back='correct:back', route='child:menu'))
        self.assertEqual([button.callback_data for button in result], ['correct:back', 'menu:home'])

    def test_footer_does_not_create_home_or_back_loop(self):
        self.assertEqual(normalize_buttons([], route='menu:home', pending=None, back='menu:home'), [])
        result = _flat(normalize_buttons([], route='screen', back='screen'))
        self.assertEqual([button.callback_data for button in result], ['menu:home'])

    def test_input_without_explicit_cancel_gets_one_cancel_and_footer(self):
        result = _flat(normalize_buttons([], pending='sourceunit:abc', back='menu:sources', route='source:input'))
        self.assertEqual([button.callback_data for button in result], ['ui:cancel', 'menu:sources', 'menu:home'])


class CrossMenuAuditTests(unittest.TestCase):
    def setUp(self):
        from thesaurus.telegram_bot import BotControls
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('menu-audit-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, 'subscriber', subscriptions='all')
        self.bot = BotControls(self.store)
        self.client = Mock()
        sequence = count(100)
        self.client.send_message.side_effect = lambda *args, **kwargs: SimpleNamespace(id=next(sequence))
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=1,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def click(self, data, actor=2):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=record.get('message_id', 100), chat=SimpleNamespace(id=actor, type='private')),
            answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self):
        call = next(call for call in reversed(self.client.mock_calls) if call[0] in {'send_message', 'edit_message_text'})
        text = call.args[2] if call[0] == 'edit_message_text' else call.args[1]
        markup = call.kwargs.get('reply_markup')
        return text, _flat(markup.inline_keyboard) if markup else []

    def assert_footer(self, *, cancel=False):
        buttons = self.view()[1]
        callbacks = [button.callback_data for button in buttons if button.callback_data]
        self.assertEqual(len(callbacks), len(set(callbacks)))
        self.assertLessEqual(sum(button.text == '← Назад' for button in buttons), 1)
        self.assertEqual(sum(button.callback_data == 'menu:home' for button in buttons), 1)
        cancel_buttons = [button for button in buttons if button.text.casefold() in
                          {'отмена', 'отменить добавление', 'отменить ввод'}]
        self.assertEqual(len(cancel_buttons), 1 if cancel else 0)

    def test_wine_photo_and_manual_inputs_have_one_cancel(self):
        self.message('/cellar')
        self.click('wc:add')
        self.assert_footer()
        self.click('wc:photo')
        self.assert_footer(cancel=True)
        self.click('wc:add')
        self.assert_footer()
        self.click('wc:manual')
        self.assert_footer(cancel=True)
        self.message('Merlot')
        self.assert_footer(cancel=True)
        self.assertEqual(self.client.send_message.call_count, 1)

    def test_bottle_price_cancel_returns_to_its_card_without_duplicate_back(self):
        from thesaurus import wine_cellar
        lot = wine_cellar.add_bottle(self.store, 2, {'name': 'Merlot', 'quantity': 1, 'purchase_price': 1000})
        self.message('/cellar')
        self.click(f'wc:lot:2:{lot["id"]}')
        self.click(f'wc:price:{lot["id"]}')
        self.assert_footer(cancel=True)
        self.click(f'wc:lot:2:{lot["id"]}')
        self.assertIn('В запасе: 1', self.view()[0])
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assert_footer()

    def test_vivino_upload_and_chart_dates_use_single_generic_cancel(self):
        self.message('/vivino')
        self.click('vi:upload')
        self.assert_footer(cancel=True)
        self.click('ui:cancel')
        self.assertIn('Vivino', self.view()[0])
        self.assert_footer()
        self.message('/chart ruonia')
        self.click('ch:view:ruonia:period')
        self.click('ch:custom:ruonia')
        self.assert_footer(cancel=True)
        self.click('ui:cancel')
        self.assert_footer()

    def test_admin_user_id_prompt_and_cbr_search_have_single_cancel(self):
        self.message('/users', actor=1)
        self.click('user:add:new', actor=1)
        self.assert_footer(cancel=True)
        self.click('ui:cancel', actor=1)
        self.assertIn('Пользователи', self.view()[0])
        self.message('/projects', actor=1)
        self.click('pj:market', actor=1)
        self.click('cb:menu', actor=1)
        self.click('cb:search', actor=1)
        self.assert_footer(cancel=True)

    def test_wine_shop_selection_edits_one_message_and_one_checkbox(self):
        self.message('/wines', actor=1)
        self.click('wc:marketshops', actor=1)
        sends = self.client.send_message.call_count
        before = {button.callback_data: button.text for button in self.view()[1]}
        self.click('wc:shop:decanter', actor=1)
        after = {button.callback_data: button.text for button in self.view()[1]}
        self.assertNotEqual(before['wc:shop:decanter'], after['wc:shop:decanter'])
        self.assertTrue(after['wc:shop:decanter'].startswith('✓'))
        self.assertEqual(self.client.send_message.call_count, sends)
        self.assert_footer()

    def test_wine_hierarchy_back_returns_to_life_then_projects(self):
        self.message('/wines')
        buttons = {button.callback_data: button.text for button in self.view()[1]}
        self.assertEqual(buttons.get('pj:life'), '← Назад')
        self.click('pj:life')
        self.assertIn('Жизнь', self.view()[0])
        self.assert_footer()
        self.click('menu:home')
        self.assertIn('Главное меню', self.view()[0])

    def test_advanced_shop_domains_cancel_returns_to_shop_presets(self):
        self.message('/wines', actor=1)
        self.click('wc:marketshops', actor=1)
        self.click('wc:shops', actor=1)
        self.assert_footer(cancel=True)
        self.click('ui:cancel', actor=1)
        self.assertIn('Магазины и справочники вина', self.view()[0])
        self.assert_footer()

    def test_life_period_form_returns_to_period_picker_with_one_cancel(self):
        self.message('/life')
        self.click('lf:home')
        self.assert_footer()
        self.click('lf:period')
        self.click('lf:period:custom')
        self.assert_footer(cancel=True)
        self.click('ui:cancel')
        self.assertIn('Период учёта', self.view()[0])
        self.assert_footer()
        sends = self.client.send_message.call_count
        self.click('lf:period:all')
        self.assertIn('Вся история', self.view()[0])
        self.assertEqual(self.client.send_message.call_count, sends)

    def test_life_manual_review_cancel_prevents_old_confirmation(self):
        from thesaurus import life_ledger
        self.message('/expenses')
        self.click('lf:add:expense')
        self.assert_footer(cancel=True)
        self.message('1200,50')
        self.click('lf:today')
        self.click('lf:category:0')
        self.click('lf:nonote')
        self.assertIn('Проверка операции', self.view()[0])
        self.assert_footer(cancel=True)
        save = next(button.callback_data for button in self.view()[1] if button.callback_data.startswith('lf:save:'))
        self.click('lf:home')
        self.assert_footer()
        self.assertIsNone(self.store.get_setting('life_entry:2'))
        self.click(save)
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)

    def test_receipt_item_checkbox_and_volume_cancel_preserve_review_context(self):
        from thesaurus import life_ledger
        token = uuid4().hex
        life_ledger.save_receipt_draft(self.store, 2, {
            'status': 'parsed', 'fingerprint': 'b'*64, 'date': '2026-10-01', 'merchant': 'Магазин',
            'currency': 'RUB', 'total': 1200, 'warnings': [],
            'items': [{'name': 'Merlot', 'quantity': 1, 'amount': 1200, 'unit_price': 1200,
                       'is_wine': True, 'volume_ml': None}]}, token)
        self.click('rc:review:'+token)
        self.assert_footer()
        self.click(f'rc:item:{token}:0')
        self.assert_footer()
        sends = self.client.send_message.call_count
        self.click(f'rc:toggle:{token}:0')
        buttons = {button.callback_data: button.text for button in self.view()[1]}
        self.assertTrue(buttons[f'rc:toggle:{token}:0'].startswith('✓'))
        self.assertEqual(self.client.send_message.call_count, sends)
        self.click(f'rc:edit:{token}:volume_ml:0')
        self.assert_footer(cancel=True)
        self.click('ui:cancel')
        self.assertIn('Товар 1', self.view()[0])
        self.assert_footer()
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)


if __name__ == '__main__':
    unittest.main()
