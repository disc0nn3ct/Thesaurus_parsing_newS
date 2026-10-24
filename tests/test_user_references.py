"""Username-based access uses fresh Telegram resolution and stable numeric IDs."""
import unittest
from types import SimpleNamespace

import test_subscription_access as fixtures

from thesaurus import feature_access, subscription_access, user_management


class UserReferenceTests(unittest.TestCase):
    setUp = fixtures.SubscriptionAccessTests.setUp
    message = fixtures.SubscriptionAccessTests.message
    click = fixtures.SubscriptionAccessTests.click
    view = fixtures.SubscriptionAccessTests.view

    def account(self, **changes):
        fields = {'id': 99, 'first_name': 'Иван', 'last_name': 'Петров', 'username': 'ivan_test',
                  'is_bot': False, 'is_deleted': False}
        fields.update(changes)
        return SimpleNamespace(**fields)

    def test_add_form_accepts_username_and_displays_resolved_identity(self):
        self.client.get_users.return_value = self.account()
        self.click('user:add:new', 1)
        self.assertIn('@username', self.view()[0])
        self.message('@IVAN_TEST', 1)
        self.client.get_users.assert_called_once_with('@IVAN_TEST')
        row = self.store.one('SELECT role,enabled FROM users WHERE user_id=99')
        self.assertEqual(row, {'role': 'subscriber', 'enabled': 1})
        self.assertEqual(subscription_access.assigned(self.store, 99), [])
        self.assertEqual(feature_access.assigned(self.store, 99), [])
        text, buttons = self.view()
        self.assertIn('Иван Петров · @ivan_test · ID 99', text)
        self.assertIn('ua:open:99', buttons)
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=1')['pending'])
        self.assertIsNone(self.store.one('SELECT last_seen FROM user_profiles WHERE user_id=99')['last_seen'])
        self.assertFalse(any(call.args[0] == 99 for call in self.client.send_message.call_args_list))

    def test_numeric_add_still_works(self):
        self.client.get_users.return_value = self.account(username=None)
        self.click('user:add:new', 1)
        self.message('99', 1)
        self.assertTrue(self.store.authorize(99))
        self.client.get_users.assert_called_once_with(99)
        self.assertIn('Иван Петров · ID 99', self.view()[0])

    def test_allow_and_deny_commands_accept_username(self):
        self.client.get_users.return_value = self.account()
        self.message('/allow @ivan_test', 1)
        self.assertTrue(self.store.authorize(99))
        self.assertIn('@ivan_test · ID 99', self.view()[0])
        self.message('/deny @ivan_test', 1)
        self.assertFalse(self.store.authorize(99))
        self.assertIn('Доступ отключён:', self.view()[0])
        self.assertIn('@ivan_test · ID 99', self.view()[0])

    def test_failed_lookup_keeps_form_and_never_uses_stale_cached_username(self):
        user_management.remember(self.store, self.account(id=88), interacted=False)
        self.client.get_users.side_effect = TimeoutError('private-api-details')
        self.click('user:add:new', 1)
        self.message('@ivan_test', 1)
        self.assertFalse(self.store.authorize(88))
        self.assertFalse(self.store.authorize(99))
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=1')['pending'], 'add_user')
        text = self.view()[0]
        self.assertIn('/start', text)
        self.assertNotIn('private-api-details', text)
        self.assertFalse(self.store.query('SELECT * FROM access_audit'))
        self.message('/cancel', 1)
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=1')['pending'])

    def test_current_owner_wins_over_old_profile(self):
        user_management.remember(self.store, self.account(id=88), interacted=False)
        self.client.get_users.return_value = self.account()
        self.message('/allow @ivan_test', 1)
        self.assertFalse(self.store.authorize(88))
        self.assertTrue(self.store.authorize(99))
        self.assertEqual(self.store.one('SELECT target_id FROM access_audit')['target_id'], 99)

    def test_invalid_input_bots_deleted_and_mismatched_profiles_cannot_grant_access(self):
        for value in ('0', '-1', '@bad name', '@кириллица', '@', 'ivan_test', 'https://t.me/ivan_test'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                user_management.resolve_user_reference(self.store, self.client, value)
        self.client.get_users.assert_not_called()
        for account in (self.account(is_bot=True), self.account(is_deleted=True),
                        self.account(username='somebody_else'), self.account(id=-99),
                        SimpleNamespace(id=99), self.account(id=True)):
            self.client.get_users.return_value = account
            with self.subTest(account=account), self.assertRaises(ValueError):
                user_management.resolve_user_reference(self.store, self.client, '@ivan_test')
        self.assertFalse(self.store.authorize(99))
        self.assertFalse(self.store.one('SELECT * FROM user_profiles WHERE user_id=99'))

    def test_lookup_errors_are_actionable_and_do_not_expose_telegram_errors(self):
        from pyrogram.errors import FloodWait, UsernameInvalid, UsernameNotOccupied
        for error, expected in ((UsernameInvalid(), 'не найден'), (UsernameNotOccupied(), 'не найден'),
                                (FloodWait(25), '25 с')):
            self.client.get_users.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaisesRegex(ValueError, expected):
                user_management.resolve_user_reference(self.store, self.client, '@ivan_test')

    def test_subscriber_cannot_resolve_or_grant_a_username(self):
        self.message('/allow @ivan_test', 2)
        self.click('user:add:new', 2)
        self.bot.market_controls.pending(2, 'add_user')
        self.message('@ivan_test', 2)
        self.client.get_users.assert_not_called()
        self.assertFalse(self.store.authorize(99))
