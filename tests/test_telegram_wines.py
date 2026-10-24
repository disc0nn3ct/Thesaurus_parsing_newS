"""Cellar journeys exercise reviewed forms without Telegram or shop requests."""
import json
import shutil
import unittest
from datetime import datetime, timedelta, timezone
from io import BytesIO
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import wine_cellar as cellar
from thesaurus.store import Store, utcnow
from thesaurus.telegram_wines import (
    WineControls,
    availability_addresses,
    availability_label,
)


class WineControlTests(unittest.TestCase):
    def test_drink_manual_value_and_quantity_freeze_result_and_keep_depleted_purchase_visible(self):
        lot = self.lot()
        self.click(f'wc:stock:{lot["id"]}:drink')
        token = self.store.get_setting('wine_draft:2')['token']
        self.assertIn('Нет отдельной оценки', self.view()[0])
        self.click('wc:consume:quantity:'+token)
        self.input('2')
        self.click('wc:consume:price:'+token)
        self.input('1250')
        self.assertIn('+500.00 ₽', self.view()[0])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        self.confirm('wc:apply:')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 0)
        self.click('wc:purchases:2:0')
        self.assertIn('Куплено: 2; выпито: 2; осталось: 0', self.view()[0])
        self.assertIn(f'wc:lot:2:{lot["id"]}', self.view()[1])
        self.click('wc:consumed:2:0')
        self.assertIn('+500.00 ₽', self.view()[0])
        closure = next(data for data in self.view()[1] if data.startswith('wc:consumption:'))
        cellar.record_price(self.store, 2, lot['id'], 2000)
        self.click(closure)
        self.assertIn('1 250.00 ₽', self.view()[0])
        self.assertIn('+500.00 ₽', self.view()[0])
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))

    def test_drink_date_reselects_only_eligible_quote_and_unknown_result_is_clear(self):
        lot = self.lot()
        cellar.record_price(self.store, 2, lot['id'], 1500, price_date='2024-02-01')
        self.click(f'wc:stock:{lot["id"]}:drink')
        token = self.store.get_setting('wine_draft:2')['token']
        self.assertIn('Оценка старше даты списания', self.view()[0])
        self.click('wc:consume:date:'+token)
        self.input('01.01.2024')
        self.assertIn('результат не рассчитан', str(self.view()[1]))
        self.confirm('wc:apply:')
        self.assertIsNone(cellar.consumptions(self.store, 2)[0]['result'])

    def test_drink_cancel_does_not_change_stock_and_changed_price_does_not_change_preview(self):
        lot = self.lot()
        cellar.record_price(self.store, 2, lot['id'], 1200)
        self.click(f'wc:stock:{lot["id"]}:drink')
        self.click(f'wc:lot:2:{lot["id"]}')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        self.click(f'wc:stock:{lot["id"]}:drink')
        cellar.record_price(self.store, 2, lot['id'], 1800)
        self.confirm('wc:apply:')
        self.assertEqual(cellar.consumptions(self.store, 2)[0]['valuation_price'], 1200)

    def test_purchases_and_consumptions_pagination_do_not_expose_other_users(self):
        for index in range(9):
            lot = self.lot(name='Wine '+str(index))
            cellar.consume_bottles(self.store, 2, lot['id'], price=1500, operation_id='drink-'+str(index))
        self.click('wc:purchases:2:0')
        self.assertIn('wc:purchases:2:1', self.view()[1])
        self.click('wc:consumed:2:0')
        self.assertIn('wc:consumed:2:1', self.view()[1])
        self.assertLess(len(self.view()[0]), 4096)
        self.assertTrue(all(len(data.encode('utf-8')) <= 64 for data in self.view()[1]))
        for route in ('wc:purchases:2:0', 'wc:consumed:2:0', 'wc:consumption:2:1'):
            with self.assertRaises(PermissionError):
                self.click(route, actor=3)
        self.click('wc:consumed:2:0', actor=1)
        self.assertIn('Выпито', self.view()[0])

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('wine-ui-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        for actor, role in [(1, 'admin'), (2, 'subscriber'), (3, 'subscriber')]:
            self.store.add_user(actor, role, subscriptions='all')
        self.reply = Mock()
        def pending(actor, value):
            self.store.execute('INSERT INTO bot_sessions(user_id,last_seen,pending) VALUES(?,?,?) '
                               'ON CONFLICT(user_id) DO UPDATE SET pending=excluded.pending', (actor, utcnow(), value))
        self.bot = SimpleNamespace(store=self.store, _reply=self.reply, market_controls=SimpleNamespace(pending=pending))
        self.controls = WineControls(self.bot)
        self.client = Mock()
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No external requests')))

    def view(self):
        call = self.reply.call_args
        return call.args[2], {button.callback_data: button.text for row in call.args[3].inline_keyboard for button in row}

    def click(self, data, actor=2):
        self.controls.callback(self.client, actor, data)

    def input(self, text, actor=2):
        self.assertTrue(self.controls.input(self.client, actor, text))

    def lot(self, actor=2, **fields):
        return cellar.add_bottle(self.store, actor, {'name': 'Merlot', 'winery': 'Example', 'vintage': 2021,
                                'quantity': 2, 'purchase_price': 1000, 'purchase_date': '2020-01-01', **fields})

    def form(self):
        self.click('wc:add')
        self.click('wc:manual')
        self.input('Merlot')
        self.input('Example')
        self.input('2021')
        self.input('2')
        self.input('750')
        self.input('1250,50')
        self.input('01.01.2020')
        self.input('Шкаф')

    def confirm(self, prefix='wc:save:'):
        data = next(data for data in self.view()[1] if data.startswith(prefix))
        self.click(data)
        return data

    def test_project_home_separates_cellar_from_vivino(self):
        self.controls.home(self.client, 2)
        text, buttons = self.view()
        self.assertTrue(text.startswith('Вино'))
        self.assertIn('wc:menu', buttons)
        self.assertIn('vi:menu', buttons)
        self.assertNotIn('wc:shops', buttons)
        self.assertNotIn('wc:marketshops', buttons)

    def test_manual_add_requires_final_review_and_is_not_repeated(self):
        self.form()
        self.assertEqual(cellar.lots(self.store, 2), [])
        text, _ = self.view()
        self.assertIn('Проверьте добавление', text)
        self.assertIn('1 250.50', text)
        confirmation = self.confirm()
        lot = cellar.lots(self.store, 2)[0]
        self.assertEqual(lot['quantity'], 2)
        self.assertEqual(lot['purchase_price'], 1250.5)
        with self.assertRaises(ValueError):
            self.click(confirmation)
        self.assertEqual(len(cellar.lots(self.store, 2)), 1)

    def test_optional_unknown_values_do_not_become_zero_or_fake_purchase_date(self):
        self.click('wc:add')
        self.click('wc:manual')
        self.input('Вино без сведений')
        self.click('wc:skip')
        self.click('wc:skip')
        self.input('1')
        self.click('wc:volume:750')
        for _ in range(3):
            self.click('wc:skip')
        self.confirm()
        lot = cellar.lots(self.store, 2)[0]
        self.assertIsNone(lot['purchase_price'])
        self.assertFalse(lot['purchase_date'])
        self.assertIsNone(cellar.portfolio(self.store, 2)['current_value'])

    def test_editing_a_review_field_returns_directly_to_review(self):
        self.form()
        self.click('wc:edit:quantity')
        self.input('3')
        self.assertIn('Проверьте добавление', self.view()[0])
        self.assertIn('Бутылок: 3', self.view()[0])
        self.confirm()
        self.assertEqual(cellar.lots(self.store, 2)[0]['quantity'], 3)

    def test_invalid_numbers_and_dates_preserve_form_and_stock(self):
        self.click('wc:add')
        self.click('wc:manual')
        self.input('Wine')
        self.click('wc:skip')
        self.click('wc:skip')
        for bad in ('0', '-1', '2.5', '10001', 'NaN'):
            with self.assertRaises(ValueError):
                self.controls.input(self.client, 2, bad)
        self.input('1')
        self.input('750')
        for bad in ('-100', 'NaN', 'inf'):
            with self.assertRaises(ValueError):
                self.controls.input(self.client, 2, bad)
        self.click('wc:skip')
        with self.assertRaises(ValueError):
            self.controls.input(self.client, 2, '31.02.2020')
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_cancel_and_previous_do_not_add_bottle(self):
        self.click('wc:add')
        self.click('wc:manual')
        self.input('Wine')
        self.click('wc:previous')
        self.assertIn('Как называется', self.view()[0])
        self.input('/cancel')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_drink_and_add_quantity_have_review_confirmation(self):
        lot = self.lot()
        self.click(f'wc:stock:{lot["id"]}:drink')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        confirmation = self.confirm('wc:apply:')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        with self.assertRaises(ValueError):
            self.click(confirmation)
        self.click(f'wc:stock:{lot["id"]}:add')
        self.input('4')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        self.confirm('wc:apply:')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 5)

    def test_returning_to_card_cancels_input(self):
        lot = self.lot()
        self.click(f'wc:stock:{lot["id"]}:add')
        self.click(f'wc:lot:2:{lot["id"]}')
        self.assertFalse(self.controls.input(self.client, 2, '5'))
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)

    def test_manual_quote_has_date_and_does_not_replace_purchase_cost(self):
        lot = self.lot()
        self.click(f'wc:price:{lot["id"]}')
        self.input('1500 01.01.2021')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['latest_price']['amount'], 1000)
        self.confirm('wc:apply:')
        result = cellar.get_lot(self.store, 2, lot['id'])
        self.assertEqual(result['purchase_price'], 1000)
        self.assertEqual(result['latest_price']['amount'], 1500)
        self.assertEqual(result['latest_price']['price_date'], '2021-01-01')

    def test_photo_candidates_need_owner_specific_review_and_never_autocreate(self):
        self.click('wc:add')
        self.click('wc:photo')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'wine:photo')
        self.store.set_setting('wine_recognition:2:testtoken', {'candidate': {'name': 'Recognized Wine', 'winery': 'Example', 'vintage': 2021},
                                                              'warnings': ['Проверьте год']})
        with self.assertRaises(ValueError):
            self.click('wc:recognized:testtoken', actor=3)
        self.click('wc:recognized:testtoken')
        self.assertIn('Проверьте год', self.view()[0])
        self.assertEqual(cellar.lots(self.store, 2), [])
        self.confirm()
        self.assertEqual(cellar.lots(self.store, 2)[0]['name'], 'Recognized Wine')

    def test_revoked_feature_cannot_confirm_existing_draft(self):
        self.form()
        data = next(data for data in self.view()[1] if data.startswith('wc:save:'))
        from thesaurus.feature_access import change
        change(self.store, 1, 2, 'wine_cellar', False)
        with self.assertRaises(PermissionError):
            self.click(data)
        self.assertEqual(self.store.query('SELECT * FROM wine_lots'), [])

    def test_admin_readonly_other_cellar_and_subscriber_owner_tampering(self):
        lot = self.lot()
        with self.assertRaises(PermissionError):
            self.click(f'wc:lot:2:{lot["id"]}', actor=3)
        self.click(f'wc:lot:2:{lot["id"]}', actor=1)
        text, buttons = self.view()
        self.assertIn('Просмотр администратора', text)
        self.assertFalse(any(data.startswith(('wc:stock:', 'wc:price:', 'wc:refresh:')) for data in buttons))
        with self.assertRaises((ValueError, PermissionError)):
            self.click(f'wc:stock:{lot["id"]}:drink', actor=1)
        self.click('wc:users:0', actor=1)
        self.assertIn('wc:list:2:0', self.view()[1])
        with self.assertRaises(PermissionError):
            self.click('wc:users:0', actor=2)

    def test_queued_price_refresh_and_export_are_deduplicated_and_offline(self):
        lot = self.lot()
        self.click(f'wc:refresh:{lot["id"]}')
        self.click(f'wc:refresh:{lot["id"]}')
        self.click('wc:export:2:xlsx')
        jobs = self.store.query('SELECT kind,payload FROM jobs ORDER BY id')
        self.assertEqual([job['kind'] for job in jobs], ['wine_price_refresh', 'wine_cellar_export'])
        self.assertEqual(json.loads(jobs[0]['payload'])['lot_id'], lot['id'])
        self.assertEqual(json.loads(jobs[1]['payload'])['format'], 'xlsx')
        with self.assertRaises(ValueError):
            self.click('wc:export:2:executable')
        with self.assertRaises(PermissionError):
            self.click('wc:export:3:csv')

    def test_paginated_holdings_and_empty_valuation_are_understood(self):
        for index in range(10):
            self.lot(name=f'Wine {index}')
        self.click('wc:menu')
        self.assertFalse(any(data.startswith('wc:lot:') for data in self.view()[1]))
        self.click('wc:list:2:0')
        text, buttons = self.view()
        self.assertIn('Страница 1/2', text)
        self.assertEqual(len([value for value in buttons if value.startswith('wc:lot:')]), 8)
        self.click('wc:list:2:1')
        self.assertIn('wc:list:2:0', self.view()[1])
        self.assertTrue(all(len(data.encode('utf-8')) <= 64 for data in self.view()[1]))
        self.controls.valuation(self.client, 3)
        self.assertIn('Истории запасов и стоимости пока нет', self.view()[0])
        self.assertNotIn('wc:export:3:chart', self.view()[1])

    def test_new_review_proposal_only_consumes_after_specific_confirmation(self):
        lot = self.lot()
        cellar.sync_reviews(self.store, 2, [], 'baseline')
        proposals = cellar.sync_reviews(self.store, 2, [{'name': 'Merlot', 'winery': 'Example', 'vintage': '2021', 'rating': 4}], 'next')
        identifier = proposals[0]['id']
        self.click(f'wc:proposal:{identifier}')
        self.click(f'wc:proposalchoose:{identifier}:{lot["id"]}')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        self.confirm('wc:apply:')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        self.assertEqual(cellar.proposals(self.store, 2), [])

    def test_shops_require_admin_and_validate_hosts(self):
        with self.assertRaises(PermissionError):
            self.click('wc:shops')
        self.click('wc:shops', actor=1)
        with self.assertRaises(ValueError):
            self.controls.input(self.client, 1, 'https://shop.example/path,127.0.0.1')
        self.input('simplewine.ru, winelab.ru', actor=1)
        self.assertEqual(self.store.get_setting('wine_price_allowed_hosts'), ['simplewine.ru', 'winelab.ru'])

    def test_shop_reset_uses_server_configuration_without_promising_supported_stores(self):
        self.click('wc:shops', actor=1)
        self.assertIn('WINE_PRICE_ALLOWED_HOSTS', self.view()[0])
        self.assertIn('он может быть пустым', self.view()[0])
        self.assertNotIn('стандартный список', self.view()[0])
        self.store.set_setting('wine_price_allowed_hosts', ['example.com'])
        self.click('wc:shops', actor=1)
        self.input('сброс', actor=1)
        self.assertIsNone(self.store.get_setting('wine_price_allowed_hosts'))
        self.store.set_setting('wine_price_allowed_hosts', [])
        self.click('wc:shops', actor=1)
        self.assertIn('магазины не разрешены', self.view()[0])

    def test_photo_volume_is_reviewed_and_same_result_cannot_be_added_twice(self):
        self.store.set_setting('wine_recognition:2:half-bottle', {'candidate': {'name': 'Small Bottle', 'volume_ml': 375}})
        self.click('wc:recognized:half-bottle')
        self.assertIn('375 мл', self.view()[0])
        self.click('wc:edit:volume_ml')
        with self.assertRaises(ValueError):
            self.controls.input(self.client, 2, '0')
        self.input('1500')
        self.assertIn('Проверьте добавление', self.view()[0])
        self.confirm()
        self.assertEqual(cellar.lots(self.store, 2)[0]['volume_ml'], 1500)
        with self.assertRaises(ValueError):
            self.click('wc:recognized:half-bottle')
        self.assertEqual(len(cellar.lots(self.store, 2)), 1)

    def test_unknown_purchase_date_is_labelled_in_card_without_claiming_new_shop_price(self):
        lot = self.lot(purchase_date=None)
        self.click(f'wc:lot:2:{lot["id"]}')
        self.assertIn('дата покупки неизвестна', self.view()[0])
        self.assertIn('Это не новая цена магазина', self.view()[0])

    def test_declining_a_vivino_proposal_preserves_stock(self):
        lot = self.lot()
        cellar.sync_reviews(self.store, 2, [], 'baseline')
        proposals = cellar.sync_reviews(self.store, 2, [{'name': 'Merlot', 'winery': 'Example', 'vintage': 2021, 'review': 'В ресторане'}], 'next')
        self.click(f'wc:proposalignore:{proposals[0]["id"]}')
        self.assertEqual(cellar.proposals(self.store, 2), [])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)

    def test_pending_admin_configuration_rechecks_demotion(self):
        self.click('wc:shops', actor=1)
        self.store.execute("UPDATE users SET role='subscriber' WHERE user_id=1")
        with self.assertRaises(PermissionError):
            self.controls.input(self.client, 1, 'example.com')
        self.assertIsNone(self.store.get_setting('wine_price_allowed_hosts'))

    def test_leaving_review_invalidates_its_old_confirmation(self):
        self.form()
        data = next(data for data in self.view()[1] if data.startswith('wc:save:'))
        self.click('wc:menu')
        with self.assertRaises(ValueError):
            self.click(data)
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_shop_presets_show_selection_separately_from_adapter_health(self):
        catalog = [{'id': 'store', 'title': 'Проверенный магазин', 'enabled': False, 'automatic': True,
                    'status_label': 'поиск доступен', 'reason': 'Москва, текущий каталог',
                    'last_check': {'status': 'unavailable', 'reason': 'Защитная страница', 'checked_at': '2026-10-11T10:00:00+00:00'}},
                   {'id': 'manual', 'title': 'Магазин с защитой', 'enabled': True, 'automatic': False,
                    'status_label': 'автопоиск заблокирован', 'reason': 'Требуется ручная проверка'}]
        with patch('thesaurus.wine_retail.shops', return_value=catalog), patch('thesaurus.wine_retail.configure_shop') as configure:
            self.click('wc:marketshops', actor=1)
            text, buttons = self.view()
            self.assertIn('○ Проверенный магазин', buttons['wc:shop:store'])
            self.assertIn('✓ Магазин с защитой', buttons['wc:shop:manual'])
            self.assertIn('автопоиск заблокирован', text)
            self.assertIn('Защитная страница', text)
            self.assertIn('Последняя проверка', text)
            self.assertIn('wc:shops', buttons)
            self.click('wc:shop:store', actor=1)
            configure.assert_called_once_with(self.store, 1, 'store', True)
            with self.assertRaises(PermissionError):
                self.click('wc:marketshops', actor=2)
            with self.assertRaises(PermissionError):
                self.click('wc:shop:store', actor=2)

    def test_moscow_retail_card_shows_blocked_and_disabled_sources_without_fake_search(self):
        lot = self.lot()
        catalog = [{'id': 'blocked', 'title': 'WineLab', 'enabled': True, 'automatic': False,
                    'status_label': 'ручной режим', 'reason': 'Источник блокирует автоматические запросы'},
                   {'id': 'off', 'title': 'Проверенный', 'enabled': False, 'automatic': True, 'status_label': 'поиск доступен'}]
        with patch('thesaurus.wine_retail.shops', return_value=catalog):
            self.click(f'wc:retail:{lot["id"]}')
            text, buttons = self.view()
            self.assertIn('Цены в Москве', text)
            self.assertIn('блокирует', text)
            self.assertNotIn(f'wc:search:{lot["id"]}', buttons)
            self.assertIn(f'wc:link:{lot["id"]}', buttons)
            with self.assertRaises(ValueError):
                self.click(f'wc:search:{lot["id"]}')
            self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_moscow_search_is_queued_once_and_rechecks_lot_owner(self):
        lot = self.lot()
        with patch('thesaurus.wine_retail.shops', return_value=[{'id': 'store', 'enabled': True, 'automatic': True}]):
            self.click(f'wc:search:{lot["id"]}')
            self.click(f'wc:search:{lot["id"]}')
            with self.assertRaises((PermissionError, ValueError)):
                self.click(f'wc:search:{lot["id"]}', actor=3)
        jobs = self.store.query('SELECT kind,payload FROM jobs')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['kind'], 'wine_price_search')
        self.assertEqual(json.loads(jobs[0]['payload'])['lot_id'], lot['id'])

    def test_price_adapter_without_search_does_not_offer_or_queue_automatic_discovery(self):
        lot = self.lot()
        catalog = [{'id': 'amwine', 'title': 'Ароматный мир', 'enabled': True, 'automatic': True,
                    'search_supported': False, 'price_supported': True, 'reason': 'Цена по прямой ссылке'}]
        with patch('thesaurus.wine_retail.shops', return_value=catalog):
            self.click('wc:marketshops', actor=1)
            self.assertIn('цены по ссылке', self.view()[1]['wc:shop:amwine'])
            self.click(f'wc:retail:{lot["id"]}')
            self.assertNotIn(f'wc:search:{lot["id"]}', self.view()[1])
            self.assertIn(f'wc:link:{lot["id"]}', self.view()[1])
            with self.assertRaises(ValueError):
                self.click(f'wc:search:{lot["id"]}')
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_found_offers_show_conditions_and_need_match_before_confirmation(self):
        lot = self.lot()
        token = uuid4().hex
        quote = {'shop_title': 'Магазин', 'title': 'Merlot Example 2021 0.75 л', 'winery': 'Example',
                 'vintage': 2021, 'volume_ml': 750, 'amount': 1500, 'city': 'Москва',
                 'price_kind': 'обычная публичная цена', 'availability_scope': 'наличие в конкретном магазине не подтверждено',
                 'checked_at': '2026-10-11T12:00:00+00:00', 'url': 'https://example.com/wine', 'match': True}
        result = {'lot_id': lot['id'], 'candidates': [quote, {**quote, 'vintage': 2022, 'match': False,
                  'match_reasons': ['Не совпадает год урожая']}], 'sources': [{'title': 'WineLab', 'status': 'blocked', 'reason': 'Защита сайта'}]}
        with patch('thesaurus.wine_retail.candidates', return_value=result), patch('thesaurus.wine_retail.confirm_candidate') as confirm:
            self.click(f'wc:offers:{token}')
            self.assertIn('Защита сайта', self.view()[0])
            self.click(f'wc:offer:{token}:1')
            text, buttons = self.view()
            self.assertIn('Не совпадает год', text)
            self.assertFalse(any(value.startswith('wc:offerconfirm:') for value in buttons))
            with self.assertRaises(ValueError):
                self.click(f'wc:offerconfirm:{token}:1')
            confirm.assert_not_called()
            self.click(f'wc:offer:{token}:0')
            text, buttons = self.view()
            self.assertIn('обычная публичная цена', text)
            self.assertIn('конкретном магазине не подтверждено', text)
            self.assertIn('Москва', text)
            self.assertIn(f'wc:offerconfirm:{token}:0', buttons)
            self.assertTrue(all(len(value.encode('utf-8')) <= 64 for value in buttons))
            confirm.assert_not_called()
            self.click(f'wc:offerconfirm:{token}:0')
            confirm.assert_called_once_with(self.store, 2, token, 0)

    def test_empty_search_results_keep_saved_price_and_explain_source_errors(self):
        lot = self.lot()
        token = uuid4().hex
        with patch('thesaurus.wine_retail.candidates', return_value={'lot_id': lot['id'], 'candidates': [],
                   'sources': [{'title': 'Магазин', 'status': 'error', 'reason': 'Ответ недоступен'}]}):
            self.click(f'wc:offers:{token}')
            self.assertIn('Ответ недоступен', self.view()[0])
            self.assertIn('Сохранённые цены не изменены', self.view()[0])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['latest_price']['amount'], 1000)

    def _saved_exact_offer(self, lot):
        from thesaurus.wine_retail import configure_shop
        configure_shop(self.store, 1, 'decanter', True)
        token = uuid4().hex
        quote = {'shop_id': 'decanter', 'shop_title': 'Декантер', 'title': 'Merlot Example 2021 750 мл',
                 'winery': 'Example', 'vintage': 2021, 'volume_ml': 750, 'amount': 1500, 'currency': 'RUB',
                 'city': 'Москва', 'price_kind': 'public', 'availability': 'InStock', 'status': 'ok',
                 'availability_scope': 'магазин в Москве', 'as_of': utcnow(),
                 'url': 'https://decanter.ru/wine/example-2021', 'match': True}
        self.store.set_setting(f'wine_retail_search:2:{token}', {'actor_id': 2, 'lot_id': lot['id'],
            'created_at': utcnow(), 'expires': (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            'candidates': [quote], 'sources': []})
        return token

    def test_real_offer_confirmation_saves_one_link_and_one_quote_without_network(self):
        lot = self.lot()
        token = self._saved_exact_offer(lot)
        self.click(f'wc:offer:{token}:0')
        self.assertIn('обычная публичная цена', self.view()[0])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['latest_price']['amount'], 1000)
        self.click(f'wc:offerconfirm:{token}:0')
        self.click(f'wc:offerconfirm:{token}:0')
        result = cellar.get_lot(self.store, 2, lot['id'])
        self.assertEqual(result['latest_price']['amount'], 1500)
        self.assertEqual(result['purchase_price'], 1000)
        self.assertEqual(result['quantity'], 2)
        self.assertEqual(len(cellar.price_links(self.store, 2, lot['id'])), 1)
        self.assertEqual(len(self.store.query("SELECT * FROM wine_prices WHERE source='shop_confirmed'")), 1)

    def test_real_offer_confirmation_rechecks_disabled_source_and_result_owner(self):
        from thesaurus.wine_retail import configure_shop
        lot = self.lot()
        token = self._saved_exact_offer(lot)
        with self.assertRaises((PermissionError, ValueError)):
            self.click(f'wc:offer:{token}:0', actor=3)
        self.click(f'wc:offer:{token}:0')
        configure_shop(self.store, 1, 'decanter', False)
        with self.assertRaises(ValueError):
            self.click(f'wc:offerconfirm:{token}:0')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['latest_price']['amount'], 1000)
        self.assertEqual(cellar.price_links(self.store, 2, lot['id']), [])

    def test_store_availability_scopes_and_addresses_are_readable_in_offer(self):
        lot = self.lot()
        token = uuid4().hex
        quotes = [
            {'title': 'Merlot', 'amount': 1500, 'match': True, 'availability_scope': 'moscow_branches',
             'available_shops': ['Москва, ул. Примерная, 1', 'Москва, ул. Винная, 2']},
            {'title': 'Merlot', 'amount': 1600, 'match': True, 'availability_scope': 'selected_moscow_shop',
             'shop_address': 'Москва, проспект Тестовый, 3', 'available_shops': ['Москва, проспект Тестовый, 3']},
        ]
        with patch('thesaurus.wine_retail.candidates', return_value={'lot_id': lot['id'], 'candidates': quotes}):
            self.click(f'wc:offer:{token}:0')
            self.assertIn('в магазинах Москвы', self.view()[0])
            self.assertIn('Москва, ул. Винная, 2', self.view()[0])
            self.assertNotIn('moscow_branches', self.view()[0])
            self.click(f'wc:offer:{token}:1')
            text = self.view()[0]
            self.assertIn('в выбранном магазине Москвы', text)
            self.assertEqual(text.count('Москва, проспект Тестовый, 3'), 1)
            self.assertNotIn('selected_moscow_shop', text)
        self.assertIn('адрес конкретного магазина не подтверждён', availability_label({'availability': 'InStock'}))
        self.assertEqual(availability_addresses({'shop_address': 'Адрес', 'available_shops': ['Адрес']}), ['Адрес'])

    def test_saved_quote_card_translates_its_scope_without_claiming_all_shops(self):
        lot = self.lot()
        cellar.record_price(self.store, 2, lot['id'], 1700, price_date='2021-01-01', source='shop_confirmed',
                            metadata={'shop_title': 'Магазин', 'availability_scope': 'selected_moscow_shop'})
        self.click(f'wc:lot:2:{lot["id"]}')
        text = self.view()[0]
        self.assertIn('в выбранном магазине Москвы', text)
        self.assertNotIn('selected_moscow_shop', text)


class WineBotIntegrationTests(unittest.TestCase):
    def test_real_callback_flow_consumes_with_frozen_value_and_keeps_all_purchases_after_restart(self):
        lot = cellar.add_bottle(self.store, 2, {'name': 'Merlot', 'quantity': 2, 'purchase_price': 1000})
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.click(f'wc:lot:2:{lot["id"]}')
        self.click(f'wc:stock:{lot["id"]}:drink')
        token = self.store.get_setting('wine_draft:2')['token']
        self.click('wc:consume:price:'+token)
        self.assertEqual(sum(label == 'Отмена' for label in self.view()[1].values()), 1)
        self.message('1250')
        self.assertIn('+250.00 ₽', self.view()[0])
        self.click('wc:apply:'+token)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        self.message('/cellar')
        self.click('wc:consumed:2:0')
        self.assertIn('+250.00 ₽', self.view()[0])
        self.store = Store(self.root/'state.sqlite3')
        from thesaurus.telegram_bot import BotControls
        self.bot = BotControls(self.store)
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.click('wc:purchases:2:0')
        self.assertIn('Куплено: 2; выпито: 1; осталось: 1', self.view()[0])
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))

    def test_cancel_and_return_from_consumption_input_are_safe_and_revocation_blocks_confirmation(self):
        from thesaurus.feature_access import change
        lot = cellar.add_bottle(self.store, 2, {'name': 'Merlot', 'quantity': 2, 'purchase_price': 1000})
        self.message('/cellar')
        self.click('wc:list:2:0')
        self.click(f'wc:lot:2:{lot["id"]}')
        self.click(f'wc:stock:{lot["id"]}:drink')
        token = self.store.get_setting('wine_draft:2')['token']
        self.click('wc:consume:price:'+token)
        self.click('wc:consume:review:'+token)
        self.assertIn('Списание выпитого вина', self.view()[0])
        self.click('wc:consume:price:'+token)
        self.message('/cancel')
        self.assertIn('В запасе: 2', self.view()[0])
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.click(f'wc:stock:{lot["id"]}:drink')
        token = self.store.get_setting('wine_draft:2')['token']
        change(self.store, 1, 2, 'wine_cellar', False)
        event = self.click('wc:apply:'+token)
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertEqual(self.store.one('SELECT quantity FROM wine_lots WHERE id=?', (lot['id'],))['quantity'], 2)
        self.assertFalse(self.store.query('SELECT * FROM wine_consumptions'))

    def setUp(self):
        from thesaurus.telegram_bot import BotControls
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('wine-bot-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, 'subscriber', subscriptions='all')
        self.bot = BotControls(self.store)
        self.client = Mock()
        sequence = count(100)
        self.client.send_message.side_effect = lambda *args, **kwargs: SimpleNamespace(id=next(sequence))
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No external requests')))

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=1,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def click(self, data, actor=2, private=True, identifier=None):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=identifier or record.get('message_id', 100),
                                    chat=SimpleNamespace(id=actor, type='private' if private else 'group')), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self):
        rpc = next(call for call in reversed(self.client.mock_calls) if call[0] in {'send_message', 'edit_message_text'})
        text = rpc.args[2] if rpc[0] == 'edit_message_text' else rpc.args[1]
        markup = rpc.kwargs.get('reply_markup')
        return text, {button.callback_data: button.text for row in markup.inline_keyboard for button in row} if markup else {}

    def test_home_to_wines_to_cellar_and_vivino_have_consistent_back(self):
        self.message('/projects')
        self.assertEqual(self.view()[1]['pj:life'], 'Жизнь')
        self.click('pj:life')
        self.assertEqual(self.view()[1]['pj:vivino'], 'Вино')
        self.click('pj:vivino')
        self.assertIn('wc:menu', self.view()[1])
        self.assertIn('vi:menu', self.view()[1])
        self.click('wc:menu')
        self.assertEqual(self.view()[1]['pj:vivino'], '← Назад')
        self.click('pj:vivino')
        self.click('vi:menu')
        self.assertIn('географический HTML-атлас', self.view()[0])
        self.assertEqual(self.view()[1]['pj:vivino'], '← Назад')

    def test_actual_form_routes_messages_into_review_and_inventory(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:manual')
        for value in ('Merlot', 'Example', '2021', '2', '750', '1000', '01.01.2020', 'Шкаф'):
            self.assertIn('пример', self.view()[0].casefold())
            self.message(value)
        text, buttons = self.view()
        self.assertIn('Проверьте добавление', text)
        self.assertEqual(cellar.lots(self.store, 2), [])
        self.click(next(data for data in buttons if data.startswith('wc:save:')))
        self.assertIn('В запасе: 2', self.view()[0])
        self.assertEqual(len(cellar.lots(self.store, 2)), 1)
        lot = cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['winery'], lot['vintage'], lot['quantity'], lot['volume_ml'],
                          lot['purchase_price'], lot['purchase_date'], lot['location']),
                         ('Merlot', 'Example', '2021', 2, 750, 1000, '2020-01-01', 'Шкаф'))

    def _review(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:manual')
        for value in ('Merlot', 'Example', '2021', '2', '750', '1000', '01.01.2020', 'Шкаф'):
            self.message(value)
        return next(data for data in self.view()[1] if data.startswith('wc:save:'))

    def test_cancel_during_review_clears_draft_and_old_message_cannot_save(self):
        confirmation = self._review()
        self.assertIsNotNone(self.store.get_setting('wine_draft:2'))
        self.message('/cancel')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.click(confirmation, identifier=9000)
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_project_navigation_clears_review_and_old_confirmation(self):
        confirmation = self._review()
        self.click('menu:home')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.click('pj:market')
        self.assertIn('Рынок и экономика', self.view()[0])
        self.click(confirmation, identifier=9001)
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_slash_vivino_clears_unconfirmed_cellar_draft(self):
        self._review()
        self.message('/vivino')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.assertIn('географический HTML-атлас', self.view()[0])

    def test_cellar_callbacks_still_require_personal_chat(self):
        self.message('/wines')
        query = self.click('wc:menu', private=False)
        query.answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.assertEqual(self.store.query('SELECT * FROM wine_lots'), [])

    def test_feature_only_cellar_does_not_show_vivino_upload(self):
        from thesaurus.feature_access import key
        self.store.set_setting(key(2), ['wine_cellar'])
        self.message('/wines')
        self.assertIn('wc:menu', self.view()[1])
        self.assertNotIn('vi:menu', self.view()[1])
        self.click('wc:menu')
        self.assertIn('wc:add', self.view()[1])

    def test_cellar_has_one_add_action_with_methods_inside_the_chooser(self):
        self.message('/cellar')
        buttons = self.view()[1]
        self.assertEqual(sum(label == 'Добавить вино' for label in buttons.values()), 1)
        self.assertIn('wc:add', buttons)
        self.assertFalse({'wc:manual', 'wc:photo', 'lf:receipt'} & buttons.keys())
        self.click('wc:add')
        self.assertTrue({'wc:manual', 'wc:photo', 'lf:receipt', 'wc:example'} <= self.view()[1].keys())
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_manual_examples_are_readable_before_starting_one_field_at_a_time(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:example')
        text, buttons = self.view()
        for number in range(1, 9):
            self.assertIn(f'{number}.', text)
        self.assertIn('wc:manual', buttons)
        self.assertEqual(buttons['wc:add'], '← Назад')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.click('wc:manual')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'wine:add:name')
        self.assertIn('1/8', self.view()[0])
        self.assertIn('пример', self.view()[0].casefold())
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_method_back_clears_pending_and_returns_to_add_chooser(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:manual')
        first_token = self.store.get_setting('wine_draft:2')['token']
        self.assertEqual(self.view()[1]['wc:add'], '← Назад')
        self.click('wc:add')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.click('wc:photo')
        self.assertEqual(self.view()[1]['wc:add'], '← Назад')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'wine:photo')
        self.click('wc:add')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.click('wc:manual')
        self.assertNotEqual(self.store.get_setting('wine_draft:2')['token'], first_token)
        self.click('wc:cancel')
        self.assertIsNone(self.store.get_setting('wine_draft:2'))
        self.assertIn('wc:add', self.view()[1])
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_receipt_method_requires_finance_permission_and_does_not_self_grant(self):
        from thesaurus.feature_access import key
        self.store.set_setting(key(2), ['wine_cellar'])
        self.message('/cellar')
        self.click('wc:add')
        self.assertTrue({'wc:manual', 'wc:photo'} <= self.view()[1].keys())
        self.assertNotIn('lf:receipt', self.view()[1])
        query = self.click('lf:receipt')
        self.assertTrue(query.answer.call_args.kwargs.get('show_alert'))
        self.assertEqual(self.store.get_setting(key(2)), ['wine_cellar'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.store.set_setting(key(2), ['wine_cellar', 'life_finance'])
        self.message('/cellar')
        self.click('wc:add')
        self.assertIn('lf:receipt', self.view()[1])
        self.click('lf:receipt')
        self.assertIn('Добавление по чеку', self.view()[0])
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'life:receipt')
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_receipt_from_wine_back_returns_to_add_chooser_without_creating_data(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('lf:receipt')
        self.assertEqual(self.view()[1]['wc:add'], '← Назад')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'wc:add')
        self.click('wc:add').answer.assert_called_once_with()
        self.assertTrue(self.view()[0].startswith('Добавить вино'))
        self.assertTrue({'wc:manual', 'wc:photo', 'lf:receipt'} <= self.view()[1].keys())
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertEqual(cellar.lots(self.store, 2), [])

    def test_receipt_cancel_after_wine_revocation_falls_back_to_personal_finance(self):
        from thesaurus.feature_access import change, key
        self.store.set_setting(key(2), ['wine_cellar', 'life_finance'])
        self.message('/cellar')
        self.click('wc:add')
        self.click('lf:receipt')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'wc:add')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'life:receipt')
        change(self.store, 1, 2, 'wine_cellar', False)
        self.click('ui:cancel').answer.assert_called_once_with()
        self.assertTrue(self.view()[0].startswith('Доходы и расходы'))
        self.assertEqual(self.store.get_setting('ui_screen:2')['route'], 'lf:home')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertEqual(self.store.get_setting(key(2)), ['life_finance'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])

    def test_receipt_cancel_after_wine_and_finance_revocation_returns_to_projects(self):
        from thesaurus.feature_access import change, key
        self.store.set_setting(key(2), ['wine_cellar', 'life_finance'])
        self.message('/cellar')
        self.click('wc:add')
        self.click('lf:receipt')
        change(self.store, 1, 2, 'wine_cellar', False)
        change(self.store, 1, 2, 'life_finance', False)
        self.click('ui:cancel').answer.assert_called_once_with()
        self.assertIn('Главное меню', self.view()[0])
        self.assertEqual(self.store.get_setting('ui_screen:2')['route'], 'menu:home')
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertFalse({'wc:add', 'lf:home', 'pj:life', 'pj:vivino'} & self.view()[1].keys())
        self.assertEqual(self.store.get_setting(key(2)), [])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])

    def _photo(self, actor=2, size=100, private=True):
        return SimpleNamespace(id=77, from_user=SimpleNamespace(id=actor),
                               chat=SimpleNamespace(id=actor, type='private' if private else 'group'),
                               photo=SimpleNamespace(file_size=size))

    def test_photo_upload_queues_private_owner_but_does_not_add_inventory(self):
        from PIL import Image
        output = BytesIO()
        Image.new('RGB', (32, 48), 'white').save(output, format='JPEG')
        raw = output.getvalue()
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(raw)
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:photo')
        message = self._photo(size=len(raw))
        self.bot.handle_wine_photo(self.client, message)
        self.bot.handle_wine_photo(self.client, message)
        jobs = self.store.query("SELECT payload FROM jobs WHERE kind='wine_recognize'")
        self.assertEqual(len(jobs), 1)
        payload = json.loads(jobs[0]['payload'])
        self.assertEqual(payload['actor_id'], 2)
        self.assertTrue(Path(payload['path']).is_relative_to(self.root / 'wine_photos' / '2'))
        self.assertEqual(Path(payload['path']).read_bytes(), raw)
        self.assertEqual(cellar.lots(self.store, 2), [])
        self.assertIn('Запас пока не изменён', self.view()[0])

    def test_photo_upload_checks_private_chat_and_feature_before_download(self):
        self.bot.handle_wine_photo(self.client, self._photo(private=False))
        self.client.download_media.assert_not_called()
        from thesaurus.feature_access import change
        change(self.store, 1, 2, 'wine_cellar', False)
        self.bot.handle_wine_photo(self.client, self._photo())
        self.client.download_media.assert_not_called()
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_invalid_and_oversize_photo_uploads_do_not_queue_work(self):
        self.bot.handle_wine_photo(self.client, self._photo(size=8 * 1024 * 1024 + 1))
        self.client.download_media.assert_not_called()
        self.client.download_media.return_value = BytesIO(b'not an image')
        self.bot.handle_wine_photo(self.client, self._photo(size=12))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertFalse((self.root / 'wine_photos').exists())


if __name__ == '__main__':
    unittest.main()
