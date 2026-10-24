"""Life BotControls journeys with private synthetic receipts and no Telegram I/O."""
import json
import tempfile
import unittest
from io import BytesIO
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image

from thesaurus import feature_access, receipt_wines, wine_cellar, wine_retail
from thesaurus import life_ledger as ledger
from thesaurus.store import Store
from thesaurus.telegram_bot import BotControls


class TelegramLifeTests(unittest.TestCase):
    def test_receipt_field_picker_offers_found_years_and_saves_only_the_selected_year(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt(volume=750)
        result['items'][0].update(winery='',vintage=None)
        token = self.open_receipt(result)
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        variants = [candidate(id=str(i)*32,published_vintage=year) for i,year in enumerate([2015,2016,2017,2019,2024])]
        with patch('thesaurus.wine_enrichment.lookup',return_value=lookup_response(*variants)):
            receipt_wines.run_lookup(self.store,2,payload)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:edit:{token}:vintage:0')
        text,buttons = self.view()
        self.assertIn('Сейчас: не указано',text)
        self.assertIn('с вашей этикетки',text)
        choices = {label:route for route,label in buttons.items() if route.startswith('rc:pick:')}
        self.assertEqual(set(choices),{'2015','2016','2017','2019','2024'})
        self.assertTrue(all(len(route.encode())<=64 for route in choices.values()))
        before = self.store.get_setting(f'receipt_ui:2:{token}')['overrides']['0'].copy()
        self.click(choices['2024'])
        after = self.store.get_setting(f'receipt_ui:2:{token}')['overrides']['0']
        self.assertEqual(after['vintage'],'2024')
        self.assertEqual({key:value for key,value in after.items() if key!='vintage'},before)
        self.assertIn('Год: не указано → 2024',self.view()[0])
        self.assertFalse(wine_cellar.lots(self.store,2))
        self.click(f'rc:edit:{token}:vintage:0')
        self.assertIn('Сейчас: 2024',self.view()[0])
        self.message('2021')
        self.assertIn('Год: не указано → 2021',self.view()[0])

    def test_enriched_receipt_shows_before_after_and_final_name_in_confirmation(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt(volume=750)
        result['items'][0].update(name='Мерло из чека', winery='', vintage=None)
        token = self.open_receipt(result)
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(candidate(name='Merlot Reserva'))):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click(f'rc:item:{token}:0')
        text, _ = self.view()
        self.assertIn('Название: Мерло из чека → Merlot Reserva', text)
        self.assertIn('Производитель: не указано → Bodega Ejemplo', text)
        self.assertIn('Страна: не указано → Испания', text)
        self.assertIn('Регион: не указано → Риоха', text)
        self.assertIn('Сорта винограда: не указано → Мерло', text)
        self.assertNotIn('Год: не указано → 2020', text)
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.assertEqual(self.view()[1][f'rc:item:{token}:0'], '✓ Merlot Reserva')
        self.click(f'rc:changes:{token}:0')
        self.assertIn('Название: Мерло из чека → Merlot Reserva', self.view()[0])
        self.assertIn('Пока в запасы ничего не добавлено', self.view()[0])
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.click('rc:review:'+token)
        self.click('rc:confirm:'+token)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['name'], 'Merlot Reserva')
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)

    def test_manual_name_changes_survive_restart_and_later_enrichment(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt(volume=750)
        result['items'][0].update(name='Название 1', winery='', vintage=None)
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        for name in ('Название 2', 'Итоговое название'):
            self.click(f'rc:edit:{token}:name:0')
            self.message(name)
        self.store = Store(self.root/'life-ui.sqlite3')
        self.bot = BotControls(self.store)
        self.click(f'rc:item:{token}:0')
        self.assertIn('Название: Название 1 → Итоговое название', self.view()[0])
        self.click(f'rc:enrich:{token}:0')
        payload = json.loads(self.store.query("SELECT payload FROM jobs WHERE kind='life_wine_enrich' ORDER BY id DESC")[0]['payload'])
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(candidate(name='Название из интернета'))):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click(f'rc:item:{token}:0')
        self.assertIn('Название: Название 1 → Итоговое название', self.view()[0])
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.click('rc:confirm:'+token)
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['winery']), ('Итоговое название', 'Bodega Ejemplo'))

    def test_legacy_stock_draft_compares_with_imported_receipt_and_keeps_manual_name(self):
        from tests.test_receipt_wines import candidate, lookup_response
        token = self.open_receipt(self.receipt(volume=750))
        self.click('rc:confirm:'+token)
        new_token = ledger.prepare_receipt_wines(self.store, 2, token)
        self.click('rc:review:'+new_token, identifier=99999)
        result = ledger.get_receipt_draft(self.store, 2, new_token)['result']
        result['items'][0]['name'] = 'Название уже исправлено'
        ledger.save_receipt_draft(self.store, 2, result, new_token)
        ui = self.store.get_setting(f'receipt_ui:2:{new_token}')
        ui.pop('original_items', None)
        self.store.set_setting(f'receipt_ui:2:{new_token}', ui)
        self.click(f'rc:item:{new_token}:0')
        self.assertIn('Название: Reserva 2020 → Название уже исправлено', self.view()[0])
        self.click(f'rc:enrich:{new_token}:0')
        payload = json.loads(self.store.query("SELECT payload FROM jobs WHERE kind='life_wine_enrich' ORDER BY id DESC")[0]['payload'])
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(candidate(name='Из источника'))):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click(f'rc:review:{new_token}')
        self.click(f'rc:confirm:{new_token}')
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['name'], 'Название уже исправлено')
        imported = self.store.one('SELECT items FROM life_receipt_imports WHERE user_id=2')
        self.assertEqual(json.loads(imported['items'])[0]['name'], 'Reserva 2020')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)

    def test_changes_page_navigation_and_private_owner_checks(self):
        token = self.open_receipt(self.receipt(volume=750))
        for index, name in ((0, 'Итоговое вино'), (1, 'Итоговый сыр')):
            self.click(f'rc:item:{token}:{index}')
            self.click(f'rc:edit:{token}:name:{index}')
            self.message(name)
            self.click('rc:review:'+token)
        self.click(f'rc:changes:{token}:0')
        self.assertIn('Название: Reserva 2020 → Итоговое вино', self.view()[0])
        self.click(f'rc:changes:{token}:1')
        self.assertIn('Название: Сыр → Итоговый сыр', self.view()[0])
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'rc:review:'+token)
        self.message('/cellar', actor=3)
        event = self.click(f'rc:changes:{token}:0', actor=3, identifier=99999)
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertNotIn('Итоговое вино', self.view(3)[0])
        self.assertFalse(wine_cellar.lots(self.store, 2))

    def test_automatic_metadata_requires_saved_preference_and_shows_destinations(self):
        self.store.set_setting('wine_metadata_auto:2', False)
        token = self.open_receipt()
        self.assertFalse(self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'"))
        self.click(f'rc:item:{token}:0')
        self.assertIn('Сорта винограда: не найдены', self.view()[0])
        self.click(f'rc:autoinfo:{token}:0')
        self.assertIn('https://decanter.ru/', self.view()[0])
        self.assertIn('Суммы, фискальные номера', self.view()[0])
        self.click(f'rc:autoon:{token}:0')
        self.assertTrue(self.store.get_setting('wine_metadata_auto:2'))
        self.assertEqual(len(self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'")), 1)
        self.click(f'rc:autoinfo:{token}:0')
        self.click(f'rc:autooff:{token}:0')
        self.assertFalse(self.store.get_setting('wine_metadata_auto:2'))
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        with patch('thesaurus.wine_enrichment.lookup') as lookup, self.assertRaises(PermissionError):
            receipt_wines.run_lookup(self.store, 2, payload)
        lookup.assert_not_called()

    def test_country_region_grapes_edits_save_into_wine_statistics_and_export(self):
        result = self.receipt(volume=750)
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        for field, value in [('country', 'Argentina'), ('region', 'Mendoza'), ('grapes', 'Malbec, Merlot')]:
            self.click(f'rc:edit:{token}:{field}:0')
            self.message(value)
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.click('rc:confirm:'+token)
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['country'], lot['region'], lot['grapes']), ('Аргентина', 'Мендоса', ['Мальбек', 'Мерло']))
        self.message('/cellar')
        self.click('wc:stats:2:country')
        self.assertIn('Аргентина: 2 / 2 / 0', self.view()[0])
        self.click('wc:stats:2:grapes')
        self.assertIn('Мальбек: 2 / 2 / 0', self.view()[0])
        self.assertIn('Купаж учитывается', self.view()[0])
        self.message('/cellar', actor=3)
        event = self.click('wc:stats:2:country', actor=3, identifier=99999)
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertNotIn('Аргентина: 2', self.view(3)[0])

    def test_cached_item_button_reopens_accounted_receipt_and_allows_explicit_stock_confirmation(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click(f'rc:confirm:{token}')
        self.message('/cellar')
        self.click('wc:exports:2')
        event = self.click(f'rc:toggle:{token}:0')
        event.answer.assert_called_once_with()
        text, buttons = self.view()
        self.assertIn('Это прежняя карточка', text)
        self.assertIn('сохранён только расход', text)
        self.assertIn('rc:cellar:'+token, buttons)
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.click('rc:cellar:'+token)
        confirmation = self.prefix('rc:confirm:')
        self.click(confirmation)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['quantity'], 2)
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)

    def test_cached_confirmation_only_reopens_draft_and_clears_unrelated_pending_input(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click(f'rc:select:{token}')
        self.click(f'rc:edit:{token}:total')
        event = self.click(f'rc:confirm:{token}')
        event.answer.assert_called_once_with()
        self.assertIn('действие со старой кнопки не выполнено', self.view()[0])
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.click(f'rc:confirm:{token}')
        self.assertEqual(len(wine_cellar.lots(self.store, 2)), 1)

    def test_old_confirmed_item_cannot_edit_selection_or_start_lookup(self):
        token = self.open_receipt(self.receipt(volume=750))
        jobs_before = self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'")
        self.click(f'rc:confirm:{token}')
        before = self.store.get_setting(f'receipt_ui:2:{token}')
        for action in ('toggle', 'enrich'):
            self.click(f'rc:{action}:{token}:0', identifier=99999)
            self.assertIn('уже учтён', self.view()[0])
            self.assertIn(f'rc:cellar:{token}', self.view()[1])
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{token}'), before)
        self.assertEqual(self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'"), jobs_before)
        self.assertFalse(wine_cellar.lots(self.store, 2))

    def test_cached_receipt_buttons_still_enforce_owner_expiry_and_feature_access(self):
        token = self.open_receipt()
        self.message('/expenses', actor=4)
        event = self.click(f'rc:confirm:{token}', actor=4)
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertNotIn('Reserva', self.view(4)[0])
        self.message('/cellar')
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01' WHERE token=?", (token,))
        event = self.click(f'rc:toggle:{token}:0')
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        feature_access.change(self.store, 1, 2, 'life_finance', False)
        event = self.click(f'rc:review:{token}')
        self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))
        self.assertFalse(wine_cellar.lots(self.store, 2))

    def test_wine_selection_is_labelled_as_unsaved_and_has_visible_confirmation_step(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click(f'rc:item:{token}:0')
        self.assertEqual(self.view()[1][f'rc:toggle:{token}:0'], 'Выбрать для винотеки')
        self.click(f'rc:toggle:{token}:0')
        text, buttons = self.view()
        self.assertIn('ещё не сохранено', text)
        self.assertEqual(buttons['rc:review:'+token], 'К подтверждению выбранных вин')
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.click('rc:review:'+token)
        self.assertIn('Галочки означают только выбор', self.view()[0])
        self.click('rc:confirm:'+token)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['quantity'], 2)

    def test_empty_cellar_and_exports_explain_saved_receipt_and_offer_direct_recovery(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click('rc:confirm:'+token)
        self.message('/cellar')
        self.assertIn('Расход сохранён', self.view()[0])
        self.assertIn('rc:cellar:'+token, self.view()[1])
        self.click('wc:exports:2')
        self.assertIn('после первого сохранённого добавления', self.view()[0])
        self.assertNotIn('wc:export:2:chart', self.view()[1])
        self.assertIn('rc:cellar:'+token, self.view()[1])
        self.click('wc:export:2:chart', identifier=99999)
        self.assertIn('Истории запасов и стоимости пока нет', self.view()[0])
        self.assertFalse(self.store.query("SELECT * FROM jobs WHERE kind='wine_cellar_export'"))

    def test_receipt_recovery_survives_store_reopen_without_losing_corrections(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click('rc:confirm:'+token)
        self.message('/cellar')
        self.click('rc:cellar:'+token)
        new_token = self.prefix('rc:confirm:').split(':')[2]
        self.click(f'rc:item:{new_token}:0')
        self.click(f'rc:edit:{new_token}:winery:0')
        self.message('С этикетки')
        self.store = Store(self.root/'life-ui.sqlite3')
        self.bot = BotControls(self.store)
        self.message('/cellar')
        self.assertIn('Продолжить добавление из чека', self.view()[1].values())
        self.click('rc:review:'+new_token)
        self.click(f'rc:item:{new_token}:0')
        self.assertIn('С этикетки', self.view()[0])
        self.assertEqual(ledger.prepare_receipt_wines(self.store, 2, token), new_token)
        self.click('rc:review:'+new_token)
        self.click('rc:confirm:'+new_token)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['winery'], 'С этикетки')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)

    def test_saved_expense_can_add_wines_from_cellar_without_duplicate_expense(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.click(f'rc:confirm:{token}')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.message('/cellar')
        self.click('wc:add')
        self.click('lf:receipts:0')
        self.click('rc:cellar:'+token)
        text, buttons = self.view()
        new_token = self.prefix('rc:confirm:').split(':')[2]
        self.assertIn('Расход уже учтён', text)
        self.assertIn('без повторного расхода', buttons[f'rc:confirm:{new_token}'])
        self.assertNotIn(f'rc:edit:{new_token}:total', buttons)
        self.click(f'rc:item:{new_token}:0')
        self.assertNotIn(f'rc:edit:{new_token}:quantity:0', self.view()[1])
        self.click(f'rc:review:{new_token}')
        self.click(f'rc:confirm:{new_token}')
        self.assertIn('повторной записи нет', self.view()[0])
        self.assertIn('Моя винотека', self.view()[0])
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['quantity'], 2)

    def test_wine_receipt_returns_to_cellar_instead_of_income_controls(self):
        token = self.open_receipt(self.receipt(volume=750))
        self.store.set_setting(f'receipt_ui:2:{token}', {'selected': [0], 'overrides': {}, 'return_to': 'wc:menu'})
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:review:{token}')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'wc:menu')
        self.click(f'rc:confirm:{token}')
        text, buttons = self.view()
        self.assertIn('Чек сохранён', text)
        self.assertIn('Моя винотека', text)
        self.assertNotIn('lf:add:income', buttons)
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['quantity'], 2)
        self.message('/expenses')
        self.assertIn('lf:add:income', self.view()[1])

    def test_wine_upload_keeps_cellar_origin_in_job_and_saved_receipt_ui(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('lf:receipt')
        raw = b'%PDF-1.4\n%%EOF'
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(raw)
        self.bot.handle_document(self.client, self.event(document=SimpleNamespace(file_name='receipt.pdf', mime_type='application/pdf', file_size=len(raw)), photo=None))
        payload = json.loads(self.store.query("SELECT payload FROM jobs WHERE kind='life_receipt'")[0]['payload'])
        self.assertEqual(payload['return_to'], 'wc:menu')
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{payload["request_id"]}')['return_to'], 'wc:menu')
        self.assertEqual(self.store.get_setting('ui_screen:2')['back'], 'wc:menu')
        self.assertNotIn('lf:home', self.view()[1])

    def test_legacy_receipt_volume_is_visible_without_reupload_or_financial_writes(self):
        result = self.receipt(quantity=1)
        result['items'][0]['name'] = 'Вино КазилДьябло Мерло кр.сух.0.75'
        token = self.open_receipt(result)
        # Simulate an already persisted draft from the previous parser version.
        saved = self.store.one('SELECT result FROM life_receipt_drafts WHERE token=?', (token,))
        legacy = json.loads(saved['result'])
        legacy['items'][0].update(volume_ml=None, volume_source=None, volume_requires_confirmation=False)
        self.store.execute('UPDATE life_receipt_drafts SET result=? WHERE token=?', (json.dumps(legacy), token))
        self.click(f'rc:item:{token}:0')
        self.assertIn('750 мл', self.view()[0])
        self.assertIn('проверьте по бутылке', self.view()[0])
        self.assertEqual(self.store.one('SELECT result FROM life_receipt_drafts WHERE token=?', (token,))['result'], json.dumps(legacy))
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))

    def setUp(self):
        parent = Path(__file__).resolve().parents[1]/'.test-tmp'
        parent.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root/'life-ui.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions=['vivino'])
        self.store.add_user(4, subscriptions=['life'])
        self.store.add_user(5, subscriptions=['market'])
        wine_retail.configure_shop(self.store, 1, 'decanter', True)
        self.store.set_setting('wine_metadata_auto:2', True)
        self.bot = BotControls(self.store)
        self.client = Mock()
        ids = count(100)
        self.client.send_message.side_effect = lambda *args, **kwargs: SimpleNamespace(id=next(ids))
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network in Life tests')))

    def event(self, actor=2, kind='private', chat=None, **kwargs):
        return SimpleNamespace(id=17, from_user=SimpleNamespace(id=actor),
                               chat=SimpleNamespace(id=actor if chat is None else chat, type=kind), **kwargs)

    def message(self, text, actor=2, **kwargs):
        self.bot.handle_message(self.client, self.event(actor, text=text, **kwargs))

    def click(self, data, actor=2, *, identifier=None, kind='private', chat=None):
        record = self.store.get_setting(f'ui_screen:{actor}', {})
        # Explicit IDs exercise forged/old-message permissions beyond the active-screen guard.
        if identifier is not None and identifier == record.get('message_id'):
            identifier += 1
        event = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
                                message=SimpleNamespace(id=identifier or record.get('message_id', 100),
                                chat=SimpleNamespace(id=actor if chat is None else chat, type=kind)), answer=Mock())
        self.bot.handle_callback(self.client, event)
        return event

    def view(self, actor=2):
        calls = [call for call in self.client.mock_calls if call[0] in {'send_message', 'edit_message_text'} and call.args[0] == actor]
        call = calls[-1]
        text = call.args[2] if call[0] == 'edit_message_text' else call.args[1]
        markup = call.kwargs.get('reply_markup')
        return text, {button.callback_data: button.text for row in markup.inline_keyboard for button in row} if markup else {}

    def prefix(self, prefix, actor=2):
        return next(data for data in self.view(actor)[1] if data.startswith(prefix))

    def receipt(self, *, fingerprint='a'*64, total=2200, quantity=2, volume=None):
        return {'fingerprint': fingerprint, 'date': '2024-02-01', 'merchant': 'Тестовый магазин', 'total': total, 'currency': 'RUB',
                'items': [{'name': 'Reserva 2020', 'winery': 'Bodega', 'vintage': 2020, 'volume_ml': volume,
                           'quantity': quantity, 'unit_price': 1500, 'amount': 2000, 'is_wine': True},
                          {'name': 'Сыр', 'quantity': 0.2, 'unit_price': 1000, 'amount': 200, 'is_wine': False}], 'warnings': []}

    def open_receipt(self, result=None, *, actor=2, token='receipt-test'):
        ledger.save_receipt_draft(self.store, actor, result or self.receipt(), token)
        self.click('rc:review:'+token, actor, identifier=99999)
        return token

    def manual_review(self, *, actor=2, kind='expense', amount='1250,50', category='Продукты'):
        self.message('/expenses', actor)
        self.click('lf:add:'+kind, actor)
        self.message(amount, actor)
        self.click('lf:today', actor)
        self.message(category, actor)
        self.click('lf:nonote', actor)

    def photo_bytes(self):
        stream = BytesIO()
        Image.new('RGB', (80, 100), 'white').save(stream, format='PNG')
        return stream.getvalue()

    def test_life_hierarchy_has_wine_and_finance_children_with_consistent_back(self):
        self.message('/start')
        text, buttons = self.view()
        self.assertIn('pj:life', buttons)
        self.assertNotIn('pj:vivino', buttons)
        self.click('pj:life')
        text, buttons = self.view()
        self.assertTrue(text.startswith('Жизнь'))
        self.assertTrue({'pj:vivino', 'lf:home', 'lf:receipt', 'menu:home'} <= buttons.keys())
        self.click('pj:vivino')
        self.assertEqual(self.view()[1]['pj:life'], '← Назад')
        self.click('pj:life')
        self.click('lf:home')
        self.assertEqual(self.view()[1]['pj:life'], '← Назад')

    def test_wine_only_finance_only_and_market_only_rights(self):
        self.message('/life', actor=3)
        self.assertIn('pj:vivino', self.view(3)[1])
        self.assertNotIn('lf:home', self.view(3)[1])
        self.click('lf:home', 3, identifier=99999).answer.assert_called_once()
        self.message('/life', actor=4)
        self.assertIn('lf:home', self.view(4)[1])
        self.assertNotIn('pj:vivino', self.view(4)[1])
        self.message('/start', actor=5)
        self.assertNotIn('pj:life', self.view(5)[1])
        self.message('/expenses', actor=5)
        self.assertIn('не подключ', self.view(5)[0])

    def test_manual_income_and_expense_have_review_and_idempotent_save(self):
        self.manual_review()
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.assertIn('Проверка операции', self.view()[0])
        confirmation = self.prefix('lf:save:')
        self.click(confirmation)
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 1250.5)
        self.click(confirmation, identifier=99999).answer.assert_called_once()
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.manual_review(kind='income', amount='5000', category='Зарплата')
        self.click(self.prefix('lf:save:'))
        totals = ledger.summary(self.store, 2)
        self.assertEqual((totals['income'], totals['expenses'], totals['balance']), (5000, 1250.5, 3749.5))

    def test_manual_money_rounds_before_review_and_cancel_leaves_no_entry(self):
        self.manual_review(amount='1.005')
        self.assertIn('1,01 ₽', self.view()[0])
        self.click(self.prefix('lf:save:'))
        self.assertEqual(ledger.entries(self.store, 2)[0]['amount'], 1.01)
        self.message('/expenses')
        self.click('lf:add:expense')
        self.message('NaN')
        self.assertIn('Введите число', self.view()[0])
        self.message('/cancel')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])

    def test_custom_period_monthly_analytics_history_pagination_and_manual_void(self):
        for index in range(12):
            ledger.add_entry(self.store, 2, 'expense', index+1, date='2024-01-01', category='Еда')
        ledger.add_entry(self.store, 2, 'income', 1000, date='2024-02-01')
        self.message('/expenses')
        self.click('lf:period')
        self.click('lf:period:custom')
        self.message('01.01.2024 31.01.2024')
        self.assertIn('78,00 ₽', self.view()[0])
        self.click('lf:history:0')
        first = {data for data in self.view()[1] if data.startswith('lf:entry:')}
        self.assertEqual(len(first), 10)
        self.click('lf:history:1')
        second = {data for data in self.view()[1] if data.startswith('lf:entry:')}
        self.assertEqual(len(second), 2)
        self.assertFalse(first & second)
        entry_route = next(iter(second))
        removed_amount = ledger.get_entry(self.store, 2, int(entry_route.rsplit(':', 1)[-1]))['amount']
        self.click(entry_route)
        self.click(self.prefix('lf:voidask:'))
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 78)
        self.click(self.prefix('lf:void:'))
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 78-removed_amount)
        self.click('lf:analytics')
        self.assertIn('2024-01', self.view()[0])
        self.assertNotIn('2024-02:', self.view()[0])

    def test_receipt_selection_and_volume_are_explicit_until_atomic_confirmation(self):
        token = self.open_receipt()
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{token}', {'selected': []})['selected'], [])
        self.assertEqual(wine_cellar.lots(self.store, 2), [])
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.click(f'rc:select:{token}')
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{token}')['selected'], [0])
        self.click(f'rc:volumeall:{token}')
        self.assertEqual(wine_cellar.lots(self.store, 2), [])
        self.click(f'rc:confirm:{token}')
        self.assertIn('Чек сохранён', self.view()[0])
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['quantity'], lot['volume_ml'], lot['purchase_price']), (2, 750, 1000))
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)

    def test_receipt_quantity_and_cost_corrections_keep_whole_receipt_expense(self):
        token = self.open_receipt(self.receipt(quantity=1.5))
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:edit:{token}:quantity:0')
        self.message('2')
        self.assertIn('Количество: 2', self.view()[0])
        self.click(f'rc:edit:{token}:amount:0')
        self.message('1800')
        self.click(f'rc:toggle:{token}:0')
        self.click(f'rc:volume:{token}:0')
        self.click('rc:review:'+token)
        self.click(f'rc:edit:{token}:total')
        self.message('2000')
        self.click(f'rc:confirm:{token}')
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2000)
        self.assertEqual(wine_cellar.lots(self.store, 2)[0]['purchase_price'], 900)

    def test_receipt_wine_lookup_is_requested_and_repeated_taps_only_queue_one_job(self):
        token = self.open_receipt()
        with patch('thesaurus.wine_enrichment.lookup') as lookup:
            self.click(f'rc:item:{token}:0')
            self.assertEqual(self.view()[1][f'rc:enrich:{token}:0'], 'Найти сведения о вине')
            jobs = self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'")
            self.assertEqual(len(jobs), 1)
            self.assertTrue(json.loads(jobs[0]['payload'])['automatic'])
            self.click(f'rc:enrich:{token}:0')
            self.click(f'rc:enrich:{token}:0')
        lookup.assert_not_called()
        jobs = self.store.query("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")
        self.assertEqual(len(jobs), 1)
        payload = json.loads(jobs[0]['payload'])
        self.assertEqual((payload['actor_id'], payload['owner_id'], payload['receipt_token'], payload['index']),
                         (2, 2, token, 0))
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.assertEqual(wine_cellar.lots(self.store, 2), [])

    def test_receipt_lookup_variants_require_choice_then_final_purchase_confirmation(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt()
        result['items'][0].update(winery='', vintage=None, name='Merlot Reserva')
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:enrich:{token}:0')
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(
                   candidate(match_status='possible'), candidate(id='c'*32, winery='Второй производитель',
                   source_url='https://decanter.ru/wine/second', match_status='possible'))):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click('rc:review:'+token)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:info:{token}:0')
        self.assertIn('Год в источнике · сверьте с бутылкой', self.view()[0])
        self.assertIn(f'rc:useyear:{token}:0:0', self.view()[1])
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.click(f'rc:info:{token}:0:1')
        self.click(f'rc:useinfo:{token}:0:1')
        self.assertIn('Второй производитель', self.view()[0])
        self.assertEqual(wine_cellar.lots(self.store, 2), [])
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.click(f'rc:confirm:{token}')
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], lot['quantity'], lot['purchase_price'], lot['purchase_date']),
                         ('Merlot Reserva', 2, 1000, result['date']))
        self.assertFalse(lot['vintage'])
        self.assertIn('https://decanter.ru/wine/second', lot['notes'])
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)

    def test_receipt_wine_information_paginates_long_variants_with_one_explicit_choice(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt()
        result['items'][0].update(winery='', vintage=None, name='Merlot Reserva')
        token = self.open_receipt(result, token=uuid4().hex)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:enrich:{token}:0')
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        variants = [candidate(id=str(index)*32, name=f'Вино {index+1}', winery='Винодельня '+('А'*80),
                              region='Регион '+('Б'*80), description='Описание вина. '*40,
                              grapes=['Мерло', 'Каберне Совиньон'],
                              source_url=f'https://decanter.ru/wine/variant-{index}', match_status='possible')
                    for index in range(8)]
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(*variants)):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click('rc:review:'+token)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:info:{token}:0')
        for index, variant in enumerate(variants):
            with self.subTest(page=index):
                text, buttons = self.view()
                self.assertLessEqual(len(text), 4096)
                self.assertIn(variant['source_url'], text)
                for other in variants:
                    if other is not variant:
                        self.assertNotIn(other['source_url'], text)
                self.assertIn('Мерло, Каберне Совиньон', text)
                self.assertNotIn("['Мерло'", text)
                choices = [data for data in buttons if data.startswith('rc:useinfo:')]
                self.assertEqual(choices, [f'rc:useinfo:{token}:0:{index}'])
                self.assertIn(f'rc:item:{token}:0', buttons)
                self.assertTrue(all(len(data.encode('utf-8')) <= 64 for data in buttons))
                self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
                if index < len(variants)-1:
                    self.click(f'rc:info:{token}:0:{index+1}')
        self.click(f'rc:info:{token}:0:6')
        self.assertIn(variants[6]['source_url'], self.view()[0])
        self.assertFalse(self.store.query('SELECT * FROM wine_lots'))

    def test_receipt_lookup_callbacks_reject_missing_functions_other_owners_and_groups(self):
        token = self.open_receipt()
        jobs_before = self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'")
        with patch('thesaurus.wine_enrichment.lookup') as lookup:
            for actor in (1, 3, 4, 5, 99):
                for action in ('enrich', 'info', 'useinfo'):
                    data = f'rc:{action}:{token}:0'+(':0' if action == 'useinfo' else '')
                    event = self.click(data, actor=actor, identifier=99999)
                    self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
            event = self.click(f'rc:enrich:{token}:0', kind='group', chat=-100)
            self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        lookup.assert_not_called()
        self.assertEqual(self.store.query("SELECT * FROM jobs WHERE kind='life_wine_enrich'"), jobs_before)
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)

    def test_renaming_enriched_wine_discards_automatic_identity_but_keeps_purchase_record(self):
        from tests.test_receipt_wines import candidate, lookup_response
        result = self.receipt()
        result['items'][0].update(winery='', vintage=None, name='Merlot Reserva')
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:enrich:{token}:0')
        payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich'")['payload'])
        with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(candidate())):
            receipt_wines.run_lookup(self.store, 2, payload)
        self.click(f'rc:edit:{token}:name:0')
        self.message('Другое вино')
        ui = self.store.get_setting(f'receipt_ui:2:{token}')
        self.assertFalse(ui['overrides'].get('0', {}).get('winery'))
        self.assertFalse(ui['overrides'].get('0', {}).get('volume_ml'))
        self.assertFalse(ui.get('wine_info', {}).get('0'))
        self.assertNotIn(f'rc:info:{token}:0', self.view()[1])
        stale = self.click(f'rc:useinfo:{token}:0:0', identifier=99999)
        self.assertTrue(stale.answer.call_args.kwargs.get('show_alert'))
        saved = ledger.get_receipt_draft(self.store, 2, token)['result']
        self.assertEqual((saved['items'][0]['name'], saved['items'][0]['quantity'], saved['items'][0]['amount'],
                          saved['date'], saved['total']), ('Другое вино', 2, 2000, result['date'], 2200))
        self.assertEqual(wine_cellar.lots(self.store, 2), [])

    def test_receipt_rename_recomputes_name_derived_vintage_and_volume(self):
        from thesaurus.receipts import parse_receipt_text
        result = parse_receipt_text('Дата: 01.02.2024\nВино Merlot 2020 0.75\n2 x 1000 = 2000\nИТОГО 2000')
        self.assertEqual((str(result['items'][0]['vintage']), result['items'][0]['volume_ml']), ('2020', 750))
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:edit:{token}:name:0')
        self.message('Вино Cabernet 2021 1.5 л')
        item = ledger.get_receipt_draft(self.store, 2, token)['result']['items'][0]
        self.assertEqual((str(item['vintage']), item['volume_ml']), ('2021', 1500))
        self.assertEqual(item['volume_source'], 'name_explicit_unit')
        self.assertFalse(item['volume_requires_confirmation'])
        self.assertIn('Год: 2021', self.view()[0])
        self.assertIn('1500 мл', self.view()[0])
        self.assertEqual((item['quantity'], item['amount']), (2, 2000))
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))
        self.assertFalse(self.store.query('SELECT * FROM wine_lots'))

    def test_receipt_rename_keeps_explicit_manual_vintage_volume_and_producer(self):
        from thesaurus.receipts import parse_receipt_text
        result = parse_receipt_text('Дата: 01.02.2024\nВино Merlot 2020 0.75\n2 x 1000 = 2000\nИТОГО 2000')
        token = self.open_receipt(result)
        self.click(f'rc:item:{token}:0')
        for field, value in (('winery', 'С этикетки'), ('vintage', '2018'), ('volume_ml', '375')):
            self.click(f'rc:edit:{token}:{field}:0')
            self.message(value)
        self.click(f'rc:edit:{token}:name:0')
        self.message('Вино Cabernet 2021 1.5 л')
        self.assertIn('Год: 2018', self.view()[0])
        self.assertIn('375 мл', self.view()[0])
        self.assertIn('С этикетки', self.view()[0])
        self.assertFalse(self.store.query('SELECT * FROM wine_lots'))
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.click(f'rc:confirm:{token}')
        lot = wine_cellar.lots(self.store, 2)[0]
        self.assertEqual((lot['name'], str(lot['vintage']), lot['volume_ml'], lot['winery']),
                         ('Вино Cabernet 2021 1.5 л', '2018', 375, 'С этикетки'))
        self.assertEqual((lot['quantity'], lot['purchase_price']), (2, 1000))
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2000)

    def test_receipt_volume_shortcuts_clear_automatic_information_and_keep_manual_fields(self):
        from tests.test_receipt_wines import candidate, lookup_response
        for action in ('volume', 'volumeall'):
            with self.subTest(action=action):
                result = self.receipt()
                result['items'][0].update(name='Merlot Reserva', winery='', vintage=None)
                token = self.open_receipt(result, token=uuid4().hex)
                self.store.set_setting(f'receipt_ui:2:{token}', {'selected': [0], 'overrides': {
                    '0': {'winery': 'С этикетки', 'vintage': '2019', 'notes': 'Мои заметки'}}})
                self.click(f'rc:item:{token}:0')
                self.click(f'rc:enrich:{token}:0')
                payload = json.loads(self.store.one("SELECT payload FROM jobs WHERE kind='life_wine_enrich' ORDER BY id DESC")['payload'])
                with patch('thesaurus.wine_enrichment.lookup', return_value=lookup_response(candidate(volume_ml=1500))):
                    receipt_wines.run_lookup(self.store, 2, payload)
                self.click('rc:review:'+token)
                if action == 'volume':
                    self.click(f'rc:item:{token}:0')
                self.click(f'rc:{action}:{token}'+(':0' if action == 'volume' else ''))
                ui = self.store.get_setting(f'receipt_ui:2:{token}')
                self.assertEqual(ui['overrides']['0'], {'winery': 'С этикетки', 'vintage': '2019',
                                                        'notes': 'Мои заметки', 'volume_ml': 750})
                self.assertFalse(ui.get('wine_info', {}).get('0'))
                self.assertEqual(ui['wine_lookups']['0']['status'], 'pending')
                self.assertFalse(ui['wine_lookups']['0']['candidates'])
                self.assertEqual(ui['selected'], [0])
                self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))
                self.assertFalse(self.store.query('SELECT * FROM wine_lots'))

    def test_unselect_after_volume_override_saves_expense_without_wine(self):
        token = self.open_receipt()
        self.click(f'rc:select:{token}')
        self.click(f'rc:volumeall:{token}')
        self.click(f'rc:item:{token}:0')
        self.click(f'rc:toggle:{token}:0')
        self.click('rc:review:'+token)
        self.click(f'rc:confirm:{token}')
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)
        self.assertEqual(wine_cellar.lots(self.store, 2), [])

    def test_duplicate_receipt_different_upload_is_not_readded_and_confirmed_screen_is_read_only(self):
        token = self.open_receipt()
        self.click(f'rc:confirm:{token}')
        self.open_receipt(token='same-fiscal-different-file')
        self.click('rc:confirm:same-fiscal-different-file')
        self.assertIn('уже учтён', self.view()[0])
        self.assertEqual(ledger.summary(self.store, 2)['count'], 1)
        self.click('rc:review:'+token, identifier=99999)
        self.assertIn('уже сохранён', self.view()[0])
        self.assertFalse(any(route.startswith(('rc:edit:', 'rc:confirm:', 'rc:toggle:')) for route in self.view()[1]))

    def test_revoked_wine_access_can_clear_selection_and_keep_expense(self):
        token = self.open_receipt()
        self.click(f'rc:select:{token}')
        self.click(f'rc:volumeall:{token}')
        feature_access.change(self.store, 1, 2, 'wine_cellar', False)
        self.click('rc:review:'+token, identifier=99999)
        self.assertIn('Доступ к винотеке отключён', self.view()[0])
        self.assertNotIn(f'rc:confirm:{token}', self.view()[1])
        self.click(f'rc:clear:{token}')
        self.assertIn(f'rc:confirm:{token}', self.view()[1])
        self.click(f'rc:confirm:{token}')
        self.assertEqual(ledger.summary(self.store, 2)['expenses'], 2200)
        self.assertEqual(self.store.query('SELECT * FROM wine_lots WHERE user_id=2'), [])

    def test_draft_list_only_exposes_owned_unconfirmed_unexpired_receipts(self):
        self.open_receipt(token='owned-active')
        ledger.save_receipt_draft(self.store, 4, self.receipt(), 'other-owner')
        ledger.save_receipt_draft(self.store, 2, self.receipt(fingerprint='b'*64), 'owned-expired')
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01' WHERE token='owned-expired'")
        ledger.save_receipt_draft(self.store, 2, self.receipt(fingerprint='c'*64), 'owned-confirmed')
        ledger.confirm_receipt(self.store, 2, 'owned-confirmed', selected_wine_items=[])
        self.message('/expenses')
        self.click('lf:drafts')
        routes = {route for route in self.view()[1] if route.startswith('rc:review:')}
        self.assertEqual(routes, {'rc:review:owned-active'})
        self.click('rc:review:owned-active')
        self.assertIn('Проверка чека', self.view()[0])

    def test_manual_previous_step_back_and_cancel_never_records_draft(self):
        self.message('/expenses')
        self.click('lf:add:expense')
        self.message('100')
        self.assertIn('lf:step:amount', self.view()[1])
        self.click('lf:step:amount')
        self.message('250')
        self.click('lf:today')
        self.assertIn('lf:step:date', self.view()[1])
        self.click('lf:step:date')
        self.message('01.02.2024')
        self.click('lf:category:0')
        self.assertIn('lf:step:category', self.view()[1])
        self.click('lf:step:category')
        self.message('Другая категория')
        self.message('/cancel')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.assertIsNone(self.store.get_setting('life_entry:2'))
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])

    def test_receipt_unknown_volume_and_noninteger_volume_do_not_invent_stock(self):
        token = self.open_receipt()
        self.click(f'rc:select:{token}')
        self.click(f'rc:confirm:{token}').answer.assert_called_once()
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.click(f'rc:item:{token}:0', identifier=99999)
        self.click(f'rc:edit:{token}:volume_ml:0')
        self.message('750.9')
        self.assertIn('целое число', self.view()[0])
        self.assertEqual(wine_cellar.lots(self.store, 2), [])
        self.message('375')
        self.assertIn('375 мл', self.view()[0])

    def test_wrong_receipt_and_entry_ids_never_read_or_write_another_user(self):
        token = self.open_receipt()
        self.click('rc:review:'+token, actor=4, identifier=99999).answer.assert_called_once()
        self.click('rc:confirm:'+token, actor=4, identifier=99999).answer.assert_called_once()
        other = ledger.add_entry(self.store, 4, 'expense', 777)
        self.click(f'lf:entry:{other["id"]}', identifier=99999).answer.assert_called_once()
        self.click(f'lf:void:{other["id"]}', identifier=99999).answer.assert_called_once()
        self.assertEqual(ledger.summary(self.store, 4)['expenses'], 777)
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)

    def test_finance_only_receipt_has_no_wine_actions_and_cannot_forge_stock_write(self):
        token = self.open_receipt(actor=4)
        self.assertNotIn('rc:select:'+token, self.view(4)[1])
        self.click(f'rc:select:{token}', actor=4, identifier=99999).answer.assert_called_once()
        self.click(f'rc:confirm:{token}', actor=4, identifier=99999)
        self.assertEqual(ledger.summary(self.store, 4)['expenses'], 2200)
        self.assertEqual(self.store.query('SELECT * FROM wine_lots WHERE user_id=4'), [])

    def test_revocation_blocks_pending_manual_and_receipt_confirmation(self):
        self.manual_review()
        confirmation = self.prefix('lf:save:')
        feature_access.change(self.store, 1, 2, 'life_finance', False)
        self.click(confirmation, identifier=99999).answer.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])
        feature_access.change(self.store, 1, 2, 'life_finance', True)
        token = self.open_receipt()
        self.click(f'rc:select:{token}')
        self.click(f'rc:volumeall:{token}')
        feature_access.change(self.store, 1, 2, 'wine_cellar', False)
        self.click(f'rc:confirm:{token}', identifier=99999).answer.assert_called_once()
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])
        self.assertEqual(self.store.query('SELECT * FROM wine_lots'), [])

    def test_all_life_callbacks_require_private_matching_chat_and_allowed_user(self):
        for actor, kind, chat in ((99, 'private', 99), (2, 'group', -100), (2, 'private', 4)):
            event = self.click('lf:home', actor=actor, kind=kind, chat=chat)
            event.answer.assert_called_once_with('Нет доступа', show_alert=True)
        self.message('/expenses', kind='group', chat=-100)
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])

    def test_pdf_and_finance_photo_queue_receipt_without_stock_or_expense(self):
        self.message('/receipt')
        raw = b'%PDF-1.4\n% synthetic receipt, parser mocked in worker tests\n%%EOF'
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(raw)
        document = SimpleNamespace(file_name='receipt.pdf', mime_type='application/pdf', file_size=len(raw))
        self.bot.handle_document(self.client, self.event(document=document, photo=None))
        jobs = self.store.query('SELECT kind,payload FROM jobs')
        self.assertEqual([row['kind'] for row in jobs], ['life_receipt'])
        payload = json.loads(jobs[0]['payload'])
        self.assertEqual((payload['actor_id'], payload['owner_id']), (2, 2))
        self.assertEqual(Path(payload['path']).parent, self.root/'receipts'/'2')
        self.assertEqual(ledger.summary(self.store, 2)['count'], 0)
        self.message('/expenses')
        image = self.photo_bytes()
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(image)
        self.bot.handle_wine_photo(self.client, self.event(photo=SimpleNamespace(file_size=len(image)), document=None))
        self.assertEqual([row['kind'] for row in self.store.query('SELECT kind FROM jobs')], ['life_receipt', 'life_receipt'])
        self.assertEqual(wine_cellar.lots(self.store, 2), [])

    def test_wine_context_photo_and_vivino_csv_keep_their_own_upload_routes(self):
        self.message('/cellar')
        self.click('wc:add')
        self.click('wc:photo')
        image = self.photo_bytes()
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(image)
        self.bot.handle_wine_photo(self.client, self.event(photo=SimpleNamespace(file_size=len(image)), document=None))
        self.assertEqual(self.store.query('SELECT kind FROM jobs')[0]['kind'], 'wine_recognize')
        self.message('/vivino')
        with patch.object(self.bot.vivino_controls, 'document') as importer:
            document = SimpleNamespace(file_name='wines.csv', mime_type='text/csv', file_size=10)
            self.bot.handle_document(self.client, self.event(document=document, photo=None))
            importer.assert_called_once()
        self.assertEqual(len(self.store.query('SELECT * FROM jobs')), 1)

    def test_media_rejects_wrong_rights_group_bad_format_and_oversize_before_queue(self):
        raw = b'%PDF-1.4\n%%EOF'
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(raw)
        pdf = SimpleNamespace(file_name='receipt.pdf', mime_type='application/pdf', file_size=len(raw))
        self.bot.handle_document(self.client, self.event(actor=3, document=pdf, photo=None))
        self.client.download_media.assert_not_called()
        self.bot.handle_document(self.client, self.event(kind='group', chat=-100, document=pdf, photo=None))
        self.client.download_media.assert_not_called()
        self.message('/receipt')
        big = SimpleNamespace(file_name='receipt.pdf', mime_type='application/pdf', file_size=11*1024*1024)
        self.bot.handle_document(self.client, self.event(document=big, photo=None))
        self.client.download_media.assert_not_called()
        self.client.download_media.side_effect = lambda *args, **kwargs: BytesIO(b'not a photo or PDF')
        self.bot.handle_document(self.client, self.event(document=pdf, photo=None))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_finance_revocation_during_download_prevents_retention_and_queue(self):
        self.message('/receipt')
        raw = b'%PDF-1.4\n%%EOF'
        def revoke_then_download(*args, **kwargs):
            feature_access.change(self.store, 1, 2, 'life_finance', False)
            return BytesIO(raw)
        self.client.download_media.side_effect = revoke_then_download
        document = SimpleNamespace(file_name='receipt.pdf', mime_type='application/pdf', file_size=len(raw))
        self.bot.handle_document(self.client, self.event(document=document, photo=None))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])
        self.assertFalse((self.root/'receipts'/'2').exists())
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])


if __name__ == '__main__':
    unittest.main()
