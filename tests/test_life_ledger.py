"""Offline personal ledger, receipt privacy, atomic stock and duplicate imports."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from thesaurus import feature_access, life_ledger, subscription_access, wine_cellar
from thesaurus.store import Store, dumps


class LifeLedgerTests(unittest.TestCase):
    def test_receipt_wine_sources_deduplicate_and_only_offer_own_missing_stock(self):
        self.draft()
        self.confirm(selected=[])
        self.draft(token='duplicate-upload')
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 2), [
            {'token': 'duplicate-upload', 'kind': 'saved', 'wine_count': 1}])
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 4), [])
        token = life_ledger.prepare_receipt_wines(self.store, 2, 'draft-token')
        ui = self.store.get_setting(f'receipt_ui:2:{token}')
        ui['overrides'] = {'0': {'winery': 'Моя правка'}}
        self.store.set_setting(f'receipt_ui:2:{token}', ui)
        self.assertEqual(life_ledger.prepare_receipt_wines(self.store, 2, 'duplicate-upload'), token)
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 2), [
            {'token': token, 'kind': 'draft', 'wine_count': 1}])
        self.confirm(token=token, add_wines_to_existing=True)
        lot = wine_cellar.lots(self.store, 2)[0]
        wine_cellar.adjust_stock(self.store, 2, lot['id'], -lot['quantity'], operation_id='drink-all')
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 2), [])

    def test_expired_receipt_is_not_offered_but_accounted_receipt_remains_available(self):
        self.draft()
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01'")
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 2), [])
        self.draft(token='saved')
        self.confirm(token='saved', selected=[])
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01'")
        self.assertEqual(life_ledger.receipt_wine_sources(self.store, 2), [
            {'token': 'saved', 'kind': 'saved', 'wine_count': 1}])

    def test_add_wines_to_accounted_receipt_is_explicit_idempotent_and_private(self):
        self.draft()
        original = self.confirm(selected=[])
        self.assertEqual(original['wine_lot_ids'], [])
        new_token = life_ledger.prepare_receipt_wines(self.store, 2, 'draft-token')
        self.assertNotEqual(new_token, 'draft-token')
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{new_token}')['selected'], [0])
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)
        self.assertFalse(wine_cellar.lots(self.store, 2))
        for actor in (1, 3, 4, 5):
            with self.subTest(actor=actor), self.assertRaises((ValueError, PermissionError)):
                life_ledger.prepare_receipt_wines(self.store, actor, 'draft-token')
        added = self.confirm(token=new_token, add_wines_to_existing=True)
        repeated = self.confirm(token=new_token, add_wines_to_existing=True)
        self.assertEqual(len(added['added_wine_lot_ids']), 1)
        self.assertEqual(repeated['added_wine_lot_ids'], [])
        self.assertEqual((added['receipt_id'], added['entry_id']), (original['receipt_id'], original['entry_id']))
        self.assertEqual((life_ledger.summary(self.store, 2)['count'], life_ledger.summary(self.store, 2)['expenses']), (1, 2200))
        self.assertEqual(len(wine_cellar.lots(self.store, 2)), 1)
        another = life_ledger.prepare_receipt_wines(self.store, 2, 'draft-token')
        self.assertEqual(self.store.get_setting(f'receipt_ui:2:{another}')['selected'], [])

    def test_accounted_receipt_stock_cannot_change_financial_data_or_create_expense(self):
        self.draft()
        with self.assertRaises(ValueError):
            self.confirm(add_wines_to_existing=True)
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)
        self.confirm(selected=[])
        token = life_ledger.prepare_receipt_wines(self.store, 2, 'draft-token')
        with self.assertRaises(ValueError):
            self.confirm(token=token, add_wines_to_existing=True, overrides={0: {'quantity': 3}})
        result = life_ledger.get_receipt_draft(self.store, 2, token)['result']
        result['items'][0]['amount'] = 1999
        life_ledger.save_receipt_draft(self.store, 2, result, token)
        with self.assertRaises(ValueError):
            self.confirm(token=token, add_wines_to_existing=True)
        self.assertFalse(wine_cellar.lots(self.store, 2))
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)

    def test_failed_stock_append_rolls_back_all_lots_prices_and_receipt_links(self):
        receipt = self.receipt()
        receipt['items'][1] = {**receipt['items'][0], 'amount': 200, 'quantity': 1, 'name': 'Second'}
        self.draft(receipt)
        original = self.confirm(selected=[])
        token = life_ledger.prepare_receipt_wines(self.store, 2, 'draft-token')
        with self.assertRaises(ValueError):
            self.confirm(token=token, selected=[0, 1], overrides={1: {'name': ''}}, add_wines_to_existing=True)
        for table in ('wine_lots', 'wine_events', 'wine_prices', 'wine_snapshots'):
            self.assertFalse(self.store.query('SELECT * FROM '+table))
        saved = self.store.one('SELECT wine_lot_ids FROM life_receipt_imports WHERE id=?', (original['receipt_id'],))
        self.assertEqual(saved['wine_lot_ids'], '[]')
        self.assertEqual(life_ledger.get_receipt_draft(self.store, 2, token)['status'], 'draft')
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)

    def setUp(self):
        # Keep the domain tests independent of concurrent menu/catalog integration.
        sections = patch.dict(subscription_access.CATALOG, {'life': 'Жизнь · доходы и расходы'})
        features = patch.dict(feature_access.CATALOG, {'life_finance': ('Доходы и расходы', ('life',))})
        sections.start()
        features.start()
        self.addCleanup(sections.stop)
        self.addCleanup(features.stop)
        parent = Path(__file__).resolve().parents[1]/'.test-tmp'
        parent.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)/'life.sqlite3'
        self.store = Store(self.path)
        life_ledger.migrate(self.store)
        self.store.add_user(1, 'admin')
        self.store.add_user(2, subscriptions=['life', 'vivino'])
        self.store.add_user(3, subscriptions=['life', 'vivino'])
        self.store.add_user(4, subscriptions=['life'])
        self.store.add_user(5, subscriptions=['vivino'])

    def receipt(self, **changes):
        result = {'status': 'parsed', 'fingerprint': 'a'*64, 'date': '2024-02-01', 'merchant': 'Магазин',
                  'currency': 'RUB', 'total': 2200,
                  'items': [{'name': 'Reserva 2020', 'quantity': 2, 'unit_price': 1500, 'amount': 2000,
                             'is_wine': True, 'winery': 'Bodega', 'vintage': 2020, 'volume_ml': 750},
                            {'name': 'Сыр', 'quantity': 0.2, 'unit_price': 1000, 'amount': 200, 'is_wine': False}],
                  'raw_text': 'SHOULD_NOT_PERSIST', 'fiscal_fn': 'PRIVATE_FISCAL_ID', 'warnings': []}
        result.update(changes)
        return result

    def draft(self, result=None, actor=2, token='draft-token'):
        return life_ledger.save_receipt_draft(self.store, actor, result or self.receipt(), token)

    def confirm(self, actor=2, token='draft-token', selected=None, **kwargs):
        return life_ledger.confirm_receipt(self.store, actor, token, selected_wine_items=[0] if selected is None else selected, **kwargs)

    def test_migration_backup_is_idempotent_and_rejects_unknown_version(self):
        backups = list(self.path.parent.glob('*.life-backup-*.sqlite3'))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as conn:
            self.assertFalse(conn.execute("SELECT name FROM sqlite_master WHERE name='life_ledger_entries'").fetchone())
        life_ledger.migrate(self.store)
        self.assertEqual(len(list(self.path.parent.glob('*.life-backup-*.sqlite3'))), 1)
        self.store.set_setting('life_ledger_version', 99)
        with self.assertRaises(RuntimeError):
            life_ledger.migrate(self.store)

    def test_migration_rolls_back_all_tables_on_failure(self):
        path = self.path.parent/'legacy.sqlite3'
        with patch.object(life_ledger, 'migrate'):
            other = Store(path)
        with patch.object(life_ledger, 'SCHEMA', life_ledger.SCHEMA+'; INVALID SQL'), self.assertRaises(sqlite3.DatabaseError):
            life_ledger.migrate(other)
        self.assertFalse(other.one("SELECT name FROM sqlite_master WHERE name='life_ledger_entries'"))
        self.assertIsNone(other.get_setting('life_ledger_version'))

    def test_income_expenses_exact_money_filters_and_categories(self):
        life_ledger.add_entry(self.store, 2, 'income', '1000,01', date='2024-01-01', category='Зарплата')
        life_ledger.add_entry(self.store, 2, 'expense', '200.05', date='2024-02-01', category='Еда')
        life_ledger.add_entry(self.store, 2, 'expense', 100, date='2024-03-01', category='Еда')
        result = life_ledger.summary(self.store, 2)
        self.assertEqual((result['income'], result['expenses'], result['balance'], result['count']), (1000.01, 300.05, 699.96, 3))
        self.assertIn({'kind': 'expense', 'category': 'Еда', 'amount': 300.05, 'count': 2}, result['by_category'])
        result = life_ledger.summary(self.store, 2, start='2024-02-01', end='2024-02-29')
        self.assertEqual(result['expenses'], 200.05)
        self.assertEqual(len(life_ledger.entries(self.store, 2, kind='income')), 1)
        self.assertEqual(life_ledger.entries(self.store, 2, limit=1)[0]['date'], '2024-03-01')

    def test_offset_pagination_and_direct_get_do_not_truncate_or_expose_other_owner(self):
        identities = [life_ledger.add_entry(self.store, 2, 'expense', value+1, date='2024-01-01')['id'] for value in range(12)]
        first = life_ledger.entries(self.store, 2, limit=5)
        second = life_ledger.entries(self.store, 2, limit=5, offset=5)
        third = life_ledger.entries(self.store, 2, limit=5, offset=10)
        self.assertEqual([row['id'] for row in first+second+third], list(reversed(identities)))
        self.assertEqual(life_ledger.get_entry(self.store, 2, identities[0])['amount'], 1)
        self.assertEqual(life_ledger.get_entry(self.store, 1, identities[0], owner=2)['amount'], 1)
        for actor in (1, 3):
            with self.assertRaises(ValueError):
                life_ledger.get_entry(self.store, actor, identities[0])
        for offset in (-1, True, '1', 1_000_001):
            with self.assertRaises(ValueError):
                life_ledger.entries(self.store, 2, offset=offset)
        with self.assertRaises(ValueError):
            life_ledger.entries(self.store, 2, limit=1001)

    def test_monthly_summary_aggregates_all_active_entries_with_date_filter_and_privacy(self):
        life_ledger.add_entry(self.store, 2, 'income', 1000, date='2024-01-01')
        life_ledger.add_entry(self.store, 2, 'expense', '10.01', date='2024-01-31')
        life_ledger.add_entry(self.store, 2, 'expense', '20.02', date='2024-02-01')
        cancelled = life_ledger.add_entry(self.store, 2, 'expense', 99, date='2024-02-01')
        life_ledger.void_entry(self.store, 2, cancelled['id'])
        life_ledger.add_entry(self.store, 3, 'expense', 9999, date='2024-02-01')
        self.assertEqual(life_ledger.monthly_summary(self.store, 2), [
            {'month': '2024-01', 'income': 1000, 'expenses': 10.01, 'balance': 989.99, 'count': 2},
            {'month': '2024-02', 'income': 0, 'expenses': 20.02, 'balance': -20.02, 'count': 1}])
        self.assertEqual(len(life_ledger.monthly_summary(self.store, 1, owner=2, start='2024-02-01')), 1)
        self.assertEqual(life_ledger.monthly_summary(self.store, 2, end='2023-12-31'), [])
        with self.assertRaises(PermissionError):
            life_ledger.monthly_summary(self.store, 3, owner=2)

    def test_manual_entry_idempotency_does_not_mask_changed_operation(self):
        first = life_ledger.add_entry(self.store, 2, 'expense', 123, operation_id='manual')
        second = life_ledger.add_entry(self.store, 2, 'expense', 123, operation_id='manual')
        self.assertEqual(first['id'], second['id'])
        with self.assertRaises(ValueError):
            life_ledger.add_entry(self.store, 2, 'expense', 124, operation_id='manual')
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)

    def test_void_manual_entry_preserves_audit_and_is_idempotent(self):
        entry = life_ledger.add_entry(self.store, 2, 'expense', 123, operation_id='manual')
        first = life_ledger.void_entry(self.store, 2, entry['id'])
        second = life_ledger.void_entry(self.store, 2, entry['id'])
        self.assertTrue(first['voided'])
        self.assertFalse(first['already_voided'])
        self.assertTrue(second['already_voided'])
        self.assertEqual(life_ledger.entries(self.store, 2), [])
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 0)
        self.assertTrue(life_ledger.entries(self.store, 2, include_void=True)[0]['voided'])
        self.assertEqual(self.store.one('SELECT COUNT(*) AS n FROM life_ledger_actions')['n'], 1)
        self.assertTrue(life_ledger.add_entry(self.store, 2, 'expense', 123, operation_id='manual')['voided'])
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)

    def test_void_is_own_only_and_never_breaks_receipt_stock_link(self):
        manual = life_ledger.add_entry(self.store, 2, 'income', 100)
        for actor in (1, 3):
            with self.assertRaises(ValueError):
                life_ledger.void_entry(self.store, actor, manual['id'])
        self.draft()
        imported = self.confirm()
        with self.assertRaises(ValueError):
            life_ledger.void_entry(self.store, 2, imported['entry_id'])
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 2200)

    def test_v1_migration_upgrades_manual_void_without_losing_entries(self):
        path = self.path.parent/'v1.sqlite3'
        with patch.object(life_ledger, 'migrate'):
            legacy = Store(path)
        old_schema = life_ledger.SCHEMA.replace(',voided_at TEXT', '')
        with patch.object(life_ledger, 'VERSION', 1), patch.object(life_ledger, 'SCHEMA', old_schema):
            life_ledger.migrate(legacy)
        legacy.add_user(2, subscriptions=['life'])
        legacy.execute('''INSERT INTO life_ledger_entries(user_id,kind,amount_cents,date,category,note,operation_id,created_at)
            VALUES(2,'expense',10000,'2024-01-01','Еда','','legacy','2024-01-01')''')
        life_ledger.migrate(legacy)
        self.assertEqual(legacy.get_setting('life_ledger_version'), 2)
        self.assertEqual(life_ledger.summary(legacy, 2)['expenses'], 100)
        entry = life_ledger.entries(legacy, 2)[0]
        life_ledger.void_entry(legacy, 2, entry['id'])
        self.assertEqual(life_ledger.summary(legacy, 2)['count'], 0)

    def test_invalid_money_kind_dates_and_limits(self):
        for amount in (True, False, -1, 0, 'nan', 'inf', '1e999', '1000000001'):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                life_ledger.add_entry(self.store, 2, 'expense', amount)
        with self.assertRaises(ValueError):
            life_ledger.add_entry(self.store, 2, 'other', 10)
        with self.assertRaises(ValueError):
            life_ledger.add_entry(self.store, 2, 'expense', 10, date='2999-01-01')
        with self.assertRaises(ValueError):
            life_ledger.entries(self.store, 2, start='2024-03-01', end='2024-01-01')
        with self.assertRaises(ValueError):
            life_ledger.entries(self.store, 2, limit=True)

    def test_privacy_requires_finance_and_life_and_admin_other_read_only(self):
        life_ledger.add_entry(self.store, 2, 'expense', 100)
        self.assertEqual(life_ledger.entries(self.store, 3), [])
        with self.assertRaises(PermissionError):
            life_ledger.entries(self.store, 3, owner=2)
        self.assertEqual(life_ledger.summary(self.store, 1, owner=2)['expenses'], 100)
        with self.assertRaises(PermissionError):
            life_ledger.require_owner(self.store, 1, owner=2, write=True)
        with self.assertRaises(PermissionError):
            life_ledger.entries(self.store, 5)
        self.store.set_setting(feature_access.key(4), [])
        with self.assertRaises(PermissionError):
            life_ledger.entries(self.store, 4)
        for actor in (True, '2', 0, -1):
            with self.assertRaises(PermissionError):
                life_ledger.entries(self.store, actor)
        self.assertEqual(life_ledger.users(self.store, 1)[0]['user_id'], 2)
        with self.assertRaises(PermissionError):
            life_ledger.users(self.store, 2)

    def test_draft_private_allowlist_no_raw_or_fiscal_ids_and_expiry(self):
        draft = self.draft()
        self.assertEqual(draft['status'], 'draft')
        self.assertNotIn('SHOULD_NOT_PERSIST', dumps(draft))
        self.assertNotIn('PRIVATE_FISCAL_ID', dumps(draft))
        with self.assertRaises(ValueError):
            life_ledger.get_receipt_draft(self.store, 3, 'draft-token')
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01T00:00:00+00:00' WHERE user_id=2")
        with self.assertRaises(ValueError):
            life_ledger.get_receipt_draft(self.store, 2, 'draft-token')
        with self.assertRaises(ValueError):
            self.confirm()

    def test_draft_can_be_corrected_until_confirmed_without_fingerprint_change(self):
        self.draft(self.receipt(total=None, date=None))
        with self.assertRaises(ValueError):
            self.confirm()
        updated = self.draft()
        self.assertEqual(updated['result']['total'], 2200)
        with self.assertRaises(ValueError):
            self.draft(self.receipt(fingerprint='b'*64))
        confirmation = self.confirm()
        updated = self.draft(self.receipt(total=999))
        self.assertEqual(updated['status'], 'confirmed')
        self.assertEqual(updated['receipt_id'], confirmation['receipt_id'])
        self.assertEqual(updated['result']['total'], 2200)

    def test_confirm_creates_whole_expense_once_and_selected_discounted_wine(self):
        self.draft()
        result = self.confirm()
        self.assertFalse(result['already_imported'])
        self.assertEqual(result['total'], 2200)
        expense = life_ledger.entries(self.store, 2)[0]
        self.assertEqual((expense['amount'], expense['receipt_id'], expense['merchant']), (2200, result['receipt_id'], 'Магазин'))
        lot = wine_cellar.get_lot(self.store, 2, result['wine_lot_ids'][0])
        self.assertEqual((lot['quantity'], lot['purchase_price'], lot['volume_ml']), (2, 1000, 750))
        self.assertEqual(wine_cellar.portfolio(self.store, 2)['purchase_cost'], 2000)
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 2200)
        self.assertEqual(self.store.one('SELECT COUNT(*) AS n FROM wine_snapshots')['n'], 1)

    def test_receipt_never_automatically_adds_detected_wines(self):
        self.draft()
        result = self.confirm(selected=[])
        self.assertEqual(result['wine_lot_ids'], [])
        self.assertEqual(wine_cellar.lots(self.store, 2), [])
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 2200)

    def test_repeated_confirmation_and_different_file_same_fiscal_fingerprint_are_idempotent(self):
        self.draft()
        first = self.confirm()
        second = self.confirm()
        self.draft(token='other-file-token')
        third = self.confirm(token='other-file-token')
        self.assertTrue(second['already_imported'])
        self.assertTrue(third['already_imported'])
        self.assertEqual(first['receipt_id'], third['receipt_id'])
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)
        self.assertEqual(len(wine_cellar.lots(self.store, 2)), 1)
        self.assertEqual(life_ledger.get_receipt_draft(self.store, 2, 'other-file-token')['status'], 'confirmed')

    def test_same_fingerprint_different_users_is_private_independent_import(self):
        self.draft()
        first = self.confirm()
        self.draft(actor=3)
        second = self.confirm(actor=3)
        self.assertNotEqual(first['receipt_id'], second['receipt_id'])
        self.assertEqual(life_ledger.summary(self.store, 3)['expenses'], 2200)
        self.assertEqual(len(wine_cellar.lots(self.store, 3)), 1)

    def test_invalid_selected_item_or_duplicate_or_override_is_rejected(self):
        self.draft()
        for selected in ([0, 0], [-1], [True], [99], '0'):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.confirm(selected=selected)
        with self.assertRaises(ValueError):
            self.confirm(overrides={1: {'name': 'Other'}})
        with self.assertRaises(ValueError):
            self.confirm(overrides={0: {'purchase_price': 0}})
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)

    def test_unknown_volume_requires_explicit_override_unknown_vintage_stays_unknown(self):
        receipt = self.receipt()
        receipt['items'][0].update(volume_ml=None, vintage=None)
        self.draft(receipt)
        with self.assertRaises(ValueError):
            self.confirm()
        result = self.confirm(overrides={'0': {'volume_ml': 375, 'name': 'Manual review'}})
        lot = wine_cellar.get_lot(self.store, 2, result['wine_lot_ids'][0])
        self.assertEqual((lot['volume_ml'], lot['vintage'], lot['name']), (375, '', 'Manual review'))

    def test_fractional_wine_and_unreconciled_discounts_require_review(self):
        receipt = self.receipt()
        receipt['items'][0]['quantity'] = 1.5
        self.draft(receipt)
        with self.assertRaises(ValueError):
            self.confirm()
        self.draft(self.receipt(total=2100))
        with self.assertRaises(ValueError):
            self.confirm()
        receipt = self.receipt()
        receipt['items'][1]['amount'] = None
        self.draft(receipt)
        with self.assertRaises(ValueError):
            self.confirm()
        self.confirm(selected=[])
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 2200)

    def test_failed_second_wine_rolls_back_expense_first_lot_quotes_and_snapshots(self):
        receipt = self.receipt()
        receipt['items'][1] = {**receipt['items'][0], 'amount': 200, 'quantity': 1, 'name': 'Second'}
        self.draft(receipt)
        with self.assertRaises(ValueError):
            self.confirm(selected=[0, 1], overrides={1: {'name': ''}})
        for table in ('life_ledger_entries', 'life_receipt_imports', 'wine_lots', 'wine_events', 'wine_prices', 'wine_snapshots'):
            self.assertEqual(self.store.one('SELECT COUNT(*) AS n FROM '+table)['n'], 0, table)
        self.assertEqual(life_ledger.get_receipt_draft(self.store, 2, 'draft-token')['status'], 'draft')
        self.confirm(selected=[0, 1])
        self.assertEqual(len(wine_cellar.lots(self.store, 2)), 2)

    def test_revoked_feature_or_user_denies_confirmation_without_partial_expense(self):
        self.draft()
        features = feature_access.assigned(self.store, 2)
        self.store.set_setting(feature_access.key(2), [name for name in features if name != 'wine_cellar'])
        with self.assertRaises(PermissionError):
            self.confirm()
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 0)
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        with self.assertRaises(PermissionError):
            self.confirm(selected=[])
        self.assertEqual(self.store.one('SELECT COUNT(*) AS n FROM life_ledger_entries')['n'], 0)

    def test_finance_only_user_can_record_receipt_but_cannot_add_wine(self):
        self.draft(actor=4)
        with self.assertRaises(PermissionError):
            self.confirm(actor=4)
        self.confirm(actor=4, selected=[])
        self.assertEqual(life_ledger.summary(self.store, 4)['expenses'], 2200)

    def test_rounding_is_disclosed_but_expense_grand_total_stays_exact(self):
        receipt = self.receipt(total=2200.01)
        receipt['items'][0]['amount'] = 2000.01
        self.draft(receipt)
        result = self.confirm()
        self.assertEqual(result['rounding_difference_cents'], 1)
        self.assertEqual(life_ledger.summary(self.store, 2)['expenses'], 2200.01)

    def test_draft_cleanup_never_mutates_other_users(self):
        self.draft(actor=3)
        self.store.execute("UPDATE life_receipt_drafts SET expires_at='2000-01-01T00:00:00+00:00' WHERE user_id=3")
        self.draft()
        self.assertTrue(self.store.one('SELECT 1 FROM life_receipt_drafts WHERE user_id=3'))

    def test_receipt_operation_collision_and_invalid_fingerprint(self):
        self.draft()
        self.confirm(operation_id='one-operation')
        self.draft(self.receipt(fingerprint='b'*64), token='another')
        with self.assertRaises(ValueError):
            self.confirm(token='another', operation_id='one-operation')
        with self.assertRaises(ValueError):
            self.draft(self.receipt(fingerprint='invalid'))
        self.assertEqual(life_ledger.summary(self.store, 2)['count'], 1)


if __name__ == '__main__':
    unittest.main()
