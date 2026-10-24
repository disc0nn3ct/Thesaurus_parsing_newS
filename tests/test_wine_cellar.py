"""Offline personal stock, dated valuation and opt-in Vivino reconciliation."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from thesaurus import wine_cellar as cellar
from thesaurus.feature_access import key
from thesaurus.store import Store


class WineCellarTests(unittest.TestCase):
    def test_consumption_freezes_gain_and_remaining_portfolio_changes_independently(self):
        lot = self.add(quantity=3)
        quote = cellar.record_price(self.store, 2, lot['id'], 1200, '2024-04-01', source='shop')
        cellar.consume_bottles(self.store, 2, lot['id'], 1, '2024-04-02', price_id=quote['id'], operation_id='drink-a')
        cellar.record_price(self.store, 2, lot['id'], 900, source='manual')
        closed = cellar.consumptions(self.store, 2)[0]
        self.assertEqual((closed['purchase_price'], closed['valuation_price'], closed['result']), (1000, 1200, 200))
        self.assertEqual(closed['price_date'], '2024-04-01')
        values = cellar.portfolio(self.store, 2)
        self.assertEqual((values['purchased_quantity'], values['consumed_quantity'], values['quantity']), (3, 1, 2))
        self.assertEqual((values['open_result'], values['closed_result'], values['combined_result']), (-200, 200, 0))
        self.assertEqual(values['total_purchase_cost'], 3000)
        self.assertFalse(self.store.query('SELECT * FROM life_ledger_entries'))
        cellar.consume_bottles(self.store, 2, lot['id'], 2, price=900, operation_id='drink-b')
        self.assertEqual(cellar.portfolio(self.store, 2)['closed_result'], 0)
        self.assertEqual(cellar.purchase_history(self.store, 2)[0]['consumed_quantity'], 3)
        self.assertEqual(cellar.purchase_history(self.store, 2)[0]['quantity'], 0)

    def test_no_market_quote_or_unknown_purchase_does_not_create_zero_profit(self):
        lot = self.add(quantity=1)
        self.assertIsNone(cellar.consumption_preview(self.store, 2, lot['id'])['valuation_price'])
        cellar.consume_bottles(self.store, 2, lot['id'], operation_id='no-quote')
        self.assertIsNone(cellar.consumptions(self.store, 2)[0]['result'])
        other = self.add(quantity=1, purchase_price=None)
        cellar.consume_bottles(self.store, 2, other['id'], price=500, operation_id='no-cost')
        values = cellar.portfolio(self.store, 2)
        self.assertEqual(values['closed_unknown_quantity'], 2)
        self.assertIsNone(values['closed_result'])
        self.assertIsNone(values['combined_result'])

    def test_explicit_zero_valuation_is_a_loss_and_retries_cannot_change_frozen_price(self):
        lot = self.add(quantity=1)
        kwargs = {'price': 0, 'operation_id': 'zero-price'}
        cellar.consume_bottles(self.store, 2, lot['id'], **kwargs)
        cellar.consume_bottles(self.store, 2, lot['id'], **kwargs)
        self.assertEqual(len(cellar.consumptions(self.store, 2)), 1)
        self.assertEqual(cellar.consumptions(self.store, 2)[0]['result'], -1000)
        with self.assertRaises(ValueError):
            cellar.consume_bottles(self.store, 2, lot['id'], price=1, operation_id='zero-price')

    def test_consumption_date_and_quote_must_belong_to_owned_bottle_and_valid_period(self):
        lot = self.add(quantity=1)
        other = self.add(actor=3)
        foreign_quote = cellar.record_price(self.store, 3, other['id'], 1500)
        for kwargs in ({'consumed_on': '2024-02-01'}, {'price_id': foreign_quote['id']}, {'quantity': 2},
                       {'price': float('nan')}, {'unpriced': True, 'price': 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cellar.consume_bottles(self.store, 2, lot['id'], **kwargs)
        quote = cellar.record_price(self.store, 2, lot['id'], 1200, '2024-05-01')
        with self.assertRaises(ValueError):
            cellar.consume_bottles(self.store, 2, lot['id'], consumed_on='2024-04-01', price_id=quote['id'])
        with self.assertRaises(ValueError):
            cellar.consume_bottles(self.store, 3, lot['id'], price=1500)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        self.assertFalse(cellar.consumptions(self.store, 2))

    def test_explicit_unpriced_confirmation_stays_unpriced_if_quote_arrives_later(self):
        lot = self.add(quantity=1)
        cellar.record_price(self.store, 2, lot['id'], 1200)
        cellar.consume_bottles(self.store, 2, lot['id'], unpriced=True, operation_id='no-valuation')
        cellar.record_price(self.store, 2, lot['id'], 1500)
        self.assertIsNone(cellar.consumptions(self.store, 2)[0]['valuation_price'])

    def test_consumption_is_persistent_private_and_admin_can_only_read_others(self):
        lot = self.add(quantity=1)
        cellar.consume_bottles(self.store, 2, lot['id'], price=1250, operation_id='persisted')
        self.store = Store(self.path)
        self.assertEqual(cellar.consumptions(self.store, 2)[0]['result'], 250)
        self.assertEqual(cellar.consumptions(self.store, 1, owner=2)[0]['result'], 250)
        with self.assertRaises(PermissionError):
            cellar.consumptions(self.store, 3, owner=2)
        self.assertEqual(cellar.purchase_history(self.store, 3), [])
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        with self.assertRaises(PermissionError):
            cellar.consume_bottles(self.store, 2, lot['id'], price=100)

    def test_upgrade_preserves_stock_and_uses_only_quotes_known_before_legacy_consumption(self):
        lot = self.add(quantity=2)
        old = cellar.record_price(self.store, 2, lot['id'], 1200, '2024-04-01')
        self.store.execute("UPDATE wine_prices SET retrieved_at='2024-04-01T00:00:00+00:00' WHERE id=?", (old['id'],))
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'consumed', 'old-consumption')
        self.store.execute("UPDATE wine_events SET created_at='2024-04-02T00:00:00+00:00' WHERE operation_id='old-consumption'")
        later = cellar.record_price(self.store, 2, lot['id'], 9999, '2024-04-01')
        self.store.execute("UPDATE wine_prices SET retrieved_at='2024-05-01T00:00:00+00:00' WHERE id=?", (later['id'],))
        before = {table: self.store.one('SELECT COUNT(*) AS n FROM '+table)['n']
                  for table in ('wine_lots', 'wine_events', 'wine_prices', 'wine_snapshots', 'users')}
        self.store.execute('DROP TABLE wine_consumptions')
        self.store.set_setting('wine_cellar_version', 1)
        cellar.migrate(self.store)
        after = {table: self.store.one('SELECT COUNT(*) AS n FROM '+table)['n'] for table in before}
        self.assertEqual(before, after)
        self.assertEqual(self.store.get_setting('wine_cellar_version'), cellar.VERSION)
        self.assertEqual(cellar.consumptions(self.store, 2)[0]['valuation_price'], 1200)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 1)
        cellar.migrate(self.store)
        self.assertEqual(len(cellar.consumptions(self.store, 2)), 1)

    def test_failed_upgrade_rolls_back_and_missing_historical_quote_stays_unknown(self):
        lot = self.add(quantity=1)
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'consumed', 'old-unknown')
        self.store.execute("UPDATE wine_events SET created_at='2024-04-02T00:00:00+00:00' WHERE operation_id='old-unknown'")
        cellar.record_price(self.store, 2, lot['id'], 1500, price_date='2024-04-01')
        self.store.execute('DROP TABLE wine_consumptions')
        self.store.set_setting('wine_cellar_version', 1)
        with patch.object(cellar, 'CONSUMPTION_SCHEMA', cellar.CONSUMPTION_SCHEMA+'; INVALID SQL'), self.assertRaises(sqlite3.DatabaseError):
            cellar.migrate(self.store)
        self.assertFalse(self.store.one("SELECT name FROM sqlite_master WHERE name='wine_consumptions'"))
        self.assertEqual(self.store.get_setting('wine_cellar_version'), 1)
        cellar.migrate(self.store)
        self.assertIsNone(cellar.consumptions(self.store, 2)[0]['result'])

    def test_shop_value_change_uses_remaining_bottles_and_keeps_finance_unchanged(self):
        self.store.set_setting('wine_price_allowed_hosts', ['shop.example'])
        lot = self.add(quantity=3, purchase_price=1000)
        self.assertIsNone(cellar.portfolio(self.store, 2)['shop_price_change'])
        cellar.record_price(self.store, 2, lot['id'], 1200, source='shop', url='https://shop.example/wine')
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'one-drunk')
        values = cellar.portfolio(self.store, 2)
        self.assertEqual(values['shop_compared_quantity'], 2)
        self.assertEqual(values['shop_price_change'], 400)
        self.assertEqual(values['shop_price_change_pct'], 20)
        self.assertEqual(len(values['shop_price_dates']), 1)
        self.assertEqual(self.store.query('SELECT * FROM life_ledger_entries'), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'cellar.sqlite3'
        self.store = Store(self.path)
        self.store.add_user(1, 'admin')
        self.store.add_user(2, subscriptions=['vivino'])
        self.store.add_user(3, subscriptions=['vivino'])
        self.store.add_user(4)
        self.fields = {'name': 'Reserva', 'winery': 'Bodega Test', 'vintage': 2020,
                       'quantity': 3, 'purchase_price': 1000, 'purchase_date': '2024-03-01'}

    def add(self, actor=2, **changes):
        return cellar.add_bottle(self.store, actor, {**self.fields, **changes})

    def review(self, **changes):
        return {'name': 'Reserva', 'winery': 'Bodega Test', 'vintage': '2020',
                'review': 'Приятное вино', 'rating': 4, **changes}

    def test_migration_backup_is_idempotent_and_version_checked(self):
        backups = list(self.path.parent.glob('*.wine-backup-*.sqlite3'))
        self.assertEqual(len(backups), 1)
        self.add()
        cellar.migrate(self.store)
        self.assertEqual(len(list(self.path.parent.glob('*.wine-backup-*.sqlite3'))), 1)
        with closing(sqlite3.connect(backups[0])) as backup:
            self.assertFalse(backup.execute("SELECT name FROM sqlite_master WHERE name='wine_lots'").fetchone())
        self.store.set_setting('wine_cellar_version', 99)
        with self.assertRaises(RuntimeError):
            cellar.migrate(self.store)

    def test_failed_migration_rolls_back_schema_and_settings(self):
        other_path = self.path.parent/'legacy.sqlite3'
        # Simulate a database predating both the cellar and operational history.
        with patch.object(cellar, 'migrate'), patch('thesaurus.monitoring.migrate'):
            legacy = Store(other_path)
        with patch.object(cellar, 'SCHEMA', cellar.SCHEMA+'; THIS IS INVALID SQL'), self.assertRaises(sqlite3.DatabaseError):
            cellar.migrate(legacy)
        self.assertFalse(legacy.one("SELECT name FROM sqlite_master WHERE name='wine_lots'"))
        self.assertIsNone(legacy.get_setting('wine_cellar_version'))
        cellar.migrate(legacy)
        self.assertEqual(legacy.get_setting('wine_cellar_version'), cellar.VERSION)

    def test_migration_baselines_existing_vivino_import(self):
        other_path = self.path.parent/'legacy.sqlite3'
        with patch.object(cellar, 'migrate'), patch('thesaurus.monitoring.migrate'):
            legacy = Store(other_path)
        legacy.add_user(2, subscriptions=['vivino'])
        legacy.set_setting('project:vivino:2:dataset', {'rows': [self.review()], 'generation': 'old'})
        cellar.migrate(legacy)
        cellar.add_bottle(legacy, 2, self.fields)
        self.assertEqual(cellar.sync_reviews(legacy, 2, [self.review()], 'new'), [])
        self.assertEqual(cellar.lots(legacy, 2)[0]['quantity'], 3)

    def test_no_feature_no_access_disabled_no_access_and_numeric_actor(self):
        with self.assertRaises(PermissionError):
            cellar.lots(self.store, 4)
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        with self.assertRaises(PermissionError):
            self.add()
        with self.assertRaises(PermissionError):
            cellar.require_owner(self.store, True)

    def test_owner_privacy_and_admin_read_only_other_collections(self):
        own = self.add()
        self.assertEqual(cellar.lots(self.store, 3), [])
        with self.assertRaises(PermissionError):
            cellar.lots(self.store, 3, owner=2)
        with self.assertRaises(ValueError):
            cellar.get_lot(self.store, 3, own['id'])
        self.assertEqual(cellar.get_lot(self.store, 1, own['id'], owner=2)['name'], 'Reserva')
        with self.assertRaises(PermissionError):
            cellar.require_owner(self.store, 1, owner=2, write=True)
        with self.assertRaises(ValueError):
            cellar.adjust_stock(self.store, 1, own['id'], -1, 'consume', 'admin-other')
        self.assertEqual(cellar.owners(self.store, 1), [{'user_id': 2, 'quantity': 3, 'lot_count': 1}])
        with self.assertRaises(PermissionError):
            cellar.owners(self.store, 2)

    def test_add_purchase_quote_and_idempotent_retry(self):
        lot = cellar.add_bottle(self.store, 2, self.fields, 'add-once')
        again = cellar.add_bottle(self.store, 2, self.fields, 'add-once')
        self.assertEqual(lot, again)
        self.assertEqual(lot['latest_price']['source'], 'purchase')
        self.assertEqual(lot['latest_price']['amount'], 1000)
        self.assertEqual(lot['volume_ml'], 750)
        self.assertEqual(len(cellar.valuation_history(self.store, 2)), 1)
        with self.assertRaises(ValueError):
            cellar.add_bottle(self.store, 2, {**self.fields, 'name': 'Different'}, 'add-once')
        with self.assertRaises(ValueError):
            cellar.add_bottle(self.store, 2, {**self.fields, 'quantity': 5}, 'add-once')

    def test_stock_never_negative_retry_not_double_consumption(self):
        lot = self.add(quantity=1)
        result = cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'consume-once')
        self.assertEqual(result['quantity'], 0)
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'consume-once')
        self.assertEqual(cellar.lots(self.store, 2), [])
        self.assertEqual(len(cellar.lots(self.store, 2, include_empty=True)), 1)
        with self.assertRaises(ValueError):
            cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'consume-again')
        with self.assertRaises(ValueError):
            cellar.adjust_stock(self.store, 2, lot['id'], 1, 'buy', 'consume-once')
        self.assertEqual(len(self.store.query('SELECT * FROM wine_events')), 2)
        self.assertEqual(len(cellar.valuation_history(self.store, 2)), 2)

    def test_validation_unknown_prices_not_zero_and_only_rubles(self):
        for changes in ({'quantity': True}, {'quantity': -1}, {'quantity': 1.2}, {'volume_ml': 0},
                        {'purchase_price': float('nan')}, {'purchase_price': float('inf')},
                        {'purchase_price': -1}, {'currency': 'USD'}, {'vintage': 100},
                        {'purchase_date': '2999-01-01'}, {'name': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.add(**changes)
        lot = self.add(purchase_price=None, vintage='NV')
        self.assertIsNone(lot['purchase_price'])
        result = cellar.portfolio(self.store, 2)
        self.assertIsNone(result['current_value'])
        self.assertEqual(result['known_current_value'], 0)
        self.assertIsNone(result['purchase_cost'])
        self.assertEqual(result['unpriced_quantity'], 3)
        self.assertIsNone(result['net_flows'])
        self.assertEqual(self.add(actor=3, purchase_price='100.125')['purchase_price'], 100.13)

    def test_prices_are_dated_idempotent_and_metadata_retained(self):
        lot = self.add()
        price = cellar.record_price(self.store, 2, lot['id'], 1200, '2025-01-01', 'shop',
                                   operation_id='quote-once', title='Reserva', vintage=2020, volume_ml=750,
                                   observed_at='2025-01-01T12:00:00+00:00')
        self.assertEqual(price['metadata']['title'], 'Reserva')
        cellar.record_price(self.store, 2, lot['id'], 1200, '2025-01-01', 'shop', operation_id='quote-once')
        self.assertEqual(len(cellar.valuation_history(self.store, 2)), 2)
        cellar.record_price(self.store, 2, lot['id'], 800, '2024-06-01', 'manual')
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['latest_price']['amount'], 1200)
        for kwargs in ({'available': False}, {'vintage': 2021}, {'volume_ml': 1500}, {'currency': 'EUR'}, {'price_date': '2999-01-01'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cellar.record_price(self.store, 2, lot['id'], 1500, **kwargs)
        with self.assertRaises(ValueError):
            cellar.record_price(self.store, 2, lot['id'], 1900, '2025-01-01', operation_id='quote-once')

    def test_cost_stock_flow_and_price_movement_are_separate(self):
        lot = self.add(quantity=2, purchase_price=1000)
        cellar.record_price(self.store, 2, lot['id'], 1200)
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'drink1')
        self.add(name='Another', quantity=1, purchase_price=3000)
        result = cellar.portfolio(self.store, 2)
        self.assertEqual(result['quantity'], 2)
        self.assertEqual(result['purchase_cost'], 4000)
        self.assertEqual(result['current_value'], 4200)
        self.assertEqual(result['price_change'], 200)
        self.assertEqual(result['net_flows'], 4000)
        self.assertEqual([r['amount'] for r in result['price_movement']], [400, 0, 0])
        self.assertEqual([e['flow'] for e in result['events']], [2000, -1000, 3000])
        self.assertEqual(cellar.valuation_history(self.store, 1, owner=2)[-1]['value'], 4200)

    def test_zero_price_is_known_price_and_partial_portfolio_remains_partial(self):
        self.add(name='Gift', quantity=1, purchase_price=0)
        unknown = self.add(name='Unknown', quantity=1, purchase_price=None)
        result = cellar.portfolio(self.store, 2)
        self.assertEqual(result['priced_quantity'], 1)
        self.assertIsNone(result['current_value'])
        cellar.record_price(self.store, 2, unknown['id'], 500)
        result = cellar.portfolio(self.store, 2)
        self.assertEqual(result['current_value'], 500)
        self.assertIsNone(result['purchase_cost'])
        self.assertIsNone(result['price_change'])

    def test_missing_purchase_date_is_not_invented(self):
        lot = self.add(purchase_date=None)
        self.assertIsNone(lot['purchase_date'])
        self.assertEqual(lot['latest_price']['source'], 'purchase_undated')
        self.assertTrue(lot['latest_price']['metadata']['purchase_date_unknown'])

    def test_quote_offer_metadata_is_bounded_and_allowlisted(self):
        lot = self.add()
        metadata = {'selection': 'lowest_matched_available_offer', 'checked_offer_count': 2,
                    'currency': 'RUB', 'availability': 'InStock', 'access_token': 'do-not-store',
                    'candidate_offers': [{'amount': 1200, 'currency': 'RUB', 'access_token': 'omit'}]}
        price = cellar.record_price(self.store, 2, lot['id'], 1200, metadata=metadata)
        self.assertEqual(price['metadata']['checked_offer_count'], 2)
        self.assertNotIn('access_token', price['metadata'])
        self.assertNotIn('access_token', price['metadata']['candidate_offers'][0])
        for invalid in ({'currency': 'USD'}, {'availability': 'OutOfStock'}, {'title': 'x'*40_000},
                        {'amount': float('nan')}, {'candidate_offers': [{}]*101}):
            with self.subTest(invalid=list(invalid)), self.assertRaises(ValueError):
                cellar.record_price(self.store, 2, lot['id'], 1500, metadata=invalid)

    def test_revocation_blocks_all_mutations_and_read(self):
        lot = self.add()
        self.store.set_setting(key(2), ['vivino_map'])
        for action in (lambda: cellar.get_lot(self.store, 2, lot['id']),
                       lambda: cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'x'),
                       lambda: cellar.record_price(self.store, 2, lot['id'], 300),
                       lambda: cellar.sync_reviews(self.store, 2, [], 'generation'),
                       lambda: cellar.snapshot(self.store, 2), lambda: cellar.export_rows(self.store, 2)):
            with self.assertRaises(PermissionError):
                action()
        self.assertEqual(cellar.get_lot(self.store, 1, lot['id'], owner=2)['quantity'], 3)

    def test_first_sync_is_baseline_no_automatic_consumption(self):
        self.add()
        self.assertEqual(cellar.sync_reviews(self.store, 2, [self.review()], 'baseline'), [])
        self.assertEqual(cellar.sync_reviews(self.store, 2, [self.review(review='Edited review')], 'new'), [])
        self.assertEqual(cellar.lots(self.store, 2)[0]['quantity'], 3)

    def test_new_review_proposes_one_bottle_and_confirmation_is_idempotent(self):
        lot = self.add()
        cellar.sync_reviews(self.store, 2, [self.review(review='', rating=None)], 'baseline')
        proposed = cellar.sync_reviews(self.store, 2, [self.review(), self.review()], 'new')
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]['lot_ids'], [lot['id']])
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 3)
        self.assertEqual(cellar.resolve_proposal(self.store, 2, proposed[0]['id'], True)['state'], 'consumed')
        cellar.resolve_proposal(self.store, 2, proposed[0]['id'], True)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 2)
        self.assertEqual(cellar.sync_reviews(self.store, 2, [self.review()], 'newer'), [])

    def test_review_exact_producer_vintage_matching_and_user_isolation(self):
        self.add()
        self.add(actor=3)
        cellar.sync_reviews(self.store, 2, [], 'baseline')
        self.assertEqual(cellar.sync_reviews(self.store, 2, [self.review(vintage='2021'), self.review(winery='Other')], 'new'), [])
        proposed = cellar.sync_reviews(self.store, 2, [self.review(name='  RESERVA  ', winery='BODEGA TEST')], 'new2')
        self.assertEqual(len(proposed), 1)
        self.assertEqual(cellar.proposals(self.store, 3), [])
        with self.assertRaises(ValueError):
            cellar.resolve_proposal(self.store, 3, proposed[0]['id'], True)

    def test_ambiguous_lots_require_explicit_selection(self):
        first, second = self.add(), self.add(location='Second shelf')
        cellar.sync_reviews(self.store, 2, [], 'baseline')
        proposal = cellar.sync_reviews(self.store, 2, [self.review()], 'new')[0]
        with self.assertRaises(ValueError):
            cellar.resolve_proposal(self.store, 2, proposal['id'], True)
        cellar.resolve_proposal(self.store, 2, proposal['id'], second['id'])
        self.assertEqual(cellar.get_lot(self.store, 2, first['id'])['quantity'], 3)
        self.assertEqual(cellar.get_lot(self.store, 2, second['id'])['quantity'], 2)

    def test_dismissed_or_depleted_proposals_do_not_subtract_twice(self):
        lot = self.add(quantity=1)
        cellar.sync_reviews(self.store, 2, [], 'baseline')
        proposal = cellar.sync_reviews(self.store, 2, [self.review()], 'new')[0]
        cellar.adjust_stock(self.store, 2, lot['id'], -1, 'drink', 'manual1')
        with self.assertRaises(ValueError):
            cellar.resolve_proposal(self.store, 2, proposal['id'], True)
        self.assertEqual(cellar.resolve_proposal(self.store, 2, proposal['id'], False)['state'], 'dismissed')
        cellar.resolve_proposal(self.store, 2, proposal['id'], True)
        self.assertEqual(cellar.get_lot(self.store, 2, lot['id'])['quantity'], 0)
        self.assertEqual(cellar.proposals(self.store, 2), [])

    def test_public_link_validation_and_allowlist_handoff(self):
        lot = self.add()
        for url in ('http://shop.example/wine', 'https://127.0.0.1/wine', 'https://192.168.1.1/wine',
                    'https://user:password@shop.example/wine', 'https://localhost/wine', 'https://[::1]/wine'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                cellar.validate_price_url(url)
        self.assertEqual(cellar.validate_price_url('https://shop.example/wine#details'), 'https://shop.example/wine')
        self.store.set_setting('wine_price_allowed_hosts', ['shop.example'])
        with patch('thesaurus.wine_tools.validate_product_url', return_value='https://shop.example/wine') as validate:
            first = cellar.add_price_link(self.store, 2, lot['id'], 'https://shop.example/wine')
            second = cellar.add_price_link(self.store, 2, lot['id'], 'https://shop.example/wine')
            self.assertEqual(first['id'], second['id'])
            self.assertEqual(validate.call_args.kwargs['allowed_hosts'], ['shop.example'])
        self.assertEqual(cellar.price_links(self.store, 1, lot['id'], owner=2)[0]['url'], 'https://shop.example/wine')

    def test_export_contains_units_dated_provenance_and_no_other_owner(self):
        self.add(name='=HYPERLINK("url")')
        self.add(actor=3, name='Private')
        rows = cellar.export_rows(self.store, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['currency'], 'RUB')
        self.assertEqual(rows[0]['price_source'], 'purchase')
        self.assertEqual(rows[0]['price_date'], '2024-03-01')
        self.assertEqual(rows[0]['volume_ml'], 750)
        # Export renderer adds formula protection, preserving original data here.
        self.assertTrue(rows[0]['name'].startswith('='))


if __name__ == '__main__':
    unittest.main()
