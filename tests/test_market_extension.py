"""Long horizons, catalogue navigation and staged integrity checks: offline only."""
import json
import shutil
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pandas as pd

from thesaurus import (
    integrity,
    market,
    market_analysis,
    market_incremental,
    market_sources,
    market_store,
)
from thesaurus.charts import run_chart_job
from thesaurus.config import Settings
from thesaurus.forecast_horizons import targets, validate
from thesaurus.source_catalog import GROUPS, browse
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_charts import ChartControls


def meta(identity='test.rate', frequency='monthly', **extra):
    return {'id': identity, 'dataset': identity, 'source': 'cbr', 'title': identity, 'kind': 'series',
            'frequency': frequency, 'calendar': 'calendar', 'unit': '%', 'adapter': 'currency',
            'url': 'https://cbr.ru/hd_base/keyrate/', 'aggregation': 'last', **extra}


def data(n=80, frequency='monthly'):
    return [{'date': d.date().isoformat(), 'value': 10+i*.01, 'retrieved_at': '2026-10-01T00:00:00+00:00'}
            for i, d in enumerate(pd.date_range('2018-01-01', periods=n, freq={'daily': 'D', 'monthly': 'MS', 'quarterly': 'QS', 'annual': 'YS'}[frequency]))]


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('extension-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.settings = Settings(root=self.root)
        market_store.register(self.store, [meta()])
        market.enable(self.store, 'test.rate', True)
        self.reply = Mock()
        self.controls = ChartControls(self.store, self.reply)

    def test_duration_limits_and_types(self):
        for spec in ({'value': 24, 'unit': 'months'}, {'value': 104, 'unit': 'weeks'}, {'value': 8, 'unit': 'quarters'}, {'value': 2, 'unit': 'years'}):
            self.assertEqual(validate(spec), spec)
        for spec in ({'value': 25, 'unit': 'months'}, {'value': True, 'unit': 'months'}, {'value': -1, 'unit': 'weeks'}, {'value': 1, 'unit': 'days'}):
            with self.assertRaises(ValueError):
                validate(spec)

    def test_administrative_value_last_changed_years_ago_still_has_current_scenario(self):
        predictions = market_analysis.forecast([{'date': '2020-01-01', 'value': .9}],
            meta(frequency='daily', administrative=True, calendar='events'), created_at='2026-10-07',
            horizon={'value': 24, 'unit': 'months'})
        self.assertEqual(predictions[-1]['target'], '2028-10-07')
        self.assertTrue(all(p['prediction'] == .9 and p['lower'] is None for p in predictions))
        self.assertEqual(predictions[0]['origin'], '2020-01-01')

    def test_monthly_administrative_rate_has_no_statistical_interval_or_validation(self):
        rows = data(156)
        predictions = market_analysis.forecast(rows, meta(administrative=True), created_at=rows[-1]['date'],
            horizon={'value': 24, 'unit': 'months'})
        self.assertEqual(len(predictions), 24)
        self.assertTrue(all(p['prediction'] == rows[-1]['value'] and p['validation_n'] == 0 for p in predictions))
        self.assertTrue(all(p['lower'] is None and p['upper'] is None and p['validation_mae'] is None for p in predictions))
        self.assertTrue(all(p['note'] == 'Сценарий сохранения административного значения' for p in predictions))
        # Publication dates and the cutoff still control which last value can be used.
        hidden = {'date': '2099-01-01', 'value': 99, 'published_at': '2099-01-01'}
        self.assertEqual(market_analysis.forecast(rows+[hidden], meta(administrative=True), created_at=rows[-1]['date'],
            horizon={'value': 24, 'unit': 'months'}), predictions)

    def test_legacy_daily_wrapper_respects_known_publication_delay(self):
        from thesaurus.rate_forecasts import forecast_daily
        rows = [{**row, 'source': 'ruonia', 'metric': '1 мес', 'frequency': 'daily', 'unit': '%',
                 'published_at': '2099-01-01'} for row in data(100, 'daily')]
        self.assertEqual(forecast_daily(rows, created_at='2026-10-07'), [])

    def test_transfers_preserve_quarters_units_and_archive_status(self):
        item = next(x for x in market_sources.catalogue() if x['id'] == 'cbr.hd_base.tg.1.CO_total_sum')
        transport = Mock()
        transport.request.return_value = SimpleNamespace(text='<table><tr><th>Страна</th><th>I квартал 2021 г.</th><th>II квартал 2021 г.</th></tr><tr><td>Всего по странам</td><td>1 234,5</td><td>—</td></tr></table>')
        batches = market_sources.fetch(item, transport, date(2006, 1, 1), date(2026, 10, 7))
        child, rows = batches[0]
        self.assertEqual(child['data_type'], 'archive')
        self.assertEqual(child['unit'], 'млн долларов США')
        self.assertEqual(rows[0]['period_end'], '2021-03-31')
        self.assertEqual(rows[0]['value'], 1234.5)
        self.assertIsNone(rows[1]['value'])
        self.assertEqual(transport.request.call_args.kwargs['data']['ToDateKv'], '4.2021')

    def test_monthly_miacr_sends_source_form_and_keeps_four_tables_distinct(self):
        item = next(x for x in market_sources.catalogue() if x['id'] == 'cbr.hd_base.mkr.mkr_monthes')
        header = '<tr>' + ''.join('<th>' + column + '</th>' for column in item['contracts'][0]['headers']) + '</tr>'
        body = '<tr><td>сентябрь 2026 г.</td>' + '<td>14,00</td>'*6 + '</tr>'
        transport = Mock()
        transport.get.return_value = SimpleNamespace(text=('<h2>Значения за выбранные даты</h2><table>' + header + body + '</table>')*4)
        batches = market_sources.fetch(item, transport, date(2026, 9, 1), date(2026, 10, 7))
        self.assertEqual(len({child['id'] for child, _ in batches}), 24)
        params = transport.get.call_args.args[1]
        self.assertEqual(params['UniDbQuery.From'], '09.2026')
        self.assertEqual(params['UniDbQuery.st'], ['SF', 'HR', 'MB'])

    def test_reference_approval_detects_conflicting_review_and_audits_accepted_version(self):
        import hashlib
        item = meta('test.reference', kind='reference', adapter='html')
        market_store.register(self.store, [item])
        def snapshot(value):
            payload = {'tables': [[['Код', 'Описание'], ['1', value]]], 'links': [], 'headings': [], 'url': item['url']}
            return {'dataset': item['id'], 'version': hashlib.sha256(dumps(payload).encode()).hexdigest(), 'payload': payload}
        original, first, second = map(snapshot, ['исходный', 'первый', 'второй'])
        self.store.execute('INSERT INTO source_snapshots(dataset,version,received_at,payload) VALUES(?,?,?,?)',
                           (item['id'], original['version'], utcnow(), dumps(original['payload'])))
        one = integrity.incident(self.store, 'source_reference_revision', series_id=item['id'], old=original, new=first)
        two = integrity.incident(self.store, 'source_reference_revision', series_id=item['id'], old=original, new=second)
        integrity.resolve(self.store, 1, one, 'accept')
        with self.assertRaises(ValueError):
            integrity.resolve(self.store, 1, two, 'accept')
        check = integrity.enqueue(self.store, 1, [item['id']])
        with patch('thesaurus.integrity.sources.fetch', return_value=[({**item, '_snapshot': first['payload']}, [])]):
            integrity.run(self.store, {'actor_id': 1, 'check_id': check})
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_incidents')['n'], 2)

    def test_forecast_csv_distinguishes_facts_and_escapes_model_text(self):
        import csv

        from thesaurus.exporting import export_data
        path = self.root / 'forecast.csv'
        export_data(self.store, path, forecast_rows=[{'date': '2028-01-01', 'record_type': 'forecast',
                    'model': '=external()', 'value': -12.5, 'quality': 'forecast'}])
        with path.open(encoding='utf-8-sig', newline='') as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row['record_type'], 'forecast')
        self.assertEqual(row['model'], "'=external()")
        self.assertEqual(row['value'], '-12.5')

    def test_calendar_targets_and_sparse_daily_long_forecast(self):
        output = targets('2024-02-29', meta(frequency='daily'), {'value': 24, 'unit': 'months'})
        self.assertEqual(output[-1][1].date(), date(2026, 2, 28))
        self.assertLess(len(output), 40)
        business = targets('2026-09-30', meta(frequency='daily', calendar='russia_working'), {'value': 4, 'unit': 'weeks'})
        from thesaurus.rate_forecasts import working_dates
        self.assertTrue(all(working_dates(day, day) for _, day in business))

    def test_monthly_quarterly_annual_forecasts(self):
        for frequency, count, spec in [('monthly', 80, {'value': 24, 'unit': 'months'}),
                                        ('quarterly', 30, {'value': 8, 'unit': 'quarters'}),
                                        ('annual', 6, {'value': 2, 'unit': 'years'})]:
            rows = data(count, frequency)
            result = market_analysis.forecast(rows, meta(frequency=frequency), created_at=rows[-1]['date'], horizon=spec)
            self.assertEqual(len(result), spec['value'])
            self.assertTrue(all(p['target'] > rows[-1]['date'] for p in result))
            for p in result:
                if p['validation_n'] < 10:
                    self.assertEqual(p['model'], 'last')
                    self.assertIsNone(p['lower'])

    def test_long_horizon_cannot_see_future_values_or_publications(self):
        rows = data()
        cutoff = rows[65]['date']
        expected = market_analysis.forecast(rows, meta(), created_at=cutoff, horizon={'value': 12, 'unit': 'months'})
        changed = [{**row, 'value': 9999} if row['date'] > cutoff else row for row in rows]
        self.assertEqual(expected, market_analysis.forecast(changed, meta(), created_at=cutoff, horizon={'value': 12, 'unit': 'months'}))
        self.assertEqual([], market_analysis.forecast([{**r, 'published_at': '2099-01-01'} for r in rows], meta(), created_at=cutoff))

    def test_revision_and_duration_create_immutable_forecast_runs(self):
        rows = data()
        for row, stamp in zip(rows, pd.date_range(end=pd.Timestamp(utcnow()[:10]).replace(day=1), periods=len(rows), freq='MS')):
            row['date'] = stamp.date().isoformat()
        market_store.save_rows(self.store, 'test.rate', rows)
        series = self.store.one('SELECT * FROM market_series WHERE id=?', ('test.rate',))
        first = market.save_forecasts(self.store, series, rows, horizon={'value': 3, 'unit': 'months'})
        self.assertEqual(first, market.save_forecasts(self.store, series, rows, horizon={'value': 3, 'unit': 'months'}))
        market.save_forecasts(self.store, series, rows, horizon={'value': 12, 'unit': 'months'})
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM forecast_runs')['n'], 2)
        market_store.save_rows(self.store, 'test.rate', [{**rows[-1], 'value': 20, 'retrieved_at': utcnow()}])
        market.save_forecasts(self.store, series, rows, horizon={'value': 3, 'unit': 'months'})
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM forecast_runs')['n'], 3)
        self.assertEqual(len(self.store.query('SELECT * FROM forecast_points')), 18)

    def test_forecast_evaluation_waits_for_period_end_and_publication(self):
        self.store.execute('INSERT INTO forecast_runs(id,series_id,fingerprint,created_at,duration,inputs) VALUES(?,?,?,?,?,?)',
                           ('evaluation', 'test.rate', 'evaluation', '2026-01-01', '{}', '{}'))
        for target in ('2026-02-01', '2026-03-01', '2026-04-01'):
            self.store.execute('INSERT INTO forecast_points(run_id,target,payload) VALUES(?,?,?)',
                               ('evaluation', target, dumps({'target': target, 'created_at': '2026-01-01', 'prediction': 10})))
        market_store.save_rows(self.store, 'test.rate', [
            {'date': '2026-02-28', 'period_end': '2026-02-28', 'value': 11},
            {'date': '2026-03-01', 'period_end': '2099-03-31', 'value': 12},
            {'date': '2026-04-01', 'published_at': '2099-04-01', 'value': 13}])
        result = {r['target']: r['actual'] for r in market_store.forecast_history(self.store, 'test.rate')}
        self.assertEqual(result, {'2026-02-01': 11, '2026-03-01': None, '2026-04-01': None})

    def test_custom_horizon_ui_is_personal_and_invalid_input_keeps_prompt(self):
        self.controls.callback(Mock(), 2, 'ch:ownhorizon:market:months', 'x')
        with self.assertRaises(ValueError):
            self.controls.input(Mock(), 2, '25')
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'charthorizon:market:months')
        self.controls.input(Mock(), 2, '18')
        self.assertEqual(self.controls.options(2, 'market')['forecast'], {'value': 18, 'unit': 'months'})
        self.assertNotIn('forecast', self.controls.options(1, 'market'))

    def test_ruonia_terms_share_one_png_and_never_call_provider(self):
        from thesaurus.cbr_data import METRICS
        today = pd.Timestamp(utcnow()[:10])
        rows = [{'source': 'ruonia', 'metric': metric, 'frequency': 'daily', 'unit': '%',
                 'date': day.date().isoformat(), 'value': 15+i*.001, 'retrieved_at': utcnow()}
                for metric in ['1 мес', '3 мес', '6 мес'] for i, day in enumerate(pd.date_range(today-pd.Timedelta(days=99), periods=100))]
        self.store.save_observations(rows)
        self.controls.callback(Mock(), 2, 'ch:terms:ruonia', 1)
        options = self.controls.options(2, 'ruonia')
        self.assertTrue(set(options['series_ids']).issubset(METRICS))
        options['forecast'] = {'value': 6, 'unit': 'months'}
        payload = {'actor_id': 2, 'section': 'ruonia', 'identity': None, 'options': options, 'report_id': uuid4().hex}
        with patch('requests.sessions.Session.request', side_effect=AssertionError('network forbidden')):
            identifier = run_chart_job(self.store, self.settings, {'id': 42}, payload, dry_run=True)
        report = self.store.one('SELECT parts FROM reports WHERE id=?', (identifier,))
        self.assertEqual(len(json.loads(report['parts'])), 1)

    def test_all_29_roots_have_groups_and_pagination(self):
        market.initialize(self.store)
        self.assertEqual(sum(len(roots) for _, roots in GROUPS.values()), 29)
        for _, roots in GROUPS.values():
            for identity in roots:
                self.assertIsNotNone(self.store.one('SELECT id FROM market_series WHERE id=?', (identity,)))
        market_store.register(self.store, [meta(f'test.{i}', parent='cbr.hd_base.ruonia', dataset='cbr.hd_base.ruonia') for i in range(110)])
        rows, _, count, _ = browse(self.store, parent_id='cbr.hd_base.ruonia', page=1)
        self.assertEqual(len(rows), 8)
        self.assertGreater(count, 100)

    def test_complete_cache_and_event_history_make_zero_repeat_requests(self):
        event = meta('test.event', 'daily', calendar='events')
        market_store.register(self.store, [event])
        first, last = date(2025, 1, 1), date(2025, 1, 31)
        self.assertEqual(market_incremental.plan(self.store, event, first, last), [(first, last)])
        market_incremental.remember(self.store, event, first, last, first)
        self.assertEqual(market_incremental.plan(self.store, event, first, last), [])
        self.assertEqual(market_incremental.plan(self.store, event, first, date(2025, 2, 1)), [(date(2025, 2, 1), date(2025, 2, 1))])

    def seed_revision(self):
        market_store.save_rows(self.store, 'test.rate', [{'date': '2025-01-01', 'value': -10, 'retrieved_at': '2025-01-02T00:00:00+00:00'}])
        check = integrity.enqueue(self.store, 1, ['test.rate'])
        integrity.candidates(self.store, check, 'test.rate', [{'date': '2025-01-01', 'value': -12}], start='2025-01-01', end='2025-01-01')
        return self.store.one('SELECT id FROM integrity_incidents')['id']

    def test_corrections_wait_for_admin_and_keep_vintages(self):
        identity = self.seed_revision()
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], -10)
        with self.assertRaises(PermissionError):
            integrity.resolve(self.store, 2, identity, 'accept')
        self.assertTrue(integrity.resolve(self.store, 1, identity, 'accept'))
        self.assertFalse(integrity.resolve(self.store, 1, identity, 'accept'))
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], -12)
        self.assertEqual(market_store.history(self.store, 'test.rate', as_of='2025-01-03')[0]['value'], -10)

    def test_rejected_repeat_is_deduplicated(self):
        identity = self.seed_revision()
        integrity.resolve(self.store, 1, identity, 'reject')
        integrity.candidates(self.store, 1, 'test.rate', [{'date': '2025-01-01', 'value': -12}], start='2025-01-01', end='2025-01-01')
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_incidents')['n'], 1)
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], -10)

    def test_empty_and_network_failure_are_not_deletions(self):
        self.seed_revision()
        integrity.candidates(self.store, 1, 'test.rate', [], start='2025-01-01', end='2025-01-01', complete=True)
        with patch('thesaurus.integrity.sources.fetch', side_effect=TimeoutError('credential-private')):
            integrity.run(self.store, {'actor_id': 1, 'check_id': 1})
        self.assertEqual(market_store.history(self.store, 'test.rate')[0]['value'], -10)
        self.assertNotIn('credential-private', dumps(self.store.query('SELECT * FROM integrity_incidents')))
        self.assertEqual(self.store.one('SELECT state FROM integrity_checks')['state'], 'partial')

    def test_restart_uses_finished_audit_checkpoint(self):
        self.seed_revision()
        value = [{'date': '2025-01-01', 'value': -12}]
        with patch('thesaurus.integrity.sources.fetch', return_value=[(meta(), value)]) as fetch:
            integrity.run(self.store, {'actor_id': 1, 'check_id': 1})
            self.store.execute("UPDATE integrity_checks SET state='partial' WHERE id=1")
            integrity.run(self.store, {'actor_id': 1, 'check_id': 1})
        self.assertEqual(fetch.call_count, 1)

    def test_audit_separates_saved_parameter_variants(self):
        parent = meta('test.parent', kind='dataset', adapter='html')
        children = [{**parent, 'id': 'test.child.'+str(i), 'dataset': parent['id'], 'kind': 'series',
                     'column': 'Ставка', 'params': {'UniDbQuery.P1': str(i)}} for i in (1, 2)]
        market_store.register(self.store, [parent, *children])
        for child in children:
            market_store.save_rows(self.store, child['id'], [{'date': '2025-01-01', 'value': 10}])
        self.store.set_setting('market_params:' + parent['id'], {'UniDbQuery.P1': '2'})
        check = integrity.enqueue(self.store, 1, [child['id'] for child in children])
        def response(item, *_):
            variant = item['params']['UniDbQuery.P1']
            return [(next(child for child in children if child['params']['UniDbQuery.P1'] == variant),
                     [{'date': '2025-01-01', 'value': 10+int(variant)}])]
        with patch('thesaurus.integrity.sources.fetch', side_effect=response) as fetch:
            integrity.run(self.store, {'actor_id': 1, 'check_id': check})
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_incidents')['n'], 2)
        self.assertEqual(self.store.one('SELECT state FROM integrity_checks WHERE id=?', (check,))['state'], 'done')

    def test_schedule_is_disabled_and_missed_month_is_coalesced(self):
        self.seed_revision()
        now = datetime(2026, 10, 7, 10, tzinfo=timezone.utc)
        integrity.enqueue_due(self.store, now)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_checks')['n'], 1)
        self.store.execute("UPDATE jobs SET state='done'")
        self.store.set_setting('integrity_schedule', {'enabled': True, 'actor_id': 1})
        integrity.enqueue_due(self.store, now)
        integrity.enqueue_due(self.store, now)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM integrity_checks')['n'], 2)

    def test_critical_notifications_immediate_warnings_at_18_only(self):
        integrity.incident(self.store, 'source_revision', series_id='test.rate', new={'value': 1})
        integrity.dispatch(self.store, now=datetime(2026, 10, 7, 10, tzinfo=timezone.utc))
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM reports')['n'], 0)
        integrity.incident(self.store, 'sqlite_integrity', severity='critical')
        integrity.dispatch(self.store, now=datetime(2026, 10, 7, 10, tzinfo=timezone.utc))
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM reports')['n'], 1)
        integrity.dispatch(self.store, now=datetime(2026, 10, 7, 16, tzinfo=timezone.utc))
        integrity.dispatch(self.store, now=datetime(2026, 10, 7, 16, tzinfo=timezone.utc))
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM reports')['n'], 2)

    def test_official_forecast_is_not_actual_training_data(self):
        item = next(x for x in market_sources.catalogue() if x['id'] == 'cbr.statistics.pffl')
        text = '<p>За период с 07.10.2026 по 13.10.2026</p><table><tr><td>Фактор</td><td>−78</td></tr></table>'
        transport = Mock()
        transport.get.return_value = SimpleNamespace(text=text)
        rows = market_sources.fetch(item, transport, date(2026, 10, 7), date(2026, 10, 7))
        self.assertEqual(rows[0][1][0]['value'], -78)
        self.assertEqual(market_analysis.forecast(data(), {**meta(), 'data_type': 'official_forecast'}, created_at='2026-10-07'), [])

    def test_multi_table_miacr_has_distinct_series(self):
        contract = {'table': 0, 'headers': ['Дата', 'Ставка'], 'numeric': [1]}
        item = {**meta(), 'kind': 'dataset', 'adapter': 'html', 'contracts': [contract, {**contract, 'table': 1}]}
        transport = Mock()
        transport.get.return_value = SimpleNamespace(text='<h2>RUB</h2><table><tr><th>Дата</th><th>Ставка</th></tr><tr><td>01.01.2025</td><td>10</td></tr></table><h2>USD</h2><table><tr><th>Дата</th><th>Ставка</th></tr><tr><td>01.01.2025</td><td>3</td></tr></table>')
        output = market_sources.fetch(item, transport, date(2025, 1, 1), date(2025, 1, 1))
        self.assertNotEqual(output[0][0]['id'], output[1][0]['id'])

    def test_bot_integrity_rejects_subscriber(self):
        bot, client = BotControls(self.store), Mock()
        with self.assertRaises(PermissionError):
            bot.integrity_controls.callback(client, 2, 'ig:menu')

    def test_v2_upgrade_keeps_data_and_backs_up_once(self):
        path = self.root / 'old.sqlite3'
        original = Store(path)
        original.add_user(9, subscriptions='all')
        original.execute('PRAGMA user_version=2')
        before = len(list(self.root.glob('old.sqlite3.market-backup-*')))
        upgraded = Store(path)
        self.assertTrue(upgraded.authorize(9))
        self.assertEqual(upgraded.one('PRAGMA user_version')['user_version'], 3)
        self.assertEqual(len(list(self.root.glob('old.sqlite3.market-backup-*'))), before+1)
        Store(path)
        self.assertEqual(len(list(self.root.glob('old.sqlite3.market-backup-*'))), before+1)
