"""Offline acceptance of DB reuse, missing intervals and publication delays."""
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from thesaurus import cbr_incremental as incremental
from thesaurus import market, market_incremental, market_store
from thesaurus.cbr_data import parse_rates
from thesaurus.collectors import Collector, Providers
from thesaurus.config import Settings
from thesaurus.store import Store

NOW = datetime(2026, 9, 5, 16, tzinfo=timezone.utc)
FIRST, LAST = date(2026, 9, 1), date(2026, 9, 4)


def xml(days):
    return ('<RuoniaResult><NewDataSet>' + ''.join(
        f'<ro><D0>{day}</D0><ruo>14</ruo><vol>400</vol></ro>' for day in days)
        + '</NewDataSet></RuoniaResult>').encode()


class IncrementalCbrTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / '.test-tmp'
        parent.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = Store(self.root / 'state.sqlite3')
        self.transport = Mock()
        self.transport.post.return_value.content = xml(['2026-09-04'])
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')))

    def seed(self, metrics=incremental.RATE_METRICS, days=('2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04')):
        self.store.save_observations([{'source': 'ruonia', 'metric': metric, 'frequency': 'daily',
            'date': day, 'value': 14., 'retrieved_at': '2026-09-04T16:00:00+00:00',
            'unit': 'индекс' if metric == 'Индекс RUONIA' else 'млрд руб' if metric == 'Объём RUONIA' else '%'}
            for metric in metrics for day in days])

    def floors(self):
        for metric in incremental.INDEX_METRICS + incremental.RATE_METRICS:
            self.store.set_setting('ruonia_availability:' + metric, {'from': FIRST.isoformat()})

    def test_complete_db_survives_restart_without_any_requests(self):
        self.seed()
        reopened = Store(self.store.path)
        result = incremental.collect_rates(reopened, self.transport, FIRST, LAST, NOW)
        self.assertEqual(result, {'new_observations': 0, 'requests': 0})
        self.transport.post.assert_not_called()

    def test_only_new_tail_requested_and_original_timestamp_kept(self):
        self.seed(days=('2026-09-01', '2026-09-02', '2026-09-03'))
        result = incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        body = self.transport.post.call_args.args[1]
        self.assertIn('<fromDate>2026-09-04T00:00:00', body)
        self.assertIn('<ToDate>2026-09-04T00:00:00', body)
        self.assertEqual(result['new_observations'], 2)
        self.assertEqual(self.store.one("SELECT retrieved_at FROM observations WHERE date='2026-09-01'")['retrieved_at'],
                         '2026-09-04T16:00:00+00:00')

    def test_internal_null_and_absent_day_are_separate_missing_ranges(self):
        self.seed()
        self.store.execute("UPDATE observations SET value=NULL WHERE date='2026-09-02'")
        self.store.execute("DELETE FROM observations WHERE date='2026-09-04'")
        self.assertEqual(incremental.missing_ranges(self.store, incremental.RATE_METRICS, FIRST, LAST, NOW),
                         [(date(2026, 9, 2), date(2026, 9, 2)), (LAST, LAST)])
        self.transport.post.side_effect = [SimpleNamespace(content=xml(['2026-09-02'])),
                                           SimpleNamespace(content=xml(['2026-09-04']))]
        incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        self.assertEqual(self.transport.post.call_count, 2)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM observations WHERE value IS NULL')['n'], 0)

    def test_bundled_response_does_not_overwrite_existing_value(self):
        self.seed()
        self.store.execute("UPDATE observations SET value=99 WHERE metric='RUONIA' AND date='2026-09-04'")
        self.store.execute("DELETE FROM observations WHERE metric='Объём RUONIA' AND date='2026-09-04'")
        incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        self.assertEqual(self.store.one("SELECT value FROM observations WHERE metric='RUONIA' AND date='2026-09-04'")['value'], 99.)

    def test_empty_result_waits_then_retries_only_missing_day(self):
        self.seed(days=('2026-09-01', '2026-09-02', '2026-09-03'))
        self.transport.post.return_value.content = xml([])
        incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        incremental.collect_rates(Store(self.store.path), self.transport, FIRST, LAST, NOW + timedelta(minutes=30))
        self.assertEqual(self.transport.post.call_count, 1)
        self.assertIsNone(self.store.one("SELECT value FROM observations WHERE date='2026-09-04'"))
        self.transport.post.return_value.content = xml(['2026-09-04'])
        incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW + timedelta(hours=1))
        self.assertEqual(self.transport.post.call_count, 2)

    def test_timeout_does_not_mark_missing_date_as_loaded(self):
        self.seed(days=('2026-09-01', '2026-09-02', '2026-09-03'))
        self.transport.post.side_effect = TimeoutError('not printed')
        with self.assertRaises(TimeoutError):
            incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        self.assertEqual(incremental.missing_ranges(self.store, incremental.RATE_METRICS, FIRST, LAST, NOW), [(LAST, LAST)])

    def test_completed_batch_survives_failure_of_following_batch(self):
        self.seed(days=('2026-09-01', '2026-09-03'))
        self.transport.post.side_effect = [SimpleNamespace(content=xml(['2026-09-02'])), TimeoutError()]
        with self.assertRaises(TimeoutError):
            incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        self.assertEqual(incremental.missing_ranges(self.store, incremental.RATE_METRICS, FIRST, LAST, NOW), [(LAST, LAST)])

    def test_weekend_and_russian_holidays_are_not_missing_observations(self):
        self.assertEqual(incremental.missing_ranges(self.store, incremental.RATE_METRICS,
                         date(2026, 1, 1), date(2026, 1, 8), NOW), [])
        self.seed()
        incremental.collect_rates(self.store, self.transport, FIRST, date(2026, 9, 6), NOW)
        self.transport.post.assert_not_called()

    def test_old_report_cache_expiry_and_force_still_use_db(self):
        self.seed(incremental.INDEX_METRICS + incremental.RATE_METRICS)
        self.floors()
        collector = Collector(self.store, Settings(root=self.root), Providers(self.store), now=lambda: NOW)
        with patch('thesaurus.market_sources.Transport', return_value=self.transport):
            first = collector.collect('ruonia')
            collector.now = lambda: NOW + timedelta(hours=2)
            second = collector.collect('ruonia', force=True)
        self.assertTrue(first['cached'])
        self.assertTrue(second['cached'])
        self.assertEqual(second['new_observations'], 0)
        self.transport.get.assert_not_called()
        self.transport.post.assert_not_called()

    def test_index_download_queries_only_gap(self):
        self.seed(incremental.INDEX_METRICS + incremental.RATE_METRICS)
        self.floors()
        self.store.execute("DELETE FROM observations WHERE metric='1 мес' AND date='2026-09-02'")
        frame = pd.DataFrame({'Дата': ['2026-09-02'], 'Индекс': [200], '1 месяц': [15],
                              '3 месяца': [16], '6 месяцев': [17]})
        buffer = BytesIO()
        frame.to_excel(buffer, index=False)
        self.transport.get.return_value.content = buffer.getvalue()
        result = incremental.collect_report(self.store, self.transport, date(2026, 9, 5), NOW)
        params = self.transport.get.call_args.args[1]
        self.assertEqual((params['From'], params['To']), ('02.09.2026', '02.09.2026'))
        self.assertEqual(result.attrs['new_observations'], 1)
        self.assertEqual(self.store.one("SELECT value FROM observations WHERE metric='Индекс RUONIA' AND date='2026-09-02'")['value'], 14.)

    def test_invalid_empty_or_fault_xml_is_not_publication_delay(self):
        for content in (b'<root/>', b'<RuoniaResult><Fault/></RuoniaResult>', b'<bad'):
            with self.assertRaises(ValueError):
                parse_rates(content, allow_empty=True)

    def test_market_and_regular_report_share_rate_and_volume(self):
        market.initialize(self.store)
        self.floors()
        self.seed(days=('2026-09-01', '2026-09-02', '2026-09-03'))
        market.enable(self.store, 'cbr.ruonia.rate', True)
        market.enable(self.store, 'cbr.ruonia.volume', True)
        result = market.collect(self.store, transport=self.transport, today=date(2026, 9, 5))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(self.transport.post.call_count, 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_observations WHERE series_id='cbr.ruonia.rate'")['n'], 4)
        incremental.collect_rates(Store(self.store.path), self.transport, FIRST, LAST, NOW)
        self.assertEqual(self.transport.post.call_count, 1)

    def test_market_only_history_reused_by_regular_report(self):
        market.initialize(self.store)
        for metric in incremental.RATE_METRICS:
            identity = incremental.MARKET_IDS[metric]
            market_store.save_rows(self.store, identity, [{'date': day, 'value': 14., 'retrieved_at': NOW.isoformat()}
                for day in ('2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04')])
        incremental.collect_rates(self.store, self.transport, FIRST, LAST, NOW)
        self.transport.post.assert_not_called()
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM observations')['n'], 8)

    def test_index_calendar_includes_weekends(self):
        self.seed(incremental.INDEX_METRICS)
        self.assertEqual(incremental.missing_ranges(self.store, incremental.INDEX_METRICS,
                         FIRST, date(2026, 9, 6), NOW), [(date(2026, 9, 5), date(2026, 9, 6))])

    def test_all_index_columns_shared_with_market_without_new_download(self):
        market.initialize(self.store)
        self.floors()
        self.seed(incremental.INDEX_METRICS)
        market.enable(self.store, incremental.INDEX_DATASET, True)
        market.collect(self.store, transport=self.transport, today=date(2026, 9, 5))
        self.transport.get.assert_not_called()
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM market_observations WHERE series_id LIKE 'cbr.hd_base.ruonia.dynamic.%'")['n'], 16)
        index_row = self.store.one('SELECT metadata FROM market_series WHERE id=?',
                                  (incremental.MARKET_IDS['Индекс RUONIA'],))
        self.assertEqual(json.loads(index_row['metadata'])['unit'], 'индекс')

    def test_market_index_cache_reused_by_regular_report(self):
        market.initialize(self.store)
        for metric in incremental.INDEX_METRICS:
            parent = self.store.one('SELECT metadata FROM market_series WHERE id=?', (incremental.INDEX_DATASET,))
            metadata = json.loads(parent['metadata'])
            identity = incremental.MARKET_IDS[metric]
            market_store.register(self.store, [{**metadata, 'id': identity, 'kind': 'series',
                                               'dataset': incremental.INDEX_DATASET}])
            market_store.save_rows(self.store, identity, [{'date': day, 'value': 14., 'retrieved_at': NOW.isoformat()}
                for day in ('2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04')])
        incremental.collect_index(self.store, self.transport, FIRST, LAST, NOW)
        self.transport.get.assert_not_called()
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM observations')['n'], 16)

    def other_series(self, frequency='daily'):
        item = {'id': 'test.series', 'dataset': 'test.series', 'kind': 'series', 'source': 'cbr',
                'title': 'Test', 'unit': '%', 'frequency': frequency, 'calendar': 'calendar',
                'adapter': 'html', 'url': 'https://cbr.ru/hd_base/keyrate/', 'params': {}}
        market_store.register(self.store, [item])
        return item

    def test_market_gap_and_new_tail_without_overlap(self):
        item = self.other_series()
        market_store.save_rows(self.store, item['id'], [{'date': day, 'value': 1.} for day in ('2026-09-01', '2026-09-03')])
        self.assertEqual(market_incremental.plan(self.store, item, FIRST, LAST, NOW),
                         [(date(2026, 9, 2), date(2026, 9, 2)), (LAST, LAST)])

    def test_month_end_values_are_reused_and_missing_month_requested_in_full(self):
        item = self.other_series('monthly')
        market_store.save_rows(self.store, item['id'], [{'date': '2026-07-31', 'value': 1.}, {'date': '2026-09-01', 'value': 2.}])
        self.assertEqual(market_incremental.plan(self.store, item, date(2026, 7, 1), date(2026, 9, 30), NOW),
                         [(date(2026, 8, 1), date(2026, 8, 31))])

    def test_expanding_history_fetches_only_older_missing_part(self):
        item = self.other_series('monthly')
        market_store.save_rows(self.store, item['id'], [{'date': '2026-09-01', 'value': 1.}])
        market_incremental.remember(self.store, item, FIRST, LAST, FIRST, NOW)
        self.assertEqual(market_incremental.plan(self.store, item, date(2026, 8, 1), LAST, NOW),
                         [(date(2026, 8, 1), date(2026, 8, 31))])

    def test_market_complete_data_no_transport_after_cache_expiry(self):
        item = self.other_series()
        market.enable(self.store, item['id'], True)
        market_store.save_rows(self.store, item['id'], [{'date': day, 'value': 1.} for day in ('2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04')])
        state = {'floors': {item['id']: FIRST.isoformat()}, 'window_start': '2021-09-05'}
        self.store.set_setting('market_incremental:' + item['id'], state)
        # The end date is already stored; no provider call, regardless of last refresh time.
        market.collect(self.store, transport=self.transport, today=LAST)
        self.transport.get.assert_not_called()
        self.assertEqual(json.loads(self.store.one('SELECT metadata FROM market_series')['metadata'])['id'], item['id'])


if __name__ == '__main__':
    unittest.main()
