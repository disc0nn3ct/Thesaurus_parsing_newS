"""Read-only live smoke audit of connected CBR adapters; no reports/delivery."""
import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesaurus import market_incremental, market_sources, market_store
from thesaurus.source_errors import diagnostic


def audit(db, today, transport):
    catalog = {item['id']: item for item in market_sources.catalogue()}
    with closing(sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        identities = {row[0] for row in conn.execute("SELECT id FROM market_series WHERE source='cbr' AND enabled=1 UNION SELECT dataset FROM market_series WHERE source='cbr' AND enabled=1")}
        parameters = {row[0][14:]: json.loads(row[1]) for row in conn.execute("SELECT key,value FROM settings WHERE key LIKE 'market_params:%'")}
    result = []
    for identity in sorted(identities):
        item = catalog.get(identity)
        if not item or item['kind'] == 'reference' or item.get('adapter') in {'ruonia', 'stored_ruonia'}:
            continue
        end = min(today-timedelta(days=1), date.fromisoformat(item.get('available_end', '2099-01-01')))
        start = end-timedelta(days=32)
        if item['frequency'] in {'monthly', 'quarterly', 'annual'}:
            start = market_incremental.period(start, item['frequency'])
        record = {'source': identity, 'title': item['title'], 'url': item['url'], 'from': str(start), 'to': str(end)}
        try:
            batches = market_sources.fetch({**item, 'params': parameters.get(identity, {})}, transport, start, end)
            checked = [market_store.partition_rows(rows) for _, rows in batches]
            points = sum(sum(row.get('value') is not None for row in rows) for rows, _ in checked)
            conflicts = sum(len(conflicts) for _, conflicts in checked)
            record.update(status='partial' if conflicts else 'ok' if points else 'empty', series=len(batches), values=points, conflicts=conflicts)
        except Exception as exc:  # noqa: BLE001 - intentionally report only allowlisted diagnostics.
            code, message = diagnostic(exc)
            record.update(status='error', code=code, error=message)
        result.append(record)
        print(identity + ': ' + record['status'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path(__file__).resolve().parents[1] / 'data/thesaurus.sqlite3')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--today', type=date.fromisoformat, default=datetime.now(ZoneInfo('Europe/Moscow')).date())
    args = parser.parse_args()
    result = audit(args.db, args.today, market_sources.Transport())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 1 if any(row['status'] in {'error', 'partial'} for row in result) else 0


if __name__ == '__main__':
    raise SystemExit(main())
