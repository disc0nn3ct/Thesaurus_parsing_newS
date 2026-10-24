"""Opt-in official-source audit, isolated from Telegram and the application's DB.

Run with --live. Default transport respects configured proxies; --direct is an
explicit diagnostic option. Only public CBR requests are made, sequentially.
"""
import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from thesaurus.market_sources import Transport, catalogue, fetch
from thesaurus.source_catalog import GROUPS, group_for, parent


def audit(transport, output, today=None):
    today = today or datetime.now(timezone.utc).date()
    entries = {item['id']: item for item in catalogue() if item['source'] == 'cbr'}
    roots = {identity for _, identities in GROUPS.values() for identity in identities}
    results = []
    for item in entries.values():
        if item['kind'] == 'reference' or item.get('adapter') == 'stored':
            continue
        end = min(today, date.fromisoformat(item['available_end'])) if item.get('available_end') else today
        years = 2 if item['frequency'] in {'daily', 'weekly'} else 20
        start = end - timedelta(days=365 * years)
        result = {'id': item['id'], 'url': item['url'], 'group': group_for(item['id'], entries),
                  'from': start.isoformat(), 'to': end.isoformat(), 'type': item.get('data_type')}
        try:
            batches = fetch(item, transport, start, end)
            populated = [(meta, rows) for meta, rows in batches if any(row['value'] is not None for row in rows)]
            result.update(state='ok' if populated else 'empty', series=len(populated),
                          observations=sum(len(rows) for _, rows in populated),
                          units=sorted({meta['unit'] for meta, _ in populated}),
                          dates=sorted({row['date'] for _, rows in populated for row in rows})[::max(1, len({row['date'] for _, rows in populated for row in rows})-1)])
        except Exception as exc:  # noqa: BLE001 - do not print transport/proxy credentials.
            result.update(state='error', error=type(exc).__name__)
            if isinstance(exc, ValueError):
                result['reason'] = str(exc)[:160]
        identity = item['id']
        visited = set()
        while identity not in roots and identity in entries and identity not in visited:
            visited.add(identity)
            identity = parent(entries[identity], entries)
        result['root'] = identity
        results.append(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(item['id'], result['state'], result.get('observations', 0), result.get('reason', ''), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('.test-tmp/cbr-coverage/verified.json'))
    arguments = parser.parse_args()
    session = requests.Session()
    if arguments.direct:
        session.trust_env = False
    audit(Transport(session=session), arguments.output)
