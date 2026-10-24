"""Public-source smoke check in an isolated database; no bot or paid providers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesaurus.config import Settings
from thesaurus.market import collect, enable, initialize, report, selected
from thesaurus.market_jobs import export_market
from thesaurus.store import Store


def run():
    root = Path(__file__).resolve().parents[1] / '.test-tmp' / 'market-live'
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / 'market.sqlite3')
    store.add_user(1, role='admin')
    initialize(store)
    for identity in ('cbr.hd_base.keyrate', 'cbr.hd_base.infl', 'cbr.hd_base.zcyc_params',
                     'cbr.hd_base.bliquidity', 'cbr.hd_base.mrrf.mrrf_7d', 'cbr.fx.usd',
                     'cbr.ruonia.rate', 'cbr.ruonia.volume', 'moex.imoex.close'):
        enable(store, identity, True)
    result = collect(store)
    report_id = report(store, 1, settings=Settings(root=root), charts=True)
    export_market(store, root / 'market.csv', [r['id'] for r in selected(store)])
    export_market(store, root / 'market.xlsx', [r['id'] for r in selected(store)], format='xlsx')
    result['series'] = store.one('SELECT COUNT(DISTINCT series_id) n FROM market_observations')['n']
    result['observations'] = store.one('SELECT COUNT(*) n FROM market_observations')['n']
    result['forecasts'] = store.one('SELECT COUNT(*) n FROM market_forecasts')['n']
    result['report_id'] = report_id
    result['latest'] = store.query('SELECT series_id,MAX(date) date,COUNT(*) n FROM market_observations GROUP BY series_id')
    (root / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
