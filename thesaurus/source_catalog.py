"""Reviewed CBR taxonomy; parent navigation never mixes documents and series."""
import json

GROUPS = {
    'rates': ('Ставки, инфляция и долговой рынок', ['cbr.hd_base.infl', 'cbr.hd_base.ruonia', 'cbr.hd_base.mkr', 'cbr.hd_base.zcyc_params', 'cbr.hd_base.procstav', 'cbr.hd_base.impliedratecny']),
    'fx': ('Валюты и металлы', ['cbr.currency_base', 'cbr.hd_base.micex_doc', 'cbr.hd_base.seldomc', 'cbr.hd_base.metall', 'cbr.hd_base.valintbr']),
    'reserves': ('Денежная база и обязательные резервы', ['cbr.hd_base.mb_nd', 'cbr.hd_base.rreserves', 'cbr.pr', 'cbr.oper_br.o_dkp.reserve_requirements.reserv_pr4']),
    'liquidity': ('Банковская ликвидность', ['cbr.hd_base.ostat_base', 'cbr.hd_base.bliquidity', 'cbr.hd_base.bsliquidityma', 'cbr.statistics.flikvid', 'cbr.statistics.pffl']),
    'operations': ('Операции и обеспечение', ['cbr.hd_base.cntr', 'cbr.hd_base.depo', 'cbr.hd_base.obr', 'cbr.hd_base.securedloans', 'cbr.hd_base.repodb', 'cbr.hd_base.swap_info']),
    'external': ('Внешний сектор', ['cbr.hd_base.mrrf', 'cbr.hd_base.tg', 'cbr.hd_base.svs_db']),
    'moex': ('Мосбиржа', []),
    'other': ('Другие и сохранённые показатели', []),
}


def parent(meta, identities):
    if meta.get('kind') == 'series' and meta.get('dataset') in identities and meta['dataset'] != meta['id']:
        return meta['dataset']
    return meta.get('parent') if meta.get('parent') in identities else None


def group_for(identity, entries):
    visited = set()
    while identity in entries and identity not in visited:
        visited.add(identity)
        for key, (_, roots) in GROUPS.items():
            if identity in roots:
                return key
        meta = entries[identity]
        if meta.get('catalog_group') in GROUPS:
            return meta['catalog_group']
        if meta.get('source') == 'moex':
            return 'moex'
        identity = parent(meta, entries)
    return 'other'


def browse(store, *, group=None, parent_id=None, search='', page=0, source=None, loaded=False):
    rows = store.query('SELECT rowid token,* FROM market_series ORDER BY title,id')
    if source is not None:
        rows = [row for row in rows if row['source'] == source]
    from .search_provider import disabled
    if disabled(store):
        rows = [row for row in rows if row['source'] not in {'wordstat', 'google_trends'}]
    entries = {row['id']: json.loads(row['metadata']) for row in rows}
    available = {row['series_id'] for row in store.query('SELECT DISTINCT series_id FROM market_observations WHERE value IS NOT NULL')} if loaded else set()
    matched = []
    for row in rows:
        meta = entries[row['id']]
        if loaded:
            if meta.get('kind') != 'series' or row['id'] not in available or row['status'] == 'archived':
                continue
        elif search:
            if search.casefold() not in (row['title'] + ' ' + row['id']).casefold():
                continue
        elif parent_id:
            if parent(meta, entries) != parent_id:
                continue
        elif parent(meta, entries) is not None:
            continue
        if group and group_for(row['id'], entries) != group:
            continue
        matched.append(row)
    page = min(max(page, 0), max(0, (len(matched)-1)//8))
    return matched[page*8:(page+1)*8], page, len(matched), entries


def type_label(meta):
    return {'official_forecast': 'прогноз ЦБ', 'archive': 'архив', 'reference': 'справочник'}.get(
        meta.get('data_type'), 'показатель' if meta.get('kind') == 'series' else 'набор' if meta.get('kind') == 'dataset' else 'сведения')
