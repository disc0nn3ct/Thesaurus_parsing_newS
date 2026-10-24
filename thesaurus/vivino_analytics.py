"""Personal exploration statistics; no inferred grapes or predicted taste scores."""
from collections import defaultdict
from statistics import mean

EXPLORATION_GRAPES = ('Q207310', 'Q213332', 'Q223701', 'Q213338', 'Q456471', 'Q332720', 'Q44542', 'Q519874')


def analyze(view):
    countries = {r['id']: r for r in view['countries']}
    regions = {r['id']: r for r in view['regions']}
    grapes = {r['id']: r for r in view['grapes']}
    grouped = defaultdict(list)
    for index, row in enumerate(view['wines']):
        grouped[row['id']].append((index, row))
    unique = []
    for identifier, entries in grouped.items():
        confirmed = [r for _, r in entries if r['tasted']]
        rows = confirmed or [r for _, r in entries]
        ratings = {r['rating'] for r in rows if r['rating'] is not None}
        composition = sorted({g for r in rows for g in r['grapes']})
        unique.append({'id': identifier, 'name': rows[0]['name'], 'index': entries[0][0],
            'country': rows[0]['country_id'], 'region': rows[0]['region_id'],
            'tasted': bool(confirmed), 'grapes': composition,
            'unknown_composition': not composition or any(r['unknown_grapes'] for r in rows),
            'rating': mean(ratings) if ratings else None})
    tasted = [r for r in unique if r['tasted']]
    country_stats, grape_stats = [], []
    for collection, field, result in [(countries, 'country', country_stats), (grapes, 'grapes', grape_stats)]:
        members = defaultdict(list)
        for row in tasted:
            for identifier in row[field] if field == 'grapes' else [row[field]]:
                if identifier in collection:
                    members[identifier].append(row)
        for identifier, reference in collection.items():
            rows = members[identifier]
            ratings = [r['rating'] for r in rows if r['rating'] is not None]
            result.append({'id': identifier, 'name': reference['name'], 'wine_count': len(rows),
                'rated_count': len(ratings), 'mean_rating': round(mean(ratings), 3) if ratings else None,
                'single_variety_count': sum(len(r['grapes']) == 1 and not r['unknown_composition'] for r in rows),
                'blend_count': sum(len(r['grapes']) > 1 for r in rows),
                'country_counts': {c: sum(r['country'] == c for r in rows)
                                   for c in sorted({r['country'] for r in rows if r['country']})},
                'region_count': len({r['region'] for r in rows if regions[r['region']]['name'] != 'Регион не указан'})})
    country_by_id = {r['id']: r for r in country_stats}
    preferences = sorted([r for r in grape_stats if r['rated_count'] >= 3],
                         key=lambda r: (-r['mean_rating'], -r['rated_count'], r['name']))
    suggestions, keys = [], set()

    def suggest(kind, row, reason, priority):
        key = (kind, row['id'])
        if key not in keys:
            keys.add(key)
            suggestions.append({'kind': kind, 'id': row['id'], 'name': row['name'],
                                'reason': reason, 'priority': priority})

    for kind, collection in [('grape', grapes), ('region', regions)]:
        for row in collection.values():
            if row['planned'] and not row['tried']:
                suggest(kind, row, 'Уже в вашем плане; подтверждённой дегустации пока нет.', 0)
    preferred = {r['id'] for r in preferences if r['mean_rating'] >= 4}
    for row in unique:
        if row['tasted']:
            continue
        new_grapes = [grapes[g]['name'] for g in row['grapes'] if not grapes[g]['tried']]
        if new_grapes:
            reason = 'В вашем экспорте без подтверждённой дегустации. Не отмеченные сорта: ' + ', '.join(new_grapes) + '.'
            if preferred.intersection(row['grapes']):
                reason += ' В составе также есть сорт из ваших вин с высокой средней оценкой; это не прогноз вкуса.'
            suggest('wine', row, reason, 1 if preferred.intersection(row['grapes']) else 2)
    starters = 0
    for identifier in EXPLORATION_GRAPES:
        row = grapes.get(identifier)
        if not row or row['tried'] or row['planned']:
            continue
        reason = 'Стартовая цель из фиксированного набора: сорт пока не отмечен в вашем профиле. Не прогноз вкуса.'
        if any(r['unknown_composition'] for r in tasted):
            reason += ' Состав части вин неизвестен — возможно, вы его уже пробовали; уточните данные.'
        suggest('grape', row, reason, 2.5)
        starters += 1
        if starters == 3:
            break
    # One region per country keeps exploration suggestions geographically varied.
    candidates = [r for r in regions.values() if not r['tried'] and not r['planned']
                  and r['country'] in countries and not r['id'].startswith('u')]
    candidates.sort(key=lambda r: (country_by_id[r['country']]['wine_count'], r['country'], r['name']))
    seen_countries = set()
    for row in candidates:
        if row['country'] in seen_countries:
            continue
        seen_countries.add(row['country'])
        count = country_by_id[row['country']]['wine_count']
        country = countries[row['country']]
        reason = f"Расширить географию: {country.get('ru') or country['name']}, попробовано вин — {count}. Регион не отмечен."
        suggest('region', row, reason, 3)
        if len(seen_countries) >= 6:
            break
    suggestions.sort(key=lambda r: (r['priority'], r['name']))
    unknown = sum(r['unknown_composition'] for r in tasted)
    return {'unique_wines': unique, 'countries': country_stats, 'grapes': grape_stats,
        'preferences': preferences[:10], 'recommendations': suggestions[:12],
        'summary': {'tasted_wines': len(tasted), 'countries': sum(r['wine_count'] > 0 for r in country_stats),
            'grapes': sum(r['wine_count'] > 0 for r in grape_stats), 'unknown_composition': unknown,
            'known_composition': len(tasted) - unknown,
            'unknown_country': sum(not r['country'] for r in tasted)},
        'heat_max': max((r['wine_count'] for r in country_stats), default=0),
        'method': 'Одно вино одной винодельни, года, страны и региона учитывается один раз. '
                  'Средняя оценка сорта — оценка содержащих его вин, включая купажи; это не оценка самого сорта. '
                  'Сравнение предпочтений доступно от трёх оценённых вин на сорт. '
                  'Стартовые сортовые цели берутся из фиксированного набора, а не из предсказания вкуса. '
                  'Предложения — цели для исследования, не прогноз вкуса и не проверка наличия в продаже.'}
