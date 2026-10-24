"""Build offline CC0 / public-domain atlas resources from downloaded JSON."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(source, destination):
    def read(name):
        return json.loads((source / name).read_text(encoding='utf-8'))
    countries = []
    paths = []
    features = []
    for feature in read('world.json')['features']:
        prop, geo = feature['properties'], feature['geometry']
        code = prop['ISO_A2_EH']
        if code == '-99':
            code = 'ne:' + re.sub('[^a-z]', '', prop['NAME_EN'].lower())
        countries.append({'id': code, 'name': prop['NAME_EN'], 'ru': prop['NAME_RU'],
            'aliases': sorted({prop['NAME'], prop['ADMIN'], prop['NAME_EN'], prop['NAME_RU']}),
            'lon': prop['LABEL_X'], 'lat': prop['LABEL_Y']})
        polygons = geo['coordinates'] if geo['type'] == 'MultiPolygon' else [geo['coordinates']]
        for polygon in polygons:
            points = polygon[0]
            # Split antimeridian edges rather than drawing lines across the world.
            segments, current = [], []
            for lon, lat, *_ in points:
                if current and abs(lon - current[-1][0]) > 180:
                    segments.append(current)
                    current = []
                current.append((lon, lat))
            segments.append(current)
            for segment in segments:
                if len(segment) >= 3:
                    d = 'M' + 'L'.join(f'{(lon+180)*3:.2f},{(90-lat)*3:.2f}' for lon, lat in segment) + 'Z'
                    paths.append(d)
                    features.append({'country': code, 'path': d})
    aliases = {}
    for item in read('aliases.json')['results']['bindings']:
        qid = item['item']['value'].rsplit('/', 1)[1]
        record = aliases.setdefault(qid, {'ru': '', 'aliases': []})
        if 'ru' in item:
            record['ru'] = item['ru']['value']
        if 'alias' in item:
            record['aliases'].append(item['alias']['value'])
    regions = {}
    for item in read('regions.json')['results']['bindings']:
        qid = item['item']['value'].rsplit('/', 1)[1]
        row = regions.setdefault(qid, {'id': qid, 'name': item['name']['value'],
            'country': item.get('code', {}).get('value', ''), 'description': item.get('description', {}).get('value', ''),
            'lat': None, 'lon': None, 'source': 'https://www.wikidata.org/wiki/' + qid,
            **aliases.get(qid, {'ru': '', 'aliases': []})})
        if row['name'].endswith(' wine'):
            row['aliases'].append(row['name'][:-5])
        elif row['name'].endswith(' wine region'):
            row['aliases'].append(row['name'][:-12])
        code = item.get('code', {}).get('value', '')
        if code and not any(c['id'] == code for c in countries):
            name = item.get('countryName', {}).get('value', code)
            countries.append({'id': code, 'name': name, 'ru': '', 'aliases': [name], 'lat': None, 'lon': None})
        match = re.fullmatch(r'Point\(([-\d.eE+]+) ([-\d.eE+]+)\)', item.get('coord', {}).get('value', ''))
        if match and row['lon'] is None:
            row['lon'], row['lat'] = map(float, match.groups())
    grapes = {}
    for item in read('grapes.json')['results']['bindings']:
        qid = item['item']['value'].rsplit('/', 1)[1]
        row = grapes.setdefault(qid, {'id': qid, 'name': item['name']['value'],
            'ru': item.get('ru', {}).get('value', ''), 'aliases': [],
            'description': item.get('description', {}).get('value', ''),
            'origin': item.get('originName', {}).get('value', ''),
            'vivc': item['vivc']['value'], 'source': 'https://www.wikidata.org/wiki/' + qid})
        if 'alias' in item:
            row['aliases'].append(item['alias']['value'])
    for row in [*regions.values(), *grapes.values()]:
        row['aliases'] = sorted(set(row['aliases']))
    catalog = {'version': '2026-10-05', 'complete': False,
        'sources': {'regions': 'https://www.wikidata.org/wiki/Q2140699',
                    'grapes': 'https://www.wikidata.org/wiki/Property:P3904',
                    'world': 'https://www.naturalearthdata.com/about/terms-of-use/'},
        'countries': countries, 'regions': sorted(regions.values(), key=lambda x: x['name']),
        'grapes': sorted(grapes.values(), key=lambda x: x['name'])}
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in [('vivino_catalog.json', catalog), ('vivino_world.json', {'paths': paths, 'features': features})]:
        content = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
        (destination / name).write_text(content, encoding='utf-8', newline='\n')
        print(name, len(content.encode()), hashlib.sha256(content.encode()).hexdigest())
    print('regions', len(regions), 'grapes', len(grapes), 'countries', len(countries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    build(args.source, ROOT / 'thesaurus' / 'resources')
