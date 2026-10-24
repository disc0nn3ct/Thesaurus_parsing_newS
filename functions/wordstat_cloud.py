"""Translate the documented Search API v2 contract into the existing adapters."""
from datetime import date


def request_body(path, payload, folder_id):
    methods = {'/v1/dynamics': 'dynamics', '/v1/regions': 'regions',
               '/v1/getRegionsTree': 'getRegionsTree', '/v1/topRequests': 'topRequests'}
    if path not in methods:
        raise ValueError('Unsupported Wordstat method')
    result = dict(payload)
    if 'devices' in result:
        result['devices'] = [{'all': 'DEVICE_ALL', 'desktop': 'DEVICE_DESKTOP',
                              'phone': 'DEVICE_PHONE', 'tablet': 'DEVICE_TABLET'}[item] for item in result['devices']]
    if 'regions' in result:
        result['regions'] = [str(int(item)) for item in result['regions']]
    if path == '/v1/dynamics':
        result['period'] = {'daily': 'PERIOD_DAILY', 'weekly': 'PERIOD_WEEKLY', 'monthly': 'PERIOD_MONTHLY'}[result['period']]
        for field in ('fromDate', 'toDate'):
            if field in result:
                result[field] = date.fromisoformat(result[field]).isoformat() + 'T00:00:00Z'
    if path == '/v1/regions':
        result['region'] = {'all': 'REGION_ALL', 'regions': 'REGION_REGIONS',
                            'cities': 'REGION_CITIES'}[result.pop('regionType', 'all')]
    if 'numPhrases' in result:
        result['numPhrases'] = str(int(result['numPhrases']))
    result['folderId'] = folder_id
    return 'https://searchapi.api.cloud.yandex.net/v2/wordstat/' + methods[path], result


def response_body(path, data):
    if not isinstance(data, dict):
        raise TypeError('Unexpected Search API response')
    field = 'regions' if path == '/v1/getRegionsTree' else 'results'
    if not isinstance(data.get(field), list) or any(not isinstance(item, dict) for item in data[field]):
        raise TypeError('Unexpected Search API response schema')
    def numeric(item):
        result = dict(item)
        try:
            if result.get('count') is not None:
                result['count'] = int(result['count'])
            for key in ('share', 'affinityIndex'):
                if result.get(key) is not None:
                    result[key] = float(result[key])
        except (TypeError, ValueError):
            raise TypeError('Invalid Search API numeric observation') from None
        return result
    if path == '/v1/dynamics':
        return {'dynamics': [numeric(item) for item in data['results']]}
    if path == '/v1/regions':
        return {'regions': [{**numeric(item), 'regionId': int(item['region'])} for item in data['results']]}
    if path == '/v1/getRegionsTree':
        def node(item):
            return {'value': int(item['id']), 'label': item['label'],
                    'children': [node(child) for child in item.get('children', [])]}
        return [node(item) for item in data['regions']]
    return {'totalCount': data.get('totalCount'), 'topRequests': [numeric(item) for item in data['results']],
            'associations': [numeric(item) for item in data.get('associations', [])]}
