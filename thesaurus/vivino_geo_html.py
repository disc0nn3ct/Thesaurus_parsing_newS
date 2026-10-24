"""Offline country heatmap, grape analysis and exploration snapshot."""
import json
from pathlib import Path

RESOURCES = Path(__file__).parent / 'resources'


def render(atlas):
    data = dict(atlas)
    world = json.loads((RESOURCES / 'vivino_world.json').read_text(encoding='utf-8'))
    data['world'] = world['features']
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return (RESOURCES / 'vivino_atlas_template.html').read_text(encoding='utf-8').replace('__DATA__', encoded)
