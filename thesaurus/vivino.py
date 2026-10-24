"""Bounded personal Vivino exports and local geographic maps."""
import csv
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from pathlib import PurePosixPath

from .store import dumps, utcnow
from .telegram_data_jobs import require_actor
from .telegram_projects import setting_key

MAX_UPLOAD = 10 * 1024 * 1024
MAX_CSV = 8 * 1024 * 1024
MAX_ROWS = 3000
MAP_VERSION = 3
FIELDS = {'Wine name': 'name', 'Winery': 'winery', 'Vintage': 'vintage', 'Region': 'region',
          'Country': 'country', 'Regional wine style': 'style', 'Wine type': 'type',
          'Your rating': 'rating', 'Average rating': 'average', 'Your review': 'review', 'Grapes': 'grapes',
          'Link to wine': 'link'}


def parse_upload(raw, filename):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_UPLOAD:
        raise ValueError('Отправьте ZIP до 10 МБ или CSV до 8 МБ.')
    if filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if len(entries) > 512 or sum(e.file_size for e in entries) > 256 * 1024 * 1024:
                    raise ValueError('Архив слишком большой. Оставьте только full_wine_list.csv.')
                candidates = []
                for entry in entries:
                    name = entry.filename.replace('\\', '/')
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts or ':' in name or stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError('Небезопасная структура ZIP. Пересоздайте архив с CSV.')
                    if path.name.lower() == 'full_wine_list.csv' and not entry.is_dir():
                        candidates.append(entry)
                if len(candidates) != 1:
                    raise ValueError('В ZIP должен быть ровно один full_wine_list.csv (вложенная папка допустима).')
                entry = candidates[0]
                if entry.flag_bits & 1 or entry.file_size > MAX_CSV or entry.file_size / max(1, entry.compress_size) > 200:
                    raise ValueError('CSV слишком большой или ZIP защищён паролем. Отправьте исходный CSV.')
                with archive.open(entry) as stream:
                    raw = stream.read(MAX_CSV + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError):
            raise ValueError('Не удалось прочитать ZIP. Пересоздайте архив без пароля.') from None
    elif not filename.lower().endswith('.csv'):
        raise ValueError('Нужен ZIP с full_wine_list.csv или сам CSV. RAR не поддерживается.')
    if len(raw) > MAX_CSV:
        raise ValueError('CSV должен быть не больше 8 МБ.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError:
        raise ValueError('CSV должен быть в UTF-8, как исходный экспорт Vivino.') from None
    if '\0' in text:
        raise ValueError('Файл не является текстовым экспортом Vivino.')
    if not text.strip():
        raise ValueError('CSV пустой.')
    first_line = text.splitlines()[0]
    delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not {'Wine name', 'Winery', 'Country'} <= set(reader.fieldnames)):
        raise ValueError('Не найдены столбцы Wine name, Winery, Country. Нужен full_wine_list.csv из Vivino.')
    rows, seen = [], set()
    quality = {'duplicates': 0, 'missing_names': 0, 'invalid_ratings': 0, 'reviewed': 0}
    try:
        for index, original in enumerate(reader):
            if index >= MAX_ROWS:
                raise ValueError('До 3000 записей в одной карте. Разделите экспорт.')
            if None in original or any(len(value or '') > 8000 for value in original.values()):
                raise ValueError('Некорректная строка CSV или слишком длинное поле.')
            row = {target: (original.get(source) or '').strip() for source, target in FIELDS.items()}
            row['grapes'] = list(dict.fromkeys(v.strip() for v in re.split('[;|,]', row['grapes'] or original.get('Grape varieties') or '') if v.strip()))
            from urllib.parse import urlsplit
            try:
                link = urlsplit(row['link'])
                if link.scheme != 'https' or link.hostname not in {'vivino.com', 'www.vivino.com'} or link.username or link.password:
                    row['link'] = ''
            except ValueError:
                row['link'] = ''
            if not row['name']:
                quality['missing_names'] += 1
                continue
            for key in ('rating', 'average'):
                source = row[key]
                try:
                    number = float(source.replace(',', '.')) if source else None
                    if number is not None and (not math.isfinite(number) or not 0 < number <= 5):
                        raise ValueError()
                except ValueError:
                    number = None
                    quality['invalid_ratings'] += 1
                row[key] = number
            fingerprint = dumps(row)
            if fingerprint in seen:
                quality['duplicates'] += 1
                continue
            seen.add(fingerprint)
            quality['reviewed'] += bool(row['review'])
            rows.append(row)
    except csv.Error:
        raise ValueError('Некорректный CSV. Отправьте исходный экспорт.') from None
    if not rows:
        raise ValueError('В файле нет записей с названием вина.')
    return {'rows': rows, 'quality': quality, 'hash': hashlib.sha256(raw).hexdigest()}


def active(store, reference, recipient=None):
    if not isinstance(reference, dict) or reference.get('project') != 'vivino':
        return False
    actor = reference.get('owner')
    viewer = reference.get('viewer', actor)
    from .subscription_access import allowed
    if not allowed(store, viewer, 'vivino'):
        return False
    if (type(actor) is not int or type(viewer) is not int or not store.authorize(viewer)
            or (actor != viewer and not store.authorize(viewer, admin=True))
            or (recipient is not None and str(viewer) != str(recipient))):
        return False
    dataset = store.get_setting(setting_key(actor, 'vivino', 'dataset'))
    return bool(dataset and dataset['generation'] == reference.get('generation'))


def coordinates(rows):
    if len(rows) == 1:
        return [[0.0, 0.0]]
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits
    texts = [(' '.join([row['review']] * 2 + [row['style'], row['type'], row['country'], row['region']])
              .strip() or row['name']) for row in rows]
    with threadpool_limits(limits=4):
        matrix = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=6000).fit_transform(texts)
        if matrix.shape[1] < 2:
            return [[0.0, 0.0] for _ in rows]
        result = TruncatedSVD(n_components=2, random_state=0).fit_transform(matrix)
    return [[float(value) for value in point] for point in result]


def run_job(store, settings, job, payload, *, dry_run=False):
    from .delivery import prepare_delivery
    from .vivino_atlas import model, subject
    from .vivino_geo_html import render
    actor = payload['actor_id']
    require_actor(store, actor)
    owner = subject(store, actor, payload.get('owner_id', actor))
    reference = {'project': 'vivino', 'owner': owner, 'viewer': actor, 'generation': payload['generation']}
    if not active(store, reference):
        raise ValueError('Vivino dataset no longer active')
    dataset = store.get_setting(setting_key(owner, 'vivino', 'dataset'))
    from .telegram_data_jobs import report_identifier
    identifier = report_identifier(payload['report_id'])
    report = store.one('SELECT id,section,quality FROM reports WHERE id=?', (identifier,))
    if report and (report['section'] != 'vivino' or json.loads(report['quality']).get('owner') != owner
                   or json.loads(report['quality']).get('viewer', owner) != actor
                   or json.loads(report['quality']).get('generation') != payload['generation']):
        raise PermissionError('Report is not owned by this dataset')
    if not report:
        folder = store.path.parent / 'vivino' / str(owner)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (identifier + '.html')
        view = model(store, actor, owner)
        content = render(view)
        if not active(store, reference):
            raise PermissionError('Vivino access changed')
        path.write_text(content, encoding='utf-8', newline='\n')
        summary = view['analytics']['summary']
        text = (f"Vivino · географический атлас пользователя {owner}\n"
                f"Разных вин попробовано: {summary['tasted_wines']}; стран: {summary['countries']}; сортов в винах: {summary['grapes']}.\n"
                'Скачайте HTML и откройте в браузере. Страны окрашены по числу разных вин; есть сортовой срез, аналитика и предложения. Работает без интернета.\n'
                f"Неизвестен или неполон состав у {summary['unknown_composition']} вин. Уточните сорта в карточке вина. "
                'Справочник регионов неполный; точные точки регионов включаются отдельно.')
        with store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = conn.execute('SELECT value FROM settings WHERE key=?', (setting_key(owner, 'vivino', 'dataset'),)).fetchone()
            user = conn.execute('SELECT enabled,role FROM users WHERE user_id=?', (actor,)).fetchone()
            if (not current or json.loads(current[0])['generation'] != payload['generation'] or not user or not user[0]
                    or owner != actor and user['role'] != 'admin'):
                path.unlink(missing_ok=True)
                raise PermissionError('Vivino access changed')
            conn.execute('''INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,parts,quality)
                VALUES(?,'vivino',?,0,0,?,?,?,?)''', (identifier, utcnow(), text, text,
                dumps([{'kind': 'document', 'path': str(path), 'caption': 'Географический атлас Vivino · открыть HTML в браузере'}]),
                dumps({'owner': owner, 'viewer': actor, 'generation': payload['generation'], 'map_version': MAP_VERSION, **dataset['quality']})))
    if not dry_run and active(store, reference):
        prepare_delivery(store, identifier, str(actor), detail='full', project_ref=reference)
