"""Wine project navigation and reviewed, personal cellar operations."""
import json
import math
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import uuid4

from . import wine_cellar as cellar
from .feature_access import allowed, require
from .store import utcnow
from .telegram_charts import keyboard
from .telegram_navigation import screen

PAGE_SIZE = 8
FIELDS = ('name', 'winery', 'vintage', 'quantity', 'volume_ml', 'purchase_price', 'purchase_date', 'location')
EXAMPLES = {'name': 'Merlot Reserva', 'winery': 'Bodega Ejemplo', 'vintage': '2021', 'quantity': '2',
            'volume_ml': '750', 'purchase_price': '1250,50', 'purchase_date': '01.10.2025', 'location': 'Шкаф, полка 2'}
PROMPTS = {
    'name': 'Как называется вино? Перепишите название с этикетки.',
    'winery': 'Укажите производителя. Если неизвестен, нажмите «Пропустить».',
    'vintage': 'Какой год урожая? Если он не указан на этикетке, нажмите «Пропустить».',
    'quantity': 'Сколько бутылок добавить? Введите целое число.',
    'volume_ml': 'Какой объём одной бутылки в миллилитрах? 0,75 л = 750 мл; 1,5 л = 1500 мл.',
    'purchase_price': 'Сколько стоила одна бутылка в рублях? Укажите цену за бутылку, а не сумму всей покупки. Неизвестную цену можно пропустить.',
    'purchase_date': 'Когда купили вино? Введите ДД.ММ.ГГГГ или ГГГГ-ММ-ДД. Неизвестную дату можно пропустить.',
    'location': 'Где хранится вино? Например: шкаф, полка 2. Можно пропустить.',
}


def wine_edit_prompt(field, value, prompt, *, editing=True):
    labels = {'name': 'Название', 'winery': 'Производитель', 'vintage': 'Год урожая',
              'quantity': 'Количество', 'volume_ml': 'Объём бутылки', 'purchase_price': 'Цена покупки за бутылку',
              'purchase_date': 'Дата покупки', 'location': 'Место хранения', 'country': 'Страна',
              'region': 'Регион', 'grapes': 'Сорта винограда', 'amount': 'Оплачено за товар',
              'total': 'Итог чека', 'date': 'Дата чека', 'consumed_on': 'Дата списания',
              'price': 'Оценка бутылки на дату списания'}
    if value in (None, '', []):
        current = 'не указано'
    elif field in {'purchase_price', 'amount', 'total', 'price'}:
        current = f'{float(value):,.2f}'.replace(',', ' ').replace('.', ',')+' ₽'
    elif field in {'volume_ml', 'quantity'}:
        current = f'{float(value):g}'+(' мл' if field == 'volume_ml' else '')
    elif isinstance(value, list):
        current = ', '.join(map(str, value))
    else:
        current = str(value)
    title = labels[field]+(' · изменение' if editing else '')
    return f'{title}\nСейчас: {current}\n\n{prompt}'


def _id(value):
    raw = str(value)
    if not raw.isascii() or not raw.isdigit() or not 0 < int(raw) < 2**63:
        raise ValueError('Нужен положительный числовой ID.')
    return int(raw)


def _page(value):
    raw = str(value)
    if not raw.isascii() or not raw.isdigit() or int(raw) > 100000:
        raise ValueError('Неверная страница.')
    return int(raw)


def _amount(text):
    try:
        value = float(text.strip().replace(' ', '').replace(',', '.'))
    except (ValueError, TypeError):
        raise ValueError('Введите цену в рублях, например: 1250,50.') from None
    if not math.isfinite(value) or not 0 <= value <= 1_000_000_000:
        raise ValueError('Цена должна быть от 0 до 1 000 000 000 ₽.')
    return round(value, 2)


def _date(text):
    raw = text.strip()
    for fmt in ('%Y-%m-%d', '%d.%m.%Y'):
        try:
            result = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).date()
            if result > cellar._today():
                raise ValueError('Дата ещё не наступила.')
            return result.isoformat()
        except ValueError:
            pass
    raise ValueError('Введите существующую дату не позже сегодня: ДД.ММ.ГГГГ или ГГГГ-ММ-ДД.')


def _money(value):
    return 'не указана' if value is None else f'{float(value):,.2f} ₽'.replace(',', ' ')


def _result_money(value):
    return 'не рассчитан' if value is None else f'{float(value):+,.2f} ₽'.replace(',', ' ')


def _valuation_source(value):
    return {'manual_consumption': 'цена введена при списании', 'manual': 'ручная оценка',
            'shop': 'магазин', 'shop_min': 'минимальная подтверждённая цена магазина',
            'shop_confirmed': 'подтверждённая цена магазина'}.get(value, 'сохранённая оценка' if value else 'нет оценки')


def _source_status(source):
    if source.get('status_label'):
        return str(source['status_label'])
    status = source.get('status')
    return {'ok': 'проверен', 'blocked': 'автоматический доступ заблокирован', 'error': 'ошибка проверки',
            'not_checked': 'ещё не проверен', 'unsupported': 'формат не поддерживается',
            'manual': 'ручной режим', 'manual_only': 'доступен только ручной поиск',
            'disabled': 'не подключён', 'empty': 'совпадений нет', 'unavailable': 'проверяемые данные недоступны',
            'dry_run': 'проверка не запускалась', 'cached': 'сохранённый результат'}.get(status, status or 'неизвестно')


def _shop_mode(item):
    if item.get('metadata_supported'):
        return 'сведения о вине'
    search = item.get('search_supported', item.get('automatic', False))
    price = item.get('price_supported', item.get('automatic', False))
    return 'поиск и цены' if search and price else 'поиск' if search else 'цены по ссылке' if price else 'ручной режим'


def availability_label(quote):
    """Translate the published availability scope without broadening its reach."""
    scope = quote.get('availability_scope')
    if scope:
        return {'moscow_branches': 'в магазинах Москвы, перечисленных в карточке источника',
                'selected_moscow_shop': 'в выбранном магазине Москвы'}.get(scope, str(scope))
    return {'InStock': 'есть в наличии; адрес конкретного магазина не подтверждён',
            'LimitedAvailability': 'ограниченное наличие; адрес магазина не подтверждён',
            'OutOfStock': 'нет в наличии'}.get(quote.get('availability'), 'не подтверждено для конкретного магазина')


def availability_addresses(quote):
    values = quote.get('available_shops')
    values = list(values) if isinstance(values, (list, tuple)) else []
    for key in ('shop_address', 'selected_shop_address'):
        if quote.get(key):
            values.append(quote[key])
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get('address') or value.get('name')
        if isinstance(value, str):
            value = ' '.join(value.split())[:250]
            if value and value not in result:
                result.append(value)
    return result[:8]


def _name(lot):
    result = str(lot.get('name') or 'Без названия')
    if lot.get('winery'):
        result += ' · ' + str(lot['winery'])
    if lot.get('vintage'):
        result += ' · ' + str(lot['vintage'])
    return result


class WineControls:
    def __init__(self, bot):
        self.bot, self.store, self.reply = bot, bot.store, bot._reply

    def _pending(self, actor, action):
        self.bot.market_controls.pending(actor, action)

    def _leave(self, actor):
        self._pending(actor, None)
        self.store.set_setting(self._key(actor), None)

    def _key(self, actor):
        return f'wine_draft:{actor}'

    def _require(self, actor, owner=None, write=False):
        require(self.store, actor, 'wine_cellar')
        return cellar.require_owner(self.store, actor, owner, write=write)

    def _draft(self, actor):
        draft = self.store.get_setting(self._key(actor))
        if not isinstance(draft, dict) or draft.get('expires', '') < utcnow():
            self._pending(actor, None)
            raise ValueError('Форма устарела. Добавьте вино заново.')
        return draft

    def _save_draft(self, actor, draft):
        draft['expires'] = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        self.store.set_setting(self._key(actor), draft)

    def home(self, client, actor):
        from .subscription_access import require as require_section
        require_section(self.store, actor, 'vivino')
        self._leave(actor)
        screen('pj:vivino', 'pj:life')
        rows = []
        if allowed(self.store, actor, 'wine_cellar'):
            rows.append([('Моя винотека', 'wc:menu')])
        if allowed(self.store, actor, 'vivino_map') or allowed(self.store, actor, 'vivino_plan'):
            rows.append([('Vivino · карта, отзывы и план', 'vi:menu')])
        if self.store.authorize(actor, admin=True):
            rows.append([('Магазины · Москва', 'wc:marketshops')])
        self.reply(client, actor, 'Вино\n\nМоя винотека — бутылки в запасе, покупки и стоимость.\n'
                   'Vivino — попробованные вина, карта, сорта и план дегустаций.\n\n'
                   'Новый отзыв может предложить списать бутылку из запасов. Списание всегда требует подтверждения.', keyboard(rows))

    def empty_guidance(self, actor, owner):
        if owner != actor:
            return 'В запасе пользователя бутылок нет.', []
        from .feature_access import allowed
        if allowed(self.store, actor, 'life_finance'):
            from .life_ledger import receipt_wine_sources
            sources = receipt_wine_sources(self.store, actor)
            if sources:
                source = sources[0]
                if source['kind'] == 'saved':
                    return ((f'Есть учтённый чек: винных строк для добавления — {source["wine_count"]}. '
                            'Расход сохранён, но эти вина ещё не добавлены в запасы. '
                            'Откройте чек и подтвердите добавление; повторного расхода не будет.'),
                            [[('Добавить вина из учтённого чека', 'rc:cellar:'+source['token'])]])
                return ((f'Есть незавершённый чек: винных строк — {source["wine_count"]}. '
                        'Выбор галочкой ещё не добавляет бутылки. Откройте чек и подтвердите сохранение.'),
                        [[('Продолжить добавление из чека', 'rc:review:'+source['token'])]])
        if cellar.lots(self.store, actor, include_empty=True):
            return 'В запасе бутылок нет. История добавлений и списаний сохранена.', []
        return 'Бутылки ещё не добавлены. Нажмите «Добавить вино» и выберите удобный способ.', []

    def menu(self, client, actor, owner=None, page=0, notice=''):
        owner = self._require(actor, owner)
        self._leave(actor)
        screen('wc:menu','pj:vivino')
        values = cellar.portfolio(self.store,actor,owner=owner)
        from .wine_details import missing
        purchases = cellar.purchase_history(self.store,actor,owner=owner)
        incomplete = [lot for lot in purchases if missing(lot)]
        lines = ['Моя винотека' if owner==actor else 'Винотека · '+self._owner_label(owner),
                 f"В запасе: {values['quantity']} бутылок; партий: {values['lot_count']}.",
                 f"Всего куплено: {values['purchased_quantity']}; выпито: {values['consumed_quantity']}.",
                 f"Затраты на оставшийся запас: {_money(values['purchase_cost'])}.",
                 f"Оценка запаса: {_money(values['current_value'])}.",
                 (f"Открытый результат: {_result_money(values['open_result'])} "
                  f"(сопоставлено {values['open_compared_quantity']} из {values['quantity']} бутылок)."),
                 f"Зафиксировано по выпитым: {_result_money(values['closed_result'])}."]
        if values['open_unknown_quantity'] or values['closed_unknown_quantity']:
            lines.append(f"Недостаточно цен для результата: в запасе {values['open_unknown_quantity']}, выпито {values['closed_unknown_quantity']} бутылок.")
        lines.append('Оценка — по последним сохранённым ценам; первоначальная цена покупки не подтверждает изменение стоимости. '
                     'При списании разница с ценой покупки фиксируется и не меняется от следующих цен. Это условная прибыль/убыток винотеки.')
        rows = [[('Винный погреб · запасы и архив',f'wc:list:{owner}:0')]]
        if owner==actor:
            rows.append([('Добавить вино','wc:add')])
            if incomplete:
                lines.append(f'\nТребуют уточнения: {len(incomplete)} партий. Дополните производителя, год/NV, страну, регион и сорта для полной статистики.')
            rows.append([(f'Дозаполнить сведения · {len(incomplete)}','wc:incomplete:0')])
        rows += [[('Стоимость и динамика',f'wc:value:{owner}')],
                 [('Статистика · страны, регионы, сорта и годы',f'wc:stats:{owner}:country')],
                 [('Таблица и экспорт',f'wc:exports:{owner}')]]
        if owner==actor:
            rows.append([('Предложения по отзывам Vivino','wc:proposals:0')])
            if self.store.authorize(actor,admin=True):
                rows.append([('Винотеки пользователей · просмотр','wc:users:0')])
        if not values['quantity']:
            hint,recovery = self.empty_guidance(actor,owner)
            lines.append('\n'+hint)
            rows += recovery
        if notice:
            lines.insert(0,notice+'\n')
        return self.reply(client,actor,'\n'.join(lines),keyboard(rows))

    def storage(self, client, actor, owner=None, page=0, notice=''):
        owner = self._require(actor, owner)
        self._leave(actor)
        screen(f'wc:list:{owner}:{page}', 'wc:menu' if owner == actor else 'wc:users:0')
        items = cellar.lots(self.store, actor, owner=owner)
        total = len(items)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(_page(page), pages - 1)
        title = 'Винный погреб · в запасе' if owner == actor else f'Винный погреб · {self._owner_label(owner)}'
        lines = [title, f'Партий: {total}; бутылок: {sum(int(item["quantity"]) for item in items)}.']
        if notice:
            lines.insert(0, notice+'\n')
        recovery = []
        if not items:
            hint, recovery = self.empty_guidance(actor, owner)
            lines.append(hint)
        rows = [[('✓ В запасе',f'wc:list:{owner}:0'),('Архив · выпито',f'wc:consumed:{owner}:0')]]
        rows += [[(f'{_name(item)[:64]} · {item["quantity"]} шт.', f'wc:lot:{owner}:{item["id"]}')]
                for item in items[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]]
        if pages > 1:
            nav = []
            if page:
                nav.append(('‹ Предыдущие', f'wc:list:{owner}:{page - 1}'))
            if page + 1 < pages:
                nav.append(('Следующие ›', f'wc:list:{owner}:{page + 1}'))
            rows.append(nav)
            lines.append(f'Страница {page + 1}/{pages}.')
        if owner == actor:
            rows += [[('Добавить вино', 'wc:add')]]
        rows.append([('Все покупки',f'wc:purchases:{owner}:0')])
        rows += recovery
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def _owner_label(self, owner):
        from .user_management import label
        return label(self.store, owner)

    def card(self, client, actor, lot_id, owner=None):
        owner = self._require(actor, owner)
        lot = cellar.get_lot(self.store, actor, lot_id, owner=owner)
        self._leave(actor)
        previous = self.store.get_setting(f'ui_screen:{actor}', {}).get('route', '')
        back = previous if previous.startswith((f'wc:purchases:{owner}:',f'wc:list:{owner}:',f'wc:consumption:{owner}:')) else f'wc:list:{owner}:0' if lot['quantity'] else f'wc:consumed:{owner}:0'
        screen(f'wc:lot:{owner}:{lot_id}', back)
        lines = [_name(lot), f'В запасе: {lot["quantity"]} бутылок.',
                 f'Объём одной бутылки: {lot.get("volume_ml", 750)} мл.',
                 f'Покупка за бутылку: {_money(lot.get("purchase_price"))}.',
                 f'Дата покупки: {lot.get("purchase_date") or "не указана"}.',
                 f'Хранение: {lot.get("location") or "не указано"}.']
        lines += [f'Страна: {lot["country"] or "не указана"}.', f'Регион: {lot["region"] or "не указан"}.',
                  'Сорта винограда: '+(', '.join(lot['grapes']) or 'не указаны')+'.']
        history = next(row for row in cellar.purchase_history(self.store, actor, owner=owner) if row['id'] == lot_id)
        lines += [f'Всего куплено в этой партии: {history["purchased_quantity"]}; выпито: {history["consumed_quantity"]}.',
                  f'Зафиксированный результат выпитых: {_result_money(history["closed_result"])}.']
        if lot.get('notes'):
            lines.append('Сведения и заметки: '+lot['notes'])
        latest = lot.get('latest_price')
        if isinstance(latest, dict):
            if latest.get('source') == 'purchase_undated':
                lines.append(f'Начальная оценка по цене покупки: {_money(latest.get("amount"))}; дата покупки неизвестна. '
                             'Это не новая цена магазина.')
            else:
                lines.append(f'Последняя оценка за бутылку: {_money(latest.get("amount"))}; '
                             f'дата {latest.get("price_date") or "неизвестна"}.')
            metadata = latest.get('metadata')
            if isinstance(metadata, dict) and metadata.get('availability_scope'):
                lines.append(f'Магазин: {metadata.get("shop_title") or metadata.get("shop_name") or "указан в источнике"}; '
                             f'наличие: {availability_label(metadata)}.')
                lines.extend('• ' + address for address in availability_addresses(metadata))
        links = cellar.price_links(self.store, actor, lot_id, owner=owner)
        lines.append(f'Ссылок на это вино в магазинах: {len(links)}.')
        rows = []
        if owner == actor:
            if lot['quantity']:
                rows.append([('Выпил одну бутылку', f'wc:stock:{lot_id}:drink')])
            rows += [[('Добавить бутылки', f'wc:stock:{lot_id}:add')],
                     [('Уточнить характеристики', f'wc:details:{lot_id}')],
                     [('Указать цену и дату', f'wc:price:{lot_id}')],
                     [('Цены в Москве', f'wc:retail:{lot_id}')]]
        else:
            lines.append('Просмотр администратора. Изменять чужие запасы нельзя.')
        if history['consumed_quantity']:
            rows.append([('Списания и зафиксированный результат', f'wc:drinks:{owner}:{lot_id}:0')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def statistics(self, client, actor, owner, field='country'):
        owner = self._require(actor, owner)
        labels = {'country': 'Страны', 'region': 'Регионы', 'grapes': 'Сорта винограда', 'vintage': 'Годы урожая'}
        if field not in labels:
            raise ValueError('Неизвестная группировка вин')
        self._leave(actor)
        values = cellar.attribute_statistics(self.store, actor, owner=owner)
        screen(f'wc:stats:{owner}:{field}', 'wc:menu' if owner==actor else f'wc:list:{owner}:0')
        lines = ['Статистика вин · '+labels[field],
                 f'Всего куплено: {values["purchased"]}; осталось: {values["remaining"]}.',
                 'Показатели строки: куплено / в запасе / выпито.']
        groups = list(values['groups'][field].items())
        for name, count in groups[:15]:
            lines.append(f'{name[:120]}: {count["purchased"]} / {count["remaining"]} / {count["consumed"]}')
        if len(groups) > 15:
            lines.append('Показаны первые 15 групп. Полные характеристики доступны в таблице и Excel.')
        if not groups:
            lines.append('Покупок пока нет. Сведения из найденных источников сохраняются вместе с выбранными винами из чека.')
        if field == 'grapes':
            lines.append('Купаж учитывается в каждом указанном сорте; суммы по сортам могут превышать число бутылок.')
        lines.append('Неизвестные сведения остаются в группе «Не указано». Год из интернет-карточки требует сверки с вашей бутылкой.')
        rows = [[(('✓ ' if key == field else '')+label, f'wc:stats:{owner}:{key}')] for key, label in labels.items()]
        rows.append([('Таблица и экспорт', f'wc:exports:{owner}')])
        if owner == actor:
            rows.append([('Дозаполнить недостающие сведения','wc:incomplete:0')])
        return self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def purchases(self, client, actor, owner=None, page=0):
        owner = self._require(actor, owner)
        self._leave(actor)
        items = cellar.purchase_history(self.store, actor, owner=owner)
        pages = max(1, (len(items)+PAGE_SIZE-1)//PAGE_SIZE)
        page = min(_page(page), pages-1)
        screen(f'wc:purchases:{owner}:{page}', f'wc:list:{owner}:0')
        lines = ['Все покупки вина', f'Партий: {len(items)}. Здесь остаются и полностью выпитые вина.']
        rows = [[('В запасе',f'wc:list:{owner}:0'),('Архив · выпито',f'wc:consumed:{owner}:0')]]
        for lot in items[page*PAGE_SIZE:(page+1)*PAGE_SIZE]:
            lines.append(f'\n{_name(lot)[:180]}\nКуплено: {lot["purchased_quantity"]}; выпито: {lot["consumed_quantity"]}; осталось: {lot["quantity"]}. '
                         f'Покупка: {_money(lot["purchase_price"])} за бутылку; {lot.get("purchase_date") or "дата неизвестна"}.')
            rows.append([(f'{lot["name"][:42]} · осталось {lot["quantity"]}', f'wc:lot:{owner}:{lot["id"]}')])
        if not items:
            lines.append('Пока нет сохранённых покупок.')
        self._history_pages(rows, 'purchases', owner, page, pages)
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def _history_pages(self, rows, kind, owner, page, pages, lot_id=None):
        prefix = f'wc:{kind}:{owner}'+(f':{lot_id}' if lot_id is not None else '')
        nav = []
        if page:
            nav.append(('‹ Предыдущие', f'{prefix}:{page-1}'))
        if page+1 < pages:
            nav.append(('Следующие ›', f'{prefix}:{page+1}'))
        if nav:
            rows.append(nav)

    def consumed(self, client, actor, owner=None, page=0, lot_id=None):
        owner = self._require(actor, owner)
        self._leave(actor)
        items = cellar.consumptions(self.store, actor, owner=owner, lot_id=lot_id)
        pages = max(1, (len(items)+PAGE_SIZE-1)//PAGE_SIZE)
        page = min(_page(page), pages-1)
        route = f'wc:consumed:{owner}:{page}' if lot_id is None else f'wc:drinks:{owner}:{lot_id}:{page}'
        screen(route, f'wc:list:{owner}:0' if lot_id is None else f'wc:lot:{owner}:{lot_id}')
        known = sum(row['result'] for row in items if row['result'] is not None)
        missing = sum(row['quantity'] for row in items if row['result'] is None)
        known_display = None if items and not any(row['result'] is not None for row in items) else known
        lines = ['Архив · выпито и зафиксированный результат', f'Выпито бутылок: {sum(row["quantity"] for row in items)}.',
                 f'Результат по бутылкам с известными ценами: {_result_money(known_display)}.',
                 f'Без расчёта результата: {missing} бутылок.',
                 'Результат = (оценка при списании − цена покупки) × количество. Это условная прибыль/убыток винотеки, без денежного дохода.']
        rows = [[('В запасе',f'wc:list:{owner}:0'),('✓ Архив · выпито',f'wc:consumed:{owner}:0')]] if lot_id is None else []
        for item in items[page*PAGE_SIZE:(page+1)*PAGE_SIZE]:
            lines.append(f'\n{item["consumed_on"]} · {item["name"][:160]} · {item["quantity"]} шт.\n'
                         f'{_money(item["purchase_price"])} → {_money(item["valuation_price"])}; результат {_result_money(item["result"])}.')
            rows.append([(f'{item["consumed_on"]} · {item["name"][:35]}', f'wc:consumption:{owner}:{item["id"]}')])
        if not items:
            lines.append('\nСписаний пока нет. Откройте бутылку в запасах и нажмите «Выпил одну бутылку».')
        self._history_pages(rows, 'consumed' if lot_id is None else 'drinks', owner, page, pages, lot_id)
        if lot_id is None:
            rows.append([('Все покупки',f'wc:purchases:{owner}:0')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def consumption_card(self, client, actor, owner, identifier):
        owner = self._require(actor, owner)
        item = next((row for row in cellar.consumptions(self.store, actor, owner=owner) if row['id'] == identifier), None)
        if not item:
            raise ValueError('Списание не найдено в этой винотеке.')
        self._leave(actor)
        screen(f'wc:consumption:{owner}:{identifier}', f'wc:consumed:{owner}:0')
        lines = [_name(item), f'Выпито: {item["quantity"]} шт. · {item["consumed_on"]}.',
                 f'Цена покупки за бутылку: {_money(item["purchase_price"])}.',
                 f'Зафиксированная оценка за бутылку: {_money(item["valuation_price"])}.',
                 f'Условная прибыль/убыток: {_result_money(item["result"])}.',
                 f'Дата оценки: {item.get("price_date") or "не указана"}; источник: {_valuation_source(item.get("price_source"))}.',
                 'Последующие изменения магазинных цен не меняют этот результат.']
        if item.get('price_url'):
            lines.append(item['price_url'])
        self.reply(client, actor, '\n'.join(lines), keyboard([[('Карточка вина', f'wc:lot:{owner}:{item["lot_id"]}')]]))

    def add_menu(self, client, actor):
        self._require(actor, write=True)
        self._leave(actor)
        screen('wc:add', 'wc:menu')
        text = ('Добавить вино\nВыберите способ.\n\n'
                'Вручную — название, производитель, год, количество, объём, цена, дата и хранение; одно поле за шаг.\n'
                'По фото этикетки — бот предложит данные, а вы проверите их перед добавлением.')
        rows = [[('Ввести вручную', 'wc:manual')], [('По фото этикетки', 'wc:photo')]]
        if allowed(self.store, actor, 'life_finance'):
            text += '\nПо чеку — загрузите PDF или фото, проверьте покупку и выберите вина для запаса. Сохранится также расход.'
            rows.append([('По чеку · PDF или фото', 'lf:receipt')])
            rows.append([('Учтённые чеки · добавить вина', 'lf:receipts:0')])
        rows.append([('Пример ручного ввода', 'wc:example')])
        self.reply(client, actor, text+'\n\nВино появится в запасах после вашего подтверждения.', keyboard(rows))

    def manual_example(self, client, actor):
        self._require(actor, write=True)
        self._leave(actor)
        screen('wc:example', 'wc:add')
        labels = ('Название', 'Производитель', 'Год урожая', 'Количество бутылок', 'Объём одной бутылки, мл',
                  'Цена за одну бутылку, ₽', 'Дата покупки', 'Место хранения')
        text = 'Пример ручного ввода\nБот задаёт вопросы по одному. Отправляйте только значение текущего поля, отдельным сообщением.\n\n'
        text += '\n'.join(f'{i}. {label}: {EXAMPLES[field]}' for i, (label, field) in enumerate(zip(labels, FIELDS, strict=True), 1))
        text += ('\n\nЭто пример: замените значения сведениями о своём вине. Для двух бутылок по 1250,50 ₽ '
                 'в поле цены вводится 1250,50, а не общая сумма 2501. Объём 0,75 л вводится как 750 мл.\n'
                 'Неизвестные производителя, год, цену, дату и хранение можно пропустить кнопкой. '
                 'Название, количество и объём обязательны. В конце проверьте карточку и нажмите «Сохранить в винотеку».')
        self.reply(client, actor, text, keyboard([[('Начать ручной ввод', 'wc:manual')]]))

    def photo_prompt(self, client, actor):
        self._require(actor, write=True)
        self.store.set_setting(self._key(actor), None)
        self._pending(actor, 'wine:photo')
        screen('wc:photo', 'wc:add')
        self.reply(client, actor, 'Добавление по этикетке\nПришлите чёткое фото лицевой этикетки. '
                   'Название, производитель и год будут предложены для проверки. '
                   'Бутылка появится в запасах только после вашего подтверждения. '
                   'Если распознавание недоступно или ошибётся, название можно ввести вручную.',
                   keyboard([[('Добавить вручную', 'wc:manual')], [('Отмена', 'wc:cancel')]]))

    def market_shops(self, client, actor):
        from .wine_retail import shops
        self._require(actor)
        if not self.store.authorize(actor, admin=True):
            raise PermissionError('Магазины настраивает администратор.')
        self._leave(actor)
        screen('wc:marketshops', 'pj:vivino')
        items = shops(self.store)
        lines = ['Магазины и справочники вина', ('Галочка подключает источник для пользователей. '
                 'Состояние автоматического поиска и получения цены показано отдельно.')]
        rows = []
        for item in items:
            selected = bool(item.get('enabled'))
            title = str(item.get('title') or item['id'])
            mode = _shop_mode(item)
            lines.append(f'\n{title}: {"подключён" if selected else "не подключён"}. '
                         f'{item.get("status_label") or mode}.')
            if item.get('reason'):
                lines.append(str(item['reason'])[:450])
            observed = item.get('last_check')
            if isinstance(observed, dict) and observed:
                lines.append(f'Последняя проверка: {_source_status(observed)}; {observed.get("checked_at") or "дата неизвестна"}. '
                             + str(observed.get('reason') or '')[:400])
            rows.append([(f'{"✓" if selected else "○"} {title} · {mode}', f'wc:shop:{item["id"]}')])
        rows.append([('Разрешённые домены · дополнительно', 'wc:shops')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def retail(self, client, actor, lot_id):
        from .wine_retail import shops
        self._require(actor, write=True)
        lot = cellar.get_lot(self.store, actor, lot_id)
        self._leave(actor)
        screen(f'wc:retail:{lot_id}', f'wc:lot:{actor}:{lot_id}')
        items = shops(self.store)
        enabled = [item for item in items if item.get('enabled')]
        lines = ['Цены в Москве', _name(lot), (f'Год: {lot.get("vintage") or "не указан"}; '
                 f'объём: {lot.get("volume_ml", 750)} мл.'), '\nИсточники:']
        for item in items:
            lines.append(f'• {item.get("title") or item["id"]}: '
                         f'{item.get("status_label") or _shop_mode(item)} '
                         f'({"подключён" if item.get("enabled") else "не подключён"}).')
            if item.get('reason'):
                lines.append(str(item['reason'])[:400])
            observed = item.get('last_check')
            if isinstance(observed, dict) and observed:
                lines.append(f'Последняя проверка: {_source_status(observed)}; {observed.get("checked_at") or "дата неизвестна"}. '
                             + str(observed.get('reason') or '')[:400])
            if item.get('homepage') and not item.get('automatic'):
                lines.append(str(item['homepage']))
        links = cellar.price_links(self.store, actor, lot_id)
        lines.append(f'\nСохранённых ссылок для этой бутылки: {len(links)}.')
        lines.append('Найденную карточку нужно подтвердить по названию, производителю, году и объёму. '
                     'Цена конкретного магазина, клуба или другого региона не становится автоматически ценой в Москве.')
        lines.append('Поиск проверяет первые 3 карточки каждого источника с поддержкой поиска. '
                     'Результаты действуют 30 минут; если нужное вино не найдено, уточните название или сохраните точную ссылку.')
        rows = []
        if any(item.get('search_supported', item.get('automatic')) for item in enabled):
            rows.append([('Найти это вино в магазинах', f'wc:search:{lot_id}')])
        else:
            lines.append('\nАвтоматический поиск сейчас не подключён. Можно сохранить ссылку или указать цену вручную.')
        rows += [[('Добавить ссылку магазина', f'wc:link:{lot_id}')], [('Указать цену вручную', f'wc:price:{lot_id}')]]
        if links:
            rows.append([('Обновить цены по моим ссылкам', f'wc:refresh:{lot_id}')])
        if self.store.authorize(actor, admin=True):
            rows.append([('Настроить магазины', 'wc:marketshops')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def offers(self, client, actor, token):
        from .wine_retail import candidates
        self._require(actor, write=True)
        result = candidates(self.store, actor, token)
        lot_id = _id(result['lot_id'])
        cellar.get_lot(self.store, actor, lot_id)
        self._leave(actor)
        screen(f'wc:offers:{token}', f'wc:retail:{lot_id}')
        items = result.get('candidates') or []
        lines = ['Поиск вина · Москва', f'Карточек для проверки: {len(items)}.',
                 'Откройте карточку и проверьте совпадение. До подтверждения оценки винотеки не меняются.']
        lines.append('Проверены первые 3 карточки каждого источника с поддержкой поиска. Результат действует 30 минут.')
        if result.get('expires'):
            lines.append('Срок проверки результата: до ' + str(result['expires']) + '.')
        if not items:
            lines.append('Подходящие карточки не найдены. Сохранённые цены не изменены.')
        for source in result.get('sources') or []:
            if isinstance(source, dict):
                lines.append(f'\n{source.get("title") or source.get("shop_title") or source.get("id") or "Магазин"}: '
                             f'{_source_status(source)}. '
                             + str(source.get('reason') or source.get('message') or '')[:450])
        rows = []
        for index, quote in enumerate(items[:30]):
            title = str(quote.get('shop_title') or quote.get('shop_id') or quote.get('title') or quote.get('name') or 'Карточка')[:45]
            amount = quote.get('amount', quote.get('price'))
            rows.append([(f'{"Проверить" if quote.get("match") is True else "Есть расхождения"}: {title} · {_money(amount)}',
                          f'wc:offer:{token}:{index}')])
        if len(items) > 30:
            lines.append('Показаны первые 30 карточек. Уточните название вина для более точного поиска.')
        rows.append([('Повторить поиск', f'wc:search:{lot_id}')])
        rows.append([('Добавить ссылку вручную', f'wc:link:{lot_id}')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def offer(self, client, actor, token, index):
        from .wine_retail import candidates
        self._require(actor, write=True)
        result = candidates(self.store, actor, token)
        items = result.get('candidates') or []
        if index >= len(items):
            raise ValueError('Карточка магазина недоступна. Повторите поиск.')
        quote = items[index]
        lot = cellar.get_lot(self.store, actor, _id(result['lot_id']))
        self._leave(actor)
        screen(f'wc:offer:{token}:{index}', f'wc:offers:{token}')
        amount = quote.get('amount', quote.get('price'))
        price_kind = {'public': 'обычная публичная цена', 'regular': 'обычная публичная цена',
                      'club': 'цена по карте клуба', 'loyalty': 'цена для участников программы',
                      'promotion': 'акционная цена'}.get(quote.get('price_kind'), quote.get('price_kind') or 'не уточнены')
        lines = [str(quote.get('shop_title') or quote.get('shop_id') or 'Карточка магазина'),
                 str(quote.get('title') or quote.get('name') or 'Без названия'),
                 f'Производитель: {quote.get("winery") or "не указан"}.',
                 f'Год: {quote.get("vintage") or "не указан"}; объём: {quote.get("volume_ml") or "не указан"} мл.',
                 f'Цена за бутылку: {_money(amount)}.',
                 f'Город: {quote.get("city") or "не подтверждён источником"}.',
                 f'Условия цены: {price_kind}.',
                 f'Наличие: {availability_label(quote)}.',
                 f'Проверено: {quote.get("checked_at") or quote.get("as_of") or quote.get("retrieved_at") or result.get("created_at") or "дата неизвестна"}.',
                 str(quote.get('url') or quote.get('source') or ''),
                 '\nВаша бутылка: ' + _name(lot) + f' · {lot.get("volume_ml", 750)} мл.']
        addresses = availability_addresses(quote)
        if addresses:
            lines.append('\nАдреса наличия на момент проверки:')
            lines.extend('• ' + address for address in addresses)
        matches = quote.get('match') is True
        lines.append('Автоматическая проверка не нашла расхождений. Проверьте карточку перед сохранением.' if matches else
                     'Совпадение не подтверждено. Эту цену нельзя включить в оценку бутылки.')
        reasons = quote.get('match_reasons') or quote.get('reasons') or []
        if isinstance(reasons, str):
            reasons = [reasons]
        lines.extend(str(reason)[:300] for reason in reasons[:6])
        rows = [[('Это моё вино · сохранить оценку', f'wc:offerconfirm:{token}:{index}')]] if matches else []
        rows.append([('К результатам поиска', f'wc:offers:{token}')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def _begin(self, client, actor, fields=None, warnings=None, recognition_token=None):
        self._require(actor, write=True)
        token = uuid4().hex
        draft = {'token': token, 'mode': 'add', 'fields': {'quantity': 1, 'volume_ml': 750}, 'step': 'name'}
        if recognition_token:
            draft['recognition_token'] = recognition_token
        if fields:
            draft['fields'].update({key: value for key, value in fields.items()
                                    if key in {'name', 'winery', 'vintage', 'volume_ml'} and value is not None})
            draft['warnings'] = [str(value)[:200] for value in (warnings or [])[:5]]
            draft['step'] = 'review' if draft['fields'].get('name') else 'name'
        self._save_draft(actor, draft)
        if draft['step'] == 'review':
            self.review(client, actor)
        else:
            self._prompt(client, actor, draft)

    def _prompt(self, client, actor, draft):
        field = draft['step']
        self._pending(actor, f'wine:add:{field}')
        screen('wc:form', 'wc:add')
        rows = []
        if field not in {'name', 'quantity', 'volume_ml'}:
            rows.append([('Пропустить', 'wc:skip')])
        if field == 'volume_ml':
            rows.append([('Обычная бутылка · 750 мл', 'wc:volume:750')])
        index = FIELDS.index(field)
        if index:
            rows.append([('← Предыдущее поле', 'wc:previous')])
        rows.append([('Отменить добавление', 'wc:cancel')])
        example = f'\nПример ответа: {EXAMPLES[field]}'
        instruction = '\nОтправляйте по одному значению, без названия поля.' if field == 'name' else ''
        prompt = PROMPTS[field]+example+instruction
        if draft.get('return_review') or field in draft['fields']:
            prompt = wine_edit_prompt(field, draft['fields'].get(field), prompt, editing=bool(draft.get('return_review')))
        self.reply(client, actor, f'Добавление вручную · шаг {index + 1}/{len(FIELDS)}\n{prompt}', keyboard(rows))

    def _field(self, field, text):
        value = text.strip()
        if field == 'quantity':
            if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 10000:
                raise ValueError('Количество: целое число от 1 до 10 000.')
            return int(value)
        if field == 'volume_ml':
            if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 30000:
                raise ValueError('Объём: целое число миллилитров от 1 до 30 000.')
            return int(value)
        if field == 'vintage':
            if not value.isascii() or not value.isdigit() or not 1800 <= int(value) <= datetime.now(timezone.utc).year + 1:
                raise ValueError('Введите год урожая, например: 2021, или пропустите поле.')
            return int(value)
        if field == 'purchase_price':
            return _amount(value)
        if field == 'purchase_date':
            return _date(value)
        limit = 200 if field == 'name' else 160
        if not value or len(value) > limit or any(ord(char) < 32 for char in value):
            raise ValueError(f'Введите текст от 1 до {limit} символов без управляющих знаков.')
        return value

    def _advance(self, client, actor, draft, value):
        field = draft['step']
        draft['fields'][field] = value
        index = FIELDS.index(field) + 1
        draft['step'] = 'review' if draft.pop('return_review', False) or index == len(FIELDS) else FIELDS[index]
        self._save_draft(actor, draft)
        if draft['step'] == 'review':
            self.review(client, actor)
        else:
            self._prompt(client, actor, draft)

    def review(self, client, actor):
        self._require(actor, write=True)
        draft = self._draft(actor)
        fields = draft['fields']
        self._pending(actor, None)
        screen('wc:review', 'wc:menu')
        text = ('Проверьте добавление\n' + _name(fields) +
                f'\nБутылок: {fields.get("quantity", 1)}.\nЦена покупки за бутылку: {_money(fields.get("purchase_price"))}.'
                f'\nОбъём одной бутылки: {fields.get("volume_ml", 750)} мл.'
                f'\nДата покупки: {fields.get("purchase_date") or "не указана"}.'
                f'\nХранение: {fields.get("location") or "не указано"}.')
        if draft.get('warnings'):
            text += '\n\nРаспознавание: ' + '; '.join(draft['warnings'])
        rows = [[('Сохранить в винотеку', 'wc:save:' + draft['token'])]]
        rows += [[('Исправить ' + label, 'wc:edit:' + field)] for field, label in (
            ('name', 'название'), ('winery', 'производителя'), ('vintage', 'год'), ('quantity', 'количество'), ('volume_ml', 'объём'),
            ('purchase_price', 'цену покупки'), ('purchase_date', 'дату покупки'), ('location', 'хранение'))]
        rows.append([('Отмена', 'wc:cancel')])
        self.reply(client, actor, text + '\n\nПроверьте совпадение вина и года урожая. До подтверждения запасы не меняются.', keyboard(rows))

    def _operation(self, actor, kind, fields):
        token = uuid4().hex
        draft = {'token': token, 'mode': kind, 'fields': fields, 'step': 'review'}
        self._save_draft(actor, draft)
        self._pending(actor, None)
        return token

    def _stock_review(self, client, actor, lot_id, delta):
        if delta < 0:
            return self._consume_review(client, actor, lot_id, quantity=-delta)
        lot = cellar.get_lot(self.store, actor, lot_id)
        self._require(actor, write=True)
        token = self._operation(actor, 'stock', {'lot_id': lot_id, 'delta': delta})
        screen('wc:stockreview', f'wc:lot:{actor}:{lot_id}')
        text = (f'{_name(lot)}\n' + ('Списать одну выпитую бутылку?' if delta == -1 else f'Добавить {delta} бутылок?')
                + f'\nОстанется: {int(lot["quantity"]) + delta}.')
        self.reply(client, actor, text, keyboard([[('Подтвердить', 'wc:apply:' + token)], [('Отмена', f'wc:lot:{actor}:{lot_id}')]]))

    def _consume_review(self, client, actor, lot_id, quantity=1, proposal_id=None, draft=None):
        self._require(actor, write=True)
        lot = cellar.get_lot(self.store, actor, lot_id)
        if draft is None:
            self._operation(actor, 'consume', {'lot_id': lot_id, 'quantity': quantity,
                'consumed_on': cellar._today().isoformat(), 'proposal_id': proposal_id,
                'price': None, 'price_id': None, 'unpriced': False})
            draft = self._draft(actor)
        fields = draft['fields']
        preview = cellar.consumption_preview(self.store, actor, lot_id, fields['quantity'], fields['consumed_on'],
                    price=fields.get('price'), price_id=fields.get('price_id'), unpriced=fields.get('unpriced', False))
        fields['price_id'] = preview['price_id']
        fields['unpriced'] = preview['valuation_price'] is None
        draft['step'] = 'review'
        self._save_draft(actor, draft)
        self._pending(actor, None)
        token = draft['token']
        screen('wc:consumereview', f'wc:proposal:{fields["proposal_id"]}' if fields.get('proposal_id') else f'wc:lot:{actor}:{lot_id}')
        lines = ['Списание выпитого вина', _name(lot),
                 f'Выпито: {fields["quantity"]} шт.; дата: {fields["consumed_on"]}.',
                 f'Останется: {lot["quantity"]-fields["quantity"]} бутылок.',
                 f'Цена покупки за бутылку: {_money(preview["purchase_price"])}.',
                 f'Оценка при списании за бутылку: {_money(preview["valuation_price"])}.',
                 f'Зафиксировать условную прибыль/убыток: {_result_money(preview["result"])}.',
                 'Цена и результат сохранятся вместе со списанием. Будущие обновления цен их не изменят; денежный доход не записывается.']
        if preview['price_date']:
            lines.append(f'Оценка на {preview["price_date"]}; '+('введена вручную для списания.' if fields.get('price') is not None else 'сохранённая цена.'))
            if preview['price_date'] != fields['consumed_on']:
                lines.append('Оценка старше даты списания. Укажите цену на нужную дату, если она отличается.')
        if preview['valuation_price'] is None:
            lines.append('Нет отдельной оценки на дату списания. Введите цену или спишите без расчёта результата.')
        if preview['purchase_price'] is None:
            lines.append('Цена покупки неизвестна; прибыль/убыток по этой бутылке не рассчитывается.')
        rows = [[('Цена на дату списания', f'wc:consume:price:{token}')],
                [('Дата списания', f'wc:consume:date:{token}')]]
        if not fields.get('proposal_id'):
            rows.append([('Количество выпитых', f'wc:consume:quantity:{token}')])
        label = 'Подтвердить списание и результат' if preview['result'] is not None else 'Списать · результат не рассчитан'
        rows.append([(label, 'wc:apply:'+token)])
        if preview['valuation_price'] is not None:
            rows.append([('Списать без оценки', f'wc:consume:unpriced:{token}')])
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def valuation(self, client, actor, owner=None):
        owner = self._require(actor, owner)
        self._leave(actor)
        values = cellar.portfolio(self.store, actor, owner=owner)
        screen(f'wc:value:{owner}', 'wc:menu' if owner==actor else f'wc:list:{owner}:0')
        if not cellar.valuation_history(self.store, actor, owner=owner):
            hint, rows = self.empty_guidance(actor, owner)
            if owner == actor:
                rows.insert(0, [('Добавить вино', 'wc:add')])
            return self.reply(client, actor, 'Стоимость винотеки\nИстории запасов и стоимости пока нет.\n\n'+hint, keyboard(rows))
        lines = ['Стоимость винотеки', f'Бутылок: {values.get("quantity", 0)}.',
                 f'Известные затраты на покупку: {_money(values.get("known_purchase_cost", values.get("purchase_cost")))}.',
                 f'Оценка бутылок с известной ценой: {_money(values.get("known_current_value"))}.',
                 f'Без текущей цены: {values.get("unpriced_quantity", 0)} бутылок.',
                 'Полная стоимость: ' + (_money(values.get('current_value')) if values.get('current_value') is not None
                                         else 'не определена: есть бутылки без цены.')]
        open_display = None if values['quantity'] and not values['open_compared_quantity'] else values['known_open_result']
        closed_display = None if values['consumed_quantity'] and values['closed_unknown_quantity'] == values['consumed_quantity'] else values['known_closed_result']
        lines += [f'\nВсего куплено: {values["purchased_quantity"]}; выпито: {values["consumed_quantity"]}; в запасе: {values["quantity"]}.',
                  f'Вложено во все покупки: {_money(values["total_purchase_cost"])}.',
                  (f'Открытый результат по оценённым оставшимся бутылкам: {_result_money(open_display)} '
                   f'({values["open_compared_quantity"]} из {values["quantity"]} бутылок).'),
                  (f'Зафиксированный результат выпитых с известными ценами: {_result_money(closed_display)}; '
                   f'без расчёта: {values["closed_unknown_quantity"]} бутылок.'),
                  f'Общий результат: {_result_money(values["combined_result"])}.']
        if values['open_unknown_quantity'] or values['closed_unknown_quantity']:
            lines.append(f'Сумма только известной части результата: {_result_money(values["known_combined_result"])}. '
                         'Полный итог недоступен, пока есть бутылки без сопоставимых цен.')
        lines.append('Результат винотеки отражает изменение оценки; денежного дохода от выпитой бутылки нет.')
        if values.get('price_change') is not None:
            lines.append(f'Разница с ценой покупки для оценённых оставшихся бутылок: {_money(values["price_change"])}.')
        if values.get('shop_price_change') is not None:
            percent = values.get('shop_price_change_pct')
            suffix = f' ({percent:+.1f}%)' if percent is not None else ''
            lines.append(f'Изменение оценки по ценам магазинов: {_money(values["shop_price_change"])}{suffix}.')
            lines.append(f'Сопоставлено бутылок с известной ценой покупки: {values["shop_compared_quantity"]} из {values["quantity"]}. '
                         f'Даты магазинных цен: {values["shop_price_dates"][0]} — {values["shop_price_dates"][-1]}.')
        else:
            lines.append('Изменение оценки по ценам магазинов пока не рассчитано: добавьте ссылки и обновите цены.')
        lines.append('Разница = (цена магазина − цена покупки) × оставшееся количество. Это изменение оценки запасов; '
                     'в доходы и расходы оно не записывается.')
        if values.get('unknown_cost_quantity'):
            lines.append(f'Цена покупки неизвестна для {values["unknown_cost_quantity"]} бутылок; полное сравнение недоступно.')
        lines += [('\nОценка использует последние сохранённые цены с их датами. '
                  'Розничная цена замены не гарантирует стоимость продажи. '
                  'Покупки и выпитые бутылки учитываются отдельно от изменения цен.')]
        rows = [[('Таблица стоимости', f'wc:exports:{owner}')], [('График стоимости', f'wc:export:{owner}:chart')]]
        rows.append([('Все покупки', f'wc:purchases:{owner}:0'), ('Выпито и результат', f'wc:consumed:{owner}:0')])
        if owner == actor:
            rows += [[('Обновить цены по сохранённым ссылкам', 'wc:refresh:all')]]
        self.reply(client, actor, '\n'.join(lines), keyboard(rows))

    def proposals(self, client, actor, page=0):
        self._require(actor)
        self._leave(actor)
        items = cellar.proposals(self.store, actor)
        pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(_page(page), pages - 1)
        screen(f'wc:proposals:{page}', 'wc:menu')
        rows = [[(str(item.get('review_name') or item.get('name') or 'Новый отзыв')[:70], f'wc:proposal:{item["id"]}')]
                for item in items[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]]
        nav = []
        if page:
            nav.append(('‹ Предыдущие', f'wc:proposals:{page - 1}'))
        if page + 1 < pages:
            nav.append(('Следующие ›', f'wc:proposals:{page + 1}'))
        if nav:
            rows.append(nav)
        self.reply(client, actor, 'Отзывы Vivino и запасы\n' + (f'Новых предложений: {len(items)}.' if items else 'Нет новых предложений.')
                   + '\nОтзыв не доказывает, что выпита именно бутылка из ваших запасов. '
                   'Ничего не списывается автоматически. Одно подтверждение списывает одну бутылку.', keyboard(rows))

    def _proposal(self, actor, identifier):
        item = next((item for item in cellar.proposals(self.store, actor) if int(item['id']) == identifier), None)
        if not item:
            raise ValueError('Предложение уже обработано или недоступно.')
        return item

    def _show_proposal(self, client, actor, identifier):
        item = self._proposal(actor, identifier)
        screen(f'wc:proposal:{identifier}', 'wc:proposals:0')
        rows = []
        for lot_id in item.get('lot_ids', []):
            lot = cellar.get_lot(self.store, actor, int(lot_id))
            if lot['quantity']:
                rows.append([(f'Списать 1: {_name(lot)[:50]}', f'wc:proposalchoose:{identifier}:{lot_id}')])
        rows.append([('Не списывать из запасов', f'wc:proposalignore:{identifier}')])
        self.reply(client, actor, 'Новый отзыв Vivino\n' + str(item.get('review_name') or '') +
                   '\nВыберите конкретную бутылку для списания. Следующий экран попросит подтверждение. '
                   'Если пробовали в ресторане или из чужих запасов, нажмите «Не списывать».', keyboard(rows))

    def _queue(self, client, actor, kind, payload, title, back):
        owner = self._require(actor, payload.get('owner_id'))
        if kind == 'wine_price_refresh' and owner != actor:
            raise PermissionError('Изменять цены чужой винотеки нельзя.')
        self._leave(actor)
        target = {'actor_id': actor, 'owner_id': owner, **payload}
        identifier = None
        for row in self.store.query("SELECT id,payload FROM jobs WHERE kind=? AND state IN ('pending','running')", (kind,)):
            existing = json.loads(row['payload'])
            if {key: value for key, value in existing.items() if key != 'report_id'} == target:
                identifier = row['id']
                break
        if identifier is None:
            identifier = self.store.enqueue(kind, {**target, 'report_id': uuid4().hex}, f'{kind}:{actor}:{uuid4().hex}')
        screen('wc:queued', back)
        self.reply(client, actor, f'{title}: задание №{identifier} в очереди. Результат придёт сюда по готовности. '
                   'Повторное нажатие во время выполнения не создаёт копию задания.', keyboard([[('Вернуться', back)]]))

    def callback(self, client, actor, data, event_id=None):
        if data.startswith(('wc:incomplete:','wc:details:','wc:df:','wc:dp:','wc:ds:','wc:du:','wc:dd:')):
            from .telegram_wine_details import WineDetailsControls
            return WineDetailsControls(self).callback(client,actor,data)
        if data.startswith('wc:stats:'):
            parts = data.split(':')
            if len(parts) != 4:
                raise ValueError('Неизвестная группировка вин')
            return self.statistics(client, actor, _id(parts[2]), parts[3])
        if data == 'wn:menu':
            return self.home(client, actor)
        self._require(actor)
        if data == 'wc:menu':
            return self.menu(client, actor)
        if data == 'wc:portfolio':
            return self.valuation(client, actor)
        if data == 'wc:proposals':
            return self.proposals(client, actor)
        if data == 'wc:marketshops':
            return self.market_shops(client, actor)
        if data.startswith('wc:shop:'):
            from .wine_retail import configure_shop, shops
            if not self.store.authorize(actor, admin=True):
                raise PermissionError('Магазины настраивает администратор.')
            identifier = data.removeprefix('wc:shop:')
            item = next((item for item in shops(self.store) if item['id'] == identifier), None)
            if not item:
                raise ValueError('Неизвестный магазин.')
            configure_shop(self.store, actor, identifier, not item.get('enabled'))
            return self.market_shops(client, actor)
        if data == 'wc:cancel':
            self._pending(actor, None)
            self.store.set_setting(self._key(actor), None)
            return self.menu(client, actor)
        if data == 'wc:add':
            return self.add_menu(client, actor)
        if data == 'wc:example':
            return self.manual_example(client, actor)
        if data == 'wc:manual':
            return self._begin(client, actor)
        if data == 'wc:photo':
            return self.photo_prompt(client, actor)
        if data == 'wc:volume:750':
            draft = self._draft(actor)
            if draft.get('mode') != 'add' or draft.get('step') != 'volume_ml':
                raise ValueError('Форма объёма уже закрыта.')
            return self._advance(client, actor, draft, 750)
        if data == 'wc:shops':
            if not self.store.authorize(actor, admin=True):
                raise PermissionError('Магазины настраивает администратор.')
            self._pending(actor, 'wine:shops')
            screen('wc:shops', 'wc:marketshops')
            configured = self.store.get_setting('wine_price_allowed_hosts')
            current = (', '.join(configured) if configured else 'магазины не разрешены') if configured is not None else (
                'список из настроек сервера WINE_PRICE_ALLOWED_HOSTS; он может быть пустым')
            return self.reply(client, actor, 'Магазины для обновления цен\nСейчас: ' + current + '\n\n'
                              'Введите разрешённые домены через запятую, например: simplewine.ru, winelab.ru. '
                              'Бот проверяет сохранённые пользователями ссылки только на разрешённых доменах. '
                              'Доступность магазина и формат его карточек проверяются при обновлении. '
                              'Введите «сброс», чтобы использовать WINE_PRICE_ALLOWED_HOSTS из настроек сервера. '
                              'Если список пуст, обновление цен из магазинов недоступно. '
                              'Адреса локальной сети не допускаются.',
                              keyboard([[('Отмена', 'ui:cancel')]]))
        if data.startswith('wc:recognized:'):
            token = data.removeprefix('wc:recognized:')
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', token):
                raise ValueError('Некорректное распознавание.')
            result = self.store.get_setting(f'wine_recognition:{actor}:{token}')
            if not isinstance(result, dict) or result.get('consumed') or (result.get('expires') and result['expires'] < utcnow()):
                raise ValueError('Распознавание недоступно. Пришлите фото ещё раз.')
            return self._begin(client, actor, result.get('candidate') or {}, result.get('warnings'), recognition_token=token)
        if data in {'wc:skip', 'wc:previous'} or data.startswith('wc:edit:'):
            draft = self._draft(actor)
            if draft['mode'] != 'add':
                raise ValueError('Откройте добавление вина заново.')
            if data == 'wc:skip':
                if draft['step'] not in set(FIELDS) - {'name', 'quantity', 'volume_ml'}:
                    raise ValueError('Это поле нельзя пропустить.')
                return self._advance(client, actor, draft, None)
            if data == 'wc:previous':
                draft['step'] = FIELDS[max(0, FIELDS.index(draft['step']) - 1)]
            else:
                draft['step'] = data.removeprefix('wc:edit:')
                if draft['step'] not in FIELDS:
                    raise ValueError('Неизвестное поле.')
                draft['return_review'] = True
            self._save_draft(actor, draft)
            return self._prompt(client, actor, draft)
        if data.startswith(('wc:save:', 'wc:apply:')):
            token = data.rsplit(':', 1)[1]
            draft = self._draft(actor)
            if token != draft['token'] or draft['step'] != 'review':
                raise ValueError('Подтверждение устарело. Откройте винотеку заново.')
            self._require(actor, write=True)
            fields = draft['fields']
            operation = f'wine-ui:{actor}:{token}'
            if draft['mode'] == 'add' and data.startswith('wc:save:'):
                if draft.get('recognition_token'):
                    operation = f'wine-photo:{actor}:{draft["recognition_token"]}'
                lot = cellar.add_bottle(self.store, actor, fields, operation_id=operation)
                lot_id = lot['id']
                if draft.get('recognition_token'):
                    result_key = f'wine_recognition:{actor}:{draft["recognition_token"]}'
                    result = self.store.get_setting(result_key, {})
                    self.store.set_setting(result_key, {**result, 'consumed': True})
            elif draft['mode'] == 'stock' and data.startswith('wc:apply:'):
                cellar.adjust_stock(self.store, actor, fields['lot_id'], fields['delta'],
                                    'consumed' if fields['delta'] < 0 else 'addition', operation_id=operation)
                lot_id = fields['lot_id']
            elif draft['mode'] == 'consume' and data.startswith('wc:apply:'):
                kwargs = {key: fields.get(key) for key in ('price', 'price_id', 'unpriced')}
                if fields.get('proposal_id'):
                    cellar.resolve_proposal(self.store, actor, fields['proposal_id'], fields['lot_id'], operation_id=operation,
                                            consumed_on=fields['consumed_on'], **kwargs)
                    self.store.set_setting(self._key(actor), None)
                    return self.proposals(client, actor)
                cellar.consume_bottles(self.store, actor, fields['lot_id'], fields['quantity'], fields['consumed_on'],
                                       operation_id=operation, **kwargs)
                lot_id = fields['lot_id']
            elif draft['mode'] == 'proposal' and data.startswith('wc:apply:'):
                cellar.resolve_proposal(self.store, actor, fields['proposal_id'], fields['lot_id'], operation_id=operation)
                self.store.set_setting(self._key(actor), None)
                return self.proposals(client, actor)
            elif draft['mode'] == 'price' and data.startswith('wc:apply:'):
                cellar.record_price(self.store, actor, fields['lot_id'], fields['amount'],
                                    price_date=fields['price_date'], source='manual', operation_id=operation)
                lot_id = fields['lot_id']
            elif draft['mode'] == 'link' and data.startswith('wc:apply:'):
                cellar.add_price_link(self.store, actor, fields['lot_id'], fields['url'])
                lot_id = fields['lot_id']
            else:
                raise ValueError('Подтверждение не подходит к текущей форме.')
            self.store.set_setting(self._key(actor), None)
            return self.card(client, actor, lot_id)
        parts = data.split(':')
        if len(parts) == 4 and parts[1] == 'consume':
            action, token = parts[2:]
            draft = self._draft(actor)
            if draft['mode'] != 'consume' or draft['token'] != token:
                raise ValueError('Откройте актуальное списание.')
            if action == 'review':
                return self._consume_review(client, actor, draft['fields']['lot_id'], draft=draft)
            if action == 'unpriced':
                draft['fields'].update(price=None, price_id=None, unpriced=True)
                return self._consume_review(client, actor, draft['fields']['lot_id'], draft=draft)
            if action not in {'price', 'date', 'quantity'} or action == 'quantity' and draft['fields'].get('proposal_id'):
                raise ValueError('Неизвестное поле списания.')
            draft['step'] = 'consume_'+action
            self._save_draft(actor, draft)
            self._pending(actor, 'wine:consume_'+action+':'+token)
            screen('wc:consumeinput', f'wc:lot:{actor}:{draft["fields"]["lot_id"]}')
            prompts = {'price': 'Введите цену одной бутылки на дату списания, ₽. Например: 1250,50. Эта оценка сохранится только у списанной бутылки.',
                       'date': 'Введите дату, когда выпили вино: ДД.ММ.ГГГГ или ГГГГ-ММ-ДД.',
                       'quantity': 'Сколько бутылок выпито? Введите целое число в пределах остатка.'}
            key = 'consumed_on' if action == 'date' else action
            return self.reply(client, actor, wine_edit_prompt(key, draft['fields'].get(key), prompts[action]), keyboard([
                [('Вернуться к проверке списания', 'wc:consume:review:'+token)],
                [('Отмена', f'wc:lot:{actor}:{draft["fields"]["lot_id"]}')]]))
        if len(parts) == 4 and parts[1] in {'purchases', 'consumed'}:
            function = self.purchases if parts[1] == 'purchases' else self.consumed
            return function(client, actor, _id(parts[2]), _page(parts[3]))
        if len(parts) == 5 and parts[1] == 'drinks':
            return self.consumed(client, actor, _id(parts[2]), _page(parts[4]), _id(parts[3]))
        if len(parts) == 4 and parts[1] == 'consumption':
            return self.consumption_card(client, actor, _id(parts[2]), _id(parts[3]))
        if len(parts) == 3 and parts[1] == 'retail':
            return self.retail(client, actor, _id(parts[2]))
        if len(parts) == 3 and parts[1] == 'search':
            from .wine_retail import shops
            lot_id = _id(parts[2])
            self._require(actor, write=True)
            cellar.get_lot(self.store, actor, lot_id)
            if not any(item.get('enabled') and item.get('search_supported', item.get('automatic')) for item in shops(self.store)):
                raise ValueError('Автоматический поиск магазинов не подключён. Обратитесь к администратору или сохраните ссылку вручную.')
            return self._queue(client, actor, 'wine_price_search', {'lot_id': lot_id}, 'Поиск вина в Москве', f'wc:retail:{lot_id}')
        if len(parts) in {3, 4} and parts[1] in {'offers', 'offer', 'offerconfirm'}:
            token = parts[2]
            if not re.fullmatch(r'[a-f0-9]{32}', token):
                raise ValueError('Результат поиска недоступен. Повторите поиск.')
            if parts[1] == 'offers' and len(parts) == 3:
                return self.offers(client, actor, token)
            if len(parts) != 4:
                raise ValueError('Неверная карточка магазина.')
            index = _page(parts[3])
            if parts[1] == 'offer':
                return self.offer(client, actor, token, index)
            from .wine_retail import candidates, confirm_candidate
            self._require(actor, write=True)
            result = candidates(self.store, actor, token)
            items = result.get('candidates') or []
            if index >= len(items) or items[index].get('match') is not True:
                raise ValueError('Совпадение бутылки не подтверждено. Цена не сохранена.')
            confirm_candidate(self.store, actor, token, index)
            return self.card(client, actor, _id(result['lot_id']))
        if len(parts) == 4 and parts[1] == 'list':
            return self.storage(client, actor, _id(parts[2]), _page(parts[3]))
        if len(parts) == 4 and parts[1] == 'lot':
            return self.card(client, actor, _id(parts[3]), _id(parts[2]))
        if len(parts) == 3 and parts[1] == 'value':
            return self.valuation(client, actor, _id(parts[2]))
        if len(parts) == 3 and parts[1] == 'exports':
            owner = self._require(actor, _id(parts[2]))
            self._leave(actor)
            screen(f'wc:exports:{owner}', 'wc:menu' if owner==actor else f'wc:list:{owner}:0')
            if not cellar.valuation_history(self.store, actor, owner=owner):
                hint, rows = self.empty_guidance(actor, owner)
                if owner == actor:
                    rows.insert(0, [('Добавить вино', 'wc:add')])
                return self.reply(client, actor, 'Экспорт винотеки\nТаблица и график появятся после первого сохранённого добавления вина.\n\n'+hint, keyboard(rows))
            rows = [[(label, f'wc:export:{owner}:{fmt}')] for label, fmt in
                    [('CSV · таблица', 'csv'), ('Excel · таблица', 'xlsx'), ('HTML · обзор', 'html'), ('График стоимости', 'chart')]]
            return self.reply(client, actor, 'Таблица винотеки\nНазвание, производитель, год, остаток, стоимость покупки и '
                              'последняя оценка с датой и источником. Неизвестные цены остаются пустыми. '
                              'История стоимости учитывает пополнения и списания.', keyboard(rows))
        if len(parts) == 4 and parts[1] == 'export':
            if parts[3] not in {'csv', 'xlsx', 'html', 'chart'}:
                raise ValueError('Неизвестный формат.')
            owner = self._require(actor, _id(parts[2]))
            if parts[3] == 'chart' and not cellar.valuation_history(self.store, actor, owner=owner):
                return self.valuation(client, actor, owner)
            return self._queue(client, actor, 'wine_cellar_export', {'owner_id': owner, 'format': parts[3]},
                               'Выгрузка винотеки', f'wc:exports:{owner}')
        if len(parts) == 3 and parts[1] == 'refresh':
            self._require(actor, write=True)
            payload = {}
            if parts[2] != 'all':
                payload['lot_id'] = _id(parts[2])
                cellar.get_lot(self.store, actor, payload['lot_id'])
            return self._queue(client, actor, 'wine_price_refresh', payload, 'Обновление цен', 'wc:menu')
        if len(parts) == 4 and parts[1] == 'stock':
            lot_id = _id(parts[2])
            self._require(actor, write=True)
            lot = cellar.get_lot(self.store, actor, lot_id)
            if parts[3] == 'drink':
                if not lot['quantity']:
                    raise ValueError('В запасах нет бутылок для списания.')
                return self._stock_review(client, actor, lot_id, -1)
            if parts[3] != 'add':
                raise ValueError('Неизвестное изменение запаса.')
            self._pending(actor, f'wine:quantity:{lot_id}')
            screen('wc:stockquantity', f'wc:lot:{actor}:{lot_id}')
            return self.reply(client, actor, 'Сколько бутылок добавить? Введите целое число. '
                              f'Добавленные бутылки учитываются по прежней цене покупки {_money(lot.get("purchase_price"))}. '
                              'На следующем экране подтвердите изменение. Для покупки по другой цене добавьте вино новой партией.',
                              keyboard([[('Отмена', f'wc:lot:{actor}:{lot_id}')]]))
        if len(parts) == 3 and parts[1] in {'price', 'link'}:
            lot_id = _id(parts[2])
            self._require(actor, write=True)
            cellar.get_lot(self.store, actor, lot_id)
            self._pending(actor, f'wine:{parts[1]}:{lot_id}')
            screen(data, f'wc:lot:{actor}:{lot_id}')
            text = ('Введите цену за одну бутылку в рублях и дату через пробел. Например: 1500 11.10.2026. '
                    'Это ручная текущая оценка; цена покупки не изменится.' if parts[1] == 'price' else
                    'Пришлите HTTPS-ссылку на страницу именно этого вина и года в магазине. '
                    'Поиск по названию не гарантирует совпадение. Цена сохраняется с источником и датой проверки.')
            return self.reply(client, actor, text, keyboard([[('Отмена', f'wc:lot:{actor}:{lot_id}')]]))
        if len(parts) == 3 and parts[1] == 'proposals':
            return self.proposals(client, actor, _page(parts[2]))
        if len(parts) == 3 and parts[1] == 'proposal':
            return self._show_proposal(client, actor, _id(parts[2]))
        if len(parts) == 3 and parts[1] == 'proposalignore':
            identifier = _id(parts[2])
            self._require(actor, write=True)
            self._proposal(actor, identifier)
            cellar.resolve_proposal(self.store, actor, identifier, False, operation_id=f'wine-dismiss:{actor}:{identifier}')
            return self.proposals(client, actor)
        if len(parts) == 4 and parts[1] == 'proposalchoose':
            identifier, lot_id = _id(parts[2]), _id(parts[3])
            item = self._proposal(actor, identifier)
            if lot_id not in [int(value) for value in item.get('lot_ids', [])]:
                raise ValueError('Эта бутылка не относится к предложению.')
            return self._consume_review(client, actor, lot_id, proposal_id=identifier)
        if len(parts) == 3 and parts[1] == 'users':
            return self.users(client, actor, _page(parts[2]))
        raise ValueError('Неизвестное действие винотеки.')

    def users(self, client, actor, page=0):
        self._require(actor)
        if not self.store.authorize(actor, admin=True):
            raise PermissionError('Просмотр пользователей доступен администратору.')
        self._leave(actor)
        items = cellar.owners(self.store, actor)
        pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(_page(page), pages - 1)
        screen(f'wc:users:{page}', 'wc:menu')
        rows = [[(self._owner_label(int(item['user_id']))[:70], f'wc:list:{item["user_id"]}:0')]
                for item in items[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]]
        nav = []
        if page:
            nav.append(('‹ Предыдущие', f'wc:users:{page - 1}'))
        if page + 1 < pages:
            nav.append(('Следующие ›', f'wc:users:{page + 1}'))
        if nav:
            rows.append(nav)
        self.reply(client, actor, 'Винотеки пользователей\nВыберите пользователя для просмотра. '
                   'Запасы и цены другого пользователя здесь изменить нельзя.', keyboard(rows))

    def input(self, client, actor, text):
        session = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        pending = (session or {}).get('pending') or ''
        if not pending.startswith('wine:'):
            return False
        self._require(actor, write=True)
        if text == '/cancel':
            self.callback(client, actor, 'wc:cancel')
            return True
        if text.startswith('/'):
            return False
        if pending.startswith('wine:details:'):
            from .telegram_wine_details import WineDetailsControls
            WineDetailsControls(self).input(client,actor,text,pending)
            return True
        if pending == 'wine:photo':
            self.reply(client, actor, 'Сейчас ожидается фотография этикетки. Пришлите фото или выберите ручное добавление.',
                       keyboard([[('Ввести вручную', 'wc:manual')], [('Отмена', 'wc:cancel')]]))
            return True
        if pending == 'wine:shops':
            if not self.store.authorize(actor, admin=True):
                raise PermissionError('Магазины настраивает администратор.')
            raw = text.strip().lower()
            hosts = None if raw == 'сброс' else sorted({value.strip().rstrip('.') for value in raw.split(',')})
            if hosts is not None and (not 1 <= len(hosts) <= 30 or any(
                    not re.fullmatch(r'(?=.{4,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}', value)
                    for value in hosts)):
                raise ValueError('Введите до 30 доменов без https://, пути, порта или IP-адреса.')
            self.store.set_setting('wine_price_allowed_hosts', hosts)
            self._pending(actor, None)
            return self.market_shops(client, actor) or True
        parts = pending.split(':')
        if len(parts) == 3 and parts[1] in {'consume_price', 'consume_date', 'consume_quantity'}:
            draft = self._draft(actor)
            if draft['mode'] != 'consume' or draft['token'] != parts[2] or draft['step'] != parts[1]:
                raise ValueError('Форма списания устарела.')
            fields = draft['fields']
            if parts[1] == 'consume_price':
                fields.update(price=_amount(text), price_id=None, unpriced=False)
            elif parts[1] == 'consume_date':
                fields.update(consumed_on=_date(text), price=None, price_id=None, unpriced=False)
            else:
                if fields.get('proposal_id'):
                    raise ValueError('Один отзыв списывает одну бутылку.')
                fields['quantity'] = self._field('quantity', text)
            self._consume_review(client, actor, fields['lot_id'], draft=draft)
            return True
        if len(parts) == 3 and parts[1] == 'add':
            draft = self._draft(actor)
            if draft['mode'] != 'add' or parts[2] != draft['step'] or parts[2] not in FIELDS:
                raise ValueError('Форма устарела. Откройте добавление заново.')
            self._advance(client, actor, draft, self._field(parts[2], text))
            return True
        if len(parts) != 3:
            raise ValueError('Форма устарела. Откройте винотеку заново.')
        lot_id = _id(parts[2])
        cellar.get_lot(self.store, actor, lot_id)
        if parts[1] == 'quantity':
            self._stock_review(client, actor, lot_id, self._field('quantity', text))
        elif parts[1] == 'price':
            values = text.strip().split()
            if len(values) != 2:
                raise ValueError('Введите цену и дату через пробел: 1500 11.10.2026.')
            amount, price_date = _amount(values[0]), _date(values[1])
            token = self._operation(actor, 'price', {'lot_id': lot_id, 'amount': amount, 'price_date': price_date})
            screen('wc:pricereview', f'wc:lot:{actor}:{lot_id}')
            self.reply(client, actor, f'Сохранить ручную оценку {_money(amount)} за бутылку на {price_date}? '
                       'Предыдущие оценки и цена покупки сохранятся.', keyboard([[('Сохранить оценку', 'wc:apply:' + token)],
                       [('Отмена', f'wc:lot:{actor}:{lot_id}')]]))
        elif parts[1] == 'link':
            value = text.strip()
            parsed = urlsplit(value)
            if len(value) > 1500 or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Нужна HTTPS-ссылка на страницу вина без логина и пароля в адресе.')
            token = self._operation(actor, 'link', {'lot_id': lot_id, 'url': value})
            screen('wc:linkreview', f'wc:lot:{actor}:{lot_id}')
            self.reply(client, actor, f'Сохранить ссылку магазина {parsed.hostname}?\n{value}\n'
                       'Проверьте название, год и объём бутылки.', keyboard([[('Сохранить ссылку', 'wc:apply:' + token)],
                       [('Отмена', f'wc:lot:{actor}:{lot_id}')]]))
        else:
            raise ValueError('Неизвестная форма винотеки.')
        return True
