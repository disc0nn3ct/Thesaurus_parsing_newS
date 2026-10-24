"""Personal Life screens: a ledger and explicit, reviewed receipt imports."""
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from . import life_ledger as ledger
from .feature_access import allowed as feature_allowed
from .subscription_access import project_allowed
from .telegram_charts import keyboard
from .telegram_navigation import screen


def today():
    return datetime.now(ZoneInfo('Europe/Moscow')).date()


def money(value):
    from decimal import ROUND_HALF_UP, Decimal
    amount = Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return f'{amount:,.2f}'.replace(',', ' ').replace('.', ',') + ' ₽'


def number(text):
    from decimal import Decimal, InvalidOperation
    try:
        value = Decimal(str(text).replace(' ', '').replace(',', '.'))
    except InvalidOperation:
        raise ValueError('Введите число, например 1250,50.') from None
    if not value.is_finite() or value < 0 or value > 100_000_000:
        raise ValueError('Введите число от 0 до 100 000 000.')
    return float(value)


def cash(text):
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(str(number(text))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def integer(text):
    value = number(text)
    if not value.is_integer():
        raise ValueError('Введите целое число.')
    return int(value)


def day(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y'):
        try:
            result = datetime.strptime(text.strip(), fmt).replace(tzinfo=ZoneInfo('Europe/Moscow')).date()
            if result > today():
                raise ValueError('Дата покупки не может быть в будущем.')
            return result.isoformat()
        except ValueError:
            continue
    raise ValueError('Введите дату ДД.ММ.ГГГГ или ГГГГ-ММ-ДД, не позднее сегодня.')


def receipt_changes(store, actor, ui, index, item):
    from .receipt_wines import original_item
    original = original_item(store, actor, ui, index, item)
    fields = {**item, **ui.get('overrides', {}).get(str(index), {})}

    def display(key, value):
        if value in (None, '', []):
            return 'не указано'
        if key == 'amount':
            return money(value)
        if key in {'volume_ml', 'quantity'}:
            return f'{float(value):g}'+(' мл' if key == 'volume_ml' else '')
        if isinstance(value, list):
            return ', '.join(map(str, value))
        return str(value)

    changes = []
    for key, label in (('name', 'Название'), ('winery', 'Производитель'), ('vintage', 'Год'),
                       ('volume_ml', 'Объём'), ('country', 'Страна'), ('region', 'Регион'),
                       ('grapes', 'Сорта винограда'), ('quantity', 'Количество'), ('amount', 'Оплачено')):
        before, after = display(key, original.get(key)), display(key, fields.get(key))
        if before != after:
            changes.append(f'{label}: {before} → {after}')
    return changes


class LifeControls:
    def __init__(self, bot):
        self.bot, self.store = bot, bot.store

    def leave(self, actor):
        self.store.set_setting(f'life_entry:{actor}', None)
        pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        if pending and str(pending['pending'] or '').startswith('life:'):
            self.bot.market_controls.pending(actor, None)

    def home(self, client, actor):
        if not project_allowed(self.store, actor, 'life'):
            raise PermissionError('Раздел «Жизнь» пока не подключён для вас.')
        self.leave(actor)
        screen('pj:life', 'menu:home')
        rows, descriptions = [], []
        if project_allowed(self.store, actor, 'vivino'):
            rows.append([('Вино', 'pj:vivino')])
            descriptions.append('Вино — Vivino, география вкусов, запасы и стоимость бутылок.')
        if feature_allowed(self.store, actor, 'life_finance'):
            rows += [[('Доходы и расходы', 'lf:home')], [('Добавить покупки по чеку', 'lf:receipt')]]
            descriptions.append('Доходы и расходы — личный учёт покупок и поступлений. Чек сначала проверяется вами.')
        self.bot._reply(client, actor, 'Жизнь\n\n'+'\n\n'.join(descriptions), keyboard(rows))

    def bounds(self, actor):
        config = self.store.get_setting(f'life_period:{actor}', {'preset': 'month'})
        end = today()
        preset = config.get('preset', 'month')
        if preset == 'custom':
            return config['start'], config['end'], f"{config['start']} — {config['end']}"
        start = {'month': end.replace(day=1), '3months': end-timedelta(days=89),
                 'year': end.replace(month=1, day=1), 'all': None}.get(preset)
        labels = {'month': 'Этот месяц', '3months': '90 дней', 'year': 'Этот год', 'all': 'Вся история'}
        return start.isoformat() if start else None, end.isoformat(), labels.get(preset, 'Этот месяц')

    def finance(self, client, actor, notice=''):
        ledger.require_owner(self.store, actor)
        self.leave(actor)
        start, end, label = self.bounds(actor)
        totals = ledger.summary(self.store, actor, start=start, end=end)
        text = (f'Доходы и расходы · {label}\n\nДоходы: {money(totals["income"])}\n'
                f'Расходы: {money(totals["expenses"])}\nРазница: {money(totals["balance"])}\n'
                f'Операций: {totals["count"]}. Разница относится к записанным операциям, это не остаток банковского счёта.')
        categories = [row for row in totals['by_category'] if row['kind'] == 'expense']
        if categories:
            text += '\n\nРасходы по категориям:\n'+'\n'.join(f'{row["category"]}: {money(row["amount"])}' for row in categories[:8])
        if notice:
            text = notice+'\n\n'+text
        screen('lf:home', 'pj:life')
        rows = [[('Добавить расход', 'lf:add:expense'), ('Добавить доход', 'lf:add:income')],
                [('Загрузить чек', 'lf:receipt')], [('Неподтверждённые чеки', 'lf:drafts')], [('Период: '+label, 'lf:period')],
                [('История операций', 'lf:history:0')], [('Аналитика по месяцам', 'lf:analytics')],
                [('Скачать CSV', 'lf:export:csv'), ('Скачать Excel', 'lf:export:xlsx')]]
        self.bot._reply(client, actor, text, keyboard(rows))

    def period(self, client, actor):
        ledger.require_owner(self.store, actor)
        current = self.store.get_setting(f'life_period:{actor}', {'preset': 'month'}).get('preset')
        screen('lf:period', 'lf:home')
        rows = [[(('✓ ' if current == key else '')+label, 'lf:period:'+key)] for key, label in
                [('month', 'Этот месяц'), ('3months', '90 дней'), ('year', 'Этот год'), ('all', 'Вся история'), ('custom', 'Свои даты')]]
        self.bot._reply(client, actor, 'Период учёта\nОн применяется к обзору, истории, аналитике и экспорту.', keyboard(rows))

    def history(self, client, actor, page=0):
        ledger.require_owner(self.store, actor)
        if not 0 <= page <= 10000:
            raise ValueError('Неверная страница.')
        start, end, label = self.bounds(actor)
        rows = ledger.entries(self.store, actor, start=start, end=end, limit=11, offset=page*10)
        selected = rows[:10]
        text = f'История операций · {label}\n'
        buttons = []
        for row in selected:
            text += f'\n{row["date"]} · {"−" if row["kind"] == "expense" else "+"}{money(row["amount"])} · {row["category"]}'
            if row.get('note') or row.get('merchant'):
                text += '\n'+str(row.get('merchant') or row.get('note'))[:180]
            buttons.append([(f'{row["date"]} · {money(row["amount"])}', f'lf:entry:{row["id"]}')])
        if not selected:
            text += '\nПока нет записей за этот период. Добавьте операцию или загрузите чек.'
        nav = []
        if page:
            nav.append(('‹ Предыдущие', f'lf:history:{page-1}'))
        if len(rows) > 10:
            nav.append(('Следующие ›', f'lf:history:{page+1}'))
        if nav:
            buttons.append(nav)
        screen(f'lf:history:{page}', 'lf:home')
        self.bot._reply(client, actor, text, keyboard(buttons))

    def analytics(self, client, actor):
        ledger.require_owner(self.store, actor)
        start, end, label = self.bounds(actor)
        rows = ledger.monthly_summary(self.store, actor, start=start, end=end)
        text = f'Аналитика · {label}\nМесяц: доходы / расходы / разница\n'
        for row in list(reversed(rows))[:24]:
            text += f'\n{row["month"]}: {money(row["income"])} / {money(row["expenses"])} / {money(row["balance"])}'
        if len(rows) > 24:
            text += '\nПоказаны последние 24 месяца; вся история доступна в экспорте.'
        if not rows:
            text += '\nНет операций. Сначала добавьте покупки или доходы.'
        text += '\n\nУчитываются только ваши записи. Покупка вина по чеку включается в расходы один раз; оценка запаса по ценам магазинов показывается в винотеке.'
        screen('lf:analytics', 'lf:home')
        self.bot._reply(client, actor, text, keyboard([[('Изменить период', 'lf:period')]]))

    def entry(self, client, actor, identifier):
        row = ledger.get_entry(self.store, actor, identifier)
        screen(f'lf:entry:{identifier}', 'lf:history:0')
        text = (f'{"Расход" if row["kind"] == "expense" else "Доход"} · {money(row["amount"])}\n'
                f'Дата: {row["date"]}\nКатегория: {row["category"]}\n{row.get("merchant") or ""}\n{row.get("note") or ""}')
        buttons = []
        if row.get('receipt_id'):
            text += '\nЗапись связана с чеком. Бутылки управляются отдельно в винотеке; повторный импорт не создаёт дубликаты.'
            receipt = self.store.one("SELECT token FROM life_receipt_drafts WHERE user_id=? AND receipt_id=? AND status='confirmed' ORDER BY updated_at DESC LIMIT 1", (actor, row['receipt_id']))
            if receipt and feature_allowed(self.store, actor, 'wine_cellar'):
                buttons.append([('Добавить вина из этого чека', 'rc:cellar:'+receipt['token'])])
        else:
            buttons = [[('Удалить ошибочную запись', f'lf:voidask:{identifier}')]]
        self.bot._reply(client, actor, text, keyboard(buttons))

    def manual(self, client, actor, kind):
        ledger.require_owner(self.store, actor, write=True)
        if kind not in {'income', 'expense'}:
            raise ValueError('Выберите доход или расход.')
        self.store.set_setting(f'life_entry:{actor}', {'kind': kind, 'date': today().isoformat(), 'token': uuid4().hex})
        self.manual_step(client, actor, 'amount')

    def manual_step(self, client, actor, field):
        draft = self.store.get_setting(f'life_entry:{actor}')
        if not draft:
            raise ValueError('Начните добавление операции заново.')
        self.bot.market_controls.pending(actor, 'life:entry:'+field)
        previous = {'date': 'amount', 'category': 'date', 'note': 'category'}.get(field)
        screen('lf:input:'+field, 'lf:step:'+previous if previous else 'lf:home')
        texts = {'amount': 'Введите сумму в рублях, например 1250,50.',
                 'date': 'Когда была операция? Выберите сегодня или введите ДД.ММ.ГГГГ.',
                 'category': 'Выберите категорию или введите свою.',
                 'note': 'Добавьте описание, например название магазина. Можно пропустить.'}
        buttons = []
        if field == 'date':
            buttons = [[('Сегодня · '+today().strftime('%d.%m.%Y'), 'lf:today')]]
        elif field == 'category':
            categories = ['Продукты', 'Вино', 'Транспорт', 'Дом', 'Поездки', 'Здоровье', 'Прочее'] if draft['kind'] == 'expense' else ['Зарплата', 'Возврат', 'Подарок', 'Прочее']
            buttons = [[(value, 'lf:category:'+str(i))] for i, value in enumerate(categories)]
        elif field == 'note':
            buttons = [[('Без описания', 'lf:nonote')]]
        buttons.append([('Отмена', 'lf:home')])
        self.bot._reply(client, actor, ('Новый расход' if draft['kind'] == 'expense' else 'Новый доход')+'\n'+texts[field], keyboard(buttons))

    def manual_review(self, client, actor):
        draft = self.store.get_setting(f'life_entry:{actor}')
        if not draft or not all(key in draft for key in ('amount', 'date', 'category')):
            raise ValueError('Заполните сумму, дату и категорию.')
        self.bot.market_controls.pending(actor, None)
        screen('lf:entryreview', 'lf:home')
        self.bot._reply(client, actor, f'Проверка операции\n{money(draft["amount"])} · {draft["date"]}\n{draft["category"]}\n{draft.get("note", "")}\n\nЗапись появится в учёте после подтверждения.',
            keyboard([[('Сохранить операцию', 'lf:save:'+draft['token'])], [('Изменить сумму', 'lf:edit:amount'), ('Изменить дату', 'lf:edit:date')],
                      [('Категория', 'lf:edit:category'), ('Описание', 'lf:edit:note')], [('Отмена', 'lf:home')]]))

    def receipt_prompt(self, client, actor):
        ledger.require_owner(self.store, actor, write=True)
        previous = self.store.get_setting(f'ui_screen:{actor}', {})
        from .subscription_access import route_allowed
        from_wine = str(previous.get('route', '')).startswith('wc:') or (str(previous.get('route', '')).startswith('lf:receipt') and str(previous.get('back', '')).startswith('wc:'))
        back = 'wc:add' if from_wine and route_allowed(self.store, actor, 'wc:add') else 'lf:home'
        self.leave(actor)
        self.bot.market_controls.pending(actor, 'life:receipt')
        screen('lf:receipt', back)
        self.bot._reply(client, actor, 'Добавление по чеку\nПришлите PDF или чёткое фото всего чека. '
            'PDF с текстом читается локально; для фото нужен установленный локальный OCR. '
            'После распознавания проверьте дату, итог и товары. В расходах появится один чек; '
            'выбранные вами вина можно добавить в винотеку. До подтверждения учёт не меняется.',
            keyboard([[('Добавить расход вручную', 'lf:add:expense')]]))

    def receipt_back(self, actor, token):
        ui = self.receipt_ui(actor, token)
        items = ledger.get_receipt_draft(self.store, actor, token)['result']['items']
        wine_context = ui.get('return_to') == 'wc:menu' or (not ui.get('return_to') and items and all(item.get('is_wine') for item in items))
        if wine_context and feature_allowed(self.store, actor, 'wine_cellar'):
            return 'wc:menu'
        return 'lf:home'

    def receipt_ui(self, actor, token):
        ledger.get_receipt_draft(self.store, actor, token)
        return self.store.get_setting(f'receipt_ui:{actor}:{token}', {'selected': [], 'overrides': {}})

    def save_ui(self, actor, token, values):
        ledger.get_receipt_draft(self.store, actor, token)
        self.store.set_setting(f'receipt_ui:{actor}:{token}', values)

    def resume_receipt(self, client, actor, data):
        """A cached keyboard may navigate to its receipt, but never replay a write."""
        token = data.split(':')[2]
        ledger.get_receipt_draft(self.store, actor, token)
        return self.review_receipt(client, actor, token,
            'Это прежняя карточка. Открыл актуальный чек; действие со старой кнопки не выполнено. Проверьте выбор и подтвердите сохранение здесь.')

    def review_receipt(self, client, actor, token, notice=''):
        draft = ledger.get_receipt_draft(self.store, actor, token)
        from .receipt_wines import queue_automatic
        queue_automatic(self.store, actor, token)
        self.bot.market_controls.pending(actor, None)
        result = draft['result']
        ui = self.receipt_ui(actor, token)
        items = result.get('items', [])
        selected = ui['selected']
        back = self.receipt_back(actor, token)
        screen('rc:review:'+token, back)
        text = (f'Проверка чека\n{result.get("merchant") or "Магазин не распознан"}\n'
                f'Дата: {result.get("date") or "нужно указать"}\n'
                f'Итог: {money(result["total"]) if result.get("total") is not None else "нужно указать"}\n'
                f'Товаров: {len(items)}. В винотеку выбрано: {len(selected)}.\n'
                'В расходы записывается итог чека один раз. Нажмите на товар, чтобы проверить его или отметить как вино. '
                'Черновик сохраняется на сутки; к нему можно вернуться из «Неподтверждённых чеков».')
        if notice:
            text = notice+'\n\n'+text
        if ui.get('stock_only'):
            text += '\n\nРасход уже учтён. Здесь добавляются только выбранные вина; повторного расхода не будет. Суммы, количество и дата покупки остаются из сохранённого чека.'
        if draft['status'] != 'confirmed':
            text += '\n\nВина пока не сохранены в запасах. Галочки означают только выбор; завершите добавление кнопкой сохранения ниже.'
        incomplete = []
        for index in selected:
            if not 0 <= index < len(items):
                raise ValueError('Товар больше не найден. Выберите вина заново.')
            fields = {**items[index], **ui['overrides'].get(str(index), {})}
            if not fields.get('volume_ml') or not float(fields['quantity']).is_integer():
                incomplete.append(index)
        if incomplete:
            text += '\n\nПеред добавлением вин уточните объём и целое количество бутылок у товаров: '+', '.join(str(i+1) for i in incomplete)+'.'
        if selected and all(row.get('amount') is not None for row in items):
            from decimal import Decimal
            total = sum((Decimal(str(row['amount'])) for row in items), Decimal(0))
            if result.get('total') is not None and total != Decimal(str(result['total'])):
                text += '\n\nСуммы товаров не совпадают с итогом чека. Исправьте оплаченные суммы перед добавлением вин.'
        if result.get('warnings'):
            text += '\n\nПроверьте: '+'; '.join(str(value)[:220] for value in result['warnings'][:4])
        buttons = []
        changed = [index for index, item in enumerate(items) if receipt_changes(self.store, actor, ui, index, item)]
        if changed:
            text += f'\n\nИсправления у товаров: {len(changed)}. Откройте «Изменения: было → стало» для проверки.'
            buttons.append([('Изменения: было → стало', f'rc:changes:{token}:0')])
        if selected and draft['status'] != 'confirmed':
            text += '\nВ винотеку попадут итоговые названия и характеристики выбранных вин.'
        page = max(0, min(int(ui.get('page', 0)), max(0, (len(items)-1)//8)))
        for index, item in enumerate(items[page*8:page*8+8], start=page*8):
            name = ui['overrides'].get(str(index), {}).get('name', item['name'])
            buttons.append([(('✓ ' if index in selected else '')+str(name)[:65], f'rc:item:{token}:{index}')])
        nav = []
        if page:
            nav.append(('‹ Товары', f'rc:page:{token}:{page-1}'))
        if (page+1)*8 < len(items):
            nav.append(('Товары ›', f'rc:page:{token}:{page+1}'))
        if nav:
            buttons.append(nav)
        if not ui.get('stock_only'):
            buttons += [[('Дата чека', f'rc:edit:{token}:date'), ('Итог чека', f'rc:edit:{token}:total')]]
        if feature_allowed(self.store, actor, 'wine_cellar'):
            buttons += [[('Выбрать распознанные вина', f'rc:select:{token}')],
                        [('750 мл для выбранных вин', f'rc:volumeall:{token}')]] if any(row.get('is_wine') for row in items) else []
        if selected:
            buttons.append([('Снять выбор вин' if ui.get('stock_only') else 'Учесть только расход · снять выбор вин', f'rc:clear:{token}')])
            if not feature_allowed(self.store, actor, 'wine_cellar'):
                text += '\n\nДоступ к винотеке отключён. Снимите выбор вин, чтобы сохранить только расход.'
        if draft.get('status') == 'confirmed':
            text += '\n\nЧек уже сохранён. Повторно покупки не добавляются.'
            saved = self.store.one('SELECT wine_lot_ids FROM life_receipt_imports WHERE id=? AND user_id=?', (draft['receipt_id'], actor))
            if saved and saved['wine_lot_ids'] == '[]' and any(item.get('is_wine') for item in items):
                text += '\nИз этого чека сохранён только расход; вина в запасы ещё не добавлены. Нажмите «Добавить вина из учтённого чека».'
            buttons = [[('Моя винотека', back)]] if back == 'wc:menu' else [[('Доходы и расходы', back)]]
            if feature_allowed(self.store, actor, 'wine_cellar'):
                buttons.insert(0, [('Добавить вина из учтённого чека', f'rc:cellar:{token}')])
        elif result.get('total') is not None and result.get('date') and not incomplete and (not selected or feature_allowed(self.store, actor, 'wine_cellar')):
            if selected or not ui.get('stock_only'):
                label = 'Добавить выбранные вина · без повторного расхода' if ui.get('stock_only') else 'Сохранить расход и выбранные вина' if selected else 'Сохранить только расход · без вин'
                buttons.append([(label, f'rc:confirm:{token}')])
        self.bot._reply(client, actor, text, keyboard(buttons))

    def receipt_changes(self, client, actor, token, page=0):
        draft = ledger.get_receipt_draft(self.store, actor, token)
        ui = self.receipt_ui(actor, token)
        changed = [(index, receipt_changes(self.store, actor, ui, index, item))
                   for index, item in enumerate(draft['result'].get('items', []))]
        changed = [(index, lines) for index, lines in changed if lines]
        page = max(0, min(page, max(0, len(changed)-1)))
        screen(f'rc:changes:{token}:{page}', 'rc:review:'+token)
        text = 'Изменения: было → стало\nСравнение с исходной строкой чека.'
        buttons = []
        if changed:
            index, lines = changed[page]
            text += f'\n\nТовар {index+1} · исправление {page+1} из {len(changed)}\n'+'\n'.join(lines)
            text += ('\n\n✓ Выбрано для винотеки.' if index in ui['selected'] else '\n\nЭта строка не выбрана для винотеки.')
            buttons.append([('Открыть итоговую карточку', f'rc:item:{token}:{index}')])
            nav = []
            if page:
                nav.append(('‹ Предыдущий товар', f'rc:changes:{token}:{page-1}'))
            if page+1 < len(changed):
                nav.append(('Следующий товар ›', f'rc:changes:{token}:{page+1}'))
            if nav:
                buttons.append(nav)
        else:
            text += '\nИсправлений нет.'
        if draft['status'] != 'confirmed':
            text += '\n\nПри подтверждении сохраняются итоговые значения выбранных вин. Пока в запасы ничего не добавлено.'
        else:
            text += '\n\nЧек уже сохранён. Здесь показаны исправления его черновика.'
        self.bot._reply(client, actor, text, keyboard(buttons))

    def receipt_item(self, client, actor, token, index, notice=''):
        draft = ledger.get_receipt_draft(self.store, actor, token)
        result = draft['result']
        if not 0 <= index < len(result.get('items', [])):
            raise ValueError('Товар не найден.')
        item = result['items'][index]
        from .receipt_wines import queue_automatic
        queue_automatic(self.store, actor, token, index)
        ui = self.receipt_ui(actor, token)
        fields = {**item, **ui['overrides'].get(str(index), {})}
        screen(f'rc:item:{token}:{index}', 'rc:review:'+token)
        text = (f'Товар {index+1}\n{fields["name"]}\nКоличество: {item["quantity"]}\n'
                f'Оплачено: {money(item["amount"]) if item.get("amount") is not None else "не распознано"}\n'
                f'Производитель: {fields.get("winery") or "не указан"}\nГод: {fields.get("vintage") or "не указан"}\n'
                f'Объём: {str(fields["volume_ml"])+" мл" if fields.get("volume_ml") else "нужно уточнить для винотеки"}')
        grapes = fields.get('grapes') or []
        grape_text = ', '.join(grapes) if isinstance(grapes, list) else grapes
        text += (f'\nСтрана: {fields.get("country") or "не найдена"}'
                 f'\nРегион: {fields.get("region") or "не найден"}'
                 '\nСорта винограда: '+(grape_text or 'не найдены'))
        changes = receipt_changes(self.store, actor, ui, index, item)
        if changes:
            text += '\n\nИзменения: было → стало\n'+'\n'.join(changes)
        state = ui.get('wine_lookups', {}).get(str(index), {})
        if state.get('status') == 'pending':
            text += '\n\nСведения ищутся автоматически. Нажмите «Обновить карточку» через несколько секунд.'
        years = sorted({str(row['published_vintage']) for row in state.get('candidates', []) if row.get('published_vintage')})
        if years and not fields.get('vintage'):
            text += '\nГоды в найденных карточках: '+', '.join(years)+'. Сверьте с этикеткой; год вашей бутылки ещё не подтверждён.'
        if item.get('volume_requires_confirmation') and not ui['overrides'].get(str(index), {}).get('volume_ml'):
            text += '\nОбъём прочитан из сокращённого названия чека — проверьте по бутылке.'
        info = ui.get('wine_info', {}).get(str(index), {})
        if info:
            for key, label in (('name', 'Название в источнике'),):
                if info.get(key):
                    value = ', '.join(map(str, info[key])) if isinstance(info[key], list) else str(info[key])
                    text += f'\n{label}: {value}'
            if info.get('source_url') or info.get('source_title'):
                text += f'\nИсточник: {info.get("source_title") or "карточка вина"}\n{info["source_url"]}'
            elif info.get('volume_from_name'):
                text += '\nОбъём дополнен из названия чека — проверьте по бутылке.'
        sources = sorted({source['url'] for values in fields.get('metadata_provenance', {}).values()
                          if isinstance(values, list) for source in values if isinstance(source, dict) and source.get('url')})
        if sources:
            text += '\nИсточники характеристик:\n'+'\n'.join(sources[:4])
        if notice:
            text = notice+'\n\n'+text
        buttons = [[('Название', f'rc:edit:{token}:name:{index}'), ('Стоимость товара', f'rc:edit:{token}:amount:{index}')],
                   [('Количество', f'rc:edit:{token}:quantity:{index}')]]
        if ui.get('stock_only'):
            buttons = [[('Название', f'rc:edit:{token}:name:{index}')]]
        if feature_allowed(self.store, actor, 'wine_cellar'):
            selected = index in ui['selected']
            if draft['status'] != 'confirmed':
                text += ('\n\n✓ Выбрано для винотеки, но ещё не сохранено. Нажмите «К подтверждению выбранных вин».' if selected
                         else '\n\nЧтобы добавить эту строку в запасы, выберите её и затем подтвердите выбранные вина.')
            buttons += [[(('✓ Выбрано · убрать из выбора' if selected else 'Выбрать для винотеки'), f'rc:toggle:{token}:{index}')],
                        [('750 мл', f'rc:volume:{token}:{index}'), ('Другой объём', f'rc:edit:{token}:volume_ml:{index}')],
                        [('Производитель', f'rc:edit:{token}:winery:{index}'),
                         ('Год · выбрать из найденных' if years and not fields.get('vintage') else 'Год', f'rc:edit:{token}:vintage:{index}')],
                        [('Страна', f'rc:edit:{token}:country:{index}'), ('Регион', f'rc:edit:{token}:region:{index}')],
                        [('Сорта винограда', f'rc:edit:{token}:grapes:{index}')],
                        [('Найти сведения о вине', f'rc:enrich:{token}:{index}')]]
            automatic = self.store.get_setting(f'wine_metadata_auto:{actor}', False)
            buttons.append([(f'Автозаполнение: {"вкл" if automatic else "выкл"}', f'rc:autoinfo:{token}:{index}')])
            if state.get('status') == 'pending':
                buttons.append([('Обновить карточку', f'rc:item:{token}:{index}')])
            if ui.get('wine_lookups', {}).get(str(index), {}).get('candidates'):
                buttons.append([('Найденные сведения и источник', f'rc:info:{token}:{index}')])
            if ui['selected']:
                buttons.append([('К подтверждению выбранных вин', f'rc:review:{token}')])
        if draft['status'] == 'confirmed':
            text += '\n\nЧек уже сохранён. Изменения в этом черновике не применяются к запасам.'
            back = self.receipt_back(actor, token)
            buttons = [[('Моя винотека', back)]] if back == 'wc:menu' else [[('Доходы и расходы', back)]]
            if feature_allowed(self.store, actor, 'wine_cellar'):
                buttons.insert(0, [('Добавить вина из учтённого чека', f'rc:cellar:{token}')])
        self.bot._reply(client, actor, text, keyboard(buttons))

    def wine_info(self, client, actor, token, index, page=0):
        from .receipt_wines import lookup_result
        result = lookup_result(self.store, actor, token, index)
        candidates = result.get('candidates', [])[:8]
        page = max(0, min(page, max(0, len(candidates)-1)))
        screen(f'rc:info:{token}:{index}:{page}', f'rc:item:{token}:{index}')
        text = 'Сведения о вине\n'
        if result['status'] == 'pending':
            text += 'Поиск выполняется. Откройте карточку товара через несколько секунд.'
        elif result.get('candidates'):
            text += ('Проверьте название, производителя и объём. Выбор уточнит название и дополнит незаполненные поля; '
                     'название, исправленное вами вручную, сохранится. '
                     'Цена и количество остаются из чека. Год урожая сверяйте с этикеткой.\n')
        buttons = []
        for candidate in candidates[page:page+1]:
            text += f'\nВариант {page+1} из {len(candidates)}\n{candidate["name"]}'
            for key, label in (('winery', 'Производитель'), ('volume_ml', 'Объём, мл'), ('country', 'Страна'),
                               ('region', 'Регион'), ('grapes', 'Сорта'), ('published_vintage', 'Год в источнике · сверьте с бутылкой')):
                if candidate.get(key):
                    value = ', '.join(map(str, candidate[key])) if isinstance(candidate[key], list) else str(candidate[key])
                    text += f'\n{label}: {value}'
            if candidate.get('description'):
                text += '\n'+str(candidate['description'])[:300]
            text += f'\n{candidate.get("source_title") or "Источник"}: {candidate["source_url"]}\n'
            if candidate.get('match_notes'):
                text += '\n'+'; '.join(candidate['match_notes'])[:300]+'\n'
            buttons.append([('Использовать эту карточку', f'rc:useinfo:{token}:{index}:{page}')])
            if candidate.get('published_vintage'):
                buttons.append([(f'Год на моей бутылке: {candidate["published_vintage"]} · подтвердить', f'rc:useyear:{token}:{index}:{page}')])
        navigation = []
        if page:
            navigation.append(('‹ Предыдущая карточка', f'rc:info:{token}:{index}:{page-1}'))
        if page+1 < len(candidates):
            navigation.append(('Следующая карточка ›', f'rc:info:{token}:{index}:{page+1}'))
        if navigation:
            buttons.append(navigation)
        if not result.get('candidates') and result.get('status') != 'pending':
            text += 'Подходящих карточек нет. Уточните название вина или повторите поиск позже.'
        for warning in result.get('warnings', [])[:3]:
            text += '\n'+str(warning)[:250]
        self.bot._reply(client, actor, text, keyboard(buttons))

    def input(self, client, actor, text):
        row = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
        pending = (row or {}).get('pending') or ''
        if not pending.startswith('life:'):
            return False
        ledger.require_owner(self.store, actor, write=True)
        if pending == 'life:receipt':
            self.receipt_prompt(client, actor)
        elif pending == 'life:period':
            parts = text.replace(' — ', ' ').split()
            if len(parts) != 2:
                raise ValueError('Введите две даты через пробел: 01.01.2026 11.10.2026.')
            start, end = day(parts[0]), day(parts[1])
            if start > end:
                raise ValueError('Начальная дата позже конечной.')
            self.store.set_setting(f'life_period:{actor}', {'preset': 'custom', 'start': start, 'end': end})
            self.finance(client, actor)
        elif pending.startswith('life:entry:'):
            field = pending.split(':')[2]
            draft = self.store.get_setting(f'life_entry:{actor}')
            if not draft:
                raise ValueError('Начните добавление операции заново.')
            value = cash(text) if field == 'amount' else day(text) if field == 'date' else ' '.join(text.split())[:300]
            if field == 'amount' and value <= 0:
                raise ValueError('Сумма должна быть больше нуля.')
            if field == 'category' and (not value or len(value) > 80):
                raise ValueError('Укажите категорию, до 80 символов.')
            draft[field] = value
            self.store.set_setting(f'life_entry:{actor}', draft)
            next_field = {'amount': 'date', 'date': 'category', 'category': 'note'}.get(field)
            if draft.get('editing'):
                draft.pop('editing')
                self.store.set_setting(f'life_entry:{actor}', draft)
                self.manual_review(client, actor)
            elif next_field:
                self.manual_step(client, actor, next_field)
            else:
                self.manual_review(client, actor)
        elif pending.startswith('life:receiptedit:'):
            parts = pending.split(':')
            token, field = parts[2:4]
            draft = ledger.get_receipt_draft(self.store, actor, token)
            if self.receipt_ui(actor, token).get('stock_only') and field in {'date', 'total', 'amount', 'quantity'}:
                raise ValueError('Финансовые данные учтённого чека не меняются; здесь добавляются только вина.')
            result = draft['result']
            if len(parts) == 5:
                from .receipt_wines import remember_item
                remember_item(self.store, actor, token, int(parts[4]))
            if field in {'date', 'total', 'amount', 'name', 'quantity'}:
                value = (day(text) if field == 'date' else cash(text) if field in {'total', 'amount'} else
                         number(text) if field == 'quantity' else ' '.join(text.split())[:250])
                if len(parts) == 5:
                    index = int(parts[4])
                    if field == 'name':
                        from .receipt_wines import invalidate_item
                        from .receipts import receipt_identity
                        invalidate_item(self.store, actor, token, index, field)
                        result['items'][index].update(receipt_identity(value, is_wine=result['items'][index].get('is_wine', False)))
                    result['items'][index][field] = value
                else:
                    result[field] = value
                ledger.save_receipt_draft(self.store, actor, result, token)
                if field == 'name' and len(parts) == 5:
                    ui = self.receipt_ui(actor, token)
                    ui['overrides'].setdefault(str(index), {})['name'] = value
                    if str(index) not in ui.setdefault('manual_names', []):
                        ui['manual_names'].append(str(index))
                    self.save_ui(actor, token, ui)
            else:
                index = int(parts[4])
                ui = self.receipt_ui(actor, token)
                value = integer(text) if field == 'volume_ml' else text.strip()[:200]
                if field == 'grapes':
                    from .wine_cellar import _attributes
                    value = _attributes({'grapes': value})['grapes']
                if field == 'volume_ml' and not 20 <= value <= 30000:
                    raise ValueError('Укажите объём от 20 до 30000 мл.')
                if field == 'vintage' and value and value.upper() != 'NV' and (not value.isdigit() or not 1800 <= int(value) <= today().year):
                    raise ValueError('Укажите год вина или NV.')
                from .receipt_wines import invalidate_item
                invalidate_item(self.store, actor, token, index, field)
                ui = self.receipt_ui(actor, token)
                ui['overrides'].setdefault(str(index), {})[field] = value
                self.save_ui(actor, token, ui)
            self.bot.market_controls.pending(actor, None)
            if len(parts) == 5:
                self.receipt_item(client, actor, token, int(parts[4]))
            else:
                self.review_receipt(client, actor, token)
        else:
            raise ValueError('Неизвестный ввод. Нажмите «Отмена» и откройте нужный экран.')
        return True

    def callback(self, client, actor, data):
        ledger.require_owner(self.store, actor)
        parts = data.split(':')
        if parts[0] == 'rc':
            token, action = parts[2], parts[1]
            draft = ledger.get_receipt_draft(self.store, actor, token)
            ui = self.receipt_ui(actor, token)
            if draft['status'] == 'confirmed' and action not in {'review', 'cellar', 'item', 'page', 'changes'}:
                return self.review_receipt(client, actor, token, 'Этот чек уже учтён. Чтобы добавить оставшиеся вина, откройте добавление из учтённого чека ниже.')
            if action == 'review':
                return self.review_receipt(client, actor, token)
            if action == 'changes':
                return self.receipt_changes(client, actor, token, int(parts[3]) if len(parts) > 3 else 0)
            if action == 'cellar':
                new_token = ledger.prepare_receipt_wines(self.store, actor, token)
                return self.review_receipt(client, actor, new_token, 'Проверьте вина из учтённого чека. Уже добавленные строки повторно не поступят в запасы.')
            if action == 'item':
                return self.receipt_item(client, actor, token, int(parts[3]))
            if action in {'autoinfo', 'autoon', 'autooff'}:
                from .wine_cellar import require_owner
                require_owner(self.store, actor, write=True)
                index = int(parts[3])
                if not 0 <= index < len(draft['result']['items']):
                    raise ValueError('Товар не найден')
                if action != 'autoinfo':
                    self.store.set_setting(f'wine_metadata_auto:{actor}', action == 'autoon')
                    return self.receipt_item(client, actor, token, index)
                from .wine_retail import shops
                sources = [item for item in shops(self.store) if item['enabled'] and
                           (item.get('metadata_supported') or item.get('search_supported'))]
                screen(data, f'rc:item:{token}:{index}')
                text = ('Автозаполнение сведений о вине\nБот отправляет название вина и производителя в поиск подключённых источников '
                        'и читает их публичные карточки. Суммы, фискальные номера и данные пользователя не отправляются. '
                        'Подключённые источники:\n'+'\n'.join(item['homepage'] for item in sources)+
                        '\n\nСтрана, регион и сорта дополняются при однозначном совпадении. Год из карточки нужно сверить с бутылкой. '
                        'Настройка сохранится для следующих чеков. Источники подключает администратор.')
                rows = [[('Включить автозаполнение', f'rc:autoon:{token}:{index}')],
                        [('Отключить', f'rc:autooff:{token}:{index}')]]
                return self.bot._reply(client, actor, text, keyboard(rows))
            if action == 'enrich':
                from .receipt_wines import queue_lookup
                index = int(parts[3])
                identifier = queue_lookup(self.store, actor, token, index)
                return self.receipt_item(client, actor, token, index,
                    f'Поиск №{identifier} в очереди. Однозначное совпадение уточнит название и дополнит пустые поля. Исправления будут показаны «было → стало»; ручное название сохранится.')
            if action == 'info':
                return self.wine_info(client, actor, token, int(parts[3]), int(parts[4]) if len(parts) > 4 else 0)
            if action in {'useinfo', 'useyear'}:
                from .receipt_wines import apply_candidate
                index = int(parts[3])
                apply_candidate(self.store, actor, token, index, int(parts[4]), confirm_vintage=action == 'useyear')
                return self.receipt_item(client, actor, token, index, 'Сведения добавлены в черновик. Проверьте их перед сохранением чека.')
            if action == 'pick':
                from .receipt_wines import FIELD_CODES, apply_field_choice
                if len(parts) != 7 or parts[4] not in FIELD_CODES:
                    raise ValueError('Неизвестный вариант поля.')
                index,field = int(parts[3]),FIELD_CODES[parts[4]]
                pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?',(actor,))
                if not pending or pending['pending'] != f'life:receiptedit:{token}:{field}:{index}':
                    raise ValueError('Откройте изменение поля заново.')
                apply_field_choice(self.store,actor,token,index,field,int(parts[5]),parts[6])
                self.bot.market_controls.pending(actor,None)
                return self.receipt_item(client,actor,token,index,
                    'Выбранное значение записано в черновик. Остальные поля не менялись; бутылки ещё не добавлены в запасы.')
            if action == 'edit':
                field = parts[3]
                if ui.get('stock_only') and field in {'date', 'total', 'amount', 'quantity'}:
                    raise ValueError('Финансовые данные учтённого чека не меняются; здесь добавляются только вина.')
                if field not in {'date', 'total', 'name', 'amount', 'quantity', 'volume_ml', 'winery', 'vintage', 'country', 'region', 'grapes'}:
                    raise ValueError('Неизвестное поле.')
                index = int(parts[4]) if len(parts) == 5 else None
                if index is not None and not 0 <= index < len(draft['result'].get('items', [])):
                    raise ValueError('Товар не найден.')
                if field not in {'date', 'total'} and index is None:
                    raise ValueError('Сначала выберите товар.')
                self.bot.market_controls.pending(actor, f'life:receiptedit:{token}:{field}'+(f':{index}' if index is not None else ''))
                screen(data, f'rc:item:{token}:{index}' if index is not None else 'rc:review:'+token)
                prompts = {'date': 'Введите дату чека ДД.ММ.ГГГГ.', 'total': 'Введите итоговую оплаченную сумму чека, ₽.',
                           'amount': 'Введите оплаченную сумму за эту строку с учётом скидок, ₽.',
                           'quantity': 'Введите количество товара. Для винотеки нужно целое число бутылок.',
                           'name': 'Введите название товара.', 'volume_ml': 'Введите объём одной бутылки в мл.',
                           'winery': 'Введите производителя.', 'vintage': 'Введите год вина или NV.',
                           'country': 'Введите страну происхождения вина.', 'region': 'Введите винодельческий регион.',
                           'grapes': 'Введите сорта через запятую. Например: Мальбек, Каберне Совиньон.'}
                from .telegram_wines import wine_edit_prompt
                fields = draft['result'] if index is None else {
                    **draft['result']['items'][index], **ui.get('overrides', {}).get(str(index), {})}
                text = wine_edit_prompt(field, fields.get(field), prompts[field])
                buttons = []
                if index is not None and feature_allowed(self.store,actor,'wine_cellar'):
                    from .receipt_wines import FIELD_CODES, field_choices
                    try:
                        choices = field_choices(self.store,actor,token,index,field)
                    except ValueError:
                        choices = []
                    if field in FIELD_CODES.values() and not choices and ui.get('wine_lookups', {}).get(str(index), {}).get('candidates'):
                        text += '\n\nАктуальных вариантов для этого поля нет: результат мог устареть или источник отключён. Можно ввести значение вручную или обновить поиск.'
                        buttons.append([('Обновить сведения о вине',f'rc:enrich:{token}:{index}')])
                    if choices:
                        text += ('\n\nНайдено в карточках источников. Выберите значение с вашей этикетки '
                                 'или отправьте своё текстом. Порядок вариантов не означает вероятность совпадения.')
                        if len(choices)>1:
                            text += '\nТребует уточнения: в источниках есть несколько вариантов.'
                        if field == 'vintage' and not fields.get('vintage'):
                            text += '\nГод вашей бутылки требует подтверждения; найденный год не выбирается автоматически.'
                        code = next(code for code,name in FIELD_CODES.items() if name == field)
                        options = [(row['label'][:90],f"rc:pick:{token}:{index}:{code}:{row['candidate']}:{row['fingerprint']}")
                                   for row in choices]
                        width = 3 if field == 'vintage' else 1
                        buttons = [options[offset:offset+width] for offset in range(0,len(options),width)]
                return self.bot._reply(client, actor, text, keyboard(buttons))
            if action in {'select', 'toggle', 'volume', 'volumeall'}:
                from .wine_cellar import require_owner
                require_owner(self.store, actor, write=True)
                if action == 'select':
                    ui['selected'] = [i for i, row in enumerate(draft['result']['items']) if row.get('is_wine')]
                elif action == 'toggle':
                    index = int(parts[3])
                    if not 0 <= index < len(draft['result']['items']):
                        raise ValueError('Товар не найден.')
                    ui['selected'] = sorted(set(ui['selected']) ^ {index})
                else:
                    from .receipt_wines import invalidate_item
                    indices = ui['selected'] if action == 'volumeall' else [int(parts[3])]
                    for index in indices:
                        if not 0 <= index < len(draft['result']['items']):
                            raise ValueError('Товар не найден.')
                        invalidate_item(self.store, actor, token, index, 'volume_ml')
                        ui = self.receipt_ui(actor, token)
                        ui['overrides'].setdefault(str(index), {})['volume_ml'] = 750
                        self.save_ui(actor, token, ui)
                self.save_ui(actor, token, ui)
                if action in {'toggle', 'volume'}:
                    return self.receipt_item(client, actor, token, int(parts[3]))
            elif action == 'clear':
                ui['selected'] = []
                self.save_ui(actor, token, ui)
            elif action == 'page':
                ui['page'] = max(0, int(parts[3]))
                self.save_ui(actor, token, ui)
            elif action == 'confirm':
                result = ledger.confirm_receipt(self.store, actor, token, selected_wine_items=ui['selected'],
                                                overrides={int(key): value for key, value in ui['overrides'].items() if int(key) in ui['selected']}, operation_id=token,
                                                add_wines_to_existing=bool(ui.get('stock_only')))
                text = ('Этот чек уже учтён.' if result['already_imported'] else 'Чек сохранён.')+f'\nРасход: {money(result["total"])}. В винотеке: {len(result["wine_lot_ids"])} позиций.'
                if ui.get('stock_only'):
                    text = f'Добавлено в винотеку: {len(result.get("added_wine_lot_ids", []))} позиций. Расход уже учтён ранее; повторной записи нет.'
                if result.get('rounding_difference_cents'):
                    text += f'\nОкругление стоимости бутылок: {money(result["rounding_difference_cents"]/100)}; сумма расхода сохранена точно по чеку.'
                if self.receipt_back(actor, token) == 'wc:menu':
                    return self.bot.wine_controls.menu(client, actor, notice=text)
                return self.finance(client, actor, text)
            else:
                raise ValueError('Неизвестное действие с чеком.')
            return self.review_receipt(client, actor, token)
        action = parts[1]
        if action == 'home':
            return self.finance(client, actor)
        if action == 'receipt':
            return self.receipt_prompt(client, actor)
        if action == 'receipts':
            page = max(0, int(parts[2]) if len(parts) > 2 else 0)
            rows = self.store.query("SELECT token,result FROM life_receipt_drafts WHERE user_id=? AND status='confirmed' GROUP BY receipt_id ORDER BY updated_at DESC LIMIT 11 OFFSET ?", (actor, page*10))
            import json
            buttons = []
            for row in rows[:10]:
                receipt = json.loads(row['result'])
                label = (receipt.get('date') or 'Без даты')+' · '+money(receipt['total'])+' · '+str(receipt.get('merchant') or 'Чек')[:25]
                buttons.append([(label, 'rc:cellar:'+row['token'])])
            nav = []
            if page:
                nav.append(('‹ Чеки', f'lf:receipts:{page-1}'))
            if len(rows) > 10:
                nav.append(('Чеки ›', f'lf:receipts:{page+1}'))
            if nav:
                buttons.append(nav)
            screen(data, 'wc:add')
            return self.bot._reply(client, actor, 'Учтённые чеки\nВыберите чек, чтобы добавить его вина в запасы без повторного расхода.' if rows else 'Учтённых чеков пока нет.', keyboard(buttons))
        if action == 'drafts':
            from .store import utcnow
            rows = self.store.query("SELECT token,result FROM life_receipt_drafts WHERE user_id=? AND status='draft' AND expires_at>=? ORDER BY updated_at DESC LIMIT 21", (actor, utcnow()))
            import json
            buttons = []
            for row in rows[:20]:
                receipt = json.loads(row['result'])
                label = (receipt.get('date') or 'Без даты')+' · '+(money(receipt['total']) if receipt.get('total') is not None else 'Итог не указан')
                buttons.append([(label, 'rc:review:'+row['token'])])
            screen('lf:drafts', 'lf:home')
            text = 'Неподтверждённые чеки\nЧерновики хранятся сутки. Расходы и запас ещё не изменены.' if rows else 'Неподтверждённых чеков нет. Загрузите PDF или фотографию.'
            if len(rows) > 20:
                text += '\nПоказаны последние 20; остальные доступны по кнопке в доставленном результате распознавания.'
            return self.bot._reply(client, actor, text, keyboard(buttons+[[('Загрузить чек', 'lf:receipt')]]))
        if action == 'period':
            if len(parts) == 2:
                return self.period(client, actor)
            if parts[2] == 'custom':
                self.bot.market_controls.pending(actor, 'life:period')
                screen(data, 'lf:period')
                return self.bot._reply(client, actor, 'Введите начало и конец периода через пробел: 01.01.2026 11.10.2026.')
            if parts[2] not in {'month', '3months', 'year', 'all'}:
                raise ValueError('Неизвестный период.')
            self.store.set_setting(f'life_period:{actor}', {'preset': parts[2]})
            return self.finance(client, actor)
        if action == 'history':
            return self.history(client, actor, int(parts[2]))
        if action == 'analytics':
            return self.analytics(client, actor)
        if action == 'entry':
            return self.entry(client, actor, int(parts[2]))
        if action in {'voidask', 'void'}:
            identifier = int(parts[2])
            if action == 'void':
                ledger.void_entry(self.store, actor, identifier)
                return self.finance(client, actor, 'Ошибочная запись удалена из учёта.')
            screen(data, f'lf:entry:{identifier}')
            return self.bot._reply(client, actor, 'Удалить эту ручную запись из учёта?', keyboard([[('Удалить запись', f'lf:void:{identifier}')]]))
        if action == 'add':
            return self.manual(client, actor, parts[2])
        if action == 'step' and parts[2] in {'amount', 'date', 'category', 'note'}:
            return self.manual_step(client, actor, parts[2])
        if action in {'today', 'category', 'nonote', 'edit', 'save'}:
            draft = self.store.get_setting(f'life_entry:{actor}')
            if not draft:
                raise ValueError('Операция уже сохранена или ввод отменён.')
            if action == 'edit':
                draft['editing'] = True
                self.store.set_setting(f'life_entry:{actor}', draft)
                return self.manual_step(client, actor, parts[2])
            if action == 'save':
                if parts[2] != draft['token']:
                    raise ValueError('Откройте актуальный черновик операции.')
                ledger.add_entry(self.store, actor, draft['kind'], draft['amount'], date=draft['date'],
                                 category=draft['category'], note=draft.get('note', ''), operation_id=draft['token'])
                return self.finance(client, actor, 'Операция сохранена.')
            if action == 'today':
                draft['date'] = today().isoformat()
                next_field = 'category'
            elif action == 'category':
                categories = ['Продукты', 'Вино', 'Транспорт', 'Дом', 'Поездки', 'Здоровье', 'Прочее'] if draft['kind'] == 'expense' else ['Зарплата', 'Возврат', 'Подарок', 'Прочее']
                index = int(parts[2])
                if not 0 <= index < len(categories):
                    raise ValueError('Неизвестная категория.')
                draft['category'], next_field = categories[index], 'note'
            else:
                draft['note'], next_field = '', None
            editing = draft.pop('editing', False)
            self.store.set_setting(f'life_entry:{actor}', draft)
            return self.manual_review(client, actor) if editing or next_field is None else self.manual_step(client, actor, next_field)
        if action == 'export':
            if parts[2] not in {'csv', 'xlsx'}:
                raise ValueError('Выберите CSV или Excel.')
            start, end, _ = self.bounds(actor)
            identifier = self.store.enqueue('life_export', {'actor_id': actor, 'owner_id': actor, 'report_id': uuid4().hex,
                                                          'format': parts[2], 'start': start, 'end': end},
                                            dedupe_key=f'life-export:{actor}:{uuid4().hex}')
            return self.finance(client, actor, f'Экспорт №{identifier} в очереди. Файл придёт сюда по готовности.')
        raise ValueError('Неизвестный экран. Откройте /life.')
