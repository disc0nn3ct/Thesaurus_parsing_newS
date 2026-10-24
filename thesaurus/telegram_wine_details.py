"""Saved-wine completion screens; changes are reviewed before touching inventory."""
from . import wine_cellar as cellar
from . import wine_details as details
from .telegram_charts import keyboard
from .telegram_navigation import screen
from .telegram_wines import _id, _page, wine_edit_prompt

PROMPTS = {'name':'Введите название с этикетки.','winery':'Введите производителя.',
           'vintage':'Введите год с этикетки или NV, если вино без указания года.',
           'volume_ml':'Введите объём в мл, например 750.','country':'Введите страну происхождения.',
           'region':'Введите винодельческий регион.','grapes':'Введите сорта через запятую.'}


class WineDetailsControls:
    def __init__(self, wines):
        self.wines,self.store,self.reply = wines,wines.store,wines.reply

    def incomplete(self,client,actor,page=0):
        cellar.require_owner(self.store,actor,write=True)
        self.wines._pending(actor,None)
        items = [lot for lot in cellar.lots(self.store,actor,include_empty=True) if details.missing(lot)]
        page = min(_page(page),max(0,(len(items)-1)//8))
        screen(f'wc:incomplete:{page}','wc:menu')
        text = f'Вина · требуют уточнения\nПартий с незаполненными сведениями: {len(items)}.\n'
        text += 'Для полной статистики нужны производитель, год (или NV), объём, страна, регион и сорта. Уточняйте по этикетке; неизвестные сведения можно оставить пустыми.\n'
        rows = []
        for lot in items[page*8:page*8+8]:
            text += '\n'+lot['name'][:120]+'\nНе заполнено: '+', '.join(details.LABELS[key] for key in details.missing(lot))+'.\n'
            rows.append([(lot['name'][:70],f"wc:details:{lot['id']}")])
        if not items:
            text += '\nУ всех сохранённых вин заполнены характеристики.'
        if page:
            rows.append([('‹ Предыдущие',f'wc:incomplete:{page-1}')])
        if (page+1)*8<len(items):
            rows.append([('Следующие ›',f'wc:incomplete:{page+1}')])
        return self.reply(client,actor,text,keyboard(rows))

    def menu(self,client,actor,lot,notice=''):
        try:
            current = details.draft(self.store,actor,lot,create=True)
        except ValueError:
            cellar.get_lot(self.store,actor,lot)
            previous = self.store.get_setting(details.key(actor,lot),{})
            if not isinstance(previous,dict) or not previous.get('token'):
                raise
            self.wines._pending(actor,None)
            screen(f'wc:details:{lot}',f'wc:lot:{actor}:{lot}')
            return self.reply(client,actor,'Характеристики вина уже изменились. Этот черновик не применён. '
                              'Отмените его, чтобы открыть актуальные сведения.',keyboard([
                                  [('Отменить старый черновик',f"wc:dd:{lot}:{previous['token']}")]]))
        self.wines._pending(actor,None)
        screen(f'wc:details:{lot}',f'wc:lot:{actor}:{lot}')
        fields = current['fields']
        text = 'Уточнение сохранённого вина\n'+fields['name']+'\n'
        text += '\n'.join(details.LABELS[field]+': '+details.display(field,fields.get(field)) for field in details.LABELS)
        missing = details.missing(fields)
        text += '\n\n'+('Требует уточнения: '+', '.join(details.LABELS[field] for field in missing)+'.' if missing else 'Все характеристики заполнены.')
        changed = [field for field in details.LABELS if fields[field]!=current['original'][field]]
        if changed:
            text += '\n\nИзменения: было → стало\n'+'\n'.join(details.LABELS[field]+': '+details.display(field,current['original'][field])+' → '+details.display(field,fields[field]) for field in changed)
        text += '\n\nИзменения применятся после «Сохранить уточнения». Статистика пересчитается; покупка, количество и стоимость сохранятся.'
        if notice:
            text = notice+'\n\n'+text
        text += '\n○ — не заполнено; ✓ — значение указано.'
        rows = [[('Найти недостающие сведения',f"wc:ds:{lot}:{current['token']}")]]
        rows += [[(('○ ' if field in missing else '✓ ')+details.LABELS[field],f"wc:df:{lot}:{code}:{current['token']}")]
                 for code,field in sorted(details.CODES.items(),key=lambda item:item[1] not in missing)]
        found = current.get('lookup',{})
        if found:
            text += '\n\nНайденных карточек: '+str(len(found.get('candidates',[])))+'. Откройте нужное поле, чтобы выбрать вариант.'
            for warning in found.get('warnings',[]):
                text += '\n'+str(warning)[:200]
        if changed:
            rows.append([('Сохранить уточнения',f"wc:du:{lot}:{current['token']}")])
        rows.append([('Отменить уточнения',f"wc:dd:{lot}:{current['token']}")])
        return self.reply(client,actor,text,keyboard(rows))

    def field(self,client,actor,lot,code,token):
        current = details.draft(self.store,actor,lot)
        if code not in details.CODES or token!=current['token']:
            raise ValueError('Эта форма устарела.')
        field = details.CODES[code]
        self.wines._pending(actor,f'wine:details:{lot}:{code}:{token}')
        screen(f'wc:df:{lot}:{code}:{token}',f'wc:details:{lot}')
        prompt = PROMPTS[field]
        if field not in {'name','volume_ml'}:
            prompt += ' Если сведения неизвестны, можно написать «не знаю».'
        text = wine_edit_prompt(field,current['fields'].get(field),prompt)
        choices = details.choices(self.store,actor,current,field)
        rows = []
        if choices:
            text += '\n\nВыберите вариант с вашей этикетки или введите свой. Найденный год не подтверждает год вашей бутылки.'
            options = [(row['label'][:80],f"wc:dp:{lot}:{code}:{token}:{row['fingerprint']}") for row in choices]
            width = 3 if field=='vintage' else 1
            rows = [options[offset:offset+width] for offset in range(0,len(options),width)]
        elif current.get('lookup'):
            text += '\n\nАктуальных вариантов для этого поля нет. Введите сведения с этикетки или обновите поиск в карточке.'
        return self.reply(client,actor,text,keyboard(rows))

    def callback(self,client,actor,data):
        cellar.require_owner(self.store,actor,write=True)
        parts = data.split(':')
        if parts[1]=='incomplete' and len(parts)==3:
            return self.incomplete(client,actor,_page(parts[2]))
        lot = _id(parts[2])
        if parts[1]=='details' and len(parts)==3:
            return self.menu(client,actor,lot)
        if parts[1]=='dd' and len(parts)==4:
            current = self.store.get_setting(details.key(actor,lot),{})
            cellar.get_lot(self.store,actor,lot)
            if current.get('token')!=parts[3]:
                raise ValueError('Эта форма устарела.')
            self.store.set_setting(details.key(actor,lot),None)
            return self.wines.card(client,actor,lot)
        current = details.draft(self.store,actor,lot)
        if parts[1]=='df' and len(parts)==5:
            return self.field(client,actor,lot,parts[3],parts[4])
        if parts[1]=='dp' and len(parts)==6 and parts[3] in details.CODES:
            details.apply_choice(self.store,actor,lot,parts[4],details.CODES[parts[3]],parts[5])
            return self.menu(client,actor,lot)
        if len(parts)!=4 or parts[3]!=current['token']:
            raise ValueError('Эта форма устарела.')
        if parts[1]=='du':
            cellar.update_details(self.store,actor,lot,current['fields'],current['base'])
            self.store.set_setting(details.key(actor,lot),None)
            return self.menu(client,actor,lot,'Уточнения сохранены. Статистика вин обновлена.')
        if parts[1]=='ds':
            return self.wines._queue(client,actor,'wine_details_lookup',{'lot_id':lot,'details_token':current['token']},
                                     'Поиск характеристик сохранённого вина',f'wc:details:{lot}')
        raise ValueError('Неизвестное действие уточнения вина.')

    def input(self,client,actor,text,pending):
        parts = pending.split(':')
        if len(parts)!=5 or parts[3] not in details.CODES:
            raise ValueError('Эта форма устарела.')
        lot = _id(parts[2])
        details.set_field(self.store,actor,lot,parts[4],details.CODES[parts[3]],text)
        return self.menu(client,actor,lot)
