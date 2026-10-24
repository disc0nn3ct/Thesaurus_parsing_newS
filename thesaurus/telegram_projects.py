"""Explicit project registry and navigation; no imports from uploaded archives."""
from dataclasses import dataclass

from .telegram_navigation import screen


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    summary: str
    ready: bool = True

    @property
    def route(self):
        return f'pj:{self.id}'


PROJECTS = (
    Project('market', 'Рынок и экономика', 'Показатели Банка России, RUONIA и Мосбиржа.'),
    Project('life', 'Жизнь', 'Вино, доходы, расходы и чеки.'),
)
CHILD_PROJECTS = (Project('vivino', 'Вино', 'Vivino, география вкусов и моя винотека.'),)


def project(identifier):
    for item in (*PROJECTS, *CHILD_PROJECTS):
        if item.id == identifier:
            return item
    raise ValueError('Неизвестный проект. Откройте /projects.')


def setting_key(actor, identifier, name):
    """Namespace future project settings by both project and numeric owner."""
    project(identifier)
    if type(actor) is not int or actor <= 0 or not isinstance(name, str) or not name.isidentifier():
        raise ValueError('Неверная настройка проекта')
    return f'project:{identifier}:{actor}:{name}'


class ProjectControls:
    def __init__(self, bot):
        self.bot = bot

    def home(self, client, actor, identifier):
        from .telegram_charts import keyboard
        item = project(identifier)
        if not self.bot.store.authorize(actor):
            raise PermissionError('Нет доступа')
        from .subscription_access import project_allowed
        if not project_allowed(self.bot.store, actor, identifier):
            raise PermissionError('Этот проект не подключён для вас. Обратитесь к администратору.')
        if identifier not in {'market', 'vivino', 'life'}:
            raise ValueError('Инструмент ещё не подключён')
        self.bot.market_controls.pending(actor, None)
        self.bot.store.set_setting(setting_key(actor, identifier, 'visited'), True)
        screen(item.route, 'menu:home')
        if identifier == 'life':
            return self.bot.life_controls.home(client, actor)
        if identifier == 'market':
            rows = [[('Показатели Банка России', 'cb:menu')],
                [('ЦБ · с чего начать', 'cb:guide')],
                [('Построить график', 'ch:menu'), ('Получить отчёт', 'menu:report')],
                [('Доллар для поездки', 'usd:menu')],
                [('Аналитика и прогнозы', 'menu:data')],
                    [('Мои настройки и расписания', 'menu:settings')],
                    [('Помощь по проекту', 'menu:help')]]
            if self.bot.store.authorize(actor, admin=True):
                rows.append([('Источники и автосбор', 'pj:market:manage')])
            from .feature_access import CATALOG as FUNCTIONS
            from .feature_access import assigned as functions
            titles = ', '.join(FUNCTIONS[key][0] for key in functions(self.bot.store, actor) if set(FUNCTIONS[key][1]) - {'vivino', 'life'})
            from .feature_access import allowed as function_allowed
            descriptions = [('cbr', 'Начните с каталога ЦБ: готовые группы, поиск и загруженные показатели.'),
                            ('charts', 'График — показатель, период и прогноз.'),
                            ('reports', 'Отчёт — обновление данных и новый выпуск.'),
                            ('dollar', 'Доллар — обзор курса, прогноз и расчёт вашей покупки.'),
                            ('schedules', 'Мои настройки — личные фильтры, расписания и уведомления.')]
            text = f'{item.title}\nПодключено для вас: {titles}.\n\n'+'\n'.join(
                description for feature, description in descriptions if function_allowed(self.bot.store, actor, feature))
        else:
            return self.bot.wine_controls.home(client, actor)
        self.bot._reply(client, actor, text, keyboard(rows))

    def callback(self, client, actor, data):
        from .telegram_charts import keyboard
        if not self.bot.store.authorize(actor):
            raise PermissionError('Нет доступа')
        parts = data.split(':')
        if len(parts) not in {2, 3}:
            raise ValueError('Неизвестный экран проекта')
        item = project(parts[1])
        if len(parts) == 2:
            return self.home(client, actor, item.id)
        page = parts[2]
        screen(data, item.route)
        if item.id == 'market' and page == 'manage':
            self.bot._admin(actor)
            self.bot._reply(client, actor, 'Рынок и экономика · управление\n'
                'Общие источники, сбор данных и правила проекта. Личная доставка настраивается в «Моих настройках».', keyboard([
                    [('Источники ЦБ и Мосбиржи', 'menu:sources')],
                    [('Автосбор данных', 'menu:autocollect')],
                    [('Целостность данных', 'ig:menu')],
                    [('Правила Wordstat', 'menu:alerts'), ('Предложения AI', 'menu:proposals')],
                    [('Проверка источников', 'menu:verify_help')]]))
        elif item.id == 'vivino' and page == 'data':
            self.bot.vivino_controls.callback(client, actor, 'vi:upload')
        elif item.id == 'vivino' and page == 'workflow':
            self.bot.vivino_controls.callback(client, actor, 'vi:help')
        else:
            raise ValueError('Неизвестный экран проекта')
