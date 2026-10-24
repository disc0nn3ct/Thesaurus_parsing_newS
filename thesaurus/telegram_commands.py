"""Telegram's native command menu; command visibility never grants access."""
import logging

logger = logging.getLogger(__name__)
PUBLIC = [('start', 'Выбрать проект'), ('projects', 'Все проекты'), ('wines', 'Вина: Vivino и моя винотека'),
          ('life', 'Жизнь: вино, доходы и расходы'), ('expenses', 'Доходы и расходы: личный учёт'),
          ('receipt', 'Добавить покупки по чеку'),
          ('cellar', 'Моя винотека: запасы и стоимость'), ('vivino', 'Vivino: загрузить ZIP и получить карту'),
          ('dollar', 'USD/RUB: покупка долларов для поездки'),
          ('chart', 'Рынок: построить график'),
          ('graphs', 'Рынок: расписание графиков'), ('subscriptions', 'Рынок: расписание отчётов'),
          ('notifications', 'Рынок: уведомления'), ('report', 'Рынок: получить отчёт'),
          ('settings', 'Рынок: мои настройки'), ('status', 'Состояние сервиса'), ('help', 'Навигация и помощь')]
ADMIN = [('admin', 'Управление сервисом'), ('autocollect', 'Рынок: автосбор данных'),
         ('monitor', 'Мониторинг и история пользователей и сервера'),
         ('integrity', 'Рынок: сверка истории и исправления'),
         ('sources', 'Рынок: источники ЦБ и Мосбиржи'), ('users', 'Доступ ко всему боту')]


async def configure(client):
    from pyrogram import raw
    from pyrogram.types import BotCommand, BotCommandScopeAllPrivateChats
    try:
        await client.set_bot_commands([BotCommand(name, description) for name, description in PUBLIC if name in {'start', 'projects', 'status', 'help'}],
                                      scope=BotCommandScopeAllPrivateChats())
        await client.invoke(raw.functions.bots.SetBotMenuButton(
            user_id=raw.types.InputUserEmpty(), button=raw.types.BotMenuButtonCommands()))
        logger.info('Telegram commands and menu button configured')
    except Exception as exc:  # noqa: BLE001 - Menu setup must not stop collection or expose credentials.
        logger.warning('Telegram command menu setup failed (%s)', type(exc).__name__)


def configure_chat(client, store, actor):
    from pyrogram.types import BotCommand, BotCommandScopeChat

    from .subscription_access import command_route, route_allowed
    if not store.one('SELECT 1 FROM users WHERE user_id=?', (actor,)):
        return
    commands = [(name, description) for name, description in PUBLIC
                if route_allowed(store, actor, command_route(name) or 'menu:home')]
    commands += ADMIN if store.authorize(actor, admin=True) else []
    try:
        client.set_bot_commands([BotCommand(name, description) for name, description in commands],
                                scope=BotCommandScopeChat(actor))
    except Exception as exc:  # noqa: BLE001 - A temporary Telegram failure must not hide the home screen.
        logger.warning('Telegram private command menu setup failed (%s)', type(exc).__name__)
