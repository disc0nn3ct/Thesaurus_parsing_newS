"""Thread-local UI navigation, distinct from immutable report delivery."""
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from time import perf_counter

CURRENT = ContextVar('telegram_ui', default=None)

PARENTS = {
    'menu:report': 'pj:market', 'menu:data': 'pj:market', 'menu:settings': 'pj:market',
    'menu:status': 'menu:home', 'menu:help': 'pj:market', 'menu:admin': 'menu:home',
    'menu:guide': 'menu:home', 'pj:market': 'menu:home', 'pj:life': 'menu:home', 'pj:vivino': 'pj:life',
    'vi:menu': 'pj:vivino', 'wn:menu': 'pj:life', 'wc:menu': 'pj:vivino',
    'pj:market:manage': 'pj:market',
    'menu:history': 'menu:report', 'menu:subscriptions': 'menu:settings',
    'menu:regions': 'menu:settings', 'menu:keywords': 'menu:settings',
    'menu:indicators': 'menu:settings', 'menu:sources': 'pj:market:manage',
    'menu:users': 'menu:admin', 'menu:budget': 'menu:admin', 'menu:jobs': 'menu:admin',
    'menu:alerts': 'pj:market:manage', 'menu:proposals': 'pj:market:manage',
    'menu:verify_help': 'pj:market:manage', 'ch:menu': 'pj:market',
    'menu:autocollect': 'pj:market:manage',
    'menu:graphs': 'menu:settings',
    'mon:home': 'menu:admin', 'mon:server': 'mon:home',
    'mk:menu:all': 'menu:data', 'mk:open:correlations': 'menu:data',
    'mk:open:alerts': 'menu:settings', 'menu:forecasts': 'menu:data',
    'menu:ruonia_forecasts': 'menu:data', 'menu:backtest_help': 'menu:data',
    'menu:export_help': 'menu:data',
}


def screen(route, back=None):
    state = CURRENT.get()
    if state is not None:
        state['route'] = route
        state['back'] = back if back is not None else PARENTS.get(route, 'menu:home')


def route_for(data):
    if data.startswith(('ms:',)):
        return 'menu:sources'
    if data.startswith('mi:'):
        return 'menu:indicators'
    if data.startswith('ma:'):
        return 'mk:open:alerts'
    if data.startswith('user:'):
        return 'menu:users'
    if data.startswith(('history:', 'report:')):
        return 'menu:history'
    if data.startswith('request:'):
        return 'menu:report'
    if data.startswith(('sub:', 'su:')):
        return 'menu:subscriptions'
    if data.startswith('alert:'):
        return 'menu:alerts'
    if data.startswith('proposal:'):
        return 'menu:proposals'
    if data.startswith('adminpage:'):
        return 'menu:' + data.split(':')[1]
    if data.startswith(('rg:', 'kw:')):
        return 'menu:regions' if data.startswith('rg:') else 'menu:keywords'
    return data


def interaction(*, callback=False):
    """Each synchronous Pyrogram handler gets its own editing context."""
    def decorate(function):
        @wraps(function)
        def wrapped(self, client, event):
            actor = getattr(getattr(event, 'from_user', None), 'id', None)
            message = getattr(event, 'message', None) if callback else event
            chat = getattr(message, 'chat', None)
            kind = getattr(chat, 'type', None)
            private = (type(actor) is int and actor > 0 and getattr(chat, 'id', None) == actor
                       and getattr(kind, 'value', kind) == 'private')
            if type(actor) is not int or not self.store.authorize(actor):
                try:
                    return function(self, client, event)
                finally:
                    if private:
                        from .monitor_content import request_details
                        from .monitoring import interaction_action, record_interaction
                        record_interaction(self.store, actor, interaction_action(function.__name__,event,callback=callback),
                                           {'monitor_outcome':'denied','monitor_request':request_details(function.__name__,event,callback=callback)}, 0)
            with self._ui_lock_guard:
                lock = self._ui_locks.setdefault(actor, RLock())
            with lock:
                return run(self, client, event, actor, private)

        def run(self, client, event, actor, private):
            started = perf_counter()
            from .monitoring import interaction_action, record_interaction
            pending = self.store.one('SELECT pending FROM bot_sessions WHERE user_id=?', (actor,))
            action = interaction_action(function.__name__,event,pending['pending'] if pending else None,callback=callback)
            message = getattr(event, 'message', None) if callback else event
            text = getattr(message, 'text', '') or ''
            record = self.store.get_setting(f'ui_screen:{actor}', {})
            from .monitor_content import request_details
            request = request_details(function.__name__,event,pending=pending['pending'] if pending else None,
                                      record=record,callback=callback)
            data = getattr(event, 'data', '') if callback else ''
            if callback:
                target = getattr(message, 'id', None)
                # Full report buttons open a UI screen; never edit delivered reports.
                if data.startswith(('report:', 'history:')):
                    target = record.get('message_id')
                route = route_for(data)
            else:
                target = record.get('message_id') if not text.startswith('/') or text.strip() == '/cancel' else None
                command = text.strip().split(maxsplit=1)[0].lstrip('/').split('@')[0] if text.strip() else ''
                route = record.get('route', 'menu:home') if target else 'menu:' + command
            state = {'actor': actor, 'message_id': target if type(target) is int and target > 0 else None,
                     'monitor_request':request,
                     'route': route, 'back': record.get('back', 'menu:home') if not callback and target else PARENTS.get(route, 'menu:home')}
            token = CURRENT.set(state)
            try:
                return function(self, client, event)
            except Exception as exc:
                from .monitoring import mark_error
                mark_error(exc)
                raise
            finally:
                if private:
                    record_interaction(self.store,actor,action,state,(perf_counter()-started)*1000)
                CURRENT.reset(token)
        return wrapped
    return decorate


def checked(label, selected):
    return ('✓ ' if selected else '') + label


def normalize_buttons(rows, *, pending=None, back=None, route=None):
    """Deduplicate navigation and add a consistent footer after access filtering.

    Contextual cancel actions are retained because they can clear a preview or
    draft more precisely than a generic cancel. Job cancellation and pagination
    remain distinct actions. ``route=None`` performs deduplication only.
    """
    from pyrogram.types import InlineKeyboardButton

    rows = [list(row) for row in rows]

    def callback(button):
        return getattr(button, 'callback_data', None)

    def label(button):
        return ' '.join(str(getattr(button, 'text', '')).split()).casefold()

    def cancel(button):
        return callback(button) in {'ui:cancel', 'wc:cancel'} or label(button) in {
            'отмена', 'отменить ввод', 'отменить добавление', 'отменить добавление вина'}

    def previous(button):
        return label(button) in {'назад', '← назад', '‹ назад', '⬅ назад', 'вернуться'}

    buttons = [button for row in rows for button in row]
    cancels = [button for button in buttons if cancel(button)]
    selected_cancel = next((button for button in cancels if callback(button) != 'ui:cancel'),
                           cancels[0] if cancels else None)
    previous_buttons = [button for button in buttons if previous(button) and not cancel(button)]
    selected_previous = next((button for button in previous_buttons if callback(button) == back),
                             previous_buttons[0] if previous_buttons else None)
    seen = set()
    result = []
    for row in rows:
        kept = []
        for button in row:
            data = callback(button)
            if (cancel(button) and button is not selected_cancel
                    or previous(button) and not cancel(button) and button is not selected_previous):
                continue
            if data is not None:
                if data in seen:
                    continue
                seen.add(data)
            kept.append(button)
        if kept:
            result.append(kept)
    if route is not None and route != 'menu:home':
        if pending and selected_cancel is None:
            result.append([InlineKeyboardButton('Отмена', callback_data='ui:cancel')])
            seen.add('ui:cancel')
        footer = []
        if back and back not in {route, 'menu:home'} and back not in seen and selected_previous is None:
            footer.append(InlineKeyboardButton('← Назад', callback_data=back))
        if 'menu:home' not in seen:
            footer.append(InlineKeyboardButton('⌂ Проекты', callback_data='menu:home'))
        if footer:
            result.append(footer)
    return result
