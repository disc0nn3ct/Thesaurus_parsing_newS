"""Bounded, redacted descriptions of incoming requests and visible bot replies."""
import math
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

HIDDEN = '[скрыто]'
SENSITIVE_FIELD = re.compile(r'(?i)^(?:password|passwd|api_key|api_hash|api_secret|bot_token|access_token|refresh_token|client_secret|token|secret|authorization|cookie|пароль|токен)$')
SECRET_LINE = re.compile(r'(?im)(\b(?:api[_ -]?(?:key|hash|token)|bot[_ -]?token|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|'
                         r'''authorization|password|passwd|token|secret|api_secret|пароль|секрет|токен|cookie|set-cookie|session_string)\b["']?\s*(?:[:=]|\s)\s*)[^\r\n]+''')
PARAMETERS = {'section','dataset','identity','selection','scope','mode','period','period_start','period_end','start','end',
              'date_from','date_to','forecast_date','train_start','train_end','trip','horizon','horizon_days','analysis_months',
              'history','forecast','method','model','long_method','distribution','quote','total','owned','commission_pct',
              'commission_fixed','rub_yield','risk_budget','amount','currency','name','winery','vintage','volume_ml',
              'country','region','grapes','query','format','detail','frequency','enabled','times','weekdays',
              'timezone','config','options','period_days','forecast_weeks','metric','series_id','page','indicator'}


def redact_text(value, limit=4000):
    if not isinstance(value,str):
        return ''
    text = ''.join(c for c in value[:20000] if c in '\n\t' or c.isprintable() and unicodedata.category(c)!='Cf')
    text = re.sub(r'(?<![A-Za-z0-9_])(?:bot)?\d{6,14}:[A-Za-z0-9_-]{20,}\b',HIDDEN,text)
    text = re.sub(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',HIDDEN,text)
    text = re.sub(r'\b[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b',HIDDEN,text)
    text = SECRET_LINE.sub(lambda m:m[1]+HIDDEN,text)
    text = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*','Bearer '+HIDDEN,text)
    text = re.sub(r'(?i)(\b(?:ФН|ФД|ФПД?|ИНН|фискальный\s+(?:накопитель|документ|признак(?:\s+документа)?))\s*[:№=#]?\s*)\d[\d -]*',lambda m:m[1]+HIDDEN,text)
    def url(match):
        try:
            parts = urlsplit(match[0])
            host = parts.hostname or ''
            if ':' in host:
                host = '['+host+']'
            if parts.port:
                host += ':'+str(parts.port)
            return urlunsplit((parts.scheme,host,parts.path,'[параметры скрыты]' if parts.query else '',
                               '[фрагмент скрыт]' if parts.fragment else ''))
        except ValueError:
            return '[ссылка скрыта]'
    text = re.sub(r'https?://[^\s<>]+',url,text)
    return text[:limit]+('… [сокращено]' if len(text)>limit else '')


def selected_parameters(value, depth=0):
    if not isinstance(value,dict) or depth>2:
        return {}
    result = {}
    for key,item in value.items():
        if key not in PARAMETERS:
            continue
        if isinstance(item,dict):
            cleaned = selected_parameters(item,depth+1)
        elif isinstance(item,str):
            cleaned = redact_text(item,500)
        elif item is None or type(item) in {bool,int} or type(item) is float and math.isfinite(item):
            cleaned = item
        elif isinstance(item,list):
            cleaned = [redact_text(x,150) if isinstance(x,str) else x for x in item[:20]
                       if isinstance(x,str) or x is None or type(x) in {bool,int} or type(x) is float and math.isfinite(x)]
        else:
            continue
        result[key] = cleaned
        if len(result)>=40:
            break
    return result


def _positive(value):
    return value if type(value) is int and value>0 else None


def request_details(function_name, event, *, pending=None, record=None, callback=False):
    record = record if isinstance(record,dict) else {}
    result = {}
    if callback:
        message = getattr(event,'message',None)
        data = getattr(event,'data',None)
        title = None
        markup = getattr(message,'reply_markup',None)
        rows = getattr(markup,'inline_keyboard',None)
        if isinstance(rows,(list,tuple)):
            title = next((getattr(button,'text',None) for row in rows for button in row
                          if getattr(button,'callback_data',None)==data),None)
        if not title and record.get('message_id')==getattr(message,'id',None):
            title = record.get('button_labels',{}).get(data)
        result['kind'] = 'button'
        if isinstance(title,str):
            result['button_text'] = redact_text(title,300)
        identity = _positive(getattr(message,'id',None))
        if identity:
            result['menu_message_id'] = identity
    else:
        result['kind'] = 'text'
        text = getattr(event,'text',None)
        sensitive = isinstance(pending,str) and any(SENSITIVE_FIELD.fullmatch(p) for p in pending.split(':'))
        if isinstance(text,str) and text:
            result['text'] = '[Секретный ввод скрыт]' if sensitive else redact_text(text)
        for kind in ('document','photo','video','audio','voice','video_note','animation','sticker','contact','location'):
            attachment = getattr(event,kind,None)
            if attachment is None:
                continue
            # Pyrogram Message attributes are real objects; mocks/unknown placeholders are ignored.
            if kind not in {'contact','location'} and not any(type(getattr(attachment,field,None)) is int
                or isinstance(getattr(attachment,field,None),str) for field in ('file_size','file_name','width','duration','file_unique_id')):
                continue
            result['kind'] = kind
            name = getattr(attachment,'file_name',None)
            if isinstance(name,str):
                result['file_name'] = redact_text(re.split(r'[\\/]',name)[-1],240)
            mime = getattr(attachment,'mime_type',None)
            if isinstance(mime,str):
                result['mime_type'] = redact_text(mime,100)
            for field in ('file_size','width','height','duration'):
                number = getattr(attachment,field,None)
                if type(number) is int and number>=0:
                    result[field] = number
            break
        caption = getattr(event,'caption',None)
        if isinstance(caption,str) and caption:
            result['caption'] = redact_text(caption,2000)
        identity = _positive(getattr(event,'id',None))
        if identity:
            result['message_id'] = identity
        if pending and isinstance(record.get('summary'),str):
            result['form_prompt'] = redact_text(record['summary'],1000)
    return result


def response_details(text, message_id=None):
    result = {'text':redact_text(text,2000)}
    if _positive(message_id):
        result['message_id'] = message_id
    return result
