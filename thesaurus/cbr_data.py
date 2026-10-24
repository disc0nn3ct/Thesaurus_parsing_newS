"""Public CBR RUONIA rate/volume adapter and explicit metric identities."""
import math
import re
from datetime import date, timedelta
from xml.etree import ElementTree

import pandas as pd
import requests

METRICS = ["RUONIA", "Индекс RUONIA", "1 мес", "3 мес", "6 мес", "Объём RUONIA"]
RATE_URL = "https://cbr.ru/hd_base/ruonia/ruonia/"


def parse_day(value):
    """ISO dates must never be interpreted with a day-first month swap."""
    text = str(value)
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text, format="ISO8601", errors="coerce")
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def parse_rates(content, *, allow_empty=False):
    if len(content) > 10_000_000 or b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("Invalid CBR XML")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        raise ValueError("Invalid CBR XML") from None
    if any(node.tag.split('}')[-1] == 'Fault' for node in root.iter()):
        raise ValueError('CBR SOAP fault')
    rows = []
    for node in root.iter():
        fields = {child.tag.split("}")[-1]: child.text for child in node}
        if {"D0", "ruo", "vol"} <= fields.keys():
            day = date.fromisoformat(fields["D0"][:10]).isoformat()
            rows.append({"Дата": day, "Ставка RUONIA": float(fields["ruo"]),
                         "Объём RUONIA": float(fields["vol"])})
            if not all(math.isfinite(rows[-1][key]) for key in ("Ставка RUONIA", "Объём RUONIA")):
                raise ValueError("Invalid CBR numeric value")
    if not rows and not allow_empty:
        raise ValueError("CBR RUONIA response has no observations")
    if not rows and not any(node.tag.split('}')[-1] in {'RuoniaResult', 'NewDataSet', 'diffgram'} for node in root.iter()):
        raise ValueError('CBR RUONIA response has no expected result')
    return pd.DataFrame(rows, columns=['Дата', 'Ставка RUONIA', 'Объём RUONIA'])


def fetch_rates(today):
    start = today - timedelta(days=365 * 3)
    body = ('<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
            '<Ruonia xmlns="http://web.cbr.ru/"><fromDate>' + start.isoformat()
            + 'T00:00:00</fromDate><ToDate>' + today.isoformat()
            + 'T00:00:00</ToDate></Ruonia></soap:Body></soap:Envelope>')
    response = requests.post("https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx", data=body,
                             headers={"Content-Type": "text/xml; charset=utf-8",
                                      "SOAPAction": "http://web.cbr.ru/Ruonia"}, timeout=30)
    response.raise_for_status()
    return parse_rates(response.content)
