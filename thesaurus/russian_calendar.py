"""Russian working calendar, with reviewed additions missing from holidays.

Source for 2026: https://cbr.ru/other/holidays/ (reviewed 2026-10-08).
These are calendar facts, not observations inferred from missing market data.
"""
from datetime import date

import holidays

VERSION = 'cbr-2026-1'
EXTRA_DAYS_OFF = frozenset(date.fromisoformat(day) for day in (
    '2026-01-09', '2026-03-09', '2026-05-11', '2026-12-31',
))


def calendar():
    result = holidays.Russia()
    result.update({day: 'Перенесённый выходной день' for day in EXTRA_DAYS_OFF})
    return result
