"""Readable market diagnostics; machine payloads stay in SQLite/exports."""


def correlation(pair, titles):
    left, right = titles.get(pair['left'], pair['left']), titles.get(pair['right'], pair['right'])
    lines = [f'Связь: {left} ↔ {right}']
    if pair.get('status') != 'ok':
        lines.append(f"Недостаточно сопоставимых данных для расчёта (пар: {pair.get('n', 0)}).")
        return '\n'.join(lines)
    frequency = {'daily': 'дневных', 'weekly': 'недельных', 'monthly': 'месячных',
                 'quarterly': 'квартальных', 'annual': 'годовых'}.get(pair['frequency'], pair['frequency'])
    lines.append(f"Pearson {pair['pearson']:.2f}; Spearman {pair['spearman']:.2f}. Пар {pair['n']} из {pair['window']} {frequency}; покрытие {pair['coverage']:.0%}.")
    lag = pair.get('lag', 0)
    lines.append('Изменения за один период; без лага.' if not pair.get('levels') and lag == 0 else
                 'Исследование уровней; без лага.' if lag == 0 else
                 f"{'Уровни' if pair.get('levels') else 'Изменения'}; {left if lag > 0 else right} опережает {right if lag > 0 else left} на {abs(lag)} период(а).")
    if pair.get('pearson_q') is not None:
        lines.append(f"С поправкой на множественные проверки: q Pearson {pair['pearson_q']:.2g}; q Spearman {pair['spearman_q']:.2g}.")
    lines.append('Корреляция не устанавливает причину и не обещает будущего движения.')
    return '\n'.join(lines)


def errors(items, titles):
    return '\n'.join('• ' + item.get('title', titles.get(item['source'], item['source'])) + ': ' + item['error']
                     for item in items)
