"""Distinguish lack of directional signal, calendar carry and actual publications."""
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from thesaurus import dollar, dollar_models
from thesaurus.market_analysis import grid


class DollarExplanationTests(unittest.TestCase):
    def result(self, model='last'):
        point = {'model': model, 'origin': '2026-10-09', 'target': '2026-10-10', 'days_ahead': 1,
                 'prediction': 85.4173, 'baseline': 85.4173, 'validation_mae': .47,
                 'validation_n': 10, 'native_step': 1, 'lower': 84.5917, 'upper': 86.2429,
                 'features': []}
        return {'stats': {'last': 85.4173, 'date': '2026-10-09', 'coverage': .93,
                          'first': '1992-07-01', 'n': 7937, 'changes': {'30': -1.22},
                          'percentile_365': 95}, 'config': dict(dollar.DEFAULTS),
                'timing': [point], 'context': [], 'days': None}

    def test_flat_baseline_is_explicit_and_uncertainty_visible(self):
        text = dollar.summary(self.result())
        self.assertIn('одинаковые числа получаются по его формуле', text)
        self.assertIn('направление не определено', text)
        self.assertIn('84.59–86.24', text)
        self.assertIn('10 проверок — небольшая выборка', text)
        self.assertIn('не наиболее вероятный курс', text)
        self.assertIn('80-й процентиль прошлых абсолютных ошибок', text)

    def test_basic_engine_does_not_claim_external_models_were_tested(self):
        result = self.result()
        result['config']['forecast_engine'] = 'basic'
        text = dollar.summary(result)
        self.assertIn('В настройках выбран расчёт только по истории USD', text)
        self.assertNotIn('Дополнительные показатели не дали', text)

    def test_small_ridge_change_is_not_described_as_flat_baseline(self):
        result = self.result('lagged_ridge')
        result['timing'][0]['prediction'] += .2
        text = dollar.summary(result)
        self.assertIn('изменения на ближайшие дни меньше типичной ошибки', text)
        self.assertNotIn('одинаковые числа получаются', text)

    def test_known_published_course_does_not_get_model_uncertainty(self):
        result = self.result('published_cbr')
        result['timing'][0].update(known_published=True, validation_mae=None, validation_n=0,
                                  lower=None, upper=None)
        text = dollar.summary(result)
        self.assertIn('уже опубликован', text)
        self.assertNotIn('диапазон 80%', text)
        self.assertNotIn('направление не определено', text)

    def test_friday_forecast_uses_same_effective_course_saturday_to_monday(self):
        dates = grid('2025-01-01', '2026-10-09', {'frequency': 'daily', 'calendar': 'cbr_fx'})
        rows = [{'date': day.date().isoformat(), 'value': 80.+i/100} for i, day in enumerate(dates)]
        def baseline(series, *args):
            return {'prediction': float(series.iloc[-1]), 'model': 'last', 'features': []}
        with patch('thesaurus.dollar_models.select', side_effect=baseline):
            points = dollar_models.forecast(rows, {'frequency': 'daily', 'calendar': 'cbr_fx'},
                                           now='2026-10-09T09:00:00+00:00', today=date(2026, 10, 9))
        self.assertEqual([p['native_step'] for p in points], [1, 1, 1, 5])
        result = self.result()
        result['timing'] = points
        text = dollar.summary(result)
        self.assertIn('2026-10-10, 2026-10-11, 2026-10-12 соответствуют одному горизонту', text)

    def test_rejected_candidate_keeps_explanation_without_changing_baseline(self):
        series = pd.Series(80+np.arange(400)/100, index=pd.bdate_range(end='2026-10-09', periods=400))
        x = dollar_models.design(series, pd.DataFrame(index=series.index))
        def poor(prefix, *args):
            return {'prediction': float(prefix.iloc[-1]+10)}
        with patch('thesaurus.dollar_models.fit_predict', side_effect=poor):
            selected = dollar_models.select(series, x, 1, 'full')
        self.assertEqual(selected['model'], 'last')
        self.assertEqual(selected['prediction'], series.iloc[-1])
        self.assertGreater(selected['inner_candidate_mae'], selected['inner_baseline_mae'])
        self.assertEqual(selected['selection_n'], 8)
        self.assertIn('внутренней', selected['selection_reason'])


if __name__ == '__main__':
    unittest.main()
