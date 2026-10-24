"""Chart labels must not turn a baseline into a most-probable future rate."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from thesaurus.dollar_charts import near_term


class NearTermChartTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series([84., 85.], index=pd.to_datetime(['2026-10-08', '2026-10-09']))
        self.point = {'model': 'last', 'target': '2026-10-10', 'prediction': 85.,
                      'validation_n': 10, 'lower': 84., 'upper': 86.}

    def render(self, points):
        with patch.object(Figure, 'savefig', autospec=True) as save:
            part = near_term(self.series, points, Path('preview.png'))
        figure = save.call_args.args[0]
        return figure.axes[0], part

    def test_baseline_is_hollow_square_and_values_are_unchanged(self):
        points = [self.point, {**self.point, 'target': '2026-10-11'}]
        before = copy.deepcopy(points)
        axis, part = self.render(points)
        self.assertEqual(before, points)
        self.assertIn('направление не определено', axis.get_title())
        self.assertIn('Базовый ориентир · последний курс', axis.get_legend_handles_labels()[1])
        markers = [c for c in axis.collections if len(c.get_offsets()) and c.get_label().startswith('Базовый')]
        self.assertEqual(float(markers[0].get_offsets()[0, 1]), 85.)
        np.testing.assert_array_equal(markers[0].get_facecolors()[0, :3], [1, 1, 1])
        self.assertIn('не наиболее вероятный курс', part['caption'])
        self.assertIn('80-й процентиль', part['caption'])

    def test_selected_model_remains_distinct_from_baseline(self):
        axis, _ = self.render([{**self.point, 'model': 'lagged_ridge', 'prediction': 85.3}])
        self.assertNotIn('направление не определено', axis.get_title())
        self.assertIn('Точечная оценка модели', axis.get_legend_handles_labels()[1])
        self.assertAlmostEqual(float(axis.collections[0].get_offsets()[0, 1]), 85.3)

    def test_published_rate_never_inherits_stale_model_interval(self):
        axis, part = self.render([{**self.point, 'known_published': True, 'model': 'published_cbr'}])
        labels = axis.get_legend_handles_labels()[1]
        self.assertIn('Опубликованный курс ЦБ', labels)
        self.assertFalse(any('Диапазон' in label for label in labels))
        self.assertIn('фиолетовый ромб', part['caption'])
        self.assertNotIn('оранжевая', part['caption'])
        self.assertNotIn('квадрат', part['caption'])
        self.assertNotIn('Полоса', part['caption'])
        self.assertNotIn('оценка метода', axis.figure.axes[-1].texts[0].get_text())

    def test_unvalidated_model_never_shows_interval(self):
        axis, part = self.render([{**self.point, 'model': 'lagged_ridge', 'validation_n': 9}])
        labels = axis.get_legend_handles_labels()[1]
        self.assertIn('Оценка · недостаточно проверок', labels)
        self.assertFalse(any('Диапазон' in label for label in labels))
        self.assertIn('серый пустой ромб', part['caption'])
        self.assertNotIn('оранжевая', part['caption'])
        self.assertNotIn('фиолетовый', part['caption'])
        self.assertNotIn('80-й', part['caption'])

    def test_calendar_carry_is_not_a_model(self):
        axis, part = self.render([{**self.point, 'model': 'calendar_carry'}])
        labels = axis.get_legend_handles_labels()[1]
        self.assertIn('Действующий курс · календарь', labels)
        self.assertFalse(any('Диапазон' in label for label in labels))
        self.assertIn('серый крестик', part['caption'])
        self.assertNotIn('оценка модели', part['caption'])
        self.assertNotIn('80-й', part['caption'])

    def test_caption_for_user_example_only_explains_baseline_carry_and_actual_ranges(self):
        _, part = self.render([self.point, {**self.point, 'model': 'calendar_carry'}])
        caption = part['caption']
        self.assertIn('серый пустой квадрат', caption)
        self.assertIn('серый крестик', caption)
        self.assertIn('80-й процентиль', caption)
        for absent in ('фиолетовый', 'оранжевая', 'серый пустой ромб'):
            self.assertNotIn(absent, caption)
        self.assertEqual(caption.count('серый пустой квадрат'), 1)

    def test_mixed_caption_only_includes_drawn_validated_model_and_published_rate(self):
        _, part = self.render([{**self.point, 'model': 'lagged_ridge'},
                               {**self.point, 'model': 'published_cbr', 'known_published': True}])
        self.assertIn('оранжевая точка', part['caption'])
        self.assertIn('фиолетовый ромб', part['caption'])
        self.assertNotIn('серый', part['caption'])

    def test_asymmetric_bounds_are_not_recentered(self):
        axis, _ = self.render([{**self.point, 'lower': 82., 'upper': 86.}])
        interval = next(c for c in axis.collections if c.get_label().startswith('Диапазон'))
        np.testing.assert_array_equal(interval.get_segments()[0][:, 1], [82., 86.])

    def test_invalid_or_missing_bounds_are_not_drawn(self):
        for bounds in ({'lower': None}, {'upper': float('nan')}, {'lower': 87., 'upper': 84.}):
            with self.subTest(bounds=bounds):
                axis, _ = self.render([{**self.point, **bounds}])
                self.assertFalse(any('Диапазон' in label for label in axis.get_legend_handles_labels()[1]))
