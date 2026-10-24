"""Chosen forecast horizons need separate readable dates and an honest history cutoff."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from thesaurus.dollar_charts import near_term


class HorizonChartTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2025-01-01', '2026-10-11', freq='D')
        self.series = pd.Series(80.+np.sin(np.arange(len(dates))/30), index=dates)

    def point(self, days=30, **changes):
        return {'model': 'last', 'days_ahead': days,
                'target': (pd.Timestamp('2026-10-11')+pd.Timedelta(days=days)).date().isoformat(),
                'prediction': 85., 'validation_n': 10, 'lower': 83., 'upper': 88., **changes}

    def render(self, points, period=None):
        with patch.object(Figure, 'savefig', autospec=True) as save:
            part = near_term(self.series, points, Path('preview.png'), period=period)
        return save.call_args.args[0], part

    def test_selected_horizons_replace_fixed_tomorrow_title(self):
        figure, _ = self.render([self.point(1), self.point(3)])
        self.assertIn('1–3 дн.', figure.axes[0].get_title())
        self.assertNotIn('7 дней', figure.axes[0].get_title())
        figure, _ = self.render([self.point(30), self.point(90)])
        self.assertIn('30–90 дн.', figure._suptitle.get_text())

    def test_long_horizon_dates_have_distinct_rows_without_trajectory(self):
        points = [self.point(days) for days in (1, 2, 3, 7, 14, 30, 60, 90)]
        before = copy.deepcopy(points)
        figure, part = self.render(points)
        axis = figure.axes[1]
        self.assertEqual(len(axis.get_yticklabels()), 8)
        self.assertEqual(len({label.get_text() for label in axis.get_yticklabels()}), 8)
        self.assertEqual(axis.get_lines(), [])
        self.assertEqual(points, before)
        self.assertLess(len(part['caption']), 1024)
        self.assertIn('отдельная опорная дата', part['caption'])
        self.assertIn('серый пустой квадрат', part['caption'])
        self.assertNotIn('фиолетовый', part['caption'])
        self.assertNotIn('оранжевая', part['caption'])

    def test_same_baseline_values_remain_hollow_and_are_not_artificially_spread(self):
        figure, _ = self.render([self.point(30), self.point(60)])
        axis = figure.axes[1]
        markers = [c for c in axis.collections if c.get_label().startswith('Базовый')]
        self.assertEqual(float(markers[0].get_offsets()[0, 0]), 85.)
        np.testing.assert_array_equal(markers[0].get_facecolors()[0, :3], [1, 1, 1])
        self.assertIn('направление не определено', figure._suptitle.get_text())

    def test_explicit_history_limits_are_honored_with_full_selected_range(self):
        period = {'start': '2025-04-01', 'end': '2026-03-01', 'retrospective': True}
        figure, part = self.render([self.point(30)], period)
        history = figure.axes[0].lines[0].get_xdata()
        self.assertEqual(pd.Timestamp(history[0]), pd.Timestamp('2025-04-01'))
        self.assertEqual(pd.Timestamp(history[-1]), pd.Timestamp('2026-03-01'))
        self.assertGreater(len(history), 300)
        self.assertIn('ретроспектива', figure._suptitle.get_text())
        self.assertIn('Исторический расчёт', part['caption'])

    def test_short_horizon_also_respects_history_cutoff(self):
        figure, _ = self.render([self.point(3)], {'start': '2025-04-01', 'end': '2026-03-01', 'retrospective': True})
        history = figure.axes[0].lines[0].get_xdata()
        self.assertEqual(pd.Timestamp(history[0]), pd.Timestamp('2025-04-01'))
        self.assertEqual(pd.Timestamp(history[-1]), pd.Timestamp('2026-03-01'))
        self.assertIn('ретроспектива', figure._suptitle.get_text())

    def test_short_forecast_with_decades_of_history_uses_distinct_forecast_rows(self):
        self.series = pd.Series(85., index=pd.date_range('1992-07-01', '2026-10-11', freq='D'))
        figure, _ = self.render([self.point(1), self.point(2), self.point(3), self.point(7)],
                                {'start': '1992-07-01', 'end': '2026-10-11'})
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(len(figure.axes[1].get_yticklabels()), 4)
        self.assertIn('1–7 дн.', figure._suptitle.get_text())
        self.assertGreater(len(figure.axes[0].lines[0].get_xdata()), 12000)
        self.assertEqual(figure.axes[1].get_lines(), [])

    def test_default_plot_still_shows_30_days(self):
        figure, _ = self.render([self.point(30)])
        history = figure.axes[0].lines[0].get_xdata()
        self.assertEqual(pd.Timestamp(history[0]), pd.Timestamp('2026-09-11'))

    def test_ridge_published_and_unvalidated_points_have_distinct_semantics(self):
        points = [self.point(14, model='lagged_ridge'),
                  self.point(30, model='published_cbr', known_published=True),
                  self.point(60, model='ridge', validation_n=9)]
        figure, _ = self.render(points)
        axis = figure.axes[1]
        labels = axis.get_legend_handles_labels()[1]
        self.assertIn('Точечная оценка модели', labels)
        self.assertIn('Опубликованный курс ЦБ', labels)
        self.assertIn('Оценка · недостаточно проверок', labels)
        intervals = [c for c in axis.collections if hasattr(c, 'get_segments')]
        self.assertEqual(len(intervals), 1)
        np.testing.assert_array_equal(intervals[0].get_segments()[0][:, 0], [83., 88.])

    def test_invalid_bounds_are_not_presented_as_ranges(self):
        for bounds in ({'lower': None}, {'upper': float('nan')}, {'lower': 90., 'upper': 84.}):
            with self.subTest(bounds=bounds):
                figure, _ = self.render([self.point(30, **bounds)])
                labels = figure.axes[1].get_legend_handles_labels()[1]
                self.assertFalse(any('Диапазон' in label for label in labels))


if __name__ == '__main__':
    unittest.main()
