"""Scenario distributions are checked independently of point-model acceptance."""
import statistics
import unittest
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_dollar as fixtures

from thesaurus import dollar, dollar_long
from thesaurus import dollar_distribution as distribution


def series(n=420, growth=None):
    changes = np.full(n-1, growth) if growth is not None else np.resize(np.array([-.04, .01, .05, -.01]), n-1)
    return pd.Series(np.r_[100., 100*np.cumprod(1+changes)], index=pd.date_range('1992-07-31', periods=n, freq='ME'))


class DistributionTests(unittest.TestCase):
    def test_median_frequencies_and_smoothing_recomputed_from_explicit_ratios(self):
        prices = series(101)
        result = distribution.estimate(prices, 1, 'full')
        ratios = [float(prices.iloc[i]/prices.iloc[i-1]) for i in range(1, len(prices))]
        self.assertAlmostEqual(result['median'], prices.iloc[-1]*statistics.median(ratios))
        counts = [sum(r < .98 for r in ratios), sum(.98 <= r <= 1.02 for r in ratios), sum(r > 1.02 for r in ratios)]
        self.assertEqual(result['counts'], counts)
        for key, count in zip(('down', 'flat', 'up'), counts):
            self.assertAlmostEqual(result['probabilities'][key], (count+1)/(len(ratios)+3))
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.)
        self.assertLessEqual(result['lower'], result['median'])
        self.assertLessEqual(result['median'], result['upper'])

    def test_disjoint_intervals_and_missing_month_not_crossed(self):
        prices = series(100)
        ratios, periods = distribution.sample(prices, 6)
        self.assertEqual(len(ratios), 16)
        for recent, older in pairwise(periods):
            self.assertLessEqual(older['end'], recent['start'])
        prices.iloc[50] = np.nan
        missing, missing_periods = distribution.sample(prices, 6)
        self.assertLess(len(missing), len(ratios))
        absent = prices.index[50].date().isoformat()
        self.assertTrue(all(not p['start'] <= absent <= p['end'] for p in missing_periods))

    def test_small_sample_is_unavailable_not_simulated_into_thousands(self):
        result = distribution.forecast(series(), 60, 'full')
        self.assertEqual(result['status'], 'insufficient')
        self.assertNotIn('probabilities', result)
        self.assertNotIn('median', result)
        with self.assertRaises(ValueError):
            distribution.forecast(series(), 0, 'auto')

    def test_zero_and_hundred_percent_claims_prevented_in_one_sided_history(self):
        result = distribution.estimate(series(150, growth=.1), 1, 'full')
        self.assertTrue(all(0 < p < 1 for p in result['probabilities'].values()))
        shown = distribution.display_shares({'down': 13/32, 'flat': 5/32, 'up': 14/32})
        self.assertEqual(sum(int(s.rstrip('%')) for s in shown.values()), 100)
        extremes = distribution.display_shares({'down': .0001, 'flat': .0001, 'up': .9998})
        self.assertEqual(extremes['down'], '<1%')
        self.assertEqual(extremes['up'], '>99%')

    def test_estimation_and_validation_do_not_read_future_prefix(self):
        prices = series()
        before = distribution.forecast(prices.iloc[:300], 6, 'auto')
        prices.iloc[300:] = 999999.
        after = distribution.forecast(prices.iloc[:300], 6, 'auto')
        self.assertEqual(before, after)
        for i, fold in enumerate(before['validation']['folds']):
            self.assertLessEqual(fold['training_end'], fold['origin'])
            self.assertLess(fold['origin'], fold['target'])
            if i:
                self.assertLessEqual(before['validation']['folds'][i-1]['target'], fold['origin'])

    def test_auto_uses_sample_rule_in_every_past_fold_and_recent_is_strict(self):
        prices = series()
        result = distribution.forecast(prices, 6, 'auto')
        self.assertEqual(result['window'], 'ten_years')
        self.assertTrue(all(fold['window'] == 'ten_years' for fold in result['validation']['folds']))
        prices.iloc[-30] = np.nan
        self.assertEqual(distribution.forecast(prices, 12, 'auto')['window'], 'full')
        recent = distribution.forecast(prices, 12, 'recent')
        self.assertEqual(recent['status'], 'insufficient')

    def test_pinball_score_against_hand_calculation_and_quality_gate(self):
        self.assertAlmostEqual(distribution.pinball(1.2, [.9, 1., 1.1]), (.03+.1+.09)/3)
        self.assertAlmostEqual(distribution.pinball(1.2, [1., 1., 1.]), .1)
        result = distribution.forecast(series(420, growth=.01), 12, 'full')
        self.assertEqual(result['status'], 'evaluated')
        self.assertGreaterEqual(result['validation']['n'], 10)
        self.assertLess(result['validation']['brier'], result['validation']['uniform_brier'])
        self.assertLess(result['validation']['median_mae'], result['validation']['baseline_mae'])
        weak = distribution.forecast(series(150), 12, 'full')
        self.assertEqual(weak['status'], 'research')
        self.assertIn('не уверенность', '\n'.join(distribution.describe(weak)))

    def test_chart_keeps_scenarios_distinct_from_confirmed_forecasts(self):
        prices = series()
        d = distribution.forecast(prices, 12, 'full')
        points = [{'origin': prices.index[-1].date().isoformat(), 'target': '2028-06-30', 'horizon': 12,
                   'baseline': float(prices.iloc[-1]), 'distribution': d}]
        with patch('matplotlib.figure.Figure.savefig', autospec=True) as save:
            part = distribution.chart(prices, points, 'unused.png')
        axes = save.call_args.args[0].axes
        self.assertIn('не подтверждённая', axes[0].get_title())
        self.assertIn('не гарантии', axes[1].get_title())
        self.assertAlmostEqual(sum(bar.get_width() for bar in axes[1].patches), 100.)
        self.assertIn('не проверенная', part['caption'])

    def test_future_fact_scores_frozen_probabilities_without_retraining(self):
        frozen = distribution.forecast(series(), 6, 'full')
        actual = frozen['base']*1.1
        with patch('thesaurus.dollar_distribution.estimate', side_effect=AssertionError('Must not retrain')):
            result = distribution.score_actual(frozen, actual)
        self.assertEqual(result['outcome'], 'up')
        p = frozen['probabilities']
        self.assertAlmostEqual(result['brier'], p['down']**2+p['flat']**2+(1-p['up'])**2)
        self.assertEqual(result['median_absolute_error'], abs(actual-frozen['median']))
        self.assertIsNone(distribution.score_actual({'status': 'insufficient'}, actual))


class DistributionBotTests(unittest.TestCase):
    setUp = fixtures.DollarBotTests.setUp

    def test_distribution_toggle_is_personal_and_model_stays_separate(self):
        c = self.bot.dollar_controls
        c.callback(self.client, 2, 'usd:distribution')
        self.assertFalse(c.config(2)['distribution'])
        self.assertTrue(c.config(3)['distribution'])
        self.assertEqual(c.config(2)['long_method'], 'auto')
        with self.assertRaises(PermissionError):
            c.callback(self.client, 999, 'usd:distribution')

    def test_scenario_is_saved_separately_and_can_be_disabled(self):
        prices = series()
        # Contract of the long forecast's monthly input; no live source involved.
        rows = [{'date': stamp.date().isoformat(), 'value': value} for stamp, value in prices.items()]
        with patch('thesaurus.dollar_long.monthly_closes', return_value=prices):
            enabled = dollar_long.forecast(rows, {}, now=fixtures.NOW, today=pd.Timestamp(fixtures.NOW).date(),
                                          months=12, history='full', method='last')
            disabled = dollar_long.forecast(rows, {}, now=fixtures.NOW, today=pd.Timestamp(fixtures.NOW).date(),
                                           months=12, history='full', method='last', distribution=False)
        self.assertTrue(all(p['model'] == 'last' for p in enabled))
        self.assertTrue(all(p['distribution'] is not None for p in enabled))
        self.assertTrue(all(p['distribution'] is None for p in disabled))
        self.assertEqual([p['prediction'] for p in enabled], [p['prediction'] for p in disabled])
        with self.assertRaises(ValueError):
            dollar.build(self.store, {'distribution': 'yes'}, now=fixtures.NOW)


if __name__ == '__main__':
    unittest.main()
