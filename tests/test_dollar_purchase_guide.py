"""Decision explanations must not invent a directional or probability advantage."""
import copy
import json
import unittest

from thesaurus.dollar_purchase_guide import build, describe


class DollarPurchaseGuideTests(unittest.TestCase):
    def result(self, *, purchase=False, prediction=80., model='lagged_ridge'):
        return {'created_at': '2026-10-11T09:00:00+00:00',
                'config': {'mode': 'purchase' if purchase else 'overview', 'trip': '2026-11-30',
                           'quote': 86., 'amount': 1000., 'owned': 0., 'fee_pct': 0.,
                           'fee_fixed': 0., 'rub_yield': 0., 'risk': 10000.},
                'period': {'end': '2026-10-11', 'retrospective': False},
                'stats': {'last': 85., 'date': '2026-10-09', 'coverage': .97},
                'data_fresh': True, 'quote_fresh': True,
                'timing': [{'target': '2026-10-25', 'origin': '2026-10-09', 'days_ahead': 14,
                            'prediction': prediction, 'model': model, 'validation_mae': 1.,
                            'baseline_mae': 2., 'validation_n': 20, 'native_step': 10,
                            'saving': 5000., 'error_rub': 1000.}]}

    def test_overview_can_compare_without_bank_and_never_claim_optimum(self):
        result = self.result()
        result['config'].update(quote=None, trip=None)
        guide = build(result)
        self.assertEqual(guide['status'], 'consider_wait')
        self.assertFalse(guide['actionable'])
        self.assertEqual(guide['dates'], ['2026-10-25'])
        text = '\n'.join(describe(guide))
        self.assertIn('опорные даты', text)
        self.assertIn('Точный минимум между ними не найден', text)
        self.assertIn('Моя покупка', text)
        self.assertIn('типичная ошибка 1.00', text)

    def test_purchase_wait_requires_validated_exchange_and_expense_advantage(self):
        result = self.result(purchase=True)
        guide = build(result)
        self.assertEqual(guide['status'], 'consider_wait')
        self.assertTrue(guide['actionable'])
        result['timing'][0]['saving'] = 999.
        self.assertEqual(build(result)['status'], 'no_edge')

    def test_flat_baseline_with_positive_interest_is_not_price_signal(self):
        result = self.result(purchase=True, prediction=85., model='last')
        result['config']['rub_yield'] = 100.
        result['timing'][0].update(saving=10000., error_rub=100.)
        self.assertEqual(build(result)['status'], 'no_edge')

    def test_interest_alone_cannot_make_rising_fx_prediction_a_wait_signal(self):
        result = self.result(purchase=True, prediction=87.)
        result['timing'][0]['saving'] = 5000.
        self.assertEqual(build(result)['status'], 'no_edge')

    def test_rising_prediction_larger_than_error_has_now_argument(self):
        result = self.result(purchase=True, prediction=88.)
        result['timing'][0]['saving'] = -3000.
        self.assertEqual(build(result)['status'], 'consider_now')

    def test_small_prediction_or_unproven_improvement_does_not_create_edge(self):
        for updates in ({'prediction': 84.5}, {'validation_n': 9},
                        {'baseline_mae': 1.01}, {'baseline_mae': None},
                        {'validation_mae': float('nan')}):
            with self.subTest(updates=updates):
                result = self.result()
                result['timing'][0].update(updates)
                self.assertEqual(build(result)['status'], 'no_edge')

    def test_real_target_date_enforces_deadline_even_when_days_are_wrong(self):
        result = self.result(purchase=True)
        result['timing'][0].update(target='2026-12-01', days_ahead=1)
        guide = build(result)
        self.assertEqual(guide['status'], 'insufficient')
        self.assertEqual(guide['horizon_count'], 0)

    def test_no_recommendation_after_history_cutoff_or_for_future_history(self):
        for period in ({'end': '2026-10-10'}, {'end': '2026-10-11', 'retrospective': True},
                       {'end': '2026-10-12'}):
            with self.subTest(period=period):
                result = self.result(purchase=True)
                result['period'] = period
                guide = build(result)
                self.assertIn(guide['status'], ('retrospective', 'insufficient'))
                self.assertFalse(guide['actionable'])
                self.assertEqual(guide['evidence'], [])

    def test_past_target_and_missing_target_excluded(self):
        result = self.result()
        for target in ('2026-10-10', '2026-10-11', 'broken', None):
            with self.subTest(target=target):
                result['timing'][0]['target'] = target
                self.assertEqual(build(result)['status'], 'insufficient')

    def test_expired_trip_and_stale_inputs_block_purchase_guidance(self):
        for setting in ('trip', 'quote_fresh', 'data_fresh', 'coverage'):
            with self.subTest(setting=setting):
                result = self.result(purchase=True)
                if setting == 'trip':
                    result['config']['trip'] = '2026-10-10'
                elif setting == 'coverage':
                    result['stats']['coverage'] = .89
                else:
                    result[setting] = False
                guide = build(result)
                self.assertEqual(guide['status'], 'insufficient')
                self.assertFalse(guide['actionable'])

    def test_covered_sum_and_near_deadline_concern_execution(self):
        result = self.result(purchase=True)
        result['config']['owned'] = 1000.
        self.assertEqual(build(result)['status'], 'covered')
        result['config'].update(owned=0., trip='2026-10-14')
        self.assertEqual(build(result)['status'], 'near_deadline')

    def test_risk_scenario_is_explicit_and_does_not_have_assigned_probability(self):
        result = self.result(purchase=True)
        result['config']['risk'] = 5000.
        guide = build(result)
        self.assertEqual(guide['status'], 'staged')
        self.assertEqual(guide['stress_extra_rub'], 8600.)
        text = '\n'.join(describe(guide))
        self.assertIn('без оценки вероятности', text)
        self.assertIn('комиссий', text)

    def test_identical_calendar_horizons_are_grouped_without_false_extra_evidence(self):
        result = self.result()
        same = {**result['timing'][0], 'target': '2026-10-26', 'days_ahead': 15}
        result['timing'].append(same)
        guide = build(result)
        self.assertEqual(guide['horizon_count'], 1)
        self.assertEqual(guide['dates'], ['2026-10-25', '2026-10-26'])
        self.assertEqual(guide['selected']['validation_n'], 20)

    def test_already_published_course_is_not_validated_prediction_edge(self):
        result = self.result()
        result['timing'][0]['known_published'] = True
        self.assertEqual(build(result)['status'], 'insufficient')

    def test_combined_candidates_include_monthly_choice_and_not_stale_daily_list(self):
        result = self.result()
        result['purchase_candidates'] = [{**result['timing'][0], 'target': '2027-01-31',
                                          'native_step': 3, 'model': 'ridge'}]
        guide = build(result)
        self.assertEqual(guide['dates'], ['2027-01-31'])

    def test_pure_json_safe_output_does_not_mutate_inputs(self):
        result = self.result()
        before = copy.deepcopy(result)
        guide = build(result)
        json.dumps(guide, allow_nan=False)
        self.assertEqual(result, before)

    def test_moscow_date_sets_current_day_at_utc_evening(self):
        result = self.result()
        result['created_at'] = '2026-10-11T22:00:00+00:00'
        self.assertEqual(build(result)['status'], 'retrospective')


if __name__ == '__main__':
    unittest.main()
