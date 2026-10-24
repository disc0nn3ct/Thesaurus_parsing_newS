"""Offline chart controls, display windows, permissions and forecast boundaries."""
import json
import shutil
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import market, market_store
from thesaurus.charts import (
    bounds,
    chart_series,
    focus_panel,
    plot_item,
    run_chart_job,
    validate_options,
)
from thesaurus.config import Settings
from thesaurus.service import Worker
from thesaurus.store import Store, utcnow
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_charts import ChartControls


class ChartTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('charts-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.settings = Settings(root=self.root)
        self.reply = Mock()
        self.controls = ChartControls(self.store, self.reply)
        self.client = Mock()
        self.bot = BotControls(self.store)
        self.today = date.fromisoformat(utcnow()[:10])
        self.last = self.today - timedelta(days=1)
        meta = {'id': 'test.rate', 'source': 'cbr', 'dataset': 'test', 'title': 'Тестовая ставка',
                'kind': 'series', 'frequency': 'daily', 'calendar': 'calendar', 'unit': '%',
                'aggregation': 'last', 'transform': 'difference', 'url': 'https://cbr.ru/hd_base/keyrate/'}
        market_store.register(self.store, [meta])
        market.enable(self.store, 'test.rate', True)
        market_store.save_rows(self.store, 'test.rate', [
            {'date': (self.last-timedelta(days=89-i)).isoformat(), 'value': 10+i*.01, 'retrieved_at': utcnow()}
            for i in range(90)])

    def event(self, text, actor=2, private=True):
        return SimpleNamespace(text=text, id=11, from_user=SimpleNamespace(id=actor),
            chat=SimpleNamespace(id=actor, type='private' if private else 'group'))

    def click(self, data, actor=2):
        event = self.event('', actor)
        query = SimpleNamespace(id=uuid4().hex, data=data, from_user=event.from_user, message=event, answer=Mock())
        self.bot.handle_callback(self.client, query)
        return query

    def test_command_and_main_button(self):
        self.bot.handle_message(self.client, self.event('/chart'))
        markup = self.client.send_message.call_args.kwargs['reply_markup']
        data = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertIn('ch:menu:ruonia', data)
        self.assertIn('menu:home', data)

    def test_options_are_personal_and_persist(self):
        self.controls.callback(self.client, 2, 'ch:period:market:year', 1)
        self.controls.callback(self.client, 2, 'ch:horizon:market:2', 2)
        fresh = ChartControls(Store(self.store.path), self.reply)
        self.assertEqual(fresh.options(2, 'market')['period'], 'year')
        self.assertEqual(fresh.options(2, 'market')['horizon'], 2)
        self.assertEqual(fresh.options(3, 'market')['period'], 'quarter')

    def test_custom_input_invalid_keeps_prompt_and_cancel(self):
        self.controls.callback(self.client, 2, 'ch:custom:market', 1)
        self.bot.handle_message(self.client, self.event('2025-04-01 2025-01-01'))
        self.assertEqual(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'], 'chart:market')
        self.bot.handle_message(self.client, self.event('2025-01-01 2025-04-01'))
        self.assertEqual(self.controls.options(2, 'market')['date_to'], '2025-04-01')
        self.controls.callback(self.client, 2, 'ch:custom:market', 2)
        self.bot.handle_message(self.client, self.event('/cancel'))
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])

    def test_future_and_bad_dates_rejected(self):
        for start, end in [('2025-01-01', '2099-01-01'), ('bad', '2025-01-01'), (None, '2025-01-01')]:
            with self.assertRaises(ValueError):
                validate_options({'period': 'custom', 'horizon': 1, 'date_from': start, 'date_to': end})
        for horizon in [-1, 5, True]:
            with self.assertRaises(ValueError):
                validate_options({'period': 'all', 'horizon': horizon})

    def test_presets_count_calendar_days_and_all_is_unlimited(self):
        self.assertEqual(bounds({'period': 'month', 'horizon': 0}, '2025-03-30'), ('2025-03-01', '2025-03-30'))
        self.assertEqual(bounds({'period': 'all', 'horizon': 0}, '2025-03-30'), (None, '2025-03-30'))

    def test_plot_filter_does_not_shorten_training(self):
        with patch('thesaurus.charts.market.save_forecasts') as forecast:
            series, start, end = chart_series(self.store, self.settings, 2, 'market', 'test.rate', {'period': 'month', 'horizon': 4})
        self.assertEqual(len(forecast.call_args.args[2]), 90)
        self.assertEqual(len(series[0]['history']), 30)
        self.assertEqual(end, self.last.isoformat())
        self.assertEqual(start, (self.last-timedelta(days=29)).isoformat())

    def seed_forecasts(self):
        for horizon in range(1, 5):
            self.store.execute('INSERT INTO market_forecasts(series_id,origin,target,horizon,prediction,baseline,model,created_at,inputs) VALUES(?,?,?,?,?,?,?,?,?)',
                ('test.rate', self.last.isoformat(), (self.last+timedelta(weeks=horizon)).isoformat(), horizon, 12, 11, 'last', utcnow(), '{}'))

    def test_historical_chart_hides_future_trained_predictions(self):
        self.seed_forecasts()
        options = {'period': 'custom', 'date_from': (self.last-timedelta(days=40)).isoformat(),
                   'date_to': (self.last-timedelta(days=20)).isoformat(), 'horizon': 4}
        with patch('thesaurus.charts.market.save_forecasts'):
            result, _, _ = chart_series(self.store, self.settings, 2, 'market', None, options)
        self.assertEqual(len(result[0]['history']), 21)
        self.assertEqual(result[0]['forecasts'], [])

    def test_selected_forecast_horizon_and_off(self):
        self.seed_forecasts()
        with patch('thesaurus.charts.market.save_forecasts') as forecast:
            result, _, _ = chart_series(self.store, self.settings, 2, 'market', None, {'period': 'month', 'horizon': 2})
            self.assertEqual([r['horizon'] for r in result[0]['forecasts']], [1, 2])
            forecast.reset_mock()
            result, _, _ = chart_series(self.store, self.settings, 2, 'market', None, {'period': 'all', 'horizon': 0})
            self.assertEqual(result[0]['forecasts'], [])
            forecast.assert_not_called()

    def test_missing_calendar_day_remains_a_gap(self):
        missing = (self.last-timedelta(days=12)).isoformat()
        self.store.execute('DELETE FROM market_observations WHERE series_id=? AND date=?', ('test.rate', missing))
        self.store.execute('DELETE FROM market_revisions WHERE series_id=? AND date=?', ('test.rate', missing))
        result, _, _ = chart_series(self.store, self.settings, 2, 'market', None, {'period': 'month', 'horizon': 0})
        self.assertEqual(len(result[0]['history']), 30)
        self.assertEqual(result[0]['history'].isna().sum(), 1)

    def test_button_tokens_are_personal_short_and_reusable(self):
        self.controls.menu(self.client, 2, 'market', view='indicator')
        markup = self.reply.call_args.args[3]
        data = [b.callback_data for row in markup.inline_keyboard for b in row]
        token = next(d for d in data if d.startswith('ch:select:market:') and not d.endswith(':all'))
        self.assertTrue(all(len(d.encode()) <= 64 for d in data))
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 3, token, 1)
        self.controls.callback(self.client, 2, token, 2)
        self.assertEqual(self.controls.options(2, 'market')['identity'], 'test.rate')

    def test_private_chat_and_access_rechecked(self):
        self.bot.handle_message(self.client, self.event('/chart', private=False))
        self.client.send_message.assert_not_called()
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        query = self.click('ch:build:market')
        self.assertTrue(query.answer.call_args.kwargs['show_alert'])
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_queue_freezes_options_and_offline_worker_builds_png(self):
        self.controls.callback(self.client, 2, 'ch:horizon:market:0', 1)
        self.controls.callback(self.client, 2, 'ch:build:market', 'build')
        self.controls.callback(self.client, 2, 'ch:period:market:year', 2)
        payload = json.loads(self.store.one('SELECT payload FROM jobs')['payload'])
        self.assertEqual(payload['options']['period'], 'quarter')
        providers = Mock()
        worker = Worker(self.store, self.settings, providers=providers)
        with patch('thesaurus.market_sources.Transport.get', side_effect=AssertionError('No network')):
            self.assertTrue(worker.process_next())
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'done')
        report = self.store.one('SELECT * FROM reports')
        parts = json.loads(report['parts'])
        self.assertEqual(len(parts), 1)
        self.assertTrue(Path(parts[0]['path']).is_file())
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), 2)
        self.assertEqual(providers.mock_calls, [])
        self.store.execute("UPDATE jobs SET state='pending'")
        worker.process_next()
        self.assertEqual(len(self.store.query('SELECT * FROM reports')), 1)
        self.assertEqual(len(self.store.query('SELECT * FROM deliveries')), 2)

    def test_ruonia_period_preserves_negative_values_missing_and_full_training(self):
        self.store.save_observations([
            {'source': 'ruonia', 'metric': 'RUONIA', 'date': (self.last-timedelta(days=59-i)).isoformat(),
             'value': None if i == 50 else -1+i*.01, 'unit': '%', 'frequency': 'daily', 'region_id': 225}
            for i in range(60)])
        with patch('thesaurus.charts.ReportBuilder._save_forecasts') as forecast:
            result, _, _ = chart_series(self.store, self.settings, 2, 'ruonia', 'RUONIA', {'period': 'month', 'horizon': 2})
        self.assertEqual(len(forecast.call_args.args[0]), 60)
        self.assertEqual(len(result[0]['history']), 30)
        self.assertEqual(result[0]['history'].isna().sum(), 1)
        self.assertTrue((result[0]['history'].dropna() < 0).all())

    def test_restore_does_not_make_revoked_chart_deliverable(self):
        self.controls.callback(self.client, 2, 'ch:build:market', 'build')
        self.store.execute('UPDATE users SET enabled=0 WHERE user_id=2')
        Worker(self.store, self.settings).process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')
        self.assertEqual(self.store.query('SELECT * FROM reports'), [])

    def test_chart_buttons_are_paginated(self):
        base = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        for i in range(9):
            meta = {**base, 'id': f'extra.{i}', 'title': f'Дополнительный {i}'}
            market_store.register(self.store, [meta])
            market.enable(self.store, meta['id'], True)
        self.controls.menu(self.client, 2, 'market', view='indicator')
        markup = self.reply.call_args.args[3]
        callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
        self.assertIn('ch:page:market:1', callbacks)
        self.controls.callback(self.client, 2, 'ch:page:market:1', 'page')
        markup = self.reply.call_args.args[3]
        callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
        self.assertIn('ch:page:market:0', callbacks)

    def test_disabled_indicator_blocks_queued_chart(self):
        self.controls.callback(self.client, 2, 'ch:build:market', 'build')
        market.enable(self.store, 'test.rate', False)
        Worker(self.store, self.settings).process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'failed')
        self.assertEqual(self.store.query('SELECT * FROM deliveries'), [])

    def test_export_uses_selected_dates_and_series(self):
        self.controls.callback(self.client, 2, 'ch:custom:market', 1)
        self.controls.input(self.client, 2, '2025-01-01 2025-04-01')
        self.controls.callback(self.client, 2, 'ch:export:market:xlsx', 'export')
        job = self.store.one('SELECT * FROM jobs')
        payload = json.loads(job['payload'])
        self.assertEqual(job['kind'], 'market_export')
        self.assertEqual(payload['date_from'], '2025-01-01')
        self.assertEqual(payload['date_to'], '2025-04-01')
        self.assertEqual(payload['selection'], ['test.rate'])

    def test_empty_period_is_explained_without_fake_zero(self):
        self.controls.callback(self.client, 2, 'ch:custom:market', 1)
        self.controls.input(self.client, 2, '2000-01-01 2000-02-01')
        self.controls.callback(self.client, 2, 'ch:horizon:market:0', 2)
        self.controls.callback(self.client, 2, 'ch:build:market', 'build')
        Worker(self.store, self.settings).process_next()
        report = self.store.one('SELECT * FROM reports')
        self.assertEqual(json.loads(report['parts']), [])
        self.assertIn('нет данных в выбранном периоде', report['text'])

    def test_unloaded_row_does_not_cancel_other_market_charts(self):
        base = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        missing = {**base, 'id': 'empty.monthly', 'title': 'Пустой месячный ряд', 'frequency': 'monthly'}
        market_store.register(self.store, [missing])
        market.enable(self.store, missing['id'], True)
        self.controls.callback(self.client, 2, 'ch:horizon:market:0', 'off')
        self.controls.callback(self.client, 2, 'ch:build:market', 'mixed')
        with patch('thesaurus.market_sources.Transport.get', side_effect=AssertionError('No network')):
            Worker(self.store, self.settings).process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'done')
        report = self.store.one('SELECT * FROM reports')
        self.assertEqual(len(json.loads(report['parts'])), 1)
        self.assertIn('Пустой месячный ряд: нет данных', report['text'])

    def test_empty_rows_do_not_use_the_twelve_chart_slots_or_forecast_slots(self):
        base = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        for number in range(13):
            missing = {**base, 'id': f'empty.{number:02}', 'title': f'Не загружен {number}'}
            market_store.register(self.store, [missing])
            market.enable(self.store, missing['id'], True)
        self.controls.callback(self.client, 2, 'ch:build:market', 'many-empty')
        with patch('thesaurus.charts.market.save_forecasts', return_value=[]) as forecast:
            Worker(self.store, self.settings).process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'done')
        self.assertEqual(forecast.call_count, 1)
        self.assertEqual(forecast.call_args.args[1]['id'], 'test.rate')
        report = self.store.one('SELECT * FROM reports')
        self.assertEqual(len(json.loads(report['parts'])), 1)
        self.assertIn('Ещё 7 рядов без данных', report['text'])
        self.assertNotIn('Показаны первые 12 рядов', report['text'])

    def test_only_empty_market_series_returns_explanation(self):
        self.store.execute("DELETE FROM market_observations WHERE series_id='test.rate'")
        self.store.execute("DELETE FROM market_revisions WHERE series_id='test.rate'")
        self.controls.callback(self.client, 2, 'ch:build:market', 'empty')
        with patch('thesaurus.charts.market.save_forecasts') as forecast:
            Worker(self.store, self.settings).process_next()
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'done')
        report = self.store.one('SELECT * FROM reports')
        self.assertEqual(json.loads(report['parts']), [])
        self.assertIn('нет данных в выбранном периоде', report['text'])
        forecast.assert_not_called()

    def test_all_charts_have_private_pages_with_frozen_order_and_page_forecasts(self):
        base = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        observations = market_store.history(self.store, 'test.rate')
        for i in range(12):
            meta = {**base, 'id': f'page.{i}', 'title': f'Страница {i:02}'}
            market_store.register(self.store, [meta])
            market.enable(self.store, meta['id'], True)
            market_store.save_rows(self.store, meta['id'], observations)
        self.controls.callback(self.client, 2, 'ch:build:market', 'first-page')
        with patch('thesaurus.charts.market.save_forecasts', return_value=[]) as forecast:
            Worker(self.store, self.settings).process_next()
        self.assertEqual(forecast.call_count, 12)
        report = self.store.one('SELECT * FROM reports')
        quality = json.loads(report['quality'])
        self.assertEqual(quality['chart_total'], 13)
        self.assertEqual(quality['chart_pages'], 2)
        self.assertEqual(len(json.loads(report['parts'])), 12)
        navigation = json.loads(self.store.query('SELECT payload FROM deliveries ORDER BY id')[-1]['payload'])
        callback = navigation['buttons'][0][0]['callback_data']
        self.assertEqual(callback, f'ch:batch:{report["id"]}:1')
        self.assertLessEqual(len(callback.encode()), 64)
        from thesaurus.delivery import _send_part
        sender = Mock()
        sender.send_message.return_value.id = 100
        self.assertEqual(_send_part(sender, '2', navigation), 100)
        markup = sender.send_message.call_args.kwargs['reply_markup']
        self.assertEqual(markup.inline_keyboard[0][0].callback_data, callback)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 3, callback, 'intruder')
        with self.assertRaises(ValueError):
            self.controls.queue_page(self.client, 2, report['id'], 2, 'bad-page')
        with self.assertRaises(ValueError):
            self.controls.queue_page(self.client, 2, report['id'], -1, 'negative-page')
        # Changing the menu must not change the saved request or ordering.
        self.controls.callback(self.client, 2, 'ch:period:market:month', 'menu-edit')
        self.controls.callback(self.client, 2, callback, 'next-page')
        next_job = self.store.one('SELECT * FROM jobs ORDER BY id DESC')
        request = json.loads(next_job['payload'])
        self.assertEqual(request['options']['period'], 'quarter')
        self.assertEqual(request['order'], quality['chart_request']['order'])
        removed = request['order'][0]
        self.store.execute('DELETE FROM market_observations WHERE series_id=?', (removed,))
        self.store.execute('DELETE FROM market_revisions WHERE series_id=?', (removed,))
        with patch('thesaurus.charts.market.save_forecasts', return_value=[]) as forecast:
            Worker(self.store, self.settings).process_next()
        self.assertEqual(forecast.call_count, 1)
        self.assertEqual(forecast.call_args.args[1]['id'], request['order'][-1])
        last = self.store.one('SELECT * FROM reports WHERE id=?', (request['report_id'],))
        first_titles = {part['caption'] for part in json.loads(report['parts'])}
        next_titles = {part['caption'] for part in json.loads(last['parts'])}
        self.assertEqual(len(next_titles), 1)
        self.assertFalse(first_titles.intersection(next_titles))
        market.enable(self.store, request['order'][-1], False)
        with self.assertRaises(PermissionError):
            self.controls.callback(self.client, 2, callback, 'revoked-series')

    def test_sparse_observations_have_markers_without_joining_missing_dates(self):
        import pandas as pd
        from matplotlib.figure import Figure
        curve = pd.Series([4., None, None, 5.], index=pd.date_range('2025-01-01', periods=4))
        axes = Figure().subplots()
        plot_item(axes, {'history': curve, 'forecasts': [], 'title': 'Редкий ряд'})
        line = axes.lines[0]
        self.assertEqual(line.get_marker(), 'o')
        self.assertEqual(int(pd.notna(line.get_ydata()).sum()), 2)
        self.assertTrue(pd.isna(line.get_ydata()[1]))

    def test_unvalidated_tail_is_a_scenario_but_calibrated_points_keep_their_band(self):
        import pandas as pd
        from matplotlib.figure import Figure
        curve = pd.Series([10., 11.], index=pd.date_range('2025-01-01', periods=2))
        points = [{'target': '2025-01-09', 'prediction': 12., 'lower': 10., 'upper': 14., 'validation_n': 24},
                  {'target': '2025-07-02', 'prediction': 11., 'lower': None, 'upper': None, 'validation_n': 0}]
        axes = Figure().subplots()
        plot_item(axes, {'history': curve, 'forecasts': points, 'target_key': 'target', 'title': 'Тест'})
        labels = axes.get_legend_handles_labels()[1]
        self.assertIn('Прогноз', labels)
        self.assertIn('Сценарий без достаточной проверки', labels)
        self.assertIn('Диапазон 80%', labels)

    def test_administrative_scenario_is_labelled_and_not_a_dense_forecast_band(self):
        import pandas as pd
        from matplotlib.figure import Figure
        curve = pd.Series([14., 14.], index=pd.date_range('2026-07-01', periods=2, freq='MS'))
        points = [{'target': stamp.date().isoformat(), 'prediction': 14., 'lower': 12., 'upper': 16.,
                   'validation_n': 24, 'note': 'Сценарий сохранения административного значения'}
                  for stamp in pd.date_range('2026-11-01', periods=22, freq='MS')]
        axes = Figure().subplots()
        plot_item(axes, {'history': curve, 'forecasts': points, 'target_key': 'target', 'title': 'Ставка', 'unit': '%'})
        labels = axes.get_legend_handles_labels()[1]
        self.assertIn('Условие: без изменений (не прогноз)', labels)
        self.assertNotIn('Прогноз', labels)
        self.assertNotIn('Диапазон 80%', labels)
        self.assertEqual(len(axes.lines[-1].get_xdata()), 2)
        self.assertEqual(axes.lines[-1].get_marker(), 'None')
        self.assertIn('14 %', axes.texts[-1].get_text())
        single = Figure().subplots()
        plot_item(single, {'history': curve, 'forecasts': points[:1], 'target_key': 'target', 'title': 'Ставка', 'unit': '%'})
        self.assertNotIn(single.lines[-1].get_marker(), {'None', '', ' '})

    def test_recent_panel_preserves_full_selected_history(self):
        import pandas as pd
        curve = pd.Series(range(156), index=pd.date_range('2013-09-01', periods=156, freq='MS'))
        item = {'history': curve, 'forecasts': [{'target': '2028-08-01'}]}
        focused = focus_panel([item])
        self.assertEqual(len(item['history']), 156)
        self.assertEqual(len(focused[0]['history']), 25)
        self.assertEqual(focused[0]['forecasts'], item['forecasts'])
        self.assertEqual(focus_panel([{**item, 'forecasts': []}]), [])
        self.assertEqual(focus_panel([{**item, 'history': curve.tail(12)}]), [])

    def test_selected_administrative_indicator_explains_scenario_in_controls(self):
        meta = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        market_store.register(self.store, [{**meta, 'administrative': True}])
        self.store.set_setting('chart:2:market', {'identity': 'test.rate', 'period': 'all', 'horizon': 4})
        self.controls.menu(self.client, 2, 'market')
        text = self.reply.call_args.args[2]
        self.assertIn('Условный сценарий:', text)
        self.assertIn('а не будущее решение ЦБ', text)

    def test_long_chart_titles_and_date_labels_fit_the_saved_png(self):
        import pandas as pd
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        curve = pd.Series([14.] * 157, index=pd.date_range('2013-09-01', periods=157, freq='MS'))
        item = {'identity': 'test.rate', 'title': 'Ключевая ставка Банка России', 'unit': '%', 'frequency': 'monthly',
                'administrative': True, 'history': curve, 'target_key': 'target',
                'forecasts': [{'target': '2028-09-01', 'prediction': 14., 'lower': None, 'upper': None,
                               'note': 'Сценарий сохранения административного значения', 'validation_n': 0}]}
        payload = {'actor_id': 2, 'section': 'market', 'report_id': uuid4().hex, 'identity': 'test.rate',
                   'options': {'period': 'all', 'horizon': 4, 'forecast': {'value': 24, 'unit': 'months'}}}
        draw = FigureCanvasAgg.draw
        checked = []
        def inspect_canvas(canvas):
            draw(canvas)
            renderer = canvas.get_renderer()
            for ax in canvas.figure.axes:
                for label in (ax.title, ax.xaxis.label):
                    box = label.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, 0)
                    self.assertGreaterEqual(box.y0, 0)
                    self.assertLessEqual(box.x1, canvas.figure.bbox.width)
                    self.assertLessEqual(box.y1, canvas.figure.bbox.height)
            checked.append(len(canvas.figure.axes))
        with patch('thesaurus.charts.chart_series', return_value=([item], None, '2026-09-01')), \
                patch.object(FigureCanvasAgg, 'draw', inspect_canvas):
            run_chart_job(self.store, self.settings, {'id': 998}, payload, dry_run=True)
        self.assertTrue(checked)
        self.assertEqual(set(checked), {2})
        report = self.store.one('SELECT full_text FROM reports WHERE id=?', (payload['report_id'],))
        self.assertIn('вся сохранённая история; последняя дата данных — 2026-09-01', report['full_text'])
        self.assertIn('Это не предсказание решения ЦБ', report['full_text'])

    def test_single_observation_does_not_expand_date_axis_over_years(self):
        from matplotlib.figure import Figure
        self.store.execute("DELETE FROM market_observations WHERE series_id='test.rate' AND date<>?", (self.last.isoformat(),))
        self.store.execute("DELETE FROM market_revisions WHERE series_id='test.rate' AND date<>?", (self.last.isoformat(),))
        payload = {'actor_id': 2, 'section': 'market', 'report_id': uuid4().hex, 'identity': 'test.rate',
                   'options': {'period': 'all', 'horizon': 0}, 'selection': ['test.rate']}
        spans = []
        original = Figure.savefig
        def inspect_figure(figure, *args, **kwargs):
            left, right = figure.axes[0].get_xlim()
            spans.append(right-left)
            return original(figure, *args, **kwargs)
        with patch.object(Figure, 'savefig', inspect_figure):
            run_chart_job(self.store, self.settings, {'id': 999}, payload, dry_run=True)
        self.assertEqual(spans, [14])

    def test_latest_dense_series_precede_stale_sparse_series(self):
        base = json.loads(self.store.one("SELECT metadata FROM market_series WHERE id='test.rate'")['metadata'])
        meta = {**base, 'id': 'aaa.sparse', 'title': 'Редкий старый ряд'}
        market_store.register(self.store, [meta])
        market.enable(self.store, meta['id'], True)
        market_store.save_rows(self.store, meta['id'], [{'date': '2021-01-01', 'value': 99}])
        self.controls.callback(self.client, 2, 'ch:horizon:market:0', 'off')
        self.controls.callback(self.client, 2, 'ch:period:market:all', 'all-period')
        self.controls.callback(self.client, 2, 'ch:build:market', 'ordered')
        Worker(self.store, self.settings).process_next()
        report = self.store.one('SELECT * FROM reports')
        titles = [part['caption'] for part in json.loads(report['parts'])]
        self.assertEqual(titles, ['Тестовая ставка', 'Редкий старый ряд'])


if __name__ == '__main__':
    unittest.main()
