"""Personal geography, uncertain grape composition, planning and admin scopes."""
import io
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from thesaurus import vivino
from thesaurus import vivino_atlas as atlas
from thesaurus.config import Settings
from thesaurus.delivery import deliver_pending
from thesaurus.service import Worker
from thesaurus.store import Store
from thesaurus.telegram_bot import BotControls
from thesaurus.telegram_projects import setting_key
from thesaurus.vivino_geo_html import render


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.test-tmp' / ('atlas-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = Store(self.root / 'state.sqlite3')
        self.store.add_user(1, 'admin', subscriptions='all')
        self.store.add_user(2, subscriptions='all')
        self.store.add_user(3, subscriptions='all')
        self.bot = BotControls(self.store)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=100)
        self.client.send_document.return_value = SimpleNamespace(id=101)
        self.settings = Settings(root=self.root)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))
        self.dataset(2)
        self.dataset(3, review='OTHER-OWNER-REVIEW')

    def dataset(self, actor, review='OWNER-REVIEW'):
        raw = ('Wine name,Winery,Country,Region,Your rating,Your review,Grapes\n'
               f'Wine A,Winery,Spain,Rioja,4,{review},Cabernet Sauvignon\n'
               'Wine B,Winery,France,Bordeaux,,,\n').encode()
        dataset = {**vivino.parse_upload(raw, 'full_wine_list.csv'), 'generation': uuid4().hex, 'imported_at': '2026-10-05'}
        self.store.set_setting(setting_key(actor, 'vivino', 'dataset'), dataset)
        return dataset

    def click(self, data, actor=2):
        query = SimpleNamespace(data=data, id=uuid4().hex, from_user=SimpleNamespace(id=actor),
            message=SimpleNamespace(id=1000+int(uuid4().hex[:7], 16), chat=SimpleNamespace(id=actor, type='private')), answer=Mock())
        self.bot.handle_callback(self.client, query)
        return query

    def message(self, text, actor=2):
        self.bot.handle_message(self.client, SimpleNamespace(text=text, id=1,
            from_user=SimpleNamespace(id=actor), chat=SimpleNamespace(id=actor, type='private')))

    def text(self):
        rpc = next(call for call in reversed(self.client.mock_calls) if call[0] in {'send_message', 'edit_message_text'})
        return rpc.args[2] if rpc[0] == 'edit_message_text' else rpc.args[1]

    def test_catalog_identity_coordinates_license_metadata_and_no_country_collision(self):
        ref = atlas.catalog()
        self.assertGreater(len(ref['regions']), 700)
        self.assertGreater(len(ref['grapes']), 1500)
        for kind in ['countries', 'regions', 'grapes']:
            self.assertEqual(len(ref[kind]), len({r['id'] for r in ref[kind]}))
        self.assertFalse(ref['complete'])
        for row in ref['regions']:
            if row['lon'] is not None:
                self.assertTrue(-180 <= row['lon'] <= 180 and -90 <= row['lat'] <= 90)
        self.assertIn('naturalearthdata.com', ref['sources']['world'])

    def test_only_rating_review_or_explicit_confirmation_counts_as_tasted(self):
        view = atlas.model(self.store, 2)
        self.assertEqual(view['stats']['tasted'], 1)
        self.assertEqual(view['stats']['regions'], 1)
        self.assertEqual(view['stats']['grapes'], 1)
        self.assertFalse(view['wines'][1]['tasted'])
        self.assertNotEqual(view['wines'][1]['region_id'][0], 'u')  # Bordeaux alias is resolved.

    def test_no_grapes_are_invented_from_names_styles_or_geographic_origin(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][0]['name'] = 'Cabernet Sauvignon estate'
        dataset['rows'][0]['style'] = 'Spanish Rioja Red'
        dataset['rows'][0]['grapes'] = []
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        view = atlas.model(self.store, 2)
        self.assertEqual(view['stats']['grapes'], 0)
        self.assertEqual(view['unknown_grapes'], 1)
        self.assertEqual(view['wines'][0]['grapes'], [])

    def test_heat_statistics_count_distinct_tasted_wines_not_duplicate_scans(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'].append({**dataset['rows'][0], 'review': 'Second scan', 'rating': 5})
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        analysis = atlas.model(self.store, 2)['analytics']
        self.assertEqual(analysis['summary']['tasted_wines'], 1)
        country = next(c for c in analysis['countries'] if c['id'] == 'ES')
        self.assertEqual(country['wine_count'], 1)
        self.assertEqual(country['mean_rating'], 4.5)
        grape = next(c for c in analysis['grapes'] if c['id'] == 'Q207310')
        self.assertEqual(grape['wine_count'], 1)
        self.assertEqual(grape['rated_count'], 1)
        self.assertEqual(next(c for c in analysis['countries'] if c['id'] == 'FR')['wine_count'], 0)

    def test_blends_are_counted_for_each_known_grape_without_inventing_attribution(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][0]['grapes'] = ['Cabernet Sauvignon', 'Syrah']
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        analysis = atlas.model(self.store, 2)['analytics']
        for identifier in ['Q207310', 'Q332720']:
            grape = next(g for g in analysis['grapes'] if g['id'] == identifier)
            self.assertEqual(grape['wine_count'], 1)
            self.assertEqual(grape['blend_count'], 1)
            self.assertEqual(grape['single_variety_count'], 0)
            self.assertEqual(grape['country_counts'], {'ES': 1})
        self.assertEqual(analysis['summary']['grapes'], 2)
        self.assertEqual(analysis['preferences'], [])

    def test_preferences_need_three_rated_distinct_wines_per_grape(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        original = dataset['rows'][0]
        dataset['rows'] = [{**original, 'name': f'Tasted {i}', 'rating': 4+i/10} for i in range(3)]
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        prefs = atlas.model(self.store, 2)['analytics']['preferences']
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]['rated_count'], 3)
        self.assertEqual(prefs[0]['mean_rating'], 4.1)
        dataset['rows'][-1]['rating'] = None
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        self.assertEqual(atlas.model(self.store, 2)['analytics']['preferences'], [])

    def test_suggestions_prioritize_personal_plan_and_new_grapes_in_untasted_wines(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][1]['grapes'] = ['Syrah']
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        atlas.revise(self.store, 2, 'plan', ['grape:Q332720'])
        view = atlas.model(self.store, 2)
        recs = view['analytics']['recommendations']
        self.assertEqual((recs[0]['kind'], recs[0]['id']), ('grape', 'Q332720'))
        wine = next(r for r in recs if r['kind'] == 'wine')
        self.assertEqual(wine['id'], view['wines'][1]['id'])
        self.assertIn('Syrah', wine['reason'])
        atlas.revise(self.store, 2, 'progress', ['grape:Q332720'])
        updated = atlas.model(self.store, 2)['analytics']['recommendations']
        self.assertFalse(any(r['kind'] == 'wine' or r['kind'] == 'grape' and r['id'] == 'Q332720' for r in updated))

    def test_missing_composition_and_country_are_explicit_and_no_zero_ratings(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][0].update(country='Unknown', grapes=[], rating=None)
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        result = atlas.model(self.store, 2)['analytics']
        self.assertEqual(result['summary']['unknown_composition'], 1)
        self.assertEqual(result['summary']['unknown_country'], 1)
        self.assertEqual(result['heat_max'], 0)
        self.assertEqual(result['summary']['known_composition'], 0)
        self.assertEqual(result['unique_wines'][0]['rating'], None)
        self.assertFalse(any(r['kind'] == 'wine' for r in result['recommendations']))
        starters = [r for r in result['recommendations'] if r['kind'] == 'grape']
        self.assertTrue(starters)
        self.assertTrue(all('Стартовая цель' in r['reason'] and 'возможно' in r['reason'] for r in starters))

    def test_suggestions_use_existing_ratings_but_do_not_predict_new_wine_ratings(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        original = dataset['rows'][0]
        dataset['rows'] = [{**original, 'name': f'Known {i}', 'rating': 4.5} for i in range(3)]
        dataset['rows'].append({**original, 'name': 'Not tried', 'rating': None, 'review': '',
                               'grapes': ['Cabernet Sauvignon', 'Syrah']})
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        result = atlas.model(self.store, 2)['analytics']
        first = result['recommendations'][0]
        self.assertEqual(first['kind'], 'wine')
        self.assertIn('высокой средней оценкой', first['reason'])
        self.assertIn('не прогноз вкуса', first['reason'])
        self.assertIsNone(result['unique_wines'][-1]['rating'])

    def test_heat_geometries_are_bound_to_valid_country_identifiers(self):
        reference = json.loads((Path(atlas.__file__).parent/'resources/vivino_world.json').read_text(encoding='utf-8'))
        self.assertEqual(len(reference['paths']), len(reference['features']))
        countries = {c['id'] for c in atlas.catalog()['countries']}
        self.assertTrue(all(f['country'] in countries for f in reference['features']))
        self.assertEqual({f['country'] for f in reference['features']}, countries)
        page = render(atlas.model(self.store, 2))
        self.assertIn('тепловая заливка стран', page)
        self.assertIn('Сорт в составе', page)
        self.assertIn('Что попробовать дальше', page)
        self.assertNotIn('type="checkbox" checked', page)

    def test_analysis_admin_view_is_readonly_and_other_subscribers_are_denied(self):
        self.click('va:analysis:3', 1).answer.assert_called_once_with()
        self.assertIn('аналитика пользователя 3', self.text())
        self.click('va:analysis:3', 2)
        self.assertNotIn('OTHER-OWNER-REVIEW', self.text())
        self.assertIsNone(self.store.get_setting(setting_key(3, 'vivino', 'plan')))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_new_map_version_rebuilds_after_previous_cached_job(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        self.store.enqueue('vivino_map', {'actor_id': 2}, f"vivino:geo2:2:2:{dataset['generation']}")
        self.click('vi:build')
        jobs = self.store.query('SELECT * FROM jobs ORDER BY id')
        self.assertEqual(len(jobs), 2)
        self.assertEqual(json.loads(jobs[1]['payload'])['map_version'], vivino.MAP_VERSION)

    def test_region_matching_is_country_specific_and_unknown_is_not_zero_coordinate(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][0]['country'] = 'France'
        dataset['rows'][0]['region'] = 'Rioja'
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        view = atlas.model(self.store, 2)
        row = next(r for r in view['regions'] if r['id'] == view['wines'][0]['region_id'])
        self.assertIsNone(row['lon'])
        self.assertEqual(row['country'], 'FR')
        self.assertEqual(row['coordinate_quality'], 'country')

    def test_personal_plan_and_manual_progress_are_independent(self):
        region = next(r for r in atlas.model(self.store, 2)['regions'] if not r['tried'])
        self.click(f"va:plan:2:region:{region['id']}")
        self.click(f"va:done:2:region:{region['id']}")
        own = next(r for r in atlas.model(self.store, 2)['regions'] if r['id'] == region['id'])
        other = next(r for r in atlas.model(self.store, 3)['regions'] if r['id'] == region['id'])
        self.assertTrue(own['planned'] and own['manual'])
        self.assertFalse(other['planned'] or other['manual'])
        self.click(f"va:done:2:region:{region['id']}")
        self.assertNotIn('region:' + region['id'], self.store.get_setting(setting_key(2, 'vivino', 'progress')))

    def test_admin_lists_uploaders_and_can_read_disabled_owner_without_editing(self):
        self.store.add_user(3, enabled=False, subscriptions='all')
        self.click('va:users:0', 1).answer.assert_called_once_with()
        self.assertIn('ID 3', self.text())
        self.assertIn('выключен', self.text())
        self.click('va:view:3', 1).answer.assert_called_once_with()
        self.assertIn('пользователь 3', self.text())
        region = atlas.model(self.store, 1, 3)['wines'][0]['region_id']
        self.click(f'va:plan:3:region:{region}', 1).answer.assert_called_once_with(
            'Просмотр администратора не изменяет чужие данные', show_alert=True)

    def test_cross_owner_callbacks_and_admin_listing_require_admin(self):
        for callback in ['va:users:0', 'va:view:3', 'va:wines:3:0', 'va:map:3', 'va:list:3:grapes:0:all']:
            with self.subTest(callback=callback):
                event = self.click(callback)
                self.assertTrue(event.answer.call_args.kwargs.get('show_alert'))
        self.assertEqual(self.store.query('SELECT * FROM jobs'), [])

    def test_admin_map_goes_to_requesting_admin_and_preserves_subject_owner(self):
        self.store.add_user(3, enabled=False, subscriptions='all')
        self.click('va:map:3', 1)
        Worker(self.store, self.settings).drain()
        report = self.store.one("SELECT * FROM reports WHERE section='vivino'")
        self.assertEqual(json.loads(report['quality'])['owner'], 3)
        self.assertEqual(json.loads(report['quality'])['viewer'], 1)
        self.assertTrue(all(r['recipient'] == '1' for r in self.store.query('SELECT * FROM deliveries')))
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_called_once()
        self.assertEqual(self.client.send_document.call_args.args[0], 1)

    def test_role_downgrade_blocks_admin_map_and_delivery(self):
        self.click('va:map:3', 1)
        Worker(self.store, self.settings).drain()
        self.store.add_user(1, role='subscriber', subscriptions='all')
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_not_called()
        self.assertTrue(all(r['state'] == 'cancelled' for r in self.store.query('SELECT * FROM deliveries')))

    def test_grape_edit_is_owned_and_changes_generation_and_progress(self):
        view = atlas.model(self.store, 2)
        identifier = view['wines'][0]['id']
        generation = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))['generation']
        self.click(f'va:editgrapes:2:{identifier}')
        self.message('Cabernet Sauvignon; Syrah')
        updated = atlas.model(self.store, 2)
        self.assertEqual(updated['stats']['grapes'], 2)
        self.assertNotEqual(self.store.get_setting(setting_key(2, 'vivino', 'dataset'))['generation'], generation)
        self.click(f'va:editgrapes:2:{identifier}', 3).answer.assert_called_once()
        self.assertIsNone(self.store.get_setting(setting_key(3, 'vivino', 'annotations')))

    def test_annotation_change_invalidates_old_delivery_snapshot(self):
        self.click('vi:build')
        Worker(self.store, self.settings).drain()
        atlas.revise(self.store, 2, 'plan', ['region:Q1569384'])
        deliver_pending(self.store, self.client)
        self.client.send_document.assert_not_called()
        self.assertTrue(all(r['state'] == 'cancelled' for r in self.store.query('SELECT * FROM deliveries')))

    def test_custom_targets_cover_missing_reference_regions_and_varieties(self):
        self.click('va:new:2:region')
        self.message('Личный регион; FR; 44.5; 1.2; Описание моей цели')
        custom = self.store.get_setting(setting_key(2, 'vivino', 'custom'))[0]
        self.assertEqual(custom['lon'], 1.2)
        self.assertIn('region:' + custom['id'], self.store.get_setting(setting_key(2, 'vivino', 'plan')))
        self.click('va:new:2:grape')
        self.message('Личный сорт; описание')
        self.assertEqual(len(self.store.get_setting(setting_key(2, 'vivino', 'custom'))), 2)
        self.assertIsNone(self.store.get_setting(setting_key(3, 'vivino', 'custom')))

    def test_custom_coordinate_validation_keeps_input_pending(self):
        self.click('va:new:2:region')
        self.message('Bad; FR; 91; NaN; description')
        self.assertIn('Широта', self.text())
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'custom')))
        self.message('Good; FR; ; ; description')
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'custom'))[0]['lon'])

    def test_personal_coordinate_correction_does_not_change_other_users(self):
        self.click('va:new:2:region')
        self.message('Rioja; ES; 42.5; -2.5; Моя уточнённая точка')
        own = atlas.model(self.store, 2)
        other = atlas.model(self.store, 3)
        region = next(r for r in own['regions'] if r['id'] == own['wines'][0]['region_id'])
        self.assertEqual(region['coordinate_quality'], 'user')
        self.assertEqual(region['lon'], -2.5)
        self.assertNotEqual(own['wines'][0]['region_id'], other['wines'][0]['region_id'])

    def test_unknown_countries_do_not_merge_same_named_regions(self):
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        for index, row in enumerate(dataset['rows']):
            row.update(country=f'Unlisted country {index}', region='Central')
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        view = atlas.model(self.store, 2)
        self.assertNotEqual(view['wines'][0]['region_id'], view['wines'][1]['region_id'])

    def test_variety_aliases_count_once_and_are_searchable(self):
        grape = next(r for r in atlas.catalog()['grapes'] if r['id'] == 'Q207310')
        dataset = self.store.get_setting(setting_key(2, 'vivino', 'dataset'))
        dataset['rows'][0]['grapes'] = [grape['name'], grape['id']]
        self.store.set_setting(setting_key(2, 'vivino', 'dataset'), dataset)
        view = atlas.model(self.store, 2)
        self.assertEqual(view['wines'][0]['grapes'], ['Q207310'])
        self.assertEqual(next(r for r in view['grapes'] if r['id'] == 'Q207310')['tasted'], 1)
        self.click('va:search:2:grapes')
        self.message(grape['aliases'][0])
        self.assertIn('Найдено: 1;', self.text())

    def test_delete_erases_personal_notes_plan_and_admin_job_for_this_subject(self):
        atlas.revise(self.store, 2, 'progress', ['grape:Q207310'])
        self.click('va:map:2', 1)
        self.bot.vivino_controls.delete(2)
        self.assertIsNone(self.store.get_setting(setting_key(2, 'vivino', 'progress')))
        self.assertEqual(self.store.one('SELECT state FROM jobs')['state'], 'cancelled')
        self.assertIsNotNone(self.store.get_setting(setting_key(3, 'vivino', 'dataset')))

    def test_html_has_world_geometry_private_scope_sources_and_escaped_content(self):
        own = atlas.model(self.store, 2)
        own['wines'][0]['review'] = '</script><img onerror=alert(1)>'
        page = render(own)
        self.assertIn('Географическая карта', page)
        self.assertIn('Natural Earth', page)
        self.assertIn('Происхождение сорта', page)
        self.assertNotIn('OTHER-OWNER-REVIEW', page)
        self.assertNotIn('</script><img', page)
        self.assertNotIn('innerHTML', page)
        self.assertNotIn('fetch(', page)

    def test_callbacks_stay_under_telegram_limit_and_search_returns_in_same_screen(self):
        self.click('va:list:2:regions:0:all')
        for call in self.client.mock_calls:
            markup = call.kwargs.get('reply_markup')
            if markup:
                self.assertTrue(all(len(b.callback_data.encode()) <= 64 for r in markup.inline_keyboard for b in r))
        self.click('va:search:2:regions')
        self.message('Rioja')
        self.assertIn('Поиск: Rioja', self.text())
        self.assertIsNone(self.store.one('SELECT pending FROM bot_sessions WHERE user_id=2')['pending'])

    def test_unexpected_file_on_admin_profile_cannot_change_owner_dataset(self):
        self.click('va:view:3', 1)
        event = SimpleNamespace(id=10, from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=1, type='private'),
                                document=SimpleNamespace(file_name='export.zip', file_size=10))
        self.client.download_media.return_value = io.BytesIO(b'not-a-zip')
        self.bot.handle_document(self.client, event)
        self.client.download_media.assert_not_called()
