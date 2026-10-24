"""Operational audit, safe metadata, history isolation and live heartbeat tests."""
import asyncio
import csv
import io
import json
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import test_subscription_access as fixtures

from thesaurus import monitoring
from thesaurus.market_store import change_access
from thesaurus.runtime_monitor import RuntimeMonitor, resources
from thesaurus.store import Store, dumps, utcnow
from thesaurus.telegram_monitor import MonitoringControls


class MonitoringTests(unittest.TestCase):
    setUp = fixtures.SubscriptionAccessTests.setUp
    message = fixtures.SubscriptionAccessTests.message
    click = fixtures.SubscriptionAccessTests.click
    view = fixtures.SubscriptionAccessTests.view

    def test_job_transitions_are_atomic_ordered_and_payload_is_private(self):
        job = self.store.enqueue('wine_photo',{'actor_id':2,'path':'private-label.png','text':'private-input'})
        self.store.execute("UPDATE jobs SET state='running',started_at=? WHERE id=?",(utcnow(),job))
        self.store.execute("UPDATE jobs SET state='failed',finished_at=?,error=? WHERE id=?",(utcnow(),'secret-provider-error',job))
        self.store.execute("UPDATE jobs SET error='changed-secret' WHERE id=?",(job,))
        rows = self.store.query("SELECT * FROM monitor_events WHERE entity_type='jobs' ORDER BY id")
        self.assertEqual([row['outcome'] for row in rows],['pending','running','failed'])
        self.assertTrue(all(row['subject_id']==2 for row in rows))
        self.assertEqual(rows[-1]['level'],'error')
        self.assertGreaterEqual(rows[-1]['duration_ms'],0)
        self.assertNotIn('private',dumps(rows))
        self.assertNotIn('secret',dumps(rows))
        count = len(rows)
        with self.assertRaises(RuntimeError), self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute("UPDATE jobs SET state='pending' WHERE id=?",(job,))
            raise RuntimeError('rollback')
        self.assertEqual(self.store.one('SELECT state FROM jobs WHERE id=?',(job,))['state'],'failed')
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM monitor_events WHERE entity_type='jobs'")['n'],count)

    def test_access_history_has_admin_actor_and_affected_user(self):
        change_access(self.store,1,99,True)
        rows,_=monitoring.events(self.store,'u99',0,'access')
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['actor_id'],rows[0]['subject_id'],rows[0]['action']),(1,99,'allow'))
        self.assertFalse(monitoring.events(self.store,'u2',0,'access')[0])
        self.assertFalse(monitoring.events(self.store,'server',0,'access')[0])

    def test_period_category_user_and_server_filters_do_not_mix(self):
        monitoring.safe_record(self.store,'service','started')
        monitoring.safe_record(self.store,'interaction','command:start',actor=2,subject=2)
        monitoring.safe_record(self.store,'interaction','command:start',actor=3,subject=3,outcome='failed',level='error')
        self.store.execute('UPDATE monitor_events SET occurred_at=? WHERE actor_id=2',
                           ((datetime.now(timezone.utc)-timedelta(days=35)).isoformat(),))
        self.assertEqual(len(monitoring.events(self.store,'server',7)[0]),1)
        self.assertFalse(monitoring.events(self.store,'u2',7)[0])
        self.assertEqual(len(monitoring.events(self.store,'u2',0)[0]),1)
        errors,_=monitoring.events(self.store,'all',7,'errors')
        self.assertEqual([row['subject_id'] for row in errors],[3])
        for args in [('u2 OR 1=1',7,'all'),('all',365,'all'),('all',7,'evil')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                monitoring.events(self.store,*args)

    def test_snapshot_pagination_does_not_repeat_rows_after_new_events(self):
        for _ in range(18):
            monitoring.safe_record(self.store,'interaction','input:form',actor=2,subject=2)
        until=utcnow()
        first,total=monitoring.events(self.store,'u2',0,page=0,until=until)
        for _ in range(5):
            monitoring.safe_record(self.store,'interaction','command:start',actor=2,subject=2)
        second,new_total=monitoring.events(self.store,'u2',0,page=1,until=until)
        self.assertEqual(total,new_total)
        self.assertEqual(total,18)
        self.assertFalse({row['id'] for row in first} & {row['id'] for row in second})

    def test_private_interactions_log_metadata_success_errors_and_denials(self):
        self.message('/start',2)
        self.message('/allow @private_username',2)
        self.bot.market_controls.pending(1,'add_user')
        self.client.get_users.side_effect=TimeoutError('secret-network-message')
        self.message('@private_username',1)
        self.message('/start',99)
        before=self.store.one("SELECT COUNT(*) n FROM monitor_events WHERE category='interaction'")['n']
        self.bot.handle_message(self.client,SimpleNamespace(text='private-group-text',from_user=SimpleNamespace(id=2),
                                chat=SimpleNamespace(id=-100,type='group')))
        rows=self.store.query("SELECT * FROM monitor_events WHERE category='interaction' ORDER BY id")
        self.assertEqual(len(rows),before)
        self.assertEqual([row['outcome'] for row in rows],['ok','denied','invalid','denied'])
        self.assertIn('@private_username',json.loads(rows[2]['details'])['request']['text'])
        self.assertNotIn('secret-network-message',dumps(rows))
        self.assertEqual(json.loads(rows[2]['details'])['error_type'],'ValueError')

    def test_callbacks_and_uploads_do_not_record_embedded_values_or_file_names(self):
        event=SimpleNamespace(data='wc:edit:private-token:private-wine-name')
        self.assertEqual(monitoring.interaction_action('',event,callback=True),'button:wc:edit')
        event=SimpleNamespace(text='private-content',document=SimpleNamespace(file_name='private.pdf'))
        self.assertEqual(monitoring.interaction_action('handle_document',event),'upload:document')
        self.assertEqual(monitoring.interaction_action('handle_message',event,'receipt_ui:private-token'),'input:form')

    def test_admin_views_and_export_enforce_permissions_every_time(self):
        monitoring.safe_record(self.store,'interaction','command:start',actor=2,subject=2)
        controls=MonitoringControls(self.store,self.bot._reply)
        for callback in ['mon:home','mon:server','mon:users:0','mon:user:2','mon:feed:u2:0:all:0',
                         'mon:export:u2:0:all','mon:metrics:7:0']:
            with self.subTest(callback=callback),self.assertRaises(PermissionError):
                controls.callback(self.client,2,callback)
        self.message('/monitor',2)
        self.client.send_document.assert_not_called()
        controls.home(self.client,1)
        self.assertIn('Мониторинг и история',self.view()[0])
        controls.user(self.client,1,2)
        self.assertIn('mon:feed:u2:0:all:0',self.view()[1])
        controls.feed(self.client,1,'u2',0)
        controls.export(self.client,1,'u2',0,'all')
        destination,file=self.client.send_document.call_args.args[:2]
        self.assertEqual(destination,1)
        parsed=list(csv.reader(io.StringIO(file.getvalue().decode('utf-8-sig')),delimiter=';'))
        self.assertEqual(parsed[1][5:7],['2','2'])
        self.store.execute("UPDATE users SET role='subscriber' WHERE user_id=1")
        with self.assertRaises(PermissionError):
            controls.export(self.client,1,'u2',0,'all')

    def test_real_admin_navigation_reaches_user_history_and_server(self):
        self.message('/admin',1)
        self.assertIn('mon:home',self.view()[1])
        self.click('mon:home',1)
        self.assertIn('mon:users:0',self.view()[1])
        self.click('mon:users:0',1)
        self.assertIn('mon:user:2',self.view()[1])
        self.click('mon:user:2',1)
        self.assertIn('mon:feed:u2:0:all:0',self.view()[1])
        self.click('mon:feed:u2:0:all:0',1)
        self.assertIn('Журнал',self.view()[0])
        self.message('/monitor server',1)
        self.assertIn('История нагрузки',self.view()[1].values())
        self.click('mon:metrics:7:0',1)
        self.assertIn('История нагрузки',self.view()[0])

    def test_profile_changes_and_schedule_changes_have_before_and_after(self):
        from thesaurus.scheduling import seed_subscriptions, update_subscription
        from thesaurus.user_management import remember
        remember(self.store,SimpleNamespace(id=2,first_name='Иван',username='old_name'))
        remember(self.store,SimpleNamespace(id=2,first_name='Иван',username='new_name'))
        rows,_=monitoring.events(self.store,'u2',0,'identity')
        self.assertEqual(json.loads(rows[0]['details']),{'old_username':'old_name','username':'new_name'})
        seed_subscriptions(self.store,'1',1)
        update_subscription(self.store,1,'ruonia',times=['10:00'])
        update_subscription(self.store,1,'ruonia',times=['11:00'])
        rows,_=monitoring.events(self.store,'u1',0,'schedule')
        self.assertEqual(json.loads(rows[0]['details'])['old_times'],'["10:00"]')
        self.assertEqual(json.loads(rows[0]['details'])['times'],'["11:00"]')

    def test_recovery_transition_uses_change_time_not_original_queue_time(self):
        job=self.store.enqueue('dollar_analysis',{'actor_id':2})
        self.store.execute("UPDATE jobs SET state='running',created_at='2020-01-01T00:00:00+00:00' WHERE id=?",(job,))
        self.store.execute("UPDATE jobs SET state='pending',started_at=NULL WHERE id=?",(job,))
        row=self.store.one("SELECT occurred_at FROM monitor_events WHERE entity_type='jobs' ORDER BY id DESC LIMIT 1")
        self.assertGreater(datetime.fromisoformat(row['occurred_at']),datetime.now(timezone.utc)-timedelta(minutes=1))

    def test_source_failures_and_recovery_have_history_without_provider_error_text(self):
        self.store.execute('''INSERT INTO market_series(id,source,dataset,title,metadata,enabled,status)
            VALUES('test.source','cbr','test','Тестовый источник','{}',1,'not_loaded')''')
        self.store.execute("UPDATE market_series SET status='error',error='private-provider-response',updated_at=? WHERE id='test.source'",(utcnow(),))
        self.store.execute("UPDATE market_series SET status='ok',error=NULL,updated_at=? WHERE id='test.source'",(utcnow(),))
        rows=self.store.query("SELECT * FROM monitor_events WHERE entity_type='market_series' AND entity_id='test.source' ORDER BY id")
        self.assertEqual([row['outcome'] for row in rows],['error','ok'])
        self.assertEqual(rows[0]['level'],'error')
        self.assertNotIn('private-provider-response',dumps(rows))
        self.assertTrue(all(row['subject_id'] is None for row in rows))

    def test_migration_backfills_known_states_once_and_preserves_user_data(self):
        with self.store.connection() as conn:
            triggers=conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'monitor_%'").fetchall()
            for row in triggers:
                conn.execute('DROP TRIGGER "'+row['name']+'"')
            for table in ('monitor_samples','monitor_runtime','monitor_events'):
                conn.execute('DROP TABLE '+table)
            conn.execute("DELETE FROM settings WHERE key IN ('monitoring_version','monitoring_started_at')")
        job=self.store.enqueue('dollar_analysis',{'actor_id':2,'bank_price':123.45})
        self.store.execute("UPDATE jobs SET state='done',finished_at=? WHERE id=?",(utcnow(),job))
        before=self.store.query('SELECT * FROM users ORDER BY user_id')
        monitoring.migrate(self.store)
        rows=self.store.query("SELECT * FROM monitor_events WHERE entity_type='jobs'")
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['origin'],rows[0]['outcome']),('history','done'))
        self.assertNotIn('123.45',dumps(rows))
        count=self.store.one('SELECT COUNT(*) n FROM monitor_events')['n']
        Store(self.store.path)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM monitor_events')['n'],count)
        self.assertEqual(self.store.query('SELECT * FROM users ORDER BY user_id'),before)

    def test_runtime_records_connection_staleness_shutdown_and_previous_crash(self):
        client=SimpleNamespace(is_connected=True)
        runtime=RuntimeMonitor(self.store,client)
        runtime.start()
        snapshot=monitoring.server_snapshot(self.store)
        self.assertTrue(snapshot['active'])
        self.assertEqual(snapshot['runtime']['state'],'running')
        self.assertGreater(snapshot['metrics']['database_mb'],0)
        later=datetime.now(timezone.utc)+timedelta(minutes=2)
        self.assertFalse(monitoring.server_snapshot(self.store,now=later)['active'])
        client.is_connected=False
        runtime.pulse()
        self.assertEqual(self.store.one('SELECT telegram_connected FROM monitor_runtime')['telegram_connected'],0)
        next_runtime=RuntimeMonitor(self.store,client)
        next_runtime.start()
        old=self.store.one('SELECT state,finished_at FROM monitor_runtime WHERE id=?',(runtime.identity,))
        self.assertEqual(old,{'state':'interrupted','finished_at':None})
        next_runtime.finish(RuntimeError('private-runtime-failure'))
        rows,_=monitoring.events(self.store,'server',0,'errors')
        self.assertTrue(any(row['action']=='failed' for row in rows))
        self.assertNotIn('private-runtime-failure',dumps(rows))

    def test_heartbeat_runs_while_async_job_is_waiting_and_samples_are_bounded(self):
        runtime=RuntimeMonitor(self.store,SimpleNamespace(is_connected=True))
        runtime.heartbeat_interval=.005
        async def exercise():
            with patch.object(runtime,'pulse',wraps=runtime.pulse) as pulse:
                async with runtime:
                    self.assertIsNotNone(runtime._task)
                    await asyncio.to_thread(time.sleep,.1)
                    self.assertGreater(pulse.call_count,1)
                    runtime.heartbeat_interval=10
                    runtime._last_sample=-1000
                    await asyncio.to_thread(runtime.pulse)
        asyncio.run(exercise())
        self.assertEqual(self.store.one('SELECT state FROM monitor_runtime')['state'],'stopped')
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM monitor_samples')['n'],2)
        self.assertGreater(resources(self.store.path)['disk_free_gb'],0)
