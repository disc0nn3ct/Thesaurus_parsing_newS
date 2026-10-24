"""Observed history totals, user isolation and honest resource charts."""
import unittest
from datetime import datetime, timedelta, timezone

import test_subscription_access as fixtures

from thesaurus import monitoring
from thesaurus.monitor_charts import (
    activity_data,
    render_activity,
    render_resources,
    resource_data,
)
from thesaurus.store import dumps
from thesaurus.telegram_monitor import MonitoringControls


class MonitorChartTests(unittest.TestCase):
    setUp = fixtures.SubscriptionAccessTests.setUp

    def test_totals_exclude_archived_snapshots_and_isolate_users(self):
        now = datetime.now(timezone.utc)
        for actor in (2,2,3):
            monitoring.safe_record(self.store,'interaction','command:start',actor=actor,subject=actor,duration_ms=200)
        monitoring.safe_record(self.store,'job','report',actor=2,subject=2,outcome='done')
        monitoring.safe_record(self.store,'job','report',actor=3,subject=3,outcome='failed',level='error')
        self.store.execute("UPDATE monitor_events SET occurred_at=? WHERE origin='live'",((now-timedelta(hours=2)).isoformat(),))
        monitoring.safe_record(self.store,'interaction','command:old',actor=2,subject=2)
        self.store.execute("UPDATE monitor_events SET origin='history' WHERE action='command:old'")
        all_data = activity_data(self.store,'all',1,now=now+timedelta(seconds=1))
        self.assertEqual(sum(row['interactions'] for row in all_data['rows']),3)
        self.assertEqual(sum(row['users'] for row in all_data['rows']),2)
        self.assertEqual(sum(row['done'] for row in all_data['rows']),1)
        self.assertEqual(sum(row['failed'] for row in all_data['rows']),1)
        own = activity_data(self.store,'u2',1,now=now+timedelta(seconds=1))
        self.assertEqual(sum(row['interactions'] for row in own['rows']),2)
        self.assertEqual(sum(row['failed'] for row in own['rows']),0)
        self.assertEqual(sum(row['latency'] for row in own['rows']),.2)
        self.assertGreater(own['archive'],0)
        self.assertFalse(activity_data(self.store,'server',1,now=now)['rows'])
        with self.assertRaises(ValueError):
            activity_data(self.store,'u2 OR 1=1',1)

    def test_resources_preserve_peaks_missing_values_and_runtime_boundaries(self):
        now = datetime.now(timezone.utc)
        for runtime,cpu in [('one',10),('two',None)]:
            self.store.execute('INSERT INTO monitor_runtime(id,pid,host,started_at,heartbeat_at,state) VALUES(?,?,?,?,?,?)',
                               (runtime,1,'test',(now-timedelta(hours=1)).isoformat(),now.isoformat(),'running'))
            for peak in (cpu,90 if cpu is not None else None):
                self.store.execute('INSERT INTO monitor_samples(runtime_id,created_at,metrics) VALUES(?,?,?)',
                    (runtime,(now-timedelta(minutes=10)).isoformat(),dumps({'process_cpu_percent':peak,'process_rss_mb':100,'disk_free_gb':20,'database_mb':3})))
        data = resource_data(self.store,1,now=now)
        self.assertEqual(len(data['rows']),2)
        first,second = data['rows']
        self.assertEqual((first['cpu'],first['cpu_peak'],first['samples']),(50,90,2))
        self.assertIsNone(second['cpu'])
        image = render_resources(data)
        self.assertTrue(image.getvalue().startswith(b'\x89PNG'))

    def test_navigation_png_and_every_route_are_admin_only(self):
        controls = MonitoringControls(self.store,self.bot._reply)
        for route in ('mon:charts:all:7','mon:plot:all:7:activity','mon:plot:server:7:resources'):
            with self.subTest(route=route),self.assertRaises(PermissionError):
                controls.callback(self.client,2,route)
        controls.charts(self.client,1,'u2',1)
        self.assertIn('Графики истории',fixtures.SubscriptionAccessTests.view(self)[0])
        self.assertNotIn('mon:plot:u2:1:resources',fixtures.SubscriptionAccessTests.view(self)[1])
        with self.assertRaises(ValueError):
            controls.plot(self.client,1,'u2',1,'resources')
        controls.plot(self.client,1,'u2',1,'activity')
        destination,image = self.client.send_photo.call_args.args[:2]
        self.assertEqual(destination,1)
        self.assertTrue(image.getvalue().startswith(b'\x89PNG'))
        self.assertTrue(render_activity(activity_data(self.store,'u999',1),'Нет событий').getvalue().startswith(b'\x89PNG'))
