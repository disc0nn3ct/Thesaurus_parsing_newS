"""Real SQLite, mocked Telegram: delivery ambiguity and recipient baselines."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from pyrogram.errors import FloodWait, UserIsBlocked

from thesaurus import delivery as delivery_module
from thesaurus.delivery import (
    deliver_pending,
    prepare_delivery,
    recover_sending,
    resolve_uncertain,
)
from thesaurus.scheduling import seed_subscriptions
from thesaurus.store import Store, dumps, utcnow


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parent / f"delivery_test_{uuid4().hex}.sqlite3"
        self.addCleanup(self.cleanup_database)
        self.store = Store(self.path)
        self.client = Mock()
        self.client.send_message.return_value = SimpleNamespace(id=123)
        self.client.send_photo.return_value = SimpleNamespace(id=124)
        self.client.send_document.return_value = SimpleNamespace(id=125)

    def cleanup_database(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.path) + suffix).unlink(missing_ok=True)

    def report(self, text="Brief", full_text="Full", metrics=None, versions=(1, 1), parts=None,
               section="wordstat", news=None, quality=None):
        report_id = uuid4().hex
        self.store.execute(
            """INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,metrics,parts,news,quality)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (report_id, section, utcnow(), *versions, text, full_text, dumps(metrics or {}),
             dumps(parts or []), dumps(news or []), dumps(quality or {})),
        )
        return report_id

    def row(self, delivery_id):
        return self.store.one("SELECT * FROM deliveries WHERE id=?", (delivery_id,))

    def subscription(self, recipient=11, section="wordstat"):
        self.store.add_user(recipient, subscriptions='all')
        seed_subscriptions(self.store, str(recipient), recipient)
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE recipient=? AND section=?", (str(recipient), section))
        return self.store.one("SELECT id FROM subscriptions WHERE recipient=? AND section=?", (str(recipient), section))["id"]

    def test_disabled_schedule_cancels_future_retry_but_manual_request_still_sends(self):
        subscription_id = self.subscription()
        report_id = self.report()
        scheduled = prepare_delivery(self.store, report_id, 11, subscription_id=subscription_id)
        self.client.send_message.side_effect = FloodWait(3600)
        deliver_pending(self.store, self.client)
        self.assertEqual(self.row(scheduled[0])["state"], "pending")
        self.assertIsNotNone(self.row(scheduled[0])["retry_at"])
        self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
        deliver_pending(self.store, self.client)
        self.assertEqual(self.row(scheduled[0])["state"], "cancelled")
        self.client.send_message.assert_called_once()
        manual = prepare_delivery(self.store, report_id, 11)
        self.assertNotEqual(manual, scheduled)
        self.client.send_message.side_effect = None
        self.assertEqual(deliver_pending(self.store, self.client), 1)
        self.assertEqual(self.row(manual[0])["state"], "sent")
        self.store.execute("UPDATE subscriptions SET enabled=1 WHERE id=?", (subscription_id,))
        self.assertEqual(prepare_delivery(self.store, report_id, 11, subscription_id=subscription_id), scheduled)
        self.assertEqual(deliver_pending(self.store, self.client), 0)
        self.assertEqual(self.row(scheduled[0])["state"], "cancelled")

    def test_cancellation_keeps_sent_and_uncertain_parts_untouched(self):
        subscription_id = self.subscription()
        ids = prepare_delivery(self.store, self.report(text="x" * 9000), 11, subscription_id=subscription_id)
        self.store.execute("UPDATE deliveries SET state='sent',message_id=456 WHERE id=?", (ids[0],))
        self.store.execute("UPDATE deliveries SET state='uncertain' WHERE id=?", (ids[1],))
        self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
        deliver_pending(self.store, self.client)
        self.assertEqual([self.row(value)["state"] for value in ids], ["sent", "uncertain", "cancelled"])
        self.assertEqual(self.row(ids[0])["message_id"], 456)
        self.client.send_message.assert_not_called()

    def test_scheduled_delivery_cannot_use_another_recipient_or_section(self):
        subscription_id = self.subscription()
        report_id = self.report()
        wrong_owner = prepare_delivery(self.store, report_id, 12, subscription_id=subscription_id)
        self.store.execute("UPDATE reports SET section='ai' WHERE id=?", (report_id,))
        wrong_section = prepare_delivery(self.store, report_id, 11, subscription_id=subscription_id)
        deliver_pending(self.store, self.client)
        self.assertTrue(all(self.row(value)["state"] == "cancelled" for value in wrong_owner + wrong_section))
        self.client.send_message.assert_not_called()

    def test_subscription_disabled_after_claim_still_cancels_before_network(self):
        subscription_id = self.subscription()
        delivery_id = prepare_delivery(self.store, self.report(), 11, subscription_id=subscription_id)[0]
        original_claim = delivery_module._claim

        def claim_then_disable(*args):
            row = original_claim(*args)
            self.store.execute("UPDATE subscriptions SET enabled=0 WHERE id=?", (subscription_id,))
            return row

        with patch.object(delivery_module, "_claim", side_effect=claim_then_disable):
            deliver_pending(self.store, self.client)
        self.assertEqual(self.row(delivery_id)["state"], "cancelled")
        self.client.send_message.assert_not_called()

    def test_prepare_is_idempotent_and_freezes_payload(self):
        report_id = self.report()
        first = prepare_delivery(self.store, report_id, 11)
        original = self.row(first[0])["payload"]
        self.store.execute("UPDATE reports SET text=? WHERE id=?", ("changed" * 2000, report_id))
        self.assertEqual(prepare_delivery(self.store, report_id, 11), first)
        self.assertEqual(self.row(first[0])["payload"], original)

    def test_brief_and_full_have_distinct_variants_and_details_button(self):
        report_id = self.report()
        brief = prepare_delivery(self.store, report_id, 11)
        full = prepare_delivery(self.store, report_id, 11, detail="full")
        self.assertNotEqual(brief, full)
        self.assertEqual(json.loads(self.row(brief[0])["payload"])["callback_data"], f"report:{report_id}")
        self.assertNotIn("callback_data", json.loads(self.row(full[0])["payload"]))

    def test_sending_is_persisted_before_network_and_sent_is_not_repeated(self):
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]

        def send(*args, **kwargs):
            self.assertEqual(self.row(delivery_id)["state"], "sending")
            return SimpleNamespace(id=456)

        self.client.send_message.side_effect = send
        self.assertEqual(deliver_pending(self.store, self.client), 1)
        self.assertEqual(self.row(delivery_id)["message_id"], 456)
        self.assertEqual(deliver_pending(self.store, self.client), 0)
        self.client.send_message.assert_called_once()

    def test_two_workers_cannot_send_the_same_part(self):
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]
        started, release = Event(), Event()

        def send(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError("Test worker was not released")
            return SimpleNamespace(id=456)

        self.client.send_message.side_effect = send
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(deliver_pending, self.store, self.client)
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(deliver_pending(self.store, self.client), 0)
                self.assertEqual(self.row(delivery_id)["state"], "sending")
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5), 1)
        self.client.send_message.assert_called_once()

    def test_ambiguous_failure_blocks_later_parts_but_not_other_recipients(self):
        report_id = self.report(text="x" * 5000)
        first = prepare_delivery(self.store, report_id, 11)
        second = prepare_delivery(self.store, self.report(), 12)
        self.client.send_message.side_effect = [TimeoutError("secret-provider-data"), SimpleNamespace(id=2)]
        self.assertEqual(deliver_pending(self.store, self.client), 1)
        self.assertEqual(self.row(first[0])["state"], "uncertain")
        self.assertEqual(self.row(first[1])["state"], "pending")
        self.assertEqual(self.row(second[0])["state"], "sent")
        self.assertNotIn("secret-provider-data", self.row(first[0])["error"])
        deliver_pending(self.store, self.client)
        self.assertEqual(self.client.send_message.call_count, 2)

    def test_restart_requires_explicit_resolution_of_interrupted_send(self):
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]
        self.store.execute("UPDATE deliveries SET state='sending' WHERE id=?", (delivery_id,))
        self.assertEqual(recover_sending(self.store), 1)
        deliver_pending(self.store, self.client)
        self.client.send_message.assert_not_called()
        resolve_uncertain(self.store, delivery_id, "retry")
        deliver_pending(self.store, self.client)
        self.assertEqual(self.row(delivery_id)["state"], "sent")

    def test_confirmed_uncertain_send_is_never_resent(self):
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]
        self.store.execute("UPDATE deliveries SET state='uncertain' WHERE id=?", (delivery_id,))
        resolve_uncertain(self.store, delivery_id, "sent")
        deliver_pending(self.store, self.client)
        self.client.send_message.assert_not_called()
        with self.assertRaises(ValueError):
            resolve_uncertain(self.store, delivery_id, "retry")

    def test_only_explicit_floodwait_is_retried_with_a_bound(self):
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]
        self.client.send_message.side_effect = FloodWait(2)
        deliver_pending(self.store, self.client)
        self.assertEqual(self.row(delivery_id)["state"], "pending")
        self.assertIsNotNone(self.row(delivery_id)["retry_at"])
        deliver_pending(self.store, self.client)
        self.assertEqual(self.client.send_message.call_count, 1)
        for _ in range(2):
            self.store.execute("UPDATE deliveries SET retry_at=NULL WHERE id=?", (delivery_id,))
            deliver_pending(self.store, self.client)
        self.assertEqual(self.row(delivery_id)["state"], "failed")
        self.assertEqual(self.client.send_message.call_count, 3)

    def test_definite_rejection_does_not_stop_other_recipients(self):
        first = prepare_delivery(self.store, self.report(), 11)[0]
        second = prepare_delivery(self.store, self.report(), 12)[0]
        self.client.send_message.side_effect = [UserIsBlocked(), SimpleNamespace(id=2)]
        deliver_pending(self.store, self.client)
        self.assertEqual(self.row(first)["state"], "failed")
        self.assertEqual(self.row(second)["state"], "sent")

    def test_disabled_recipient_is_not_sent_queued_content(self):
        self.store.add_user(11, subscriptions='all')
        delivery_id = prepare_delivery(self.store, self.report(), 11)[0]
        self.store.add_user(11, enabled=False)
        deliver_pending(self.store, self.client)
        self.client.send_message.assert_not_called()
        self.assertEqual(self.row(delivery_id)["state"], "failed")

    def test_full_report_sends_artifacts_after_text(self):
        artifact = Path(__file__).resolve()
        report_id = self.report(parts=[{"kind": "document", "path": str(artifact), "caption": "fixture"}])
        ids = prepare_delivery(self.store, report_id, 11, detail="full")
        self.assertEqual(deliver_pending(self.store, self.client), 2)
        self.client.send_document.assert_called_once()
        self.assertTrue(all(self.row(delivery_id)["state"] == "sent" for delivery_id in ids))

    def test_missing_artifact_is_a_definite_local_failure(self):
        report_id = self.report(parts=[{"kind": "photo", "path": str(self.path.with_suffix('.missing'))}])
        ids = prepare_delivery(self.store, report_id, 11, detail="full")
        deliver_pending(self.store, self.client)
        self.client.send_photo.assert_not_called()
        self.assertEqual(self.row(ids[-1])["state"], "failed")

    def test_baseline_uses_only_recipient_fully_delivered_report(self):
        metric = lambda value: {"q": {"value": value, "metric": "query", "region_id": 1, "unit": "count"}}
        previous = self.report(metrics=metric(100))
        prepare_delivery(self.store, previous, 11)
        deliver_pending(self.store, self.client)
        # A newer undelivered report is not a valid baseline.
        prepare_delivery(self.store, self.report(metrics=metric(190)), 11)
        current = self.report(metrics=metric(150))
        own = prepare_delivery(self.store, current, 11)[0]
        other = prepare_delivery(self.store, current, 12)[0]
        text = json.loads(self.row(own)["payload"])["text"]
        self.assertIn("+50", text)
        self.assertNotIn("-40", text)
        self.assertNotIn("Изменения", json.loads(self.row(other)["payload"])["text"])

    def test_changed_cohort_is_marked_noncomparable(self):
        metrics = {"q": {"value": 100, "unit": "count"}}
        prepare_delivery(self.store, self.report(metrics=metrics, versions=(1, 1)), 11)
        deliver_pending(self.store, self.client)
        delivery_id = prepare_delivery(self.store, self.report(metrics=metrics, versions=(1, 2)), 11)[0]
        self.assertIn("изменился", json.loads(self.row(delivery_id)["payload"])["text"])

    def test_delivered_alert_does_not_hide_previous_numerical_baseline(self):
        previous = self.report(section="regions", metrics={"q:1": {"value": 100, "unit": "count"}})
        prepare_delivery(self.store, previous, 11)
        deliver_pending(self.store, self.client)
        alert = self.report(section="regions", text="Alert", metrics={})
        alert_id = prepare_delivery(self.store, alert, 11)[0]
        self.assertEqual(json.loads(self.row(alert_id)["payload"])["text"], "Alert")
        deliver_pending(self.store, self.client)
        current = self.report(section="regions", metrics={"q:1": {"value": 150, "unit": "count"}})
        delivery_id = prepare_delivery(self.store, current, 11)[0]
        self.assertIn("+50", json.loads(self.row(delivery_id)["payload"])["text"])

    def test_changed_personal_filters_fall_back_to_older_comparable_delivery(self):
        previous = self.report(metrics={"q": {"value": 100, "unit": "count"}})
        prepare_delivery(self.store, previous, 11)
        deliver_pending(self.store, self.client)
        different_filters = self.report(metrics={
            "q": {"value": 190, "unit": "count"}, "other": {"value": 10, "unit": "count"},
        })
        prepare_delivery(self.store, different_filters, 11)
        deliver_pending(self.store, self.client)
        current = self.report(metrics={"q": {"value": 150, "unit": "count"}})
        delivery_id = prepare_delivery(self.store, current, 11)[0]
        text = json.loads(self.row(delivery_id)["payload"])["text"]
        self.assertIn("+50", text)
        self.assertNotIn("-40", text)
        self.assertIn("сопоставимой", text)

    def test_baseline_skips_partial_delivery_other_recipient_and_incompatible_units_versions(self):
        metrics = {"q": {"value": 100, "unit": "count"}}
        prepare_delivery(self.store, self.report(metrics=metrics), 11)
        deliver_pending(self.store, self.client)
        incompatible = [
            self.report(metrics={"q": {"value": 190, "unit": "percent"}}),
            self.report(metrics={"q": {"value": 190, "unit": "count"}}, versions=(2, 1)),
        ]
        for report_id in incompatible:
            prepare_delivery(self.store, report_id, 11)
            deliver_pending(self.store, self.client)
        prepare_delivery(self.store, self.report(metrics={"q": {"value": 190, "unit": "count"}}), 12)
        deliver_pending(self.store, self.client)
        partial = self.report(text="x" * 5000, metrics={"q": {"value": 190, "unit": "count"}})
        partial_ids = prepare_delivery(self.store, partial, 11)
        self.store.execute("UPDATE deliveries SET state='sent' WHERE id=?", (partial_ids[0],))
        self.store.execute("UPDATE deliveries SET state='uncertain' WHERE id=?", (partial_ids[1],))
        current = self.report(metrics={"q": {"value": 150, "unit": "count"}})
        delivery_id = prepare_delivery(self.store, current, 11)[0]
        text = json.loads(self.row(delivery_id)["payload"])["text"]
        self.assertIn("+50", text)
        self.assertNotIn("-40", text)

    def test_changed_personal_filters_without_comparable_report_are_explicit(self):
        prepare_delivery(self.store, self.report(metrics={"old": {"value": 100, "unit": "count"}}), 11)
        deliver_pending(self.store, self.client)
        current = self.report(metrics={"new": {"value": 150, "unit": "count"}})
        delivery_id = prepare_delivery(self.store, current, 11)[0]
        text = json.loads(self.row(delivery_id)["payload"])["text"]
        self.assertIn("персональные фильтры", text)
        self.assertIn("Сравнение", text)

    def test_ai_brief_keeps_freshness_cache_and_partial_error_note(self):
        report_id = self.report(section="ai", news=[{
            "title": "Test update", "url": "https://example.test/story", "published_at": "2026-10-04",
        }], quality={"last_fetched_at": "2026-10-04T10:30:00+03:00", "cached": True,
                    "status": "partial", "errors": ["provider-sensitive-detail"]})
        delivery_id = prepare_delivery(self.store, report_id, 11)[0]
        text = json.loads(self.row(delivery_id)["payload"])["text"]
        self.assertIn("Новое: Test update", text)
        self.assertIn("04.10.2026 07:30 UTC", text)
        self.assertIn("сохранённые данные", text)
        self.assertIn("обновление неполное (ошибок: 1)", text)
        self.assertNotIn("provider-sensitive-detail", text)


if __name__ == "__main__":
    unittest.main()
