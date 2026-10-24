"""Sequential worker used by the one-shot CLI and the persistent Telegram service."""
import json
import logging
from uuid import uuid4

from .collectors import Collector
from .reporting import ReportBuilder, record_alerts, report_filters
from .store import SECTIONS, dumps, utcnow

logger = logging.getLogger(__name__)


def bootstrap(store, settings):
    from .market import initialize
    from .scheduling import seed_subscriptions
    initialize(store)
    store.set_setting('protected_admin_ids', list(settings.admin_ids))
    for user_id in settings.admin_ids:
        store.add_user(user_id, role="admin")
        seed_subscriptions(store, str(user_id), user_id)
    if not store.get_setting("legacy_recipients_imported", False):
        for recipient in settings.recipients:
            # A delivery address is not permission to control the bot.
            seed_subscriptions(store, recipient)
        store.set_setting("legacy_recipients", list(settings.recipients))
        store.set_setting("legacy_recipients_imported", True)


class Worker:
    def __init__(self, store, settings, providers=None, *, dry_run=False, charts=True, now=None):
        self.store, self.settings = store, settings
        from .integrity import health
        if not health(store):
            raise RuntimeError('SQLite integrity check failed; collection is stopped')
        self.dry_run, self.charts = dry_run, charts
        self.collector = Collector(store, settings, providers, now)
        self.builder = ReportBuilder(store, settings, now)

    def recover(self):
        # Only call after obtaining the process lock. Finished reports are reused
        # and uncertain Telegram sends are never automatically replayed.
        from .delivery import recover_sending
        recover_sending(self.store)
        from .user_management import recover_notices
        recover_notices(self.store)
        self.store.execute("UPDATE jobs SET state='pending',started_at=NULL WHERE state='running'")
        self.store.execute("UPDATE runs SET status='interrupted',completed_at=? WHERE status='running'", (utcnow(),))

    def process_next(self):
        with self.store.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return False
            job = dict(row)
            conn.execute("UPDATE jobs SET state='running',started_at=? WHERE id=?", (utcnow(), job["id"]))
        run_id = uuid4().hex
        try:
            payload = json.loads(job["payload"])
            from .subscription_access import allowed, job_section
            actor = payload.get('actor_id')
            if actor is None and payload.get('recipient'):
                user = self.store.one('SELECT user_id FROM users WHERE chat_id=?', (str(payload['recipient']),))
                actor = user['user_id'] if user else None
            section_access = job_section(job['kind'], payload)
            from .feature_access import allowed as feature_allowed
            from .feature_access import job_feature
            feature = job_feature(job['kind'], payload)
            features = [feature] if feature else []
            features.extend(payload.get('required_features', []))
            if 'subscription_id' in payload or 'chart_schedule' in payload:
                features.append('schedules')
            if actor is not None and self.store.authorize(actor) and any(not feature_allowed(self.store, actor, name) for name in features):
                self.store.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='feature_access_revoked' WHERE id=?", (utcnow(), job['id']))
                return True
            if actor is not None and self.store.authorize(actor) and section_access and not allowed(self.store, actor, section_access):
                self.store.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='section_access_revoked' WHERE id=?", (utcnow(), job['id']))
                return True
            from .search_provider import disabled, search_job
            if disabled(self.store) and search_job(job['kind'], payload):
                self.store.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='search_disabled' WHERE id=?", (utcnow(), job['id']))
                return True
            if job['kind'] == 'integrity_audit':
                from .integrity import run
                run(self.store, payload, dry_run=self.dry_run)
            elif job['kind'] in {'dollar_analysis', 'dollar_collect'}:
                from .dollar_jobs import run_job
                run_job(self.store, self.settings, job, payload, dry_run=self.dry_run, charts=self.charts)
            elif job["kind"] == "verify_sources":
                if "actor_id" in payload:
                    from .telegram_data_jobs import authorized_report, require_actor
                    require_actor(self.store, payload["actor_id"], admin=True)
                    authorized_report(self.store, payload["actor_id"], payload["report_id"])
                self._verify_sources(payload["report_id"])
            elif job['kind'] == 'trends_collect':
                from .trends_api import run_job
                run_job(self.store, job, payload, dry_run=self.dry_run)
            elif job['kind'] == 'trends_import':
                from .google_trends import run_import_job
                run_import_job(self.store, job, payload, dry_run=self.dry_run)
            elif job['kind'] == 'vivino_map':
                from .vivino import run_job
                run_job(self.store, self.settings, job, payload, dry_run=self.dry_run)
            elif job['kind'].startswith('wine_'):
                from .wine_jobs import run_job
                run_job(self.store, self.settings, job, payload, dry_run=self.dry_run)
            elif job['kind'].startswith('life_'):
                from .life_jobs import run_job
                run_job(self.store, self.settings, job, payload, dry_run=self.dry_run)
            elif job["kind"] == "telegram_chart":
                from .charts import run_chart_job
                run_chart_job(self.store, self.settings, job, payload, dry_run=self.dry_run)
            elif job["kind"] in {"telegram_export", "telegram_backtest"}:
                from .telegram_data_jobs import run_data_job
                run_data_job(self.store, self.settings, job, payload, dry_run=self.dry_run)
            elif job["kind"] == "propose_keywords":
                if "actor_id" in payload:
                    from .telegram_data_jobs import require_actor
                    require_actor(self.store, payload["actor_id"], admin=True)
                if self.dry_run:
                    raise ValueError("Keyword proposals require an explicitly enabled AI call")
                entries = self.collector.providers.propose_keywords(self.store.keywords())
                self.store.execute("INSERT INTO keyword_proposals(created_at,entries) VALUES(?,?)", (utcnow(), dumps(entries)))
            elif job['kind'].startswith('market_') or (job['kind'] == 'report' and payload.get('section') == 'market'):
                from .market_jobs import run_market_job
                run_market_job(self.store, self.settings, job, payload, dry_run=self.dry_run, charts=self.charts)
            elif job["kind"] == "report":
                recipient = payload.get("recipient")
                section = payload.get("section", "ruonia")
                if "subscription_id" in payload:
                    from .delivery import subscription_active
                    if not subscription_active(self.store, payload["subscription_id"], recipient, section):
                        self.store.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='subscription_inactive' WHERE id=?",
                                           (utcnow(), job["id"]))
                        return True
                user = self.store.one("SELECT enabled FROM users WHERE chat_id=?", (str(recipient),)) if recipient else None
                if user and not user["enabled"]:
                    raise PermissionError("Recipient is disabled")
                if section not in SECTIONS:
                    raise ValueError("Unknown report section")
                report_filters(self.store, section, payload.get("phrases"), payload.get("regions"))
                if payload.get("detail", "brief") not in {"brief", "full"}:
                    raise ValueError("Invalid report detail")
                self.store.execute("INSERT INTO runs(id,section,status,started_at,parameters) VALUES(?,?,'running',?,?)",
                                   (run_id, section, utcnow(), dumps(payload)))
                metadata = self.collector.collect(section, dry_run=self.dry_run)
                report_id = self.builder.build(section, metadata, phrases=payload.get("phrases"),
                    regions=payload.get("regions"), run_id=run_id, charts=self.charts)
                self.store.execute("UPDATE runs SET status=?,completed_at=?,quality=? WHERE id=?",
                                   (metadata["status"], utcnow(), dumps(metadata), run_id))
                # Collection/rendering can take long enough for a user to
                # disable this schedule while its job is already running.
                if "subscription_id" in payload and not subscription_active(self.store, payload["subscription_id"], recipient, section):
                    self.store.execute("UPDATE jobs SET state='cancelled',finished_at=?,error='subscription_inactive' WHERE id=?",
                                       (utcnow(), job["id"]))
                    return True
                if recipient and not self.dry_run:
                    from .delivery import prepare_delivery
                    prepare_delivery(self.store, report_id, str(recipient), detail=payload.get("detail", "brief"),
                                     filters={"phrases": payload.get("phrases", []), "regions": payload.get("regions", [])},
                                     subscription_id=payload.get("subscription_id"))
                if section == "regions" and not self.dry_run:
                    self._alerts()
                if section == 'ruonia' and not self.dry_run:
                    from .market_alerts import process
                    process(self.store, self.settings)
            else:
                raise ValueError("Unknown job type")
            self.store.execute("UPDATE jobs SET state='done',finished_at=?,error=NULL WHERE id=?", (utcnow(), job["id"]))
        except Exception as exc:  # noqa: BLE001 - Persist failed jobs without exposing provider response details.
            # Provider errors and payloads are never reflected into the dashboard.
            safe_error = f"Не удалось выполнить задание ({type(exc).__name__})"
            logger.error("Job %s failed (%s)", job["id"], type(exc).__name__)
            self.store.execute("UPDATE jobs SET state='failed',finished_at=?,error=? WHERE id=?", (utcnow(), safe_error, job["id"]))
            self.store.execute("UPDATE runs SET status='failed',completed_at=? WHERE id=?", (utcnow(), run_id))
        return True

    def drain(self):
        count = 0
        while self.process_next():
            count += 1
        return count

    def _alerts(self):
        from .delivery import prepare_delivery
        record_alerts(self.store)
        for pending in self.store.query("SELECT * FROM alerts WHERE dispatched=0"):
            alert = json.loads(pending["payload"])
            report_id = pending["report_id"] or uuid4().hex
            text = (f"Изменение поискового интереса: {alert['metric']}\nРегион: {alert['region_id']}\n"
                    f"Сумма за неделю: {alert['current_total']:.0f}\n"
                    f"Покрытие: {alert['coverage']:.0%}\n"
                    "Это сигнал поискового интереса, не подтверждение события.")
            if not pending["report_id"]:
                with self.store.connection() as conn:
                    conn.execute("""INSERT INTO reports(id,section,created_at,keyword_version,cohort_version,text,full_text,metrics,quality)
                        VALUES(?,'regions',?,?,?,?,?,?,?)""", (report_id, utcnow(), self.store.version("keyword"),
                        self.store.version("cohort"), text, text, dumps({}), dumps({"alert_id": pending["id"]})))
                    conn.execute("UPDATE alerts SET report_id=? WHERE id=?", (report_id, pending["id"]))
            subscriptions = self.store.query("SELECT * FROM subscriptions WHERE section='regions' AND enabled=1")
            for subscription in subscriptions:
                phrases, regions = json.loads(subscription["phrases"]), json.loads(subscription["regions"])
                if phrases and alert["metric"] not in phrases:
                    continue
                if regions and int(alert["region_id"]) not in regions:
                    continue
                prepare_delivery(self.store, report_id, subscription["recipient"], detail="brief",
                                 subscription_id=subscription["id"])
            self.store.execute("UPDATE alerts SET dispatched=1 WHERE id=?", (pending["id"],))

    def _verify_sources(self, report_id):
        from urllib.parse import urlsplit

        from functions.ai import RU_DOMAINS

        from .analysis import safe_probe_url
        report = self.store.one("SELECT news FROM reports WHERE id=?", (report_id,))
        if not report:
            raise ValueError("Report not found")
        news = json.loads(report["news"])
        for index, item in enumerate(news):
            host = (urlsplit(item.get("url") or "").hostname or "").lower()
            approved = any(host == domain or host.endswith("." + domain) for domain in RU_DOMAINS)
            if approved and index < 10:
                item["source_availability"] = safe_probe_url(item["url"], enabled=not self.dry_run)
            else:
                item["source_availability"] = {"availability": "not_checked", "reason": "domain_or_batch_limit", "claim_verified": False}
            item["claim_verified"] = False
            item["status"] = "numbers_checked" if item.get("numbers_verified") else "unverified"
            self.store.execute("UPDATE news SET payload=?,status=? WHERE report_id=? AND fingerprint=?",
                               (dumps(item), item["status"], report_id, item.get("fingerprint", "")))
        self.store.execute("UPDATE reports SET news=? WHERE id=?", (dumps(news), report_id))


def status(store):
    return {"jobs": store.query("SELECT state,COUNT(*) AS count FROM jobs GROUP BY state"),
            "deliveries": store.query("SELECT state,COUNT(*) AS count FROM deliveries GROUP BY state"),
            "observations": store.one("SELECT COUNT(*) AS count,MAX(date) AS latest_date FROM observations"),
            "reports": store.one("SELECT COUNT(*) AS count,MAX(created_at) AS latest FROM reports"),
            "usage": store.query("SELECT provider,day,SUM(calls) AS calls,SUM(tokens_in) AS tokens_in,SUM(tokens_out) AS tokens_out,SUM(cost) AS cost FROM usage GROUP BY provider,day ORDER BY day DESC LIMIT 10"),
            "budgets": store.get_setting("budgets"), "keyword_version": store.version("keyword"),
            "cohort_version": store.version("cohort"), "region_count": len(store.regions())}
