"""Nonsecret defaults and local environment configuration."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """A fixed, user-facing configuration message containing no supplied values."""


@dataclass
class Settings:
    root: Path = ROOT
    data_dir: Path | None = None
    reports_dir: Path | None = None
    dashboard_password: str = field(default="", repr=False)
    dashboard_enabled: bool = False
    admin_ids: tuple[int, ...] = ()
    recipients: tuple[str, ...] = ()
    search_provider: str = 'legacy'

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        data_dir = Path(self.data_dir or "data")
        reports_dir = Path(self.reports_dir or Path("src") / "reports")
        self.data_dir = (data_dir if data_dir.is_absolute() else self.root / data_dir).resolve()
        self.reports_dir = (reports_dir if reports_dir.is_absolute() else self.root / reports_dir).resolve()

    @classmethod
    def from_environment(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        raw_admins = os.getenv("ADMIN_TELEGRAM_IDS", "")
        try:
            admins = tuple(int(x.strip()) for x in raw_admins.split(",") if x.strip())
        except ValueError:
            raise ConfigurationError("ADMIN_TELEGRAM_IDS: укажите числовые Telegram ID через запятую в .env") from None
        if any(x <= 0 for x in admins):
            raise ConfigurationError("ADMIN_TELEGRAM_IDS: нужны положительные числовые Telegram ID в .env")
        dashboard_flag = os.getenv("DASHBOARD_ENABLED", "0").strip().lower()
        if dashboard_flag not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ConfigurationError("DASHBOARD_ENABLED: укажите 0 или 1 в .env")
        search_provider = os.getenv('SEARCH_PROVIDER', 'disabled').strip().lower()
        if search_provider not in {'disabled', 'google_trends', 'legacy'}:
            raise ConfigurationError('SEARCH_PROVIDER: укажите disabled, google_trends или legacy в .env')
        return cls(
            data_dir=Path(os.environ["THESAURUS_DATA_DIR"]) if os.getenv("THESAURUS_DATA_DIR") else None,
            dashboard_password=os.getenv("DASHBOARD_PASSWORD", ""), admin_ids=admins,
            dashboard_enabled=dashboard_flag in {"1", "true", "yes", "on"},
            recipients=tuple(x.strip() for x in os.getenv("for_whom_list", "").split(",") if x.strip()),
            search_provider=search_provider,
        )
