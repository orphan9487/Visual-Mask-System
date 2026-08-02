"""Environment-backed MySQL configuration shared by the app and setup script."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VALID_DATABASE_NAME = re.compile(r"^[A-Za-z0-9_]+$")

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        port_text = os.getenv("VMS_DB_PORT", "3306")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError("VMS_DB_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("VMS_DB_PORT must be between 1 and 65535")

        database = os.getenv("VMS_DB_NAME", "mcot_chat_db").strip()
        if not _VALID_DATABASE_NAME.fullmatch(database):
            raise ValueError("VMS_DB_NAME may contain only letters, numbers, and underscores")

        return cls(
            host=os.getenv("VMS_DB_HOST", "127.0.0.1").strip(),
            port=port,
            user=os.getenv("VMS_DB_USER", "root").strip(),
            password=os.getenv("VMS_DB_PASSWORD", ""),
            database=database,
        )

    def connection_kwargs(self, *, include_database: bool = True) -> dict:
        values = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "charset": "utf8mb4",
        }
        if include_database:
            values["database"] = self.database
        return values


database_settings = DatabaseSettings.from_env()
