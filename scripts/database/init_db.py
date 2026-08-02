"""Create the Visual Mask MySQL schema and local demonstration accounts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.database import database_settings


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def initialize_database() -> None:
    settings = database_settings
    try:
        connection = pymysql.connect(
            **settings.connection_kwargs(include_database=False),
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as exc:
        raise SystemExit(
            "無法連線 MySQL；請確認服務已啟動且 .env 的 VMS_DB_* 設定正確："
            f"{exc}"
        ) from exc

    try:
        with connection.cursor() as cursor:
            database = settings.database
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{database}`")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    account VARCHAR(50) UNIQUE NOT NULL,
                    username VARCHAR(50) NOT NULL,
                    password_hash VARCHAR(64) NOT NULL,
                    emotion_prior TEXT NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_loras (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    lora_key VARCHAR(50) NOT NULL,
                    display_name VARCHAR(100) NOT NULL,
                    is_active TINYINT(1) NOT NULL DEFAULT 0,
                    UNIQUE KEY unique_user_lora (username, lora_key)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    context TEXT,
                    wrong_label VARCHAR(50) NOT NULL,
                    correct_label VARCHAR(50) NOT NULL,
                    correction_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_td_username (username),
                    INDEX idx_td_time (correction_time)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intensity_adjustments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    emotion VARCHAR(50) NOT NULL,
                    adjusted_intensity FLOAT NOT NULL,
                    base_intensity FLOAT NOT NULL,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ia_user_emotion (username, emotion),
                    INDEX idx_ia_time (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            users = [
                ("alice_01", "Alice", "123456", "A gentle, empathetic person."),
                ("bob_99", "Bob", "123456", "A critical, occasionally sarcastic person."),
                ("charlie_x", "Charlie", "123456", "A direct, logical communicator."),
            ]
            for account, username, password, prior in users:
                cursor.execute(
                    """INSERT IGNORE INTO users
                       (account, username, password_hash, emotion_prior)
                       VALUES (%s, %s, %s, %s)""",
                    (account, username, hash_password(password), prior),
                )

            user_loras = [
                ("Alice", "human_8692", "Person 8692", 1),
                ("Bob", "human_8692", "Person 8692", 1),
                ("Bob", "ethan", "Ethan", 0),
                ("Charlie", "human_8692", "Person 8692", 1),
                ("Charlie", "yourname", "Yourname", 0),
            ]
            for row in user_loras:
                cursor.execute(
                    """INSERT IGNORE INTO user_loras
                       (username, lora_key, display_name, is_active)
                       VALUES (%s, %s, %s, %s)""",
                    row,
                )

        connection.commit()
    finally:
        connection.close()

    print(f"MySQL database '{settings.database}' initialized successfully.")


if __name__ == "__main__":
    initialize_database()
