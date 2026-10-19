"""Create the Visual Mask MySQL schema and local demonstration accounts."""

from __future__ import annotations

import sys
from pathlib import Path

import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.database import database_settings
from db.user_repo import hash_password


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
                    password_hash VARCHAR(255) NOT NULL,
                    emotion_prior TEXT NOT NULL,
                    face_profile_status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    default_mask_id VARCHAR(100) NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            # Safe migrations for databases created by older project versions.
            for statement in [
                "ALTER TABLE users MODIFY password_hash VARCHAR(255) NOT NULL",
                "ALTER TABLE users ADD COLUMN face_profile_status VARCHAR(20) NOT NULL DEFAULT 'pending'",
                "ALTER TABLE users ADD COLUMN default_mask_id VARCHAR(100) NULL",
                "ALTER TABLE users ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
            ]:
                try:
                    cursor.execute(statement)
                except pymysql.MySQLError:
                    pass
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
                CREATE TABLE IF NOT EXISTS face_profiles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL UNIQUE,
                    mask_id VARCHAR(100) NOT NULL UNIQUE,
                    reference_path TEXT NOT NULL,
                    crop_json TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'ready',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_fp_mask (mask_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_appearance_profiles (
                    user_id INT PRIMARY KEY,
                    gender_presentation VARCHAR(20) NOT NULL DEFAULT 'unspecified',
                    age_group VARCHAR(20) NOT NULL DEFAULT 'adult',
                    hair_length VARCHAR(20) NOT NULL DEFAULT 'unspecified',
                    hair_color VARCHAR(20) NOT NULL DEFAULT 'unspecified',
                    hair_style VARCHAR(20) NOT NULL DEFAULT 'unspecified',
                    glasses TINYINT(1) NOT NULL DEFAULT 0,
                    facial_hair VARCHAR(20) NOT NULL DEFAULT 'unspecified',
                    framing VARCHAR(30) NOT NULL DEFAULT 'face_only',
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_rooms (
                    id CHAR(32) PRIMARY KEY,
                    kind VARCHAR(20) NOT NULL DEFAULT 'direct',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_room_members (
                    room_id CHAR(32) NOT NULL,
                    user_id INT NOT NULL,
                    PRIMARY KEY (room_id, user_id),
                    INDEX idx_crm_user (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id CHAR(32) PRIMARY KEY,
                    room_id CHAR(32) NOT NULL,
                    sender_id INT NOT NULL,
                    text TEXT NOT NULL,
                    emotion VARCHAR(50) NULL,
                    mask_file TEXT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_cm_room_time (room_id, created_at)
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
                CREATE TABLE IF NOT EXISTS user_training_photos (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    mask_name VARCHAR(100) NOT NULL DEFAULT '未命名面具',
                    path TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'uploaded',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_utp_user (user_id),
                    INDEX idx_utp_status (status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            try:
                cursor.execute("ALTER TABLE user_training_photos ADD COLUMN mask_name VARCHAR(100) NOT NULL DEFAULT '未命名面具'")
            except pymysql.MySQLError:
                pass
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

            # ``user_loras`` is a legacy table name.  It now stores either a
            # FaceID profile or a LoRA identity.  The restored WebSocket demo
            # defaults every built-in account to Henry FaceID while retaining
            # the former LoRA identities as selectable fallbacks.
            user_loras = [
                ("Alice", "henry", "Henry FaceID", 1),
                ("Alice", "human_8692", "Person 8692 (LoRA)", 0),
                ("Bob", "henry", "Henry FaceID", 1),
                ("Bob", "human_8692", "Person 8692 (LoRA)", 0),
                ("Bob", "ethan", "Ethan", 0),
                ("Charlie", "henry", "Henry FaceID", 1),
                ("Charlie", "human_8692", "Person 8692 (LoRA)", 0),
                ("Charlie", "yourname", "Yourname", 0),
            ]
            demo_usernames = sorted({row[0] for row in user_loras})
            placeholders = ", ".join(["%s"] * len(demo_usernames))
            cursor.execute(
                f"UPDATE user_loras SET is_active = 0 "
                f"WHERE username IN ({placeholders})",
                tuple(demo_usernames),
            )
            for row in user_loras:
                cursor.execute(
                    """INSERT INTO user_loras
                       (username, lora_key, display_name, is_active)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           display_name = VALUES(display_name),
                           is_active = VALUES(is_active)""",
                    row,
                )

        connection.commit()
    finally:
        connection.close()

    print(f"MySQL database '{settings.database}' initialized successfully.")


if __name__ == "__main__":
    initialize_database()
