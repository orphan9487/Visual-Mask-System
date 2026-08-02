"""Initialize the legacy MySQL schema and demonstration users."""

import pymysql
import hashlib


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# 1. 先不指定 database，建立資料庫後再切換
try:
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print("✅ 成功連線至 MySQL 伺服器！")
except Exception as e:
    print(f"❌ 連線失敗，請確認 XAMPP 的 MySQL 是否已啟動：{e}")
    exit()

cursor = conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS mcot_chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
cursor.execute("USE mcot_chat_db;")
print("✅ 資料庫 mcot_chat_db 已就緒")

# 2. 建立 users 資料表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    account VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(50) NOT NULL,
    password_hash VARCHAR(64) NOT NULL,
    emotion_prior TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")

# 3. 準備測試資料
test_users = [
    ("alice_01", "Alice", "123456",
     "A gentle, empathetic person who always supports others."),
    ("bob_99", "Bob", "123456",
     "A highly critical and cynical person, occasionally sarcastic."),
    ("charlie_x", "Charlie", "123456",
     "A direct, logical, and emotionless communicator.")
]

# 4. 寫入資料
for account, username, password, prior in test_users:
    try:
        cursor.execute(
            "INSERT INTO users (account, username, password_hash, emotion_prior) VALUES (%s, %s, %s, %s)",
            (account, username, hash_password(password), prior)
        )
        print(f"✅ 成功新增使用者: {username}")
    except pymysql.err.IntegrityError:
        print(f"⚠️ 帳號 {account} 已存在，略過新增。")

# 5. 建立 user_loras 資料表
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_loras (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    username     VARCHAR(50)  NOT NULL,
    lora_key     VARCHAR(50)  NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    is_active    TINYINT(1)   NOT NULL DEFAULT 0,
    UNIQUE KEY unique_user_lora (username, lora_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")
print("✅ user_loras 資料表已就緒")

# 6. 每位使用者的初始 LoRA 配置
#    格式：(username, lora_key, display_name, is_active)
#    is_active=1 代表目前選用中（每人只能有一筆）
user_loras_data = [
    # Alice：只有 human_8692（預設）
    ("Alice",   "human_8692", "Person 8692", 1),

    # Bob：有 human_8692 與 ethan，預設用 human_8692
    ("Bob",     "human_8692", "Person 8692", 1),
    ("Bob",     "ethan",      "Ethan",       0),

    # Charlie：有 human_8692 與 yourname，預設用 human_8692
    ("Charlie", "human_8692", "Person 8692", 1),
    ("Charlie", "yourname",   "Yourname",    0),
]

for username, lora_key, display_name, is_active in user_loras_data:
    try:
        cursor.execute(
            """INSERT INTO user_loras (username, lora_key, display_name, is_active)
               VALUES (%s, %s, %s, %s)""",
            (username, lora_key, display_name, is_active)
        )
        status = "（使用中）" if is_active else ""
        print(f"✅ {username} ← {lora_key}{status}")
    except pymysql.err.IntegrityError:
        print(f"⚠️ {username}/{lora_key} 已存在，略過。")

# 7. 建立 training_data 資料表（情緒反饋 / 訓練資料收集）
cursor.execute("""
CREATE TABLE IF NOT EXISTS training_data (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    username       VARCHAR(50)  NOT NULL,
    context        TEXT,
    wrong_label    VARCHAR(50)  NOT NULL,
    correct_label  VARCHAR(50)  NOT NULL,
    correction_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_td_username (username),
    INDEX idx_td_time     (correction_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")
print("✅ training_data 資料表已就緒")

# 8. 建立 intensity_adjustments 資料表（強度反饋 / 自動乘數計算）
cursor.execute("""
CREATE TABLE IF NOT EXISTS intensity_adjustments (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    username            VARCHAR(50) NOT NULL,
    emotion             VARCHAR(50) NOT NULL,
    adjusted_intensity  FLOAT NOT NULL,
    base_intensity      FLOAT NOT NULL,
    timestamp           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_ia_user_emotion (username, emotion),
    INDEX idx_ia_time         (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")
print("✅ intensity_adjustments 資料表已就緒")

conn.commit()
conn.close()
print("🎉 MySQL 資料表與測試資料初始化完成！")
