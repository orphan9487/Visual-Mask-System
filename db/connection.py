import pymysql

_DB_CONF = dict(
    host="127.0.0.1",
    user="root",
    password="",
    database="mcot_chat_db",
    charset="utf8mb4",
)


def get_connection() -> pymysql.connections.Connection:
    return pymysql.connect(**_DB_CONF)
