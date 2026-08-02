import pymysql

from src.config.database import database_settings


def get_connection() -> pymysql.connections.Connection:
    return pymysql.connect(**database_settings.connection_kwargs())
