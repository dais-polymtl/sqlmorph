import sqlite3

from src.core.database.adapters.base_adapter import BaseAdapter
from src.core.logger import Logger

logger = Logger(__name__)


class SQLiteAdapter(BaseAdapter):
    def __init__(self, connection_params: dict):
        """
        connection_params might look like:
        {
            "db_path": "/absolute/path/to/sqlite.db"
        }
        """
        self.connection = None
        self.db_path = connection_params.get("db_path")

    def connect(self):
        """Connect to SQLite using the file path."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.log("info", "CONNECTION_ESTABLISHED", {"db_path": self.db_path})
        except Exception as e:
            logger.log("error", "CONNECTION_FAILED", {"error": str(e)})
            raise e

    def run_query(self, query, return_cursor=False):
        """Execute a query. If return_cursor is True, return the cursor; else return (column_names, rows)"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            logger.log("info", "QUERY_EXECUTED", {"query": " ".join(query.split())})

            if return_cursor:
                return cursor

            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return column_names, rows
        except Exception as e:
            logger.log(
                "error",
                "QUERY_EXECUTION_FAILED",
                {"query": " ".join(query.split()), "error": str(e)},
            )
            return e

    def close_connection(self):
        """Close the SQLite connection."""
        if self.connection:
            self.connection.close()
            logger.log("info", "CONNECTION_CLOSED", {"db_path": self.db_path})
        else:
            logger.log("warning", "CLOSE_ATTEMPTED_WITH_NO_CONNECTION")
