import sqlite3
from .logger import Logger

logger = Logger(__name__)

class QueryExecutor:
    """
    Abstract base class for different SQL dialect executors.
    """

    def execute_query(self, query: str):
        """
        Execute a query and return (column_names, rows).

        column_names: list of str
        rows: list of tuples (each tuple is a row)
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SQLiteQueryExecutor(QueryExecutor):
    """
    Concrete implementation for SQLite.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def execute_query(self, query: str):
        """
        Execute a query against the SQLite database.
        Returns (column_names, rows).
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        except Exception as e:
            logger.log("error", "FAILED_TO_EXECUTE_QUERY", {"QUERY": query, "ERROR": str(e)})
            column_names, rows = [], []
        finally:
            cursor.close()
        return column_names, rows
