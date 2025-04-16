import enum

from src.core.database.adapters import SQLiteAdapter, DuckDBAdapter
from src.core.logger import Logger

logger = Logger(__name__)


class DBMS(enum.Enum):
    SQLITE = "sqlite"
    DUCKDB = "duckdb"


ADAPTERS = {
    DBMS.SQLITE: SQLiteAdapter,
    DBMS.DUCKDB: DuckDBAdapter,
}


class DatabaseHandler:
    def __init__(self, dbms: DBMS, connection_params: dict):
        """
        Args:
            dbms (DBMS): A DBMS enum member.
            connection_params (dict): Connection parameters required for the given DB.
        """
        self.dbms = dbms
        self.adapter_class = ADAPTERS.get(dbms)

        if not self.adapter_class:
            logger.log(
                "error", "UNSUPPORTED_DBMS_BACKEND", {"DBMS_BACKEND": dbms.value}
            )
            raise ValueError(f"Unsupported database: {dbms.value}")

        # Instantiate the appropriate adapter and connect immediately
        self.adapter = self.adapter_class(connection_params)
        self.connect_to_database()

    def connect_to_database(self):
        """Establish the connection using the selected adapter."""
        try:
            self.adapter.connect()
        except Exception as e:
            logger.log(
                "error",
                "CONNECTION_FAILED",
                {"DBMS_BACKEND": self.dbms.value, "error": str(e)},
            )
            raise e

    def is_connection_alive(self):
        """
        Checks if the database connection is alive.
        """
        try:
            # For demonstration, attempt a trivial query:
            test_query = "SELECT 1"
            result = self.run_query(test_query)
            return result is not None
        except Exception as e:
            logger.log("error", "CONNECTION_CHECK_FAILED", {"error": str(e)})
            return False

    def run_query(self, query, return_cursor=False):
        """Run a query through the selected adapter."""
        return self.adapter.run_query(query, return_cursor)

    def close_connection(self):
        """Close the connection."""
        if self.adapter:
            self.adapter.close_connection()
        else:
            logger.log("warning", "CLOSE_ATTEMPTED_WITH_NO_CONNECTION")
