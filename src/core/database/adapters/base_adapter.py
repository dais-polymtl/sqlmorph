import abc


class BaseAdapter(abc.ABC):
    """
    Abstract base class for database adapters.
    Each adapter must implement connect, run_query, and close_connection.
    """

    @abc.abstractmethod
    def connect(self):
        """Establish the database connection."""
        pass

    @abc.abstractmethod
    def run_query(self, query, return_cursor=False):
        """
        Execute a query and optionally return a cursor or the raw results.
        """
        pass

    @abc.abstractmethod
    def close_connection(self):
        """Close the database connection."""
        pass
