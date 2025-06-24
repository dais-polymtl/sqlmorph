from typing import List
from dataclasses import dataclass
from column import Column


@dataclass
class Table:
    """
    Represents a database table.
    """

    table_name: str
    columns: List[Column]
    primary_keys: List[Column]
