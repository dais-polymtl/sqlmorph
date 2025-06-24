from dataclasses import dataclass
from column import Column


@dataclass
class ForeignKey:
    """
    Represents a foreign key constraint.
    """

    referencing_table: str
    referenced_table: str
    column: Column
    referenced_column: Column
