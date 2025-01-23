from dataclasses import dataclass


@dataclass
class Column:
    """
    Represents a database column.
    """

    table_name: str
    column_name: str
    column_type: str

    def __post_init__(self):
        """
        Classify column type as 'text', 'boolean' or 'number'. Any type other than 'number' and 'boolean' is considered 'text'.
        """
        if self.column_type not in ["boolean", "number"]:
            self.column_type = "text"
