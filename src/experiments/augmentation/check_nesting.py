from sqlglot import parse_one
from sqlglot.expressions import Select, Subquery, Exists


def is_nested(sql):
    tree = parse_one(sql)
    return any(
        isinstance(node, (Select, Subquery, Exists)) and node != tree
        for node in tree.walk()
    )


sql = """
SELECT superhero_name
FROM superhero AS T1
INNER JOIN alignment AS et ON et.id = T1.alignment_id
WHERE EXISTS (
    SELECT 1
    FROM hero_power AS T2
    INNER JOIN superpower AS T3 ON T2.power_id = T3.id
    WHERE T3.power_name = 'Super Strength' AND T1.id = T2.hero_id
)
AND EXISTS (
    SELECT 1
    FROM publisher AS T4
    WHERE T4.publisher_name = 'Marvel Comics' AND T1.publisher_id = T4.id
)
"""

print("Is nested:", is_nested(sql))

tree = parse_one(sql)
print("Subquery found:", tree.find(Subquery))
print("Exists found:", tree.find(Exists))
