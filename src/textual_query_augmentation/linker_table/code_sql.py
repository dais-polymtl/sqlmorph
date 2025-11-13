import sqlite3


db_id = "european_football_2"
# Path to your SQLite database file
db_path = f"data/benchmarks/Bird/bird_databases/{db_id}/{db_id}.sqlite"

# SQL query
query = """
SELECT DISTINCT t4.name
FROM Player_Attributes AS t1
INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id
INNER JOIN Match AS t3 ON t2.player_api_id = t3.home_player_8
INNER JOIN Country AS t4 ON t3.country_id = t4.id
INNER JOIN Team_Attributes AS t5 
    ON t5.team_api_id = t3.home_team_api_id 
    AND t5.team_fifa_api_id = t3.away_team_api_id 
    AND t5.team_fifa_api_id = t3.home_team_api_id
WHERE t1.vision > 89
"""

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Executing query on database:", db_id)
# Execute the query
cursor.execute(query)

# Fetch all results
results = cursor.fetchall()
print("Query executed successfully. Results:", len(results))
# Print results
for row in results:
    print(row[0])

# Close connection
conn.close()
