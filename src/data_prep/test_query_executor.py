import sqlite3

# Connect to the SQLite database (replace with your database path)
db_id = "european_football_2"
conn = sqlite3.connect(f'dev_databases/{db_id}/{db_id}.sqlite')  # Change the database path as needed

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Your SQL query
new_query = "SELECT COUNT(t2.id) FROM Country AS t1 INNER JOIN Match AS t2 ON t1.id = t2.country_id INNER JOIN League AS extra_table_1 ON extra_table_1.country_id = t1.id INNER JOIN League AS t2 ON t2.league_id = extra_table_2.id WHERE t1.name = 'Belgium' AND t2.season = '2008/2009'"

try:
    # Execute the query
    cursor.execute(new_query)
    
    # Fetch all the results (you can also use fetchone() for a single result)
    results = cursor.fetchall()
    
    # Process the results
    for row in results:
        print(row)  # or process the data as needed
    
except sqlite3.Error as e:
    print(f"An error occurred: {e}")

# Close the connection
conn.close()
