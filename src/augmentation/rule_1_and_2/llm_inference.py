import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_explicit_question(main_query, schema):
    """
    Generate an explicit, structured question based on SQL join relations.
    """

    prompt = f"""Your goal is to generate a natural language question for the given SQL query that involves the specified join relations. Ensure the question matches the format of the examples provided. The schema is provided as well for context:

    # Schema
    {schema}
    # Examples:
    SQL query: SELECT DISTINCT account.account_id FROM account, card, client, loan, 'order', disp WHERE loan.account_id = account.account_id AND disp.account_id = account.account_id AND account.district_id = client.district_id AND 'order'.account_id = account.account_id AND card.disp_id = disp.disp_id AND client.gender = 'M'
    Question: Construct a SQL query involving the following tables: client, loan, account, disp, card, and order. Use these join relations: 1. loan → account: Join on loan.account_id = account.account_id to identify accounts associated with loans. 2. disp → account: Join on disp.account_id = account.account_id to connect account dispositions to accounts. 3. client → account: Join on account.district_id = client.district_id to link clients to accounts based on their districts. 4. order → account: Join on order.account_id = account.account_id to include account orders. 5. card → disp: Join on card.disp_id = disp.disp_id to associate cards with account dispositions. Apply a filtering condition to include only male clients (where client.gender = 'M'). Finally, retrieve the distinct account IDs (account.account_id) that satisfy the above criteria.

    SQL query: SELECT DISTINCT drivers.surname FROM drivers, driverStandings, results, races, circuits WHERE driverStandings.driverId = drivers.driverId AND results.driverId = drivers.driverId AND races.circuitId = circuits.circuitId AND driverStandings.raceId = races.raceId AND races.name = 'Argentine Grand Prix' AND races.year = '1997'
    Question: Construct a SQL query involving the following tables: races, driverStandings, results, circuits, and drivers. Use these join relations: 1. driverStandings → drivers: Join on driverStandings.driverId = drivers.driverId to connect driver standings with driver details. 2. results → drivers: Join on results.driverId = drivers.driverId to associate race results with drivers. 3. races → circuits: Join on races.circuitId = circuits.circuitId to link races with their respective circuits. 4. driverStandings → races: Join on driverStandings.raceId = races.raceId to link driver standings to specific races. Apply filtering conditions to include only the race named 'Argentine Grand Prix' in the year 1997 (races.name = 'Argentine Grand Prix' AND races.year = '1997'). Finally, retrieve the distinct surnames of all drivers (drivers.surname) who meet the above criteria.

    SQL query: SELECT DISTINCT member.last_name FROM event, attendance, expense, member, major WHERE member.link_to_major = major.major_id AND attendance.link_to_member = member.member_id AND expense.link_to_member = member.member_id AND attendance.link_to_event = event.event_id AND event.event_name = 'Football game'
    Question: Construct a SQL query involving the following tables: expense, member, event, attendance, and major. Use these join relations: 1. member → major: Join on member.link_to_major = major.major_id to associate members with their respective majors. 2. attendance → member: Join on attendance.link_to_member = member.member_id to link member attendance records. 3. expense → member: Join on expense.link_to_member = member.member_id to connect expenses to members. 4. attendance → event: Join on attendance.link_to_event = event.event_id to link attendance records with specific events. Apply a filtering condition to include only the event named 'Football game' (event.event_name = 'Football game'). Finally, retrieve the distinct last names of all members (member.last_name) who meet the above criteria.

    SQL query: SELECT DISTINCT player.player_name FROM Match, Player_Attributes, League, Country, Player WHERE Match.away_player_11 = Player.player_api_id AND Match.country_id = Country.id AND Player_Attributes.player_api_id = Match.away_player_11 AND Match.league_id = League.id AND player_attributes.date = '2007-02-22 00:00:00'
    Question: Construct a SQL query involving the following tables: Player, Country, League, Player_Attributes, and Match. Use these join relations: 1. Match → Player: Join on Match.away_player_11 = Player.player_api_id to associate matches with the players who were the away team's 11th player. 2. Match → Country: Join on Match.country_id = Country.id to link matches to the countries where they took place. 3. Player_Attributes → Match: Join on Player_Attributes.player_api_id = Match.away_player_11 to connect player attributes to the respective players in the matches. 4. Match → League: Join on Match.league_id = League.id to link matches to their respective leagues. Apply a filtering condition to include only matches where the player attributes date is '2007-02-22 00:00:00' (Player_Attributes.date = '2007-02-22 00:00:00'). Finally, retrieve the distinct names of players (Player.player_name) who meet the above criteria.

    # Given the SQL query:
    SQL query: {main_query}

    # Instructions:
    Generate only the question with no additional explanation. Ensure that the question include correctly the tables, the join relations, the projections and the filterings of the given SQL query. Ensure the question is structured and clear, following the format of the examples provided. The question should be a single line without newlines.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Process LLM response
    return response.choices[0].message.content.strip().replace("Question:", "").strip()


def generate_dev_set_like_question(main_query, schema):
    """
    Generate a natural language question similar to those found in development sets.
    """

    prompt = f"""
    Your task is to generate a natural language question from the given SQL query, following the format of a dev set question in a Text-to-SQL benchmark as the examples below. Use the schema provided below to better understand the relationships in the database. Work through the reasoning behind the question to ensure it aligns with the structure of the SQL query, and use the examples provided to guide your understanding of the question format.

    ### Chain of Thought (CoT):
    1. **Understand the SQL query**: Analyze the SQL query and identify the key entities (tables, columns) and operations (joins, filters, aggregations). What is the query trying to extract from the database? For example, in the SQL query "SELECT COUNT(T1.Id) FROM users AS T1...," the focus is on counting the number of posts made by a user, which can help form a "How many" style question.
  
    2. **Break down the question**: What specific details or relationships from the SQL query should be captured in the question? For instance, in the query "SELECT T1.client_id FROM client AS T1...", the question needs to reflect the user (client) and the condition of being female, and it asks "Who are the female account holders who own credit cards and also have loans?"

    3. **Match the schema**: Refer to the schema to understand the relationships between tables and their attributes. For example, in the query "SELECT DISTINCT t4.name FROM Player_Attributes AS t1...," the schema tells us that "name" is an attribute of "Country," and this helps form the question around the country of the players with certain attributes.

    4. **Formulate the question**: Based on the analysis, generate a clear, concise question that corresponds to the SQL query. Ensure the question is structured similarly to the examples provided. For instance, if the SQL query is about counting or summarizing certain data (like the "wins" in the race query), the question might follow the format of "How many times did [person] do [action]?"


    ### Schema:
    {schema}

    ### Examples:

    SQL query: SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id INNER JOIN votes AS T4 ON T4.PostId = T3.Id WHERE T1.DisplayName = 'Matt Parker' GROUP BY T2.PostId, T4.Id HAVING COUNT(T4.Id) > 4
    Question: How many posts by Matt Parker have more than 4 votes?

    SQL query: SELECT T1.client_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T5 ON T2.account_id = T5.account_id INNER JOIN loan AS T3 ON T5.account_id = T3.account_id INNER JOIN card AS T4 ON T2.disp_id = T4.disp_id WHERE T1.gender = 'F'
    Question: Who are the female account holders who own credit cards and also have loans?

    SQL query: SELECT DISTINCT t4.name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id INNER JOIN Match AS t3 ON t2.player_api_id = t3.home_player_8 INNER JOIN Country AS t4 ON t3.country_id = t4.id WHERE t1.vision > 89
    Question: Locate players with vision scores of 90 and above, state the country of these players.

    SQL query: SELECT SUM(T2.wins) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T2.driverId = T1.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId INNER JOIN circuits AS T4 ON T4.circuitId = T3.circuitId WHERE T1.forename = 'Michael' AND T1.surname = 'Schumacher' AND T4.name = 'Sepang International Circuit'
    Question: How many times did Michael Schumacher won from races hosted in Sepang International Circuit?

    SQL query: SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN hero_attribute AS T3 ON T1.id = T3.hero_id INNER JOIN attribute AS T4 ON T3.attribute_id = T4.id WHERE T4.attribute_name = 'Speed' ORDER BY T3.attribute_value LIMIT 1
    Question: Which publisher published the slowest superhero?

            "SQL": "SELECT T2.forename, T2.surname FROM qualifying AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId INNER JOIN races AS T3 ON T1.raceid = T3.raceid WHERE q3 IS NOT NULL AND T3.year = 2008 AND T3.circuitId IN ( SELECT circuitId FROM circuits WHERE name = 'Marina Bay Street Circuit' ) ORDER BY CAST(SUBSTR(q3, 1, INSTR(q3, ':') - 1) AS INTEGER) * 60 + CAST(SUBSTR(q3, INSTR(q3, ':') + 1, INSTR(q3, '.') - INSTR(q3, ':') - 1) AS REAL) + CAST(SUBSTR(q3, INSTR(q3, '.') + 1) AS REAL) / 1000 ASC LIMIT 1",
        "question": "What is full name of the racer who ranked 1st in the 3rd qualifying race held in the Marina Bay Street Circuit in 2008?",


    ### Given the SQL query:
    SQL query: {main_query}

    ### Instructions:
    1. Follow the Chain of Thought reasoning steps to generate a question that matches the SQL query.
    2. Ensure that the question matches the style and format of the examples above. It should be clear, concise, and related to the SQL query.
    3. The question should be formatted as a single line with no additional explanations.
    4. Refer to the schema to better understand the relationships in the database. Make sure the question aligns with the schema and captures the important details of the query.
    """

# Examples
    # SQL query: SELECT DISTINCT account.account_id FROM account, card, client, loan, 'order', disp WHERE loan.account_id = account.account_id AND disp.account_id = account.account_id AND account.district_id = client.district_id AND 'order'.account_id = account.account_id AND card.disp_id = disp.disp_id AND client.gender = 'M'
    # Question: What are the different account IDs of male clients with accounts in the same district, linked to their dispositions, credit cards, orders, and loans?

    # SQL query: SELECT DISTINCT drivers.surname FROM drivers, driverStandings, results, races, circuits WHERE driverStandings.driverId = drivers.driverId AND results.driverId = drivers.driverId AND races.circuitId = circuits.circuitId AND driverStandings.raceId = races.raceId AND races.name = 'Argentine Grand Prix' AND races.year = '1997'
    # Question: What are the different surnames of drivers who participated in the 1997 Argentine Grand Prix, raced on its associated circuits, and had recorded results and standings?

    # SQL query: SELECT DISTINCT member.last_name FROM event, attendance, expense, member, major WHERE member.link_to_major = major.major_id AND attendance.link_to_member = member.member_id AND expense.link_to_member = member.member_id AND attendance.link_to_event = event.event_id AND event.event_name = 'Football game'
    # Question: What are the different last names of members, linked to a major, who attended the 'Football game' event and have related expenses?

    # SQL query: SELECT DISTINCT player.player_name FROM Match, Player_Attributes, League, Country, Player WHERE Match.away_player_11 = Player.player_api_id AND Match.country_id = Country.id AND Player_Attributes.player_api_id = Match.away_player_11 AND Match.league_id = League.id AND player_attributes.date = '2007-02-22 00:00:00'
    # Question: What are the different names of players who played in the match on 2007-02-22 in the respective league and country, with player attributes recorded for that date, and who are linked to the match as away_player_11?
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Process LLM response
    return response.choices[0].message.content.strip().replace("Question:", "").strip()

def generate_evidence(main_query, schema, question):

    prompt = f"""
    Your task is to generate **evidence** for the given SQL query and question.

    The evidence refers to external knowledge needed to help map the **natural language question** to its corresponding **SQL query**, especially for values in filtering conditions and the meaning of projected values (i.e., what is being selected). This is useful when such details are **implicit** or not clearly stated in the question.

    You will be provided with the database **schema** and a few **examples**.

    When writing evidence:
    - Focus only on filtering conditions and projections.
    - You may mention column names, relevant values, and the names of the tables to which those columns belong—but only if they are involved in filtering or projection, not joins.
    For example, consider the question: Which accounts are held by male clients who have loans, placed orders, and possess credit cards?
    The correct evidence should be: Male clients refer to gender = 'M'; accounts refers to account.account_id.
    There is no need to mention tables involved in join conditions, such as: accounts with loans, orders, and credit cards involve account_id in account, loan, "order", and card tables. — this would be incorrect.
    - Do not mention any join conditions or join keys used in the SQL query (e.g., `Tab1.id1 = Tab2.id2`) or refer to the keys themselves (e.g., `id1` as a join key).
    - Your output must be **a single, concise line** starting with `Evidence:` and must contain **no additional explanation**.

    ---

    ### Schema:
    {schema}

    ---

    ### Examples:

    SQL query: SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id INNER JOIN votes AS T4 ON T4.PostId = T3.Id WHERE T1.DisplayName = 'Matt Parker' GROUP BY T2.PostId, T4.Id HAVING COUNT(T4.Id) > 4
    Question: How many posts by Matt Parker have more than 4 votes?
    Evidence: more than 4 votes refer to PostId > 4; DisplayName = 'Matt Parker'

    SQL query: SELECT T1.client_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T5 ON T2.account_id = T5.account_id INNER JOIN loan AS T3 ON T5.account_id = T3.account_id INNER JOIN card AS T4 ON T2.disp_id = T4.disp_id WHERE T1.gender = 'F'
    Question: Who are the female account holders who own credit cards and also have loans?
    Evidence: Female refers to gender = 'F'

    SQL query: SELECT DISTINCT t4.name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id INNER JOIN Match AS t3 ON t2.player_api_id = t3.home_player_8 INNER JOIN Country AS t4 ON t3.country_id = t4.id WHERE t1.vision > 89
    Question: Locate players with vision scores of 90 and above, state the country of these players.
    Evidence: vision scores of 90 and above refers to vision > 89

    SQL query: SELECT SUM(T2.wins) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T2.driverId = T1.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId INNER JOIN circuits AS T4 ON T4.circuitId = T3.circuitId WHERE T1.forename = 'Michael' AND T1.surname = 'Schumacher' AND T4.name = 'Sepang International Circuit'
    Question: How many times did Michael Schumacher won from races hosted in Sepang International Circuit?
    Evidence: win from races refers to max(points)

    SQL query: SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN hero_attribute AS T3 ON T1.id = T3.hero_id INNER JOIN attribute AS T4 ON T3.attribute_id = T4.id WHERE T4.attribute_name = 'Speed' ORDER BY T3.attribute_value LIMIT 1
    Question: Which publisher published the slowest superhero?
    Evidence: the slowest superhero refers to attribute_name = 'Speed' where MIN(attribute_value); publisher refers to publisher_name

    ---

    ### Now for the following SQL query and the question:

    **SQL query:**  
    {main_query}
    **Question:**
    {question}

    ---

    **Instructions:**  
    Generate a **single-line evidence** starting with `Evidence:` based on the query above. **Do not include anything else**.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Process LLM response
    return response.choices[0].message.content.strip().replace("Evidence:", "").strip()

def generate_new_extended_question(old_query, new_query, schema, old_question):

    prompt = f"""
    You are given a text-to-SQL example where a query has been extended by adding a new table and its join conditions. Your task is to write a new natural language question that corresponds to the updated SQL query. For reference, you will be provided with the schema, the original SQL query, its associated question, and the new SQL query.
    ---

    ### Schema:
    {schema}

    ---
    ### Old query:
    {old_query}
    ### Old question:
    {old_question}

    ---
    ### New query:
    {new_query}
    ### New question:
    ### Instructions:
    Write a single-line question (no newlines) that accurately reflects the new SQL query. Return only the question — do not include explanations or commentary.    

    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Process LLM response
    return response.choices[0].message.content.strip().replace("Evidence:", "").strip()