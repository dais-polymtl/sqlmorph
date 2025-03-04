import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_explicit_question(main_query):
    """
    Generate an explicit, structured question based on SQL join relations.
    """

    prompt = f"""Your goal is to generate a natural language question for the given SQL query that involves the specified join relations. Ensure the question matches the format of the examples provided:

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


def generate_dev_set_like_question(main_query):
    """
    Generate a natural language question similar to those found in development sets.
    """

    prompt = f"""Your goal is to generate a natural language question for the given SQL query that matches the style of a development set question. Ensure the question matches the format of the examples provided:

    # Examples:

    SQL query: SELECT DISTINCT account.account_id FROM account, card, client, loan, 'order', disp WHERE loan.account_id = account.account_id AND disp.account_id = account.account_id AND account.district_id = client.district_id AND 'order'.account_id = account.account_id AND card.disp_id = disp.disp_id AND client.gender = 'M'
    Question: What are the different account IDs of male clients with accounts in the same district, linked to their dispositions, credit cards, orders, and loans?

    SQL query: SELECT DISTINCT drivers.surname FROM drivers, driverStandings, results, races, circuits WHERE driverStandings.driverId = drivers.driverId AND results.driverId = drivers.driverId AND races.circuitId = circuits.circuitId AND driverStandings.raceId = races.raceId AND races.name = 'Argentine Grand Prix' AND races.year = '1997'
    Question: What are the different surnames of drivers who participated in the 1997 Argentine Grand Prix, raced on its associated circuits, and had recorded results and standings?

    SQL query: SELECT DISTINCT member.last_name FROM event, attendance, expense, member, major WHERE member.link_to_major = major.major_id AND attendance.link_to_member = member.member_id AND expense.link_to_member = member.member_id AND attendance.link_to_event = event.event_id AND event.event_name = 'Football game'
    Question: What are the different last names of members, linked to a major, who attended the 'Football game' event and have related expenses?

    SQL query: SELECT DISTINCT player.player_name FROM Match, Player_Attributes, League, Country, Player WHERE Match.away_player_11 = Player.player_api_id AND Match.country_id = Country.id AND Player_Attributes.player_api_id = Match.away_player_11 AND Match.league_id = League.id AND player_attributes.date = '2007-02-22 00:00:00'
    Question: What are the different names of players who played in the match on 2007-02-22 in the respective league and country, with player attributes recorded for that date, and who are linked to the match as away_player_11?

    # Given the SQL query:
    SQL query: {main_query}

    # Instructions:
    Generate only the question with no additional explanation. Ensure the question is structured and clear, following the format of the examples provided. The question should be a single line without newlines.
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
