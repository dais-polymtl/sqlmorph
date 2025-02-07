import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_sql_question(main_query, example_question):
    """
    Generate a natural language question for a given SQL query.

    Args:
        main_query (str): The SQL query for which a question is generated.
        example_question (str): The example question to guide formatting.

    Returns:
        str: A generated natural language question.
    """
    prompt = f"""Your goal is to generate natural language questions for SQL queries to be included in Text-to-SQL benchmarks. Ensure the generated question follows the format of the provided example:

    SQL query: SELECT DISTINCT account.account_id FROM account, card, client, loan, 'order', disp 
    WHERE loan.account_id = account.account_id 
    AND disp.account_id = account.account_id 
    AND account.district_id = client.district_id 
    AND 'order'.account_id = account.account_id 
    AND card.disp_id = disp.disp_id 
    AND client.gender = 'M'

    Question: {example_question}

    # Given the SQL query:
    SQL query: {main_query}

    Question:
    # Instructions:
    Generate only the question with no additional explanation or formatting symbols like dashes or colons.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Process LLM response
    return response.choices[0].message.content.strip()


def generate_explicit_question(main_query):
    """
    Generate an explicit, structured question based on SQL join relations.
    """
    example_question = (
        "Construct a SQL query involving the following tables: client, loan, account, disp, card, and order. "
        "Use these join relations: 1. loan → account: Join on loan.account_id = account.account_id to identify accounts associated with loans. "
        "2. disp → account: Join on disp.account_id = account.account_id to connect account dispositions to accounts. "
        "3. client → account: Join on account.district_id = client.district_id to link clients to accounts based on their districts. "
        "4. order → account: Join on order.account_id = account.account_id to include account orders. "
        "5. card → disp: Join on card.disp_id = disp.disp_id to associate cards with account dispositions. "
        "Apply a filtering condition to include only male clients (where client.gender = 'M'). "
        "Finally, retrieve the distinct account IDs (account.account_id) that satisfy the above criteria."
    )
    return generate_sql_question(main_query, example_question)


def generate_dev_set_like_question(main_query):
    """
    Generate a natural language question similar to those found in development sets.
    """
    example_question = (
        "What are the different account IDs of male clients with accounts in the same district, "
        "linked to their dispositions, credit cards, orders, and loans?"
    )
    return generate_sql_question(main_query, example_question)
