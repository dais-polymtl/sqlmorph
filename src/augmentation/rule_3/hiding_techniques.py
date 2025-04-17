import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def apply_synonym_replacement(
    db_id, question, evidence, sql_query, candidate_table, components
):
    """Uses GPT-4o to replace references to candidate tables in questions and evidence using synonyms."""

    prompt = f"""
You are in a text-to-SQL context where each question corresponds to a SQL query based on a given database schema.

Your task is to conceal any direct reference to candidate table names in the question and the evidence while keeping them meaningful.  
Use the Synonym Replacement (SR) technique to replace words or phrases linked to the table names with their synonyms.  

**Guidelines:**
- The synonym should not refer to the candidate table itself (e.g., if the table is *cards*, do not use *playing cards*).
- Keep the meaning of the question and evidence intact.
- If the evidence directly mentions a table (e.g., cards.name), replace the table name with a synonym and imply the source with pronouns or descriptive phrases.
- If the evidence is empty, return an empty string.
- The **Candidate Table** may be a single word (e.g., *users*) or a compound name (e.g., *postHistory*).  
- You will be given **Components**, a list of parts of the candidate table name. Use them to identify what to replace.
- Do **not** mention the original table name in any way.

**Input:**
- **Database ID:** {db_id}
- **Question:** {question}
- **Evidence:** {evidence}
- **SQL Query:** {sql_query}
- **Candidate Table:** {candidate_table}
- **Components:** {components}

**Output Format:**  
New Question: modified question  
New Evidence: modified evidence (or empty string if none)
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=300, temperature=0.7
    )

    generated_text = response.choices[0].message.content.strip()

    # Extract new question and evidence
    new_question, new_evidence = "", ""
    for row in generated_text.split("\n"):
        if row.startswith("New Question:"):
            new_question = row.replace("New Question: ", "").strip()
        elif row.startswith("New Evidence:"):
            new_evidence = row.replace("New Evidence: ", "").strip()

    return new_question, new_evidence


def generate_synonym_replacement_queries(db_id, subgraphs):
    """Processes queries by applying synonym replacement and filtering unchanged ones."""

    rewritten_queries = []
    print(f"Generating new questions for {db_id}...")

    for subgraph in subgraphs:
        for query in subgraph["equivalent_queries"]:
            new_question, new_evidence = apply_synonym_replacement(
                db_id,
                query["question"],
                query["evidence"],
                query["SQL"],
                subgraph["central_table"],
                subgraph["components"],
            )

            # Ensure replacement occurred before keeping it
            if not any(comp in new_question.lower() for comp in subgraph["components"]):
                query["new_question"], query["new_evidence"] = (
                    new_question,
                    new_evidence,
                )
                rewritten_queries.append(query)

    return rewritten_queries


def hide_tables_with_synonym_replacement(data):
    """Runs synonym replacement on the dataset and returns the modified queries."""

    return {
        db_id: generate_synonym_replacement_queries(db_id, subgraphs)
        for db_id, subgraphs in data.items()
    }


def apply_backtranslation(
    db_id, question, evidence, sql_query, candidate_table, components
):
    """Uses GPT-4o to perform back translation with synonym replacement."""

    prompt = f"""
You are in a text-to-SQL context where each question corresponds to a SQL query based on a given database schema.

Your task is to conceal any direct reference to the candidate table names in the question and the evidence while keeping it meaningful.  
Use the **Back Translation (BT)** technique. First, translate the question and the evidence into French, then back to English. During the French translation, replace any reference to the candidate table with a relevant synonym.  

**Important Guidelines:**
- Do not use synonyms that refer directly to the candidate table (e.g., if the table is *cards*, do not use *cartes de jeu*).  
- Hide only the table name reference and keep the rest of the question and evidence intact.  
- If the evidence directly mentions a table name (e.g., cards.name), replace the table name with a suitable synonym and imply the source with pronouns or descriptive phrases.
- If the evidence is empty, return an empty string.
- The **Candidate Table** may be a single word (e.g., *users*) or a compound name (e.g., *postHistory*).  
- The **Components** list helps identify which words to replace.

**Input:**
- **Database ID:** {db_id}
- **Question:** {question}
- **Evidence:** {evidence}
- **SQL Query:** {sql_query}
- **Candidate Table:** {candidate_table}
- **Components:** {components}

**Output Format:**  
Question in English: [modified question]  
Evidence in English: [modified evidence] (or empty string if none)
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=300, temperature=0.7
    )

    generated_text = response.choices[0].message.content.strip()

    # Extract new question and evidence
    new_question, new_evidence = "", ""
    for row in generated_text.split("\n"):
        if row.startswith("Question in English:"):
            new_question = row.replace("Question in English: ", "").strip()
        elif row.startswith("Evidence in English:"):
            new_evidence = row.replace("Evidence in English: ", "").strip()

    return new_question, new_evidence


def generate_backtranslated_queries(db_id, subgraphs):
    """Processes queries by applying back translation and filtering unchanged ones."""

    rewritten_queries = []
    print(f"Generating backtranslated questions for {db_id}...")

    for subgraph in subgraphs:
        for query in subgraph["equivalent_queries"]:
            new_question, new_evidence = apply_backtranslation(
                db_id,
                query["question"],
                query["evidence"],
                query["SQL"],
                subgraph["central_table"],
                subgraph["components"],
            )

            # Ensure replacement occurred before keeping it
            if not any(comp in new_question.lower() for comp in subgraph["components"]):
                query["new_question"], query["new_evidence"] = (
                    new_question,
                    new_evidence,
                )
                rewritten_queries.append(query)

    return rewritten_queries


def hide_tables_with_backtranslation(data):
    """Runs back translation on the dataset, concealing table names."""

    return {
        db_id: generate_backtranslated_queries(db_id, subgraphs)
        for db_id, subgraphs in data.items()
    }


def augment_with_contextual_synonyms(
    db_id, question, evidence, sql_query, candidate_table, components
):
    """Uses GPT-4o to perform contextual augmentation by replacing table references with synonyms."""

    prompt = f"""
You are in a text-to-SQL context where each question corresponds to a SQL query based on a given database schema.

Your task is to conceal any direct reference to the candidate table names in the question and evidence while ensuring the sentence remains meaningful.  
Use **Contextual Augmentation (CA)** to replace words or phrases related to the table names with appropriate synonyms that fit naturally within the surrounding context.  

**Guidelines:**  
- Synonyms must be relevant to the database context (e.g., for *card_games*, *cards* could become *playing pieces* but not *vouchers*).  
- Synonyms should not directly reference the candidate table (e.g., *cards* should not become *playing cards*).  
- Ensure the table name is removed while keeping the question intact.  
- If the evidence contains a table name along with a column (e.g., cards.name), replace the table name and use pronouns or descriptive phrases to imply the source.
- If the evidence is empty, return an empty string.

**Input:**  
- **Database ID:** {db_id}  
- **Question:** {question}  
- **Evidence:** {evidence}  
- **SQL Query:** {sql_query}  
- **Candidate Table:** {candidate_table}  
- **Components:** {components}  

**Output Format:**  
New Question: [modified question]  
New Evidence: [modified evidence] (or empty string if none)
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=300, temperature=0.7
    )

    generated_text = response.choices[0].message.content.strip()

    # Extract new question and evidence
    new_question, new_evidence = "", ""
    for row in generated_text.split("\n"):
        if row.startswith("New Question:"):
            new_question = row.replace("New Question: ", "").strip()
        elif row.startswith("New Evidence:"):
            new_evidence = row.replace("New Evidence: ", "").strip()

    return new_question, new_evidence


def generate_contextually_augmented_queries(db_id, subgraphs):
    """Processes queries by applying contextual augmentation and filtering unchanged ones."""

    rewritten_queries = []
    print(f"Generating contextually augmented questions for {db_id}...")

    for subgraph in subgraphs:
        for query in subgraph["equivalent_queries"]:
            new_question, new_evidence = augment_with_contextual_synonyms(
                db_id,
                query["question"],
                query["evidence"],
                query["SQL"],
                subgraph["central_table"],
                subgraph["components"],
            )

            # Ensure replacement occurred before keeping it
            if not any(comp in new_question.lower() for comp in subgraph["components"]):
                query["new_question"], query["new_evidence"] = (
                    new_question,
                    new_evidence,
                )
                rewritten_queries.append(query)

    return rewritten_queries


def hide_tables_with_contextual_augmentation(data):
    """Runs contextual augmentation on the dataset to conceal table names."""

    return {
        db_id: generate_contextually_augmented_queries(db_id, subgraphs)
        for db_id, subgraphs in data.items()
    }
