import re
from typing import List
from nltk.tokenize import word_tokenize
import nltk
import sqlglot
from sqlglot.expressions import Column, Literal

nltk.download("punkt")


def tokenize_question(text):
    return set(word_tokenize(text.lower()))


def tokenize(text: str) -> List[str]:
    """Simple tokenizer to get words only (lowercase)."""
    return re.findall(r"\w+", text.lower())


def contiguous_subsequences(tokens: List[str], min_len: int = 2) -> List[List[str]]:
    """Generate all contiguous subsequences of a given minimum length."""
    subsequences = []
    for i in range(len(tokens)):
        for j in range(i + min_len, len(tokens) + 1):
            subsequences.append(tokens[i:j])
    return subsequences


# def remove_phrases_from_text(text: str, phrases: List[str]) -> str:
#     tokens = tokenize(text)
#     to_remove = [False] * len(tokens)

#     for phrase in phrases:
#         phrase_tokens = tokenize(phrase)
#         if not phrase_tokens:
#             continue

#         if len(phrase_tokens) == 1:
#             subsequences = [phrase_tokens]
#         else:
#             subsequences = contiguous_subsequences(phrase_tokens, min_len=2)
#             subsequences.insert(0, phrase_tokens)

#         subsequences.sort(key=lambda s: -len(s))  # longest first

#         for sub in subsequences:
#             i = 0
#             while i <= len(tokens) - len(sub):
#                 if tokens[i:i + len(sub)] == sub:
#                     for j in range(len(sub)):
#                         to_remove[i + j] = True
#                     i += len(sub)  # Skip over removed part
#                 else:
#                     i += 1

#     # Reconstruct cleaned text
#     original_words = re.findall(r'\w+|\W+', text)
#     token_index = 0
#     result = []

#     for word in original_words:
#         if re.match(r'\w+', word):
#             if token_index < len(to_remove) and not to_remove[token_index]:
#                 result.append(word)
#             token_index += 1
#         else:
#             result.append(word)

#     return ''.join(result).strip()


def remove_phrases_from_text(text: str, phrases: List[str]) -> str:
    tokens = tokenize(text)
    to_remove = [False] * len(tokens)

    for phrase in phrases:
        phrase_tokens = tokenize(phrase)
        if not phrase_tokens:
            continue

        # Search for exact matches of the phrase tokens
        i = 0
        while i <= len(tokens) - len(phrase_tokens):
            if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
                for j in range(len(phrase_tokens)):
                    to_remove[i + j] = True
                i += len(phrase_tokens)
            else:
                i += 1

    # Reconstruct text
    original_words = re.findall(r"\w+|\W+", text)
    token_index = 0
    result = []

    for word in original_words:
        if re.match(r"\w+", word):
            if token_index < len(to_remove) and not to_remove[token_index]:
                result.append(word)
            token_index += 1
        else:
            result.append(word)

    return "".join(result).strip()


def extract_sql_values_and_attributes(sql_query: str):
    """Extract database literal values and attributes (with or without table qualifiers) from SQL using sqlglot."""
    try:
        parsed = sqlglot.parse_one(sql_query, dialect="mysql")
    except Exception as e:
        print(f"SQL parsing error: {e}")
        return [], []

    db_values = set()
    attributes = set()

    for node in parsed.walk():
        if isinstance(node, Literal):
            val = node.this
            if val is not None:
                if isinstance(val, str):
                    db_values.add(val.strip("'\""))
                else:
                    db_values.add(str(val))
        elif isinstance(node, Column):
            table = node.table
            column = node.name
            if table:
                attributes.add(f"{table}.{column}")
            else:
                attributes.add(column)

    return list(db_values), list(attributes)


def remove_db_terms(text: str, db_values: List[str], attributes: List[str]) -> str:
    # Remove db values first
    text = remove_phrases_from_text(text, db_values)

    # Remove fully qualified attributes if they exist in text
    qualified_atts = [att for att in attributes if "." in att]
    for qatt in qualified_atts:
        if qatt in text:
            # Remove full qualified attribute as is
            text = remove_phrases_from_text(text, [qatt])
        else:
            # If full qualified not in text, try removing just unqualified attribute if present
            unqualified = qatt.split(".")[-1]
            if unqualified in text:
                text = remove_phrases_from_text(text, [unqualified])

    # Remove unqualified attributes that are not part of any qualified attribute
    unqualified_atts = [att for att in attributes if "." not in att]
    text = remove_phrases_from_text(text, unqualified_atts)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_hiding_score(new_text: str, components: list) -> int:
    tokens = set(re.findall(r"\b\w+\b", new_text.lower()))
    expected_tokens = {
        t["token"].lower()
        for comp in components
        for t in comp.get("matched_tokens", [])
    }

    # If no tokens are expected to be hidden, consider it successfully hidden
    if not expected_tokens:
        return 1.0

    # If any expected token is found in the new text, hiding failed
    if expected_tokens & tokens:
        return 0.0

    return 1.0


def hiding_success_scores(queries):
    question_scores = []
    evidence_scores = []
    queries_w_scores = []

    copied_queries = [q.copy() for q in queries]

    for q in copied_queries:
        q_comps = q.get("question_components", [])
        e_comps = q.get("evidence_components", [])

        sql = q.get("SQL", "")
        db_values, attributes = extract_sql_values_and_attributes(sql)

        # ---------- Question ----------
        raw_question = q.get("new_question", "").strip()
        if not q.get("question") or not q_comps:
            q_score = None
        elif not raw_question:
            q_score = 0.0
        else:
            q_score = compute_hiding_score(raw_question, q_comps)

        q["question_score"] = q_score
        question_scores.append(q_score)

        # ---------- Evidence ----------
        raw_evidence = q.get("new_evidence", "").strip()
        if not q.get("evidence") or not e_comps:
            e_score = None
        elif not raw_evidence:
            e_score = 0.0
        else:
            clean_evidence = remove_db_terms(raw_evidence, db_values, attributes)
            e_score = compute_hiding_score(clean_evidence, e_comps)

        q["evidence_score"] = e_score
        evidence_scores.append(e_score)
        queries_w_scores.append(q)

    # Compute averages only on non-None scores
    valid_question_scores = [s for s in question_scores if s is not None]
    valid_evidence_scores = [s for s in evidence_scores if s is not None]

    return {
        "queries_w_scores": queries_w_scores,
        "question_scores": question_scores,
        "evidence_scores": evidence_scores,
        "averages": {
            "question_score": (
                sum(valid_question_scores) / len(valid_question_scores)
                if valid_question_scores
                else None
            ),
            "evidence_score": (
                sum(valid_evidence_scores) / len(valid_evidence_scores)
                if valid_evidence_scores
                else None
            ),
        },
    }
