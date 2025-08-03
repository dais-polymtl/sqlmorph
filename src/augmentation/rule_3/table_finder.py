from typing import List, Tuple
import sqlglot
from sqlglot.expressions import Column, Literal


import nltk
import re
from nltk.tokenize import word_tokenize


import pickle
from pathlib import Path
from fuzzywuzzy import fuzz
import networkx as nx
from wordsegment import load, segment
from nltk.corpus import wordnet as wn

from nltk.corpus import words

import spacy

from src.core.logger.logger import Logger


load()

nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)


english_vocab = set(words.words())
nlp = spacy.load("en_core_web_sm")

logger = Logger(__name__)


def normalize_token(token):
    return re.sub(r"[^\w\s]", "", token.lower())


def tokenize_with_positions(raw_text):
    """Returns list of (token, start_position) using regex word boundaries."""
    tokens = []
    for match in re.finditer(r"\b\w+\b", raw_text):
        tokens.append((match.group(0), match.start()))
    return tokens


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


# def remove_phrases_from_text(text: str, phrases: List[str]) -> str:
#     tokens = tokenize(text)
#     to_remove = [False] * len(tokens)

#     for phrase in phrases:
#         phrase_tokens = tokenize(phrase)
#         if not phrase_tokens:
#             continue

#         # Search for exact matches of the phrase tokens
#         i = 0
#         while i <= len(tokens) - len(phrase_tokens):
#             if tokens[i:i + len(phrase_tokens)] == phrase_tokens:
#                 for j in range(len(phrase_tokens)):
#                     to_remove[i + j] = True
#                 i += len(phrase_tokens)
#             else:
#                 i += 1

#     # Reconstruct text
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


def remove_phrases_from_text(
    text: str, phrases: List[str]
) -> Tuple[str, List[Tuple[str, int]]]:
    tokens_with_pos = tokenize_with_positions(text)
    tokens = [t[0] for t in tokens_with_pos]
    to_remove = [False] * len(tokens)
    removed_tokens = []

    for phrase in phrases:
        phrase_tokens = tokenize(phrase)
        if not phrase_tokens:
            continue

        i = 0
        while i <= len(tokens) - len(phrase_tokens):
            if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
                for j in range(len(phrase_tokens)):
                    to_remove[i + j] = True
                    removed_tokens.append(
                        tokens_with_pos[i + j]
                    )  # (token, start_position)
                i += len(phrase_tokens)
            else:
                i += 1

    # Reconstruct cleaned text by skipping removed tokens
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

    return "".join(result).strip(), removed_tokens


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


# def remove_db_terms(text: str, db_values: List[str], attributes: List[str]) -> str:
#     # Remove db values first
#     text = remove_phrases_from_text(text, db_values)

#     # Remove fully qualified attributes if they exist in text
#     qualified_atts = [att for att in attributes if '.' in att]
#     for qatt in qualified_atts:
#         if qatt in text:
#             # Remove full qualified attribute as is
#             text = remove_phrases_from_text(text, [qatt])
#         else:
#             # If full qualified not in text, try removing just unqualified attribute if present
#             unqualified = qatt.split('.')[-1]
#             if unqualified in text:
#                 text = remove_phrases_from_text(text, [unqualified])

#     # Remove unqualified attributes that are not part of any qualified attribute
#     unqualified_atts = [att for att in attributes if '.' not in att]
#     text = remove_phrases_from_text(text, unqualified_atts)

#     # Clean extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text


def remove_db_terms(
    text: str, db_values: List[str], attributes: List[str]
) -> Tuple[str, List[Tuple[str, int]]]:
    all_removed = []

    # Remove db values first
    text, removed = remove_phrases_from_text(text, db_values)
    all_removed.extend(removed)

    # Remove fully qualified attributes if they exist in text
    qualified_atts = [att for att in attributes if "." in att]
    for qatt in qualified_atts:
        if qatt in text:
            text, removed = remove_phrases_from_text(text, [qatt])
            all_removed.extend(removed)
        else:
            unqualified = qatt.split(".")[-1]
            if unqualified in text:
                text, removed = remove_phrases_from_text(text, [unqualified])
                all_removed.extend(removed)

    # Remove unqualified attributes that are not part of any qualified attribute
    unqualified_atts = [att for att in attributes if "." not in att]
    text, removed = remove_phrases_from_text(text, unqualified_atts)
    all_removed.extend(removed)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text, all_removed


def lemmatize(text):
    lemmas = []
    for token in nlp(text):
        if token.is_alpha or any(c in token.text for c in ["_", "-", "."]):
            parts = re.split(r"[._-]", token.text)
            for part in parts:
                part = part.strip().lower()
                if not part:
                    continue
                if "part" in part:
                    lemma = "part"
                else:
                    doc = nlp(part)
                    if not doc:
                        continue
                    lemma = doc[0].lemma_.lower()
                if len(lemma) > 1 and (not nlp.vocab[lemma].is_stop or lemma == "part"):
                    lemmas.append(lemma)
    return lemmas


def is_abbreviation(token):
    token_lower = token.lower()

    known_abbreviations = {"rel", "ra", "geo"}
    not_abbreviations = {"logs", "u2base"}

    # Short tokens are suspicious

    if token_lower in known_abbreviations:
        return True
    if token_lower in not_abbreviations:
        return False

    if len(token) <= 4:

        # If not in dictionary, probably abbreviation
        if token_lower not in english_vocab:
            return True

    # Check if token is all uppercase or mixed with digits
    if token.isupper() or any(char.isdigit() for char in token):
        return True

    # Could add suffix or prefix checks here

    return False


def are_related(word1, word2, threshold=0.85):
    if word1 == word2:
        return True, 1.0
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    max_sim = 0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > max_sim:
                max_sim = sim
    return max_sim >= threshold, max_sim


def fuzzy_vowel_stripped_match(abbrev, phrase, threshold=100):
    abbrev_stripped = "".join(c for c in abbrev.lower() if c not in "aeiou")
    phrase_stripped = "".join(c for c in phrase.lower() if c not in "aeiou")

    # Avoid matching if abbrev_stripped is too short to reduce noise
    if len(abbrev_stripped) < 2 or len(phrase_stripped) < 2:
        return False

    score = fuzz.partial_ratio(abbrev_stripped, phrase_stripped)
    return score >= threshold


def should_keep(seg):
    seg_lower = seg.lower()
    return (
        (seg_lower == "part" or len(seg_lower) > 1)
        and (not nlp.vocab[seg_lower].is_stop or seg_lower == "part")
        and seg_lower.isalpha()
    )


def find_central_table_and_components(jqg, schema):
    """Find the central table in a subgraph and extract its components."""
    if jqg.number_of_nodes() < 3:
        return "", []

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(jqg)
    max_betweenness = max(betweenness_centrality.values(), default=0)

    # Get candidate tables
    candidates = [
        node
        for node, score in betweenness_centrality.items()
        if score == max_betweenness
    ]

    if not candidates:
        return "", []

    schema_lookup = {node.lower(): node for node in schema.nodes()}

    mapped_candidates = [
        schema_lookup[node[0].lower()]
        for node in candidates
        if node[0].lower() in schema_lookup
    ]

    if not mapped_candidates:
        return "", []

    # Resolve ties using schema centrality
    if len(mapped_candidates) > 1:
        schema_centrality = nx.degree_centrality(schema)
        central_table = max(
            mapped_candidates, key=lambda node: schema_centrality.get(node, 0)
        )
    else:
        central_table = mapped_candidates[0]

    if len(central_table):
        segmented = segment(central_table)
        components = [
            {"segment": seg, "is_abbreviation": is_abbreviation(seg)}
            for seg in segmented
            if should_keep(seg)
        ]
    return central_table, components


def retrieve_linker_table(jqg, schema):
    """Keep a jqg if at least one central table component is detected in the question."""
    jqg_copy = jqg.copy()
    central_table, components = find_central_table_and_components(
        jqg["jq_graph"], schema
    )
    if not central_table:
        return None

    db_values, attributes = extract_sql_values_and_attributes(jqg["SQL"])
    matched_components = {"question": [], "evidence": []}

    sources = {
        "question": jqg["question"].lower(),
        "evidence": jqg.get("evidence", "").lower() if jqg.get("evidence") else "",
    }

    for src in ["question", "evidence"]:
        raw_text = sources[src]
        raw_tokens = tokenize_with_positions(raw_text)

        if src == "evidence":
            clean_text, removed_tokens = remove_db_terms(
                raw_text, db_values, attributes
            )
            removed_set = set(
                (normalize_token(tok), pos) for tok, pos in removed_tokens
            )

            raw_tokens = [
                (tok, pos)
                for (tok, pos) in raw_tokens
                if (normalize_token(tok), pos) not in removed_set
            ]

            clean_tokens = [
                normalize_token(t) for t in re.findall(r"\b\w+\b", clean_text)
            ]
        else:
            clean_tokens = [normalize_token(t[0]) for t in raw_tokens]

        aligned = []
        used_raw_idxs = set()
        for clean_token in clean_tokens:
            for idx, (raw_tok, pos) in enumerate(raw_tokens):
                if idx in used_raw_idxs:
                    continue
                if normalize_token(raw_tok) == clean_token:
                    aligned.append((raw_tok, pos))
                    used_raw_idxs.add(idx)
                    break

        for comp in components:
            segment = comp["segment"].lower()
            segment_lemmas = (
                lemmatize(segment) if not comp["is_abbreviation"] else [segment]
            )

            matched_tokens = []
            for raw_tok, pos in aligned:
                clean_tok = normalize_token(raw_tok)
                lemma = lemmatize(clean_tok)[0] if lemmatize(clean_tok) else ""

                match_found = False
                if comp["is_abbreviation"]:
                    if fuzzy_vowel_stripped_match(segment, clean_tok, threshold=100):
                        match_found = True
                else:
                    if segment_lemmas:
                        related, sim = are_related(lemma, segment_lemmas[0], 0.95)
                        match_found = related

                if match_found:
                    matched_tokens.append({"token": raw_tok, "position": pos})

            if matched_tokens:
                comp_copy = comp.copy()
                comp_copy["matched_tokens"] = matched_tokens
                matched_components[src].append(comp_copy)

    if not matched_components["question"]:
        return None

    jqg_copy["central_table"] = central_table
    jqg_copy["question_components"] = matched_components["question"]
    jqg_copy["evidence_components"] = matched_components["evidence"]
    return jqg_copy


def process_dataset(dataset_jqgs):
    """Process train/dev/test datasets by loading schemas and filtering subgraphs."""
    dataset_jqgs_w_lt = []

    for i, jqg in enumerate(dataset_jqgs):
        logger.log(
            level="debug",
            action=f"Finding linker table for query {i + 1}/{len(dataset_jqgs)}",
        )
        db_id = jqg["db_id"]
        schema_path = Path(f"data/graph_data/bird_graphs/pickles/{db_id}_graph.pkl")

        if not schema_path.exists():
            print(f"Warning: Missing schema file for {db_id}")
            continue

        with schema_path.open("rb") as f:
            schema = pickle.load(f)

        new_jqg = retrieve_linker_table(jqg, schema)
        if new_jqg is not None:
            dataset_jqgs_w_lt.append(new_jqg)

    return dataset_jqgs_w_lt
