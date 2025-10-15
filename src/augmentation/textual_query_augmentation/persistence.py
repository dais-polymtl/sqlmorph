import json
import os


def save_questions_to_file(queries, output_base_dir, technique, dataset_type):
    output_dir = f"{output_base_dir}/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    original_questions = []
    new_questions = []
    test_stats = []
    # Iterate over db_ids and queries to collect original and new questions
    for query in queries:
        original_entry = {
            "question_id": query["question_id"],
            "db_id": query["db_id"],
            "question": query["question"],
            "evidence": query["evidence"],
            "SQL": query["SQL"],
            "difficulty": query.get("difficulty", ""),
        }
        original_questions.append(original_entry)

    for query in queries:
        new_entry = {
            "question_id": query["question_id"],
            "db_id": query["db_id"],
            "question": query["new_question"],
            "evidence": query["new_evidence"],
            "SQL": query["SQL"],
            "difficulty": query.get("difficulty", ""),
        }
        new_questions.append(new_entry)

    for query in queries:
        new_entry = {
            "question_id": query["question_id"],
            "db_id": query["db_id"],
            "old_question": query["question"],
            "old_evidence": query["evidence"],
            "central_table": query["central_table"],
            "question_components": query["question_components"],
            "evidence_components": query.get("evidence_components", []),
            "SQL": query["SQL"],
            "new_question": query["new_question"],
            "question_score": query["question_score"],
            "new_evidence": query["new_evidence"],
            "evidence_score": query["evidence_score"],
            "difficulty": query["difficulty"],
        }

        test_stats.append(new_entry)

    with open(f"{output_dir}/{dataset_type}_{technique}_stats.json", "w") as f:
        json.dump(test_stats, f, indent=4)
    # with open(f"{output_dir}/{dataset_type}_{technique}_original_questions.json", "w") as f:
    #     json.dump(original_questions, f, indent=4)
    # with open(f"{output_dir}/{dataset_type}_{technique}_queries.json", "w") as f:
    #     json.dump(new_questions, f, indent=4)
    with open(f"{output_dir}/{dataset_type}_{technique}_queries.sql", "w") as f:
        for query in new_questions:
            f.write(f"{query['SQL']}\t{query['db_id']}\n")

    print(
        f"Saved {len(original_questions)} original questions and {len(new_questions)} new questions to {output_dir}."
    )


def process_and_save_data_for_set(data, output_base_dir, dataset_type):
    """Process and save questions for a specific dataset type (train, dev, test)."""
    for technique, data in data.items():
        save_questions_to_file(data, output_base_dir, technique, dataset_type)
