import json
import os


def save_questions_to_file(queries, output_base_dir, technique):
    output_dir = f"{output_base_dir}/{technique}"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    original_questions = []
    new_questions = []
    # Iterate over db_ids and queries to collect original and new questions
    for query in queries:
        original_entry = {
            "question_id": query["question_id"],
            "db_id": query["db_id"],
            "question": query["question"],
            "evidence": query["evidence"],
            "SQL": query["SQL"],
            "difficulty": query["difficulty"],
        }
        new_entry = {
            "question_id": query["question_id"],
            "db_id": query["db_id"],
            "question": query["new_question"],
            "evidence": query["new_evidence"],
            "SQL": query["SQL"],
            "difficulty": query["difficulty"],
        }

        original_questions.append(original_entry)
        new_questions.append(new_entry)

    # Save the questions to files
    with open(f"{output_dir}/original_questions.json", "w") as f:
        json.dump(original_questions, f, indent=4)
    with open(f"{output_dir}/new_questions.json", "w") as f:
        json.dump(new_questions, f, indent=4)

    print(
        f"Saved {len(original_questions)} original questions and {len(new_questions)} new questions to {output_dir}."
    )


def process_and_save_data_for_set(data, output_base_dir):
    """Process and save questions for a specific dataset type (train, dev, test)."""
    for technique, data in data.items():
        save_questions_to_file(data, output_base_dir, technique)
