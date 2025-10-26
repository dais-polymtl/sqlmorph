import json
import difflib
import os
from termcolor import colored
from pathlib import Path
from tabulate import tabulate


def strikethrough(text):
    return "".join(c + "\u0336" for c in text)


def highlight_differences(old, new):
    diff = difflib.ndiff(old.split(), new.split())
    result = []
    for word in diff:
        if word.startswith("-"):
            removed = strikethrough(word[2:])
            result.append(colored(removed, "red", attrs=["bold"]))
        elif word.startswith("+"):
            result.append(colored(word[2:], "green", attrs=["bold"]))
        elif word.startswith(" "):
            result.append(word[2:])
    return " ".join(result)


def highlight_tokens(text, tokens):
    """Highlight matched tokens in cyan and bold."""
    positions = sorted(tokens, key=lambda t: t["position"])
    result = ""
    last_index = 0
    for token in positions:
        start = token["position"]
        word = token["token"]
        end = start + len(word)
        result += text[last_index:start]
        result += colored(f'"{word}"', "cyan", attrs=["bold"])
        last_index = end
    result += text[last_index:]
    return result


def highlight_sql_central_table(sql: str, central_table: str) -> str:
    """Highlight central table in SQL string."""
    # Case-insensitive match and highlight for table alias or full name
    words = sql.split()
    highlighted = []
    for w in words:
        if central_table.lower() in w.lower():
            highlighted.append(colored(w, "yellow", attrs=["bold"]))
        else:
            highlighted.append(w)
    return " ".join(highlighted)


def ask_score(prompt):
    while True:
        val = input(prompt + " (0, 1, or press Enter for None): ").strip()
        if val in {"0", "1"}:
            return int(val)
        elif val == "":
            return None
        else:
            print("Invalid input. Please enter 0, 1, or just press Enter for None.")


def annotate_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if "question_human_score" in item and "evidence_human_score" in item:
            continue  # Skip already annotated

        print(f"\n\n================== Example {i+1}/{len(data)} ==================\n")

        print(colored("SQL:", "magenta", attrs=["bold"]))
        highlighted_sql = highlight_sql_central_table(
            item.get("SQL", ""), item.get("central_table", "")
        )
        print(highlighted_sql)

        # ---- QUESTION ----
        print(colored("\n--- Question ---", "yellow", attrs=["bold"]))

        matched_tokens = [
            t
            for comp in item.get("question_components", [])
            for t in comp.get("matched_tokens", [])
        ]

        print(colored("Old Question with matched tokens:", "cyan"))
        print(highlight_tokens(item["old_question"], matched_tokens))

        print(colored("Diff (old → new):", "green"))
        print(highlight_differences(item["old_question"], item["new_question"]))

        q_score = ask_score("→ Your score for question")
        item["question_human_score"] = q_score

        # ---- EVIDENCE ----
        print(colored("\n--- Evidence ---", "yellow", attrs=["bold"]))
        old_evidence = item.get("old_evidence", "")
        new_evidence = item.get("new_evidence", "")

        print(colored("Old Evidence:", "cyan"))
        print(old_evidence)

        print(colored("Diff (old → new):", "green"))
        print(highlight_differences(old_evidence, new_evidence))

        e_score = ask_score("→ Your score for evidence")
        item["evidence_human_score"] = e_score

        # Save immediately after annotation
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(colored("✓ Saved progress.\n", "green", attrs=["bold"]))


def compute_and_display_stats(data, technique_name):
    def avg(scores):
        valid = [s for s in scores if s is not None]
        return sum(valid) / len(valid) if valid else 0.0

    def compare_scores(auto_scores, human_scores):
        return sum(1 for a, h in zip(auto_scores, human_scores) if a != h)

    question_auto_scores = [ex.get("question_score") for ex in data]
    evidence_auto_scores = [ex.get("evidence_score") for ex in data]
    question_human_scores = [ex.get("question_human_score") for ex in data]
    evidence_human_scores = [ex.get("evidence_human_score") for ex in data]

    avg_q_auto = avg(question_auto_scores)
    avg_q_human = avg(question_human_scores)
    avg_e_auto = avg(evidence_auto_scores)
    avg_e_human = avg(evidence_human_scores)

    diff_q = compare_scores(question_auto_scores, question_human_scores)
    diff_e = compare_scores(evidence_auto_scores, evidence_human_scores)

    table = [
        ["Automated Score", f"{avg_q_auto * 100:.1f}%", f"{avg_e_auto * 100:.1f}%"],
        ["Human Score", f"{avg_q_human * 100:.1f}%", f"{avg_e_human * 100:.1f}%"],
        [
            "Difference",
            f"{(avg_q_auto - avg_q_human) * 100:.1f}%",
            f"{(avg_e_auto - avg_e_human) * 100:.1f}%",
        ],
        ["# Diff. Examples", f"{diff_q}", f"{diff_e}"],
    ]

    print(colored(f"\nTechnique: {technique_name}\n", "cyan", attrs=["bold"]))
    print(tabulate(table, headers=["Metric", "Question", "Evidence"], tablefmt="grid"))


if __name__ == "__main__":
    technique = "context_aug"
    json_file = (
        Path(os.getenv("DATA_FOLDER"))
        / "rule_outputs"
        / "lt_elimination"
        / "dev"
        / f"dev_{technique}_stats.json"
    )
    if not os.path.exists(json_file):
        print("File not found.")
    else:
        annotate_json_file(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    compute_and_display_stats(data, technique)
    print(
        colored(
            "\n✓ Annotation and stats computation completed.", "green", attrs=["bold"]
        )
    )
