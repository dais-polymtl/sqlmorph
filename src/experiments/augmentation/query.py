import sqlite3
from pathlib import Path
from src.core.database.database_handler import DBMS
from src.evaluation import Evaluation, EvaluationTechnique
from src.core.model_manager import OpenAIModel


def calculate_execution_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> float:
    config = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
    }
    exact_evaluator = Evaluation(config)
    res = exact_evaluator.run_evaluation(
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        log=False,
    )
    return res["metrics"]["EX"]


def run_queries(db_path, query1, query2):
    conn = None
    try:
        db_path = Path(db_path).resolve()
        if not db_path.exists():
            print(f"Database file does not exist at: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Running Query 1...")
        cursor.execute(query1)
        results1 = cursor.fetchall()
        print("Results of Query 1:")
        for row in results1:
            print(row)

        print("\nRunning Query 2...")
        cursor.execute(query2)
        results2 = cursor.fetchall()
        print("Results of Query 2:")
        for row in results2:
            print(row)

        ex_acc = calculate_execution_accuracy(query1, query2, db_path)
        print("\nExecution accuracy:", ex_acc)
        print("Results are the same?", ex_acc == 1.0)

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()


def main():
    db_id = "european_football_2"  # Example database ID
    db_path = f"data/benchmarks/Bird/bird_databases/{db_id}/{db_id}.sqlite"

    query1 = """

SELECT DISTINCT t4.name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id INNER JOIN Match AS t3 ON t2.player_api_id = t3.home_player_8 INNER JOIN Country AS t4 ON t3.country_id = t4.id INNER JOIN League AS et ON et.id = t3.league_id WHERE t1.vision > 89

"""
    query2 = """
SELECT DISTINCT C.`name` AS country_name FROM Player_Attributes AS PA INNER JOIN Player AS P ON PA.`player_api_id` = P.`player_api_id` INNER JOIN Match AS M ON M.`home_player_1` = P.`player_api_id` OR M.`home_player_2` = P.`player_api_id` OR M.`home_player_3` = P.`player_api_id` OR M.`home_player_4` = P.`player_api_id` OR M.`home_player_5` = P.`player_api_id` OR M.`home_player_6` = P.`player_api_id` OR M.`home_player_7` = P.`player_api_id` OR M.`home_player_8` = P.`player_api_id` OR M.`home_player_9` = P.`player_api_id` OR M.`home_player_10` = P.`player_api_id` OR M.`home_player_11` = P.`player_api_id` OR M.`away_player_1` = P.`player_api_id` OR M.`away_player_2` = P.`player_api_id` OR M.`away_player_3` = P.`player_api_id` OR M.`away_player_4` = P.`player_api_id` OR M.`away_player_5` = P.`player_api_id` OR M.`away_player_6` = P.`player_api_id` OR M.`away_player_7` = P.`player_api_id` OR M.`away_player_8` = P.`player_api_id` OR M.`away_player_9` = P.`player_api_id` OR M.`away_player_10` = P.`player_api_id` OR M.`away_player_11` = P.`player_api_id` INNER JOIN League AS L ON M.`league_id` = L.`id` INNER JOIN Country AS C ON L.`country_id` = C.`id` WHERE PA.`vision` > 89

"""

    run_queries(db_path, query1, query2)


if __name__ == "__main__":
    main()
