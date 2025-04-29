from src.core.model_manager.utils import compose_chat_messages
from src.data_prep.database_schemas.bird_schema import BIRD_Schema
from src.core.model_manager.model_manager import ModelManager, ModelProvider, ModelType
from src.core.prompt_renderer.prompt_renderer import PromptRenderer
from src.core.model_manager.openai_model import OpenAIModel

from dotenv import load_dotenv
import os

load_dotenv()


def _generate_response(prompt_key, prompt_args, response_prefix):
    prompt = [PromptRenderer("./prompt_templates").render(prompt_key, prompt_args)]
    messages = compose_chat_messages(user_messages=prompt)
    model = ModelManager.get_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.COMPLETION,
        model_name=OpenAIModel.GPT_4O,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    response = model.get_chat_completion(
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["completion_content"][0].replace(response_prefix, "").strip()


def generate_explicit_question(main_query, db_id, data_folder):
    schema_data_path = os.path.join(
        data_folder, "benchmarks", "Bird", "dev_tables.json"
    )
    schema = BIRD_Schema(db_id, schema_data_path)

    return _generate_response(
        prompt_key="explicit_question",
        prompt_args={"schema": schema, "main_query": main_query},
        response_prefix="Question:",
    )


def generate_dev_set_like_question(main_query, db_id, data_folder):
    schema_data_path = os.path.join(
        data_folder, "benchmarks", "Bird", "dev_tables.json"
    )
    schema = BIRD_Schema(db_id, schema_data_path)
    return _generate_response(
        prompt_key="dev_set_like_question",
        prompt_args={"schema": schema, "main_query": main_query},
        response_prefix="Question:",
    )


def generate_evidence(main_query, question, db_id, data_folder):
    schema_data_path = os.path.join(
        data_folder, "benchmarks", "Bird", "dev_tables.json"
    )
    schema = BIRD_Schema(db_id, schema_data_path)
    return _generate_response(
        prompt_key="evidence",
        prompt_args={"schema": schema, "main_query": main_query, "question": question},
        response_prefix="Evidence:",
    )


def generate_new_extended_question(
    old_query, new_query, old_question, db_id, data_folder
):
    schema_data_path = os.path.join(
        data_folder, "benchmarks", "Bird", "dev_tables.json"
    )
    schema = BIRD_Schema(db_id, schema_data_path)
    return _generate_response(
        prompt_key="extended_old_question",
        prompt_args={
            "schema": schema,
            "old_query": old_query,
            "old_question": old_question,
            "new_query": new_query,
        },
        response_prefix="New question:",
    )


def gen_nl(filtered_aug, graph_first=False):
    data_folder = os.getenv("DATA_FOLDER")
    for query in filtered_aug:
        new_extended_question = generate_new_extended_question(
            old_query=query["query_first"]["SQL"],
            new_query=query["query_first"]["new_query"],
            old_question=query["query_first"]["question"],
            db_id=query["db_id"],
            data_folder=data_folder,
        )
        query["query_first"]["new_question"] = new_extended_question

        if graph_first:
            explicit_question = generate_explicit_question(
                main_query=query["graph_first"]["main_query"],
                db_id=query["graph_first"]["db_id"],
                data_folder=data_folder,
            )
            query["graph_first"]["explicit_question"] = explicit_question
            dev_set_like_question = generate_dev_set_like_question(
                main_query=query["graph_first"]["main_query"],
                db_id=query["graph_first"]["db_id"],
                data_folder=data_folder,
            )
            query["graph_first"]["dev_set_like_question"] = dev_set_like_question
            new_evidence = generate_evidence(
                main_query=query["graph_first"]["main_query"],
                schema=query["graph_first"]["schema"],
                db_id=query["graph_first"]["db_id"],
                question=new_extended_question,
            )
            query["graph_first"]["new_evidence"] = new_evidence

    return filtered_aug
