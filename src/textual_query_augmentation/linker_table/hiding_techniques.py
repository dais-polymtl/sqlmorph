import os
import openai
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Tuple
from src.core.model_manager.utils import compose_chat_messages
from src.core.model_manager.model_manager import ModelManager, ModelProvider, ModelType
from src.core.prompt_renderer.prompt_renderer import PromptRenderer
from src.core.model_manager.openai_model import OpenAIModel
from src.core.logger.logger import Logger

load_dotenv()
logger = Logger(name=__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


class TokenInfo(BaseModel):
    token: str
    position: int


def replace_tokens_in_text(
    original_text: str, new_tokens: List[Tuple[str, int]]
) -> str:
    text = list(original_text)  # Convert to list for mutable string

    # Sort by position in reverse to avoid shifting indices
    for new_token, pos in sorted(new_tokens, key=lambda x: -x[1]):
        end = pos
        while end < len(text) and text[end].isalnum():
            end += 1

        # Replace characters
        text[pos:end] = list(new_token)

    return "".join(text)


def _generate_response(prompt_key, prompt_args, model_args=None):
    prompt = [
        PromptRenderer(
            os.path.join(os.path.dirname(__file__), "prompt_templates")
        ).render(prompt_key, prompt_args)
    ]
    messages = compose_chat_messages(user_messages=prompt)
    model = ModelManager.create_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.COMPLETION,
        model_name=OpenAIModel.GPT_4O,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    response = model.get_chat_completion(
        messages=messages,
        max_tokens=3000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=model_args.get("response_format", None),
    )
    generated_text = response["completion_content"][0].strip()

    return generated_text


def hide_tables_with_synonym_replacement(jqgs):
    """Processes queries by applying synonym replacement and filtering unchanged ones."""

    def process_components(components, table_name, db_id):
        result = []
        for comp in components:
            new_tokens = _generate_response(
                prompt_key="synonym_replacement",
                prompt_args={
                    "tokens": comp["matched_tokens"],
                    "table_name": table_name,
                    "db_id": db_id,
                },
                model_args={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Test",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "response": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "token": {"type": "string"},
                                                "position": {"type": "integer"},
                                            },
                                            "required": ["token", "position"],
                                            "additionalProperties": False,
                                        },
                                    }
                                },
                                "required": ["response"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                },
            )
            comp_copy = comp.copy()
            comp_copy["new_matched_tokens"] = json.loads(new_tokens).get("response", [])
            result.append(comp_copy)
        return result

    rewritten_queries = []
    logger.log(
        level="info",
        action="hide_tables_with_synonym_replacement",
        details={"num_queries": len(jqgs), "technique": "synonym_replacement"},
    )

    for i, jqg in enumerate(jqgs):
        jqg_copy = jqg.copy()
        logger.log(
            level="debug",
            action=f"Processing query {i + 1}/{len(jqgs)}: using synonym replacement",
        )

        q_comps = process_components(
            jqg["question_components"], jqg["central_table"], jqg["db_id"]
        )
        e_comps = process_components(
            jqg.get("evidence_components", []), jqg["central_table"], jqg["db_id"]
        )

        jqg_copy["new_question"] = replace_tokens_in_text(
            jqg["question"],
            [
                (t["token"], int(t["position"]))
                for c in q_comps
                for t in c["new_matched_tokens"]
            ],
        )
        jqg_copy["new_evidence"] = replace_tokens_in_text(
            jqg["evidence"],
            [
                (t["token"], int(t["position"]))
                for c in e_comps
                for t in c["new_matched_tokens"]
            ],
        )
        rewritten_queries.append(jqg_copy)

    return rewritten_queries


def hide_tables_with_backtranslation(jqgs):
    """Processes queries by applying back translation and filtering unchanged ones."""

    rewritten_queries = []
    logger.log(
        level="info",
        action="hide_tables_with_backtranslation",
        details={"num_queries": len(jqgs), "technique": "backtranslation"},
    )

    for i, jqg in enumerate(jqgs):
        jqg_copy = jqg.copy()
        logger.log(
            level="debug",
            action=f"Processing query {i + 1}/{len(jqgs)}: using backtranslation",
        )

        common_args = {
            "db_id": jqg["db_id"],
            "sql_query": jqg["SQL"],
            "candidate_table": jqg["central_table"],
        }

        new_question = _generate_response(
            prompt_key="backtranslation",
            prompt_args={
                **common_args,
                "source": jqg["question"],
                "components": jqg["question_components"],
                "source_type": "question",
            },
            model_args={"response_format": {"type": "text"}},
        )

        new_evidence = _generate_response(
            prompt_key="backtranslation",
            prompt_args={
                **common_args,
                "source": jqg["evidence"],
                "components": jqg.get("evidence_components", []),
                "source_type": "evidence",
                "new_question": new_question,
            },
            model_args={"response_format": {"type": "text"}},
        )

        jqg_copy["new_question"] = (
            new_question.replace("New question:", "")
            .replace("Rewritten question:", "")
            .strip()
        )
        jqg_copy["new_evidence"] = (
            new_evidence.replace("New evidence:", "")
            .replace("Rewritten evidence:", "")
            .strip()
        )
        rewritten_queries.append(jqg_copy)

    return rewritten_queries


def hide_tables_with_contextual_augmentation(jqgs):
    """Processes queries by applying contextual augmentation and filtering unchanged ones."""

    rewritten_queries = []
    logger.log(
        level="info",
        action="hide_tables_with_contextual_augmentation",
        details={"num_queries": len(jqgs), "technique": "contextual_augmentation"},
    )

    for i, jqg in enumerate(jqgs):
        jqg_copy = jqg.copy()
        logger.log(
            level="debug",
            action=f"Processing query {i + 1}/{len(jqgs)} for contextual augmentation",
        )

        common_args = {
            "db_id": jqg["db_id"],
            "sql_query": jqg["SQL"],
            "candidate_table": jqg["central_table"],
        }
        model_args = {"response_format": {"type": "text"}}

        new_question = _generate_response(
            prompt_key="contextual_augmentation",
            prompt_args={
                **common_args,
                "source": jqg["question"],
                "components": jqg["question_components"],
                "source_type": "question",
            },
            model_args=model_args,
        )

        new_evidence = _generate_response(
            prompt_key="contextual_augmentation",
            prompt_args={
                **common_args,
                "source": jqg["evidence"],
                "components": jqg.get("evidence_components", []),
                "source_type": "evidence",
                "new_question": new_question,
            },
            model_args=model_args,
        )

        jqg_copy["new_question"] = (
            new_question.replace("New question:", "")
            .replace("Rewritten question:", "")
            .strip()
        )
        jqg_copy["new_evidence"] = (
            new_evidence.replace("New evidence:", "")
            .replace("Rewritten evidence:", "")
            .strip()
        )
        rewritten_queries.append(jqg_copy)

    return rewritten_queries
