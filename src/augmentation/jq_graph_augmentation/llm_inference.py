from src.core.model_manager.utils import compose_chat_messages


def _generate_response(
    prompt_key, prompt_args, prompt_renderer, model, response_prefix
):
    prompt = [prompt_renderer.render(prompt_key, prompt_args)]
    messages = compose_chat_messages(user_messages=prompt)
    response = model.get_chat_completion(
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["completion_content"][0].replace(response_prefix, "").strip()


def generate_explicit_question(main_query, schema, prompt_renderer, model):
    return _generate_response(
        prompt_key="explicit_question",
        prompt_args={"schema": schema, "main_query": main_query},
        prompt_renderer=prompt_renderer,
        model=model,
        response_prefix="Question:",
    )


def generate_dev_set_like_question(main_query, schema, prompt_renderer, model):
    return _generate_response(
        prompt_key="dev_set_like_question",
        prompt_args={"schema": schema, "main_query": main_query},
        prompt_renderer=prompt_renderer,
        model=model,
        response_prefix="Question:",
    )


def generate_evidence(main_query, schema, question, prompt_renderer, model):
    return _generate_response(
        prompt_key="evidence",
        prompt_args={"schema": schema, "main_query": main_query, "question": question},
        prompt_renderer=prompt_renderer,
        model=model,
        response_prefix="Evidence:",
    )


def generate_new_extended_question(
    old_query, new_query, schema, old_question, prompt_renderer, model
):
    return _generate_response(
        prompt_key="extended_old_question",
        prompt_args={
            "schema": schema,
            "old_query": old_query,
            "old_question": old_question,
            "new_query": new_query,
        },
        prompt_renderer=prompt_renderer,
        model=model,
        response_prefix="New question:",
    )
