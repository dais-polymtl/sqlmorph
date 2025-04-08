from typing import Dict, List


def compose_chat_messages(
    user_messages: List[str],
    assistant_messages: List[str] | None = None,
    system_message: str | None = None,
) -> List[Dict[str, str]]:
    """
    Compose a list of messages for the Chat Completions API.

    Args:
        user_messages (List[str]): List of messages from the user.
        assistant_messages (List[str], optional): List of messages from the assistant, corresponding to each user message.
        system_message (str, optional): A message from the system to set the behavior of the assistant.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries suitable for the Chat Completions API.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if assistant_messages is None:
        assistant_messages = [None] * len(user_messages)
    for user_message, assistant_message in zip(user_messages, assistant_messages):
        messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
    return messages
