from collections.abc import Sequence

from app.llm.types import ChatMessage


def split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
    system_parts = [message.content for message in messages if message.role == "system"]
    rest = [message for message in messages if message.role != "system"]
    if not system_parts:
        return None, rest
    return "\n\n".join(system_parts), rest
