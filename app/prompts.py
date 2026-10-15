"""The single prompt every provider sees, including the offline one."""

import re
from collections.abc import Sequence

from app.models import ChatMessage, RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions for the Lumen Archive demo using only the numbered excerpts. "
    "Cite supporting excerpts inline as [1] or [2]. "
    "If the excerpts do not contain the answer, reply exactly: I do not know. "
    "Do not invent sources, prices, or policies."
)

_EXCERPT = re.compile(
    r"\[(\d+)\]\nsource: (?P<source>[^\n]*)\ntitle: (?P<title>[^\n]*)\n"
    r"text:\n(?P<body>.*?)(?=\n\n---\n\n|\n\nQuestion: )",
    re.DOTALL,
)
_QUESTION = "\n\nQuestion: "


def build_messages(question: str, excerpts: Sequence[RetrievedChunk]) -> list[ChatMessage]:
    if excerpts:
        blocks = [
            (
                f"[{item.index}]\nsource: {item.source}\n"
                f"title: {item.title}\ntext:\n{item.text.strip()}"
            )
            for item in excerpts
        ]
        rendered = "\n\n---\n\n".join(blocks)
    else:
        rendered = "(no excerpts retrieved)"
    user = f"Excerpts:\n{rendered}{_QUESTION}{question.strip()}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


def question_from_messages(messages: Sequence[ChatMessage]) -> str:
    user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    if _QUESTION in user:
        return user.rsplit(_QUESTION, 1)[1].strip()
    return user.strip()


def excerpts_from_messages(messages: Sequence[ChatMessage]) -> list[tuple[int, str]]:
    user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    return [
        (int(match.group(1)), match.group("body").strip()) for match in _EXCERPT.finditer(user)
    ]
