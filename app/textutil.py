import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "where",
        "which",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, without very common function words."""
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]
