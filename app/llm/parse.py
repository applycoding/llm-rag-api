"""Turn provider stream lines into text deltas."""

import json

from app.llm.base import ProviderError


def parse_openai_data_line(line: str) -> str | None:
    payload = _sse_json(line)
    if payload is None:
        return None
    if "error" in payload and not payload.get("choices"):
        message = payload["error"]
        if isinstance(message, dict):
            message = message.get("message", message)
        raise ProviderError(f"Model host returned an error: {message}")
    choices = payload.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str) and content:
        return content
    return None


def parse_anthropic_data_line(line: str) -> str | None:
    payload = _sse_json(line)
    if payload is None or payload.get("type") != "content_block_delta":
        return None
    delta = payload.get("delta") or {}
    if delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def parse_gemini_data_line(line: str) -> str | None:
    payload = _sse_json(line)
    if payload is None:
        return None
    if payload.get("error"):
        message = payload["error"]
        if isinstance(message, dict):
            message = message.get("message", message)
        raise ProviderError(f"Gemini returned an error: {message}")
    candidates = payload.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]
    ]
    if not texts:
        return None
    return "".join(texts)


def parse_ollama_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError("Ollama returned a line that is not JSON.") from exc
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        raise ProviderError(f"Ollama returned an error: {payload['error']}")
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content:
        return content
    return None


def _sse_json(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    data = stripped[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProviderError("The model stream returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        return None
    return payload
