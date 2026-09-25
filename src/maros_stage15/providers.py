"""LLM provider adapters.

The main path is OpenAI-compatible HTTP, so the same agent code can call
local vLLM/SGLang/Qwen servers or hosted OpenAI-compatible endpoints.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence


class ChatModel(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        ...


@dataclass
class OpenAICompatibleChatModel:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    timeout_seconds: float = 120.0

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": list(messages),
            "temperature": float(self.temperature),
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach LLM endpoint {url}: {exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"unexpected chat completion payload: {payload}") from exc


class ScriptedChatModel:
    """Deterministic provider used by unit tests and protocol smoke tests."""

    def __init__(self, responses: Sequence[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("ScriptedChatModel has no response left")
        return self.responses.pop(0)
