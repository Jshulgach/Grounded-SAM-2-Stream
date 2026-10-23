"""Optional explicit Ollama phrase suggestion; direct phrases never call an LLM."""

from __future__ import annotations

import asyncio
import json
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .contracts import targets

MAX_REPLY = 16384


class PhraseError(RuntimeError):
    pass


def model_exists(model_id, models):
    """Match complete tags, accepting both current and historical Ollama tag fields."""
    expected = model_id if ":" in model_id else model_id + ":latest"
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("model") or item.get("name")
        if isinstance(name, str) and (name if ":" in name else name + ":latest") == expected:
            return True
    return False


class OllamaPhrases:
    def __init__(self, model="llama3.1", *, base_url="http://127.0.0.1:11434", timeout=30):
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("The phrase adapter accepts a local Ollama server only.")
        if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("Use a plain local Ollama origin.")
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError("Specify an installed Ollama model tag.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 1 <= timeout <= 120
        ):
            raise ValueError("Timeout must be between 1 and 120 seconds.")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.opener = build_opener(ProxyHandler({}))

    def _json(self, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_REPLY + 1)
            if len(raw) > MAX_REPLY:
                raise PhraseError("Ollama response exceeded the limit.")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object.")
            return value
        except (URLError, TimeoutError, ValueError) as exc:
            raise PhraseError("Ollama request failed or returned invalid JSON.") from exc

    def list_models(self):
        models = self._json("/api/tags").get("models")
        if not isinstance(models, list):
            raise PhraseError("Ollama returned an invalid model list.")
        return models

    def suggest(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("Query must contain 1 to 1000 characters.")
        if not model_exists(self.model, self.list_models()):
            raise PhraseError(
                "The exact model tag is not installed. No model was pulled automatically."
            )
        response = self._json(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "format": {
                    "type": "object",
                    "required": ["targets"],
                    "additionalProperties": False,
                    "properties": {
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 120},
                        }
                    },
                },
                "messages": [
                    {
                        "role": "system",
                        "content": "Suggest short visible object phrases for a camera search. Preserve explicit colors and attributes. Do not infer identities, locations, or claim an object was seen. Return only the targets JSON object.",
                    },
                    {"role": "user", "content": query},
                ],
            },
        )
        try:
            parsed = json.loads(response["message"]["content"])
            if not isinstance(parsed, dict) or set(parsed) != {"targets"}:
                raise ValueError("Invalid phrase fields.")
            phrases = targets(parsed["targets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PhraseError("Ollama did not return valid object phrases.") from exc
        return {"original_query": query, "targets": list(phrases), "needs_confirmation": True}

    async def suggest_async(self, query):
        return await asyncio.to_thread(self.suggest, query)
