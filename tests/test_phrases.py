import asyncio
import json
import time

import pytest

from grounded_stream.phrases import OllamaPhrases, PhraseError, model_exists


def test_exact_model_tags_and_schema_variants():
    assert model_exists("llama3.1", [{"model": "llama3.1:latest"}])
    assert model_exists("llama3.1:8b", [{"name": "llama3.1:8b"}])
    assert not model_exists("llama3", [{"model": "llama3.1:latest"}])
    assert not model_exists("llama3.1", [{"model": "llama3.1:8b"}])
    assert not model_exists("x", [{}, None])


def stub(monkeypatch, content):
    adapter = OllamaPhrases()

    def response(path, body=None):
        return (
            {"models": [{"model": "llama3.1:latest"}]}
            if path == "/api/tags"
            else {"message": {"content": content}}
        )

    monkeypatch.setattr(adapter, "_json", response)
    return adapter


def test_suggestion_preserves_original_and_requires_confirmation(monkeypatch):
    adapter = stub(monkeypatch, json.dumps({"targets": ["red backpack"]}))
    got = adapter.suggest("Where did I put my red backpack?")
    assert got["original_query"] == "Where did I put my red backpack?"
    assert got["targets"] == ["red backpack"] and got["needs_confirmation"]


@pytest.mark.parametrize("reply", ["bad", "[]", '{"targets":[]}', '{"targets":["toy"],"extra":1}'])
def test_invalid_structured_output(monkeypatch, reply):
    with pytest.raises(PhraseError):
        stub(monkeypatch, reply).suggest("toy")


def test_async_adapter_does_not_block_event_loop(monkeypatch):
    adapter = OllamaPhrases()

    def slow(query):
        time.sleep(0.2)
        return {"targets": [query]}

    monkeypatch.setattr(adapter, "suggest", slow)

    async def check():
        task = asyncio.create_task(adapter.suggest_async("toy"))
        start = time.perf_counter()
        await asyncio.sleep(0.03)
        assert time.perf_counter() - start < 0.15
        assert (await task)["targets"] == ["toy"]

    asyncio.run(check())


@pytest.mark.parametrize(
    "url",
    ["https://example.com", "http://localhost:11434/path", "http://user:pass@localhost:11434"],
)
def test_local_adapter_boundary(url):
    with pytest.raises(ValueError):
        OllamaPhrases(base_url=url)
