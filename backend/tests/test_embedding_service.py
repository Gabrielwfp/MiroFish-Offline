"""
Unit tests (mocked HTTP, no network) for EmbeddingService, embedding config and
the Neo4j vector-index schema.  Run:  cd backend && python -m pytest tests -q
"""

import importlib
import importlib.util

import pytest
import requests

from app.config import Config
from app.storage import embedding_service as es
from app.storage import neo4j_schema
from app.storage.embedding_service import EmbeddingError, EmbeddingService


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


def openai_payload(vectors, shuffle=False):
    items = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {"object": "list", "data": list(reversed(items)) if shuffle else items}


@pytest.fixture
def cfg(monkeypatch):
    """Isolate from the real .env: fixed test values, never real credentials."""
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "test-embed")
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "http://embed.test/v1/")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSIONS", 4)
    monkeypatch.setattr(es.time, "sleep", lambda s: None)  # no real backoff waits
    return Config


@pytest.fixture
def calls(monkeypatch):
    """Patch requests.post; tests queue responses in calls.responses."""
    class Calls(list):
        pass

    recorded = Calls()
    recorded.responses = []

    def fake_post(url, json=None, headers=None, timeout=None):
        recorded.append({"url": url, "json": json, "headers": headers})
        r = recorded.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(es.requests, "post", fake_post)
    return recorded


# --- request format / auth -------------------------------------------------

def test_posts_openai_format_with_bearer_auth(cfg, calls):
    calls.responses.append(FakeResponse(payload=openai_payload([[0.1, 0.2, 0.3, 0.4]])))
    vec = EmbeddingService().embed("  hola  ")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    c = calls[0]
    assert c["url"] == "http://embed.test/v1/embeddings"  # trailing slash stripped, no /api/embed
    assert c["json"] == {"model": "test-embed", "input": ["hola"]}
    assert c["headers"] == {"Authorization": "Bearer test-key"}


def test_no_auth_header_when_key_missing(cfg, calls, monkeypatch):
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", None)
    calls.responses.append(FakeResponse(payload=openai_payload([[1, 0, 0, 0]])))
    EmbeddingService().embed("x")
    assert calls[0]["headers"] == {}  # never sends "Bearer None"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_fail_fast_without_retry(cfg, calls, status):
    calls.responses.extend([FakeResponse(status=status, text='{"error":"nope"}')] * 3)
    with pytest.raises(EmbeddingError, match=str(status)):
        EmbeddingService().embed("x")
    assert len(calls) == 1


def test_server_errors_retry_then_succeed(cfg, calls):
    calls.responses.extend([
        FakeResponse(status=503, text="busy"),
        FakeResponse(payload=openai_payload([[1, 2, 3, 4]])),
    ])
    assert EmbeddingService().embed("x") == [1, 2, 3, 4]
    assert len(calls) == 2


def test_connection_errors_exhaust_retries(cfg, calls):
    calls.responses.extend([requests.exceptions.ConnectionError("down")] * 3)
    with pytest.raises(EmbeddingError, match="after 3 retries"):
        EmbeddingService().embed("x")
    assert len(calls) == 3


# --- response parsing / dimensions -----------------------------------------

def test_batch_respects_index_order_cache_and_empty_texts(cfg, calls):
    v1, v2 = [1, 0, 0, 0], [0, 1, 0, 0]
    calls.responses.append(FakeResponse(payload=openai_payload([v1, v2], shuffle=True)))
    svc = EmbeddingService()
    out = svc.embed_batch(["a", "", "b"])
    assert out == [v1, [0.0] * 4, v2]
    assert calls[0]["json"]["input"] == ["a", "b"]
    assert svc.embed("a") == v1 and len(calls) == 1  # served from cache


def test_dimension_mismatch_is_rejected(cfg, calls):
    calls.responses.append(FakeResponse(payload=openai_payload([[0.1] * 768])))
    with pytest.raises(EmbeddingError, match="768-dim.*EMBEDDING_DIMENSIONS=4"):
        EmbeddingService().embed("x")


def test_count_mismatch_is_rejected(cfg, calls):
    calls.responses.append(FakeResponse(payload=openai_payload([[1, 0, 0, 0]])))
    with pytest.raises(EmbeddingError, match="Expected 2"):
        EmbeddingService().embed_batch(["a", "b"])


@pytest.mark.parametrize("payload", [
    {"embeddings": [[1, 0, 0, 0]]},          # Ollama-native shape, not OpenAI
    {"data": [{"index": 0}]},               # missing "embedding"
    {"data": ["not-a-dict"]},               # wrong item type
    ValueError("not json"),                 # non-JSON body
])
def test_malformed_responses_raise_embedding_error(cfg, calls, payload):
    calls.responses.append(FakeResponse(payload=payload))
    with pytest.raises(EmbeddingError):
        EmbeddingService().embed("x")


def test_empty_text_rejected_without_request(cfg, calls):
    with pytest.raises(EmbeddingError):
        EmbeddingService().embed("   ")
    assert calls == []


# --- config / schema ---------------------------------------------------------

def _config_from_env():
    """Evaluate config.py against os.environ only (dotenv loading stripped)."""
    spec = importlib.util.find_spec("app.config")
    src = spec.loader.get_source("app.config")
    src = src.replace("load_dotenv(project_root_env, override=True)", "pass") \
             .replace("load_dotenv(override=True)", "pass")
    ns = {"__name__": "config_probe", "__file__": spec.origin}
    exec(compile(src, "config_probe", "exec"), ns)
    return ns["Config"]


def test_embedding_key_and_dims_from_env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    C = _config_from_env()
    assert C.EMBEDDING_API_KEY == "llm-test-key"   # falls back to the LLM key
    assert C.EMBEDDING_DIMENSIONS == 1024

    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-test-key")
    assert _config_from_env().EMBEDDING_API_KEY == "embed-test-key"  # independent key wins

    monkeypatch.delenv("EMBEDDING_DIMENSIONS")
    # importing app.config loaded the developer's real .env into os.environ; drop it here
    for var in ("EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    C = _config_from_env()
    assert (C.EMBEDDING_MODEL, C.EMBEDDING_DIMENSIONS) == ("qwen3-embedding:0.6b", 1024)  # consistent defaults
    assert C.EMBEDDING_BASE_URL.endswith(":11434/v1") and C.LLM_BASE_URL.endswith(":11434/v1")


def test_schema_vector_indexes_use_configured_dimension(monkeypatch):
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSIONS", 1024)
    schema = importlib.reload(neo4j_schema)
    try:
        for q in (schema.CREATE_ENTITY_VECTOR_INDEX, schema.CREATE_RELATION_VECTOR_INDEX):
            assert "`vector.dimensions`: 1024" in q
            assert "'cosine'" in q
            assert "768" not in q
        assert schema.CREATE_ENTITY_VECTOR_INDEX in schema.ALL_SCHEMA_QUERIES
        assert schema.CREATE_RELATION_VECTOR_INDEX in schema.ALL_SCHEMA_QUERIES
    finally:
        monkeypatch.undo()
        importlib.reload(neo4j_schema)
