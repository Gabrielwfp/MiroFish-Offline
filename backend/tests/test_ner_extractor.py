"""
Unit tests (mocked LLM, no network) for the NER contract, LLMClient structured
output options and Neo4jStorage.add_text failure propagation.
Run:  cd backend && python -m pytest tests -q
"""

from types import SimpleNamespace

import pytest

from app.storage.embedding_service import EmbeddingError
from app.storage.ner_extractor import NERExtractionError, NERExtractor
from app.utils.llm_client import LLMClient

ONTOLOGY = {
    "entity_types": [{"name": "Person", "description": "a person"}, {"name": "Organization"}],
    "edge_types": [{"name": "WORKS_FOR", "source_targets": [{"source": "Person", "target": "Organization"}]}],
}
VALID = {
    "entities": [
        {"name": "Ana Pérez", "type": "Person", "attributes": {}},
        {"name": "Banco Central", "type": "Organization", "attributes": {"country": "DO"}},
    ],
    "relations": [{"source": "Ana Pérez", "target": "Banco Central", "type": "WORKS_FOR", "fact": "Ana works there"}],
}


class FakeLLM:
    """Stands in for LLMClient: returns queued responses, records kwargs."""
    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# --- NER contract -------------------------------------------------------------

def test_valid_response_passes_and_sends_schema_without_thinking():
    llm = FakeLLM(VALID)
    out = NERExtractor(llm_client=llm).extract("Ana Pérez trabaja para el Banco Central.", ONTOLOGY)
    assert [e["name"] for e in out["entities"]] == ["Ana Pérez", "Banco Central"]
    assert out["relations"][0]["type"] == "WORKS_FOR"
    call = llm.calls[0]
    assert call["disable_thinking"] is True
    schema = call["schema"]
    assert schema["required"] == ["entities", "relations"]
    ent = schema["properties"]["entities"]["items"]["properties"]
    rel = schema["properties"]["relations"]["items"]["properties"]
    assert ent["type"]["enum"] == ["Person", "Organization"]  # ontology types are enforced
    assert rel["type"]["enum"] == ["WORKS_FOR"]


def test_schema_without_ontology_uses_free_strings():
    llm = FakeLLM({"entities": [], "relations": []})
    NERExtractor(llm_client=llm).extract("texto", {})
    ent = llm.calls[0]["schema"]["properties"]["entities"]["items"]["properties"]
    assert ent["type"] == {"type": "string"}


@pytest.mark.parametrize("bad", [
    [{"name": "Ana Pérez", "type": "Person"}],                                  # bare list (seen with oMLX)
    {"entities": [], "relationships": []},                                       # wrong key (seen with json_object)
    {"entities": [{"name": "Ana"}], "relations": []},                            # entity missing type
    {"entities": [], "relations": [{"subject": "a", "object": "b", "relation": "R"}]},  # wrong relation fields
    {"entities": [{"name": "", "type": "Person"}], "relations": []},              # empty name
])
def test_contract_violations_are_retried_then_raise(bad):
    llm = FakeLLM(bad, bad, bad)
    with pytest.raises(NERExtractionError, match="after 3 attempts"):
        NERExtractor(llm_client=llm, max_retries=2).extract("texto", ONTOLOGY)
    assert len(llm.calls) == 3  # bounded retries


def test_recovers_when_a_retry_complies():
    llm = FakeLLM([{"name": "x"}], ValueError("Invalid JSON format from LLM: ..."), VALID)
    out = NERExtractor(llm_client=llm, max_retries=2).extract("texto", ONTOLOGY)
    assert len(out["entities"]) == 2 and len(llm.calls) == 3


def test_llm_errors_raise_after_retries():
    llm = FakeLLM(ConnectionError("ollama down"), ConnectionError("ollama down"))
    with pytest.raises(NERExtractionError, match="ollama down"):
        NERExtractor(llm_client=llm, max_retries=1).extract("texto", ONTOLOGY)


def test_valid_empty_extraction_is_not_an_error():
    out = NERExtractor(llm_client=FakeLLM({"entities": [], "relations": []})).extract("Hola.", ONTOLOGY)
    assert out == {"entities": [], "relations": []}


def test_empty_text_skips_llm():
    llm = FakeLLM()
    assert NERExtractor(llm_client=llm).extract("   ", ONTOLOGY) == {"entities": [], "relations": []}
    assert llm.calls == []


# --- LLMClient structured output options -------------------------------------

def _client(base_url, captured):
    c = LLMClient(api_key="test", base_url=base_url, model="m", api_mode="openai")

    def create(**kwargs):
        captured.update(kwargs)
        msg = SimpleNamespace(content='{"entities": [], "relations": []}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c


def test_chat_json_with_schema_uses_json_schema_and_disables_thinking_on_ollama():
    captured = {}
    schema = {"type": "object"}
    out = _client("http://localhost:11434/v1", captured).chat_json(
        messages=[{"role": "user", "content": "x"}], schema=schema, disable_thinking=True)
    assert out == {"entities": [], "relations": []}
    assert captured["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": True}}
    assert captured["reasoning_effort"] == "none"
    assert "extra_body" not in captured  # /v1 ignores options.num_ctx; only ollama_native sends it


def test_chat_json_defaults_unchanged_for_existing_callers():
    captured = {}
    _client("http://localhost:11434/v1", captured).chat_json(messages=[{"role": "user", "content": "x"}])
    assert captured["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in captured


def test_reasoning_effort_not_sent_to_non_ollama_servers():
    captured = {}
    _client("http://other-server:8000/v1", captured).chat_json(
        messages=[{"role": "user", "content": "x"}], schema={"type": "object"}, disable_thinking=True)
    assert "reasoning_effort" not in captured


def test_chat_json_reports_truncation_instead_of_invalid_json():
    c = _client("http://localhost:11434/v1", {})
    c.client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content='{"entity_types": [{"name": "Off'))])
    with pytest.raises(ValueError, match="truncated at max_tokens=4096"):
        c.chat_json(messages=[{"role": "user", "content": "x"}])


def test_chat_handles_empty_content():
    c = _client("http://localhost:11434/v1", {})
    c.client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")])
    assert c.chat(messages=[{"role": "user", "content": "x"}]) == ""


# --- Neo4jStorage.add_text failure propagation ---------------------------------

def _storage(ner, embedding):
    from app.storage.neo4j_storage import Neo4jStorage
    st = Neo4jStorage.__new__(Neo4jStorage)  # skip __init__ (no Neo4j connection)
    st._ner, st._embedding = ner, embedding
    st._driver = SimpleNamespace(session=lambda: pytest.fail("must not write to Neo4j on failure"))
    st.get_ontology = lambda graph_id: ONTOLOGY
    return st


def test_add_text_propagates_embedding_failure_before_writing():
    class BrokenEmbedding:
        def embed_batch(self, texts):
            raise EmbeddingError("Embedding request failed (404): model not found")

    st = _storage(NERExtractor(llm_client=FakeLLM(VALID)), BrokenEmbedding())
    with pytest.raises(EmbeddingError, match="404"):
        st.add_text("g1", "Ana Pérez trabaja para el Banco Central.")


def test_add_text_propagates_ner_failure_before_writing():
    st = _storage(NERExtractor(llm_client=FakeLLM([1], [1], [1])), embedding=None)
    with pytest.raises(NERExtractionError):
        st.add_text("g1", "texto")
