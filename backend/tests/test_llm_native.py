"""
Unit tests (mocked HTTP, no network) for LLMClient's ollama_native transport:
/api/chat payload, per-operation num_ctx, context preflight, error mapping, retries,
and that context errors surface through NER, profile and simulation-config generation.
Run:  cd backend && python -m pytest tests -q
"""

from types import SimpleNamespace

import pytest
import requests

from app.config import Config
from app.utils import llm_client as lc
from app.utils.llm_client import ContextOverflowError, LLMClient


class Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text or str(payload)

    def json(self):
        return self._payload


def ok(content='{"ok": true}', prompt=100, out=20, done="stop"):
    return Resp(200, {"message": {"role": "assistant", "content": content},
                      "prompt_eval_count": prompt, "eval_count": out, "done_reason": done})


@pytest.fixture
def native(monkeypatch):
    """ollama_native client with per-operation num_ctx; requests.post captured."""
    monkeypatch.setattr(Config, "OLLAMA_NUM_CTX_BY_OPERATION",
                        {"ontology": 32768, "ner": 16384, "profile": 16384, "sim_config": 32768, "report": 32768, "chat": 32768})
    monkeypatch.setattr(Config, "OLLAMA_NUM_CTX_DEFAULT", 32768)
    monkeypatch.setattr(Config, "CONTEXT_CHARS_PER_TOKEN", 3.0)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    calls, queue = [], []

    def post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(lc.requests, "post", post)
    client = LLMClient(api_key="ollama", base_url="http://localhost:11434/v1", model="ornith-test",
                       api_mode="ollama_native", timeout=42)
    return SimpleNamespace(client=client, calls=calls, queue=queue)


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hola"}]


def test_posts_native_api_with_operation_num_ctx(native):
    native.queue.append(ok("respuesta"))
    out = native.client.chat(MSGS, temperature=0.2, max_tokens=512, operation="ner", disable_thinking=True)
    assert out == "respuesta"
    c = native.calls[0]
    assert c["url"] == "http://localhost:11434/api/chat"  # /v1 stripped
    assert c["timeout"] == 42
    body = c["json"]
    assert body["model"] == "ornith-test" and body["stream"] is False and body["think"] is False
    assert body["options"] == {"num_ctx": 16384, "num_predict": 512, "temperature": 0.2}
    assert native.client.last_usage["num_ctx"] == 16384 and native.client.last_usage["prompt_tokens"] == 100


def test_operation_on_client_and_default_fallback(native):
    native.queue.extend([ok(), ok()])
    LLMClient(api_key="x", base_url="http://h:11434/v1", model="m", api_mode="ollama_native",
              operation="profile").chat(MSGS)
    native.client.chat(MSGS, operation="unknown-op")
    assert native.calls[0]["json"]["options"]["num_ctx"] == 16384
    assert native.calls[1]["json"]["options"]["num_ctx"] == 32768  # default


def test_json_schema_and_json_mode_map_to_format(native):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    native.queue.extend([ok(), ok()])
    assert native.client.chat_json(MSGS, schema=schema, operation="ner") == {"ok": True}
    assert native.client.chat_json(MSGS, operation="ner") == {"ok": True}
    assert native.calls[0]["json"]["format"] == schema
    assert native.calls[1]["json"]["format"] == "json"


def test_preflight_rejects_without_sending(native):
    big = [{"role": "user", "content": "x" * 3 * 15000}]  # ~15k estimated tokens
    with pytest.raises(ContextOverflowError, match="num_ctx=16384") as e:
        native.client.chat(big, max_tokens=2048, operation="ner")
    assert "OLLAMA_NUM_CTX_NER" in str(e.value) and "not truncated" in str(e.value)
    assert native.calls == []


def test_preflight_counts_history_schema_and_output_reserve(native):
    history = [{"role": "user", "content": "y" * 3 * 5000}] * 2  # ~10k tokens of history
    with pytest.raises(ContextOverflowError):
        native.client.chat(history, max_tokens=8192, operation="ner")  # 10k + 8k > 16384
    native.queue.append(ok())
    native.client.chat(history, max_tokens=8192, operation="report")  # fits 32768
    assert len(native.calls) == 1


def test_ollama_context_400_maps_to_context_overflow(native):
    native.queue.append(Resp(400, None, '{"error":"request (11090 tokens) exceeds the available context size (8192 tokens)"}'))
    with pytest.raises(ContextOverflowError, match="exceeds the available context size"):
        native.client.chat(MSGS, operation="ner")
    assert len(native.calls) == 1


def test_other_client_errors_fail_fast(native):
    native.queue.append(Resp(404, None, '{"error":"model not found"}'))
    with pytest.raises(RuntimeError, match="404"):
        native.client.chat(MSGS)
    assert len(native.calls) == 1


def test_server_and_connection_errors_are_retried(native):
    native.queue.extend([Resp(500, None, "boom"), requests.ConnectionError("down"), ok("fine")])
    assert native.client.chat(MSGS) == "fine"
    assert len(native.calls) == 3


def test_retries_are_bounded(native):
    native.queue.extend([requests.ConnectionError("down")] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        native.client.chat(MSGS)


def test_full_window_is_treated_as_truncation(native):
    native.queue.append(ok(prompt=16300, out=84))  # 16384 == num_ctx for ner
    with pytest.raises(ContextOverflowError, match="may have been truncated"):
        native.client.chat(MSGS, operation="ner")


def test_output_cut_raises_in_chat_json(native):
    native.queue.append(ok('{"ok": tr', done="length"))
    with pytest.raises(ValueError, match="truncated"):
        native.client.chat_json(MSGS, operation="ner")


def test_openai_mode_does_not_send_num_ctx():
    captured = {}
    c = LLMClient(api_key="x", base_url="http://localhost:11434/v1", model="m", api_mode="openai")

    def create(**kw):
        captured.update(kw)
        msg = SimpleNamespace(content="hi")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert c.chat(MSGS, operation="ner") == "hi"
    assert "extra_body" not in captured and "options" not in captured


def test_invalid_api_mode_rejected():
    with pytest.raises(ValueError, match="LLM_API_MODE"):
        LLMClient(api_key="x", api_mode="grpc")


# --- context errors are visible (no silent fallback) --------------------------

class OverflowLLM:
    model = "m"
    last_finish_reason = None

    def __init__(self):
        self.calls = 0

    def chat_json(self, **kw):
        self.calls += 1
        raise ContextOverflowError("too big")

    def complete(self, **kw):
        self.calls += 1
        raise ContextOverflowError("too big")


def test_ner_does_not_retry_context_overflow():
    from app.storage.ner_extractor import NERExtractionError, NERExtractor
    llm = OverflowLLM()
    with pytest.raises(NERExtractionError, match="context window"):
        NERExtractor(llm_client=llm, max_retries=2).extract("texto", {})
    assert llm.calls == 1


def test_profile_generation_surfaces_context_overflow():
    from app.services.oasis_profile_generator import OasisProfileGenerator
    gen = OasisProfileGenerator(api_key="x")
    gen.llm = OverflowLLM()
    with pytest.raises(ContextOverflowError):
        gen._generate_profile_with_llm("Ana", "Person", "resumen", {}, "")
    assert gen.llm.calls == 1  # no retries, no rule-based fallback


def test_sim_config_generation_surfaces_context_overflow():
    from app.services.simulation_config_generator import SimulationConfigGenerator
    gen = SimulationConfigGenerator(api_key="x")
    gen.llm = OverflowLLM()
    with pytest.raises(ContextOverflowError):
        gen._generate_time_config("contexto", 5)  # would otherwise fall back to defaults
