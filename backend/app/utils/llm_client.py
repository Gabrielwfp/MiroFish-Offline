"""
LLM Client Wrapper

Two transports, selected with Config.LLM_API_MODE:
- 'openai' (default): OpenAI-compatible /v1 API (any provider).
- 'ollama_native': Ollama /api/chat. The only transport where options.num_ctx is
  honored (by the llama.cpp/GGUF runner; Ollama's MLX runner ignores num_ctx on
  every API). num_ctx is chosen per operation (Config.num_ctx_for) and every request
  is checked against it before sending: nothing is silently truncated.
"""

import json
import logging
import math
import re
import time
from typing import Optional, Dict, Any, List

import requests
from openai import OpenAI

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')


class ContextOverflowError(ValueError):
    """The request does not fit the configured context window (rejected, never truncated)."""


def estimate_tokens(messages: List[Dict[str, Any]], extra_text: str = "") -> int:
    """Conservative token estimate for system + history + prompt (+ schema/tools text)."""
    chars = sum(len(str(m.get("content") or "")) for m in messages) + len(extra_text)
    return math.ceil(chars / Config.CONTEXT_CHARS_PER_TOKEN) + 8 * len(messages) + 16


class LLMClient:
    """LLM Client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        operation: Optional[str] = None,
        api_mode: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.api_mode = (api_mode or Config.LLM_API_MODE).lower()
        self.operation = operation
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_finish_reason: Optional[str] = None
        self.last_usage: Dict[str, Any] = {}

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")
        if self.api_mode not in ("openai", "ollama_native"):
            raise ValueError(f"Unsupported LLM_API_MODE '{self.api_mode}' (use 'openai' or 'ollama_native')")

        if self.api_mode == "openai":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        else:
            self.client = None
            self._native_url = re.sub(r"/v1/?$", "", self.base_url.rstrip("/")) + "/api/chat"

    @property
    def last_finish_reason(self) -> Optional[str]:
        return self._last_finish_reason

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return self.api_mode == "ollama_native" or '11434' in (self.base_url or '')

    def num_ctx(self, operation: Optional[str] = None) -> int:
        return Config.num_ctx_for(operation or self.operation)

    # ------------------------------------------------------------------ transports

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        disable_thinking: bool = False,
        operation: Optional[str] = None,
    ) -> str:
        """Single chat completion on the configured transport.

        Sets last_finish_reason ('stop', 'length', ...) and, in native mode, last_usage.
        Raises ContextOverflowError when the request cannot fit the operation's num_ctx.
        """
        if self.api_mode == "ollama_native":
            content, finish = self._native_chat(messages, temperature, max_tokens, response_format,
                                                disable_thinking, operation or self.operation)
        else:
            content, finish = self._openai_chat(messages, temperature, max_tokens, response_format, disable_thinking)
        self._last_finish_reason = finish
        # Some models (like MiniMax M2.5) include <think>thinking content in response, need to remove
        return re.sub(r'<think>[\s\S]*?</think>', '', content or "").strip()

    def _openai_chat(self, messages, temperature, max_tokens, response_format, disable_thinking):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        # Ollama-specific value; other OpenAI-compatible servers may reject "none"
        if disable_thinking and self._is_ollama():
            kwargs["reasoning_effort"] = "none"
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason

    def _native_chat(self, messages, temperature, max_tokens, response_format, disable_thinking, operation):
        num_ctx = self.num_ctx(operation)
        op = operation or "default"

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature},
        }
        schema_text = ""
        if response_format:
            if response_format.get("type") == "json_schema":
                schema = response_format["json_schema"]["schema"]
                body["format"] = schema
                schema_text = json.dumps(schema)
            elif response_format.get("type") == "json_object":
                body["format"] = "json"
        if disable_thinking:
            body["think"] = False

        # Preflight: input (system + history + prompt + schema) + reserved output must fit
        estimated = estimate_tokens(messages, schema_text)
        if estimated + max_tokens > num_ctx:
            raise ContextOverflowError(
                f"[{op}] request needs ~{estimated} input tokens + {max_tokens} reserved for output, "
                f"more than num_ctx={num_ctx} (model={self.model}). Increase OLLAMA_NUM_CTX_{op.upper()} "
                f"or reduce the input; it is not truncated."
            )

        t0 = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self._native_url, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
            else:
                if resp.status_code == 200:
                    break
                text = resp.text[:500]
                if resp.status_code == 400 and "context" in text.lower():
                    raise ContextOverflowError(f"[{op}] Ollama rejected the request for num_ctx={num_ctx}: {text}")
                if resp.status_code < 500:
                    raise RuntimeError(f"Ollama /api/chat error {resp.status_code} (model={self.model}): {text}")
                last_error = RuntimeError(f"Ollama /api/chat error {resp.status_code}: {text}")
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Ollama /api/chat failed after {self.max_retries + 1} attempts: {last_error}")

        data = resp.json()
        prompt_tokens, output_tokens = data.get("prompt_eval_count") or 0, data.get("eval_count") or 0
        self.last_usage = {"operation": op, "num_ctx": num_ctx, "estimated_input": estimated,
                           "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                           "secs": round(time.time() - t0, 1)}
        logger.info(f"[llm] {self.last_usage}")
        # Older llama.cpp engines truncate the prompt instead of rejecting it: a full window is suspect
        if prompt_tokens and prompt_tokens + output_tokens >= num_ctx:
            raise ContextOverflowError(
                f"[{op}] context window filled ({prompt_tokens}+{output_tokens} >= num_ctx={num_ctx}); "
                f"the input may have been truncated by Ollama"
            )
        done = data.get("done_reason") or "stop"
        return (data.get("message") or {}).get("content") or "", ("length" if done == "length" else done)

    # ------------------------------------------------------------------ public API

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        disable_thinking: bool = False,
        operation: Optional[str] = None,
    ) -> str:
        """
        Send chat request

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count (output reserve; num_predict in native mode)
            response_format: Response format (e.g., JSON mode)
            disable_thinking: On Ollama, turn off reasoning for thinking models
                (much faster for extraction; ignored by non-thinking models)
            operation: One of Config.LLM_OPERATIONS; selects num_ctx in native mode

        Returns:
            Model response text
        """
        content = self.complete(messages, temperature, max_tokens, response_format, disable_thinking, operation)
        if self._last_finish_reason == "length":
            logger.warning(f"[llm] response cut at max_tokens={max_tokens} "
                           f"(operation={operation or self.operation}, model={self.model})")
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
        disable_thinking: bool = False,
        operation: Optional[str] = None,
    ) -> Any:
        """
        Send chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            schema: Optional JSON Schema, enforced by Ollama on the output (OpenAI
                `json_schema` response_format, or `format` in native mode). Plain JSON
                mode only guarantees *some* valid JSON. Callers must still validate.
            disable_thinking: See chat()
            operation: See chat()

        Returns:
            Parsed JSON value
        """
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        else:
            response_format = {"type": "json_object"}
        response = self.complete(messages, temperature, max_tokens, response_format, disable_thinking, operation)
        if self._last_finish_reason == "length":
            raise ValueError(
                f"LLM response truncated at max_tokens={max_tokens} (model={self.model}); "
                f"increase max_tokens or the operation's context window"
            )
        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format from LLM: {cleaned_response}")
