"""Unit tests for the ReportAgent text tool-call protocol (no LLM, no network)."""

from app.services.report_agent import ReportAgent


def _agent():
    return object.__new__(ReportAgent)  # parser needs no init state


def test_parses_tool_tag_used_by_prompts():
    resp = 'Thinking...\n<tool>\n{"name": "quick_search", "parameters": {"query": "tasas"}}\n</tool>'
    assert _agent()._parse_tool_calls(resp) == [{"name": "quick_search", "parameters": {"query": "tasas"}}]


def test_still_accepts_legacy_tool_call_tag():
    resp = '<tool_call>{"name": "panorama_search", "parameters": {}}</tool_call>'
    assert _agent()._parse_tool_calls(resp)[0]["name"] == "panorama_search"


def test_mismatched_tags_are_not_parsed_as_xml():
    resp = '<tool>{"name": "quick_search", "parameters": {}}</tool_call>'
    assert _agent()._parse_tool_calls(resp) == []


def test_prompts_no_longer_ask_for_tool_call_tag():
    import inspect
    import app.services.report_agent as ra
    src = inspect.getsource(ra)
    prompt_lines = [l for l in src.splitlines() if "<tool_call>" in l and not l.strip().startswith(("#", "(legacy", "The prompts", "tool-call parser"))]
    assert prompt_lines == [], prompt_lines
