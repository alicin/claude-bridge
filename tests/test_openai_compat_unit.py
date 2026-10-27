"""Unit tests for the OpenAI function-calling, JSON-mode and embedding translation."""

import base64
import json
import struct
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src import openai_compat
from src.models import ChatCompletionRequest

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def request(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "claude-haiku-5-5")
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return ChatCompletionRequest(**kwargs)


class FakeCLI:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts = []

    async def run_completion(self, prompt, system_prompt=None, **_):
        self.prompts.append((prompt, system_prompt))
        yield {"subtype": "success", "is_error": False, "result": self.replies.pop(0)}

    def parse_claude_message(self, chunks):
        return chunks[-1]["result"]


class TestParseToolCalls:
    def test_block_becomes_openai_tool_calls(self):
        text = 'Checking.\n<tool_calls>\n[{"name": "get_weather", "arguments": {"city": "Paris"}}]\n</tool_calls>'
        content, calls = openai_compat.parse_tool_calls(text, request(tools=[WEATHER]))
        assert content == "Checking."
        assert calls[0]["type"] == "function"
        assert calls[0]["id"].startswith("call_")
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_fenced_unclosed_block_and_unknown_names(self):
        text = '<tool_calls>\n```json\n[{"name": "rm_rf", "arguments": {}}, {"name": "get_weather", "arguments": {}}]\n```'
        _, calls = openai_compat.parse_tool_calls(text, request(tools=[WEATHER]))
        assert [c["function"]["name"] for c in calls] == ["get_weather"]

    def test_parallel_tool_calls_false_keeps_one(self):
        call = {"name": "get_weather", "arguments": {}}
        text = f"<tool_calls>{json.dumps([call, call])}</tool_calls>"
        req = request(tools=[WEATHER], parallel_tool_calls=False)
        assert len(openai_compat.parse_tool_calls(text, req)[1]) == 1

    def test_no_block_or_bad_json_is_plain_content(self):
        req = request(tools=[WEATHER])
        assert openai_compat.parse_tool_calls("Just text.", req) == ("Just text.", [])
        assert openai_compat.parse_tool_calls("<tool_calls>[oops</tool_calls>", req)[1] == []


class TestExtractJson:
    @pytest.mark.parametrize(
        "text",
        ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go: {"a": 1} hope that helps', "[1, 2]"],
    )
    def test_finds_json(self, text):
        assert json.loads(openai_compat.extract_json(text)) in ({"a": 1}, [1, 2])

    def test_rejects_non_json(self):
        assert openai_compat.extract_json("no json here") is None


class TestBuildPrompt:
    def test_single_user_turn_passes_through(self):
        prompt, system = openai_compat.build_prompt(request())
        assert prompt == "hi" and system is None

    def test_tool_transcript_and_system_instructions(self):
        req = request(
            tools=[WEATHER],
            tool_choice="required",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "4C"},
            ],
        )
        prompt, system = openai_compat.build_prompt(req)
        assert system.startswith("Be terse.")
        assert "get_weather" in system and "must call at least one function" in system
        assert "single valid JSON object" in system
        assert "<user>\nWeather?\n</user>" in prompt
        assert '"arguments": {"city": "Oslo"}' in prompt
        assert '<tool_result tool_call_id="c1" name="get_weather">\n4C' in prompt

    def test_tool_choice_none_hides_tools(self):
        _, system = openai_compat.build_prompt(request(tools=[WEATHER], tool_choice="none"))
        assert system is None


class TestRunChat:
    @pytest.mark.asyncio
    async def test_forced_call_retries_once(self):
        cli = FakeCLI(
            "No call here.", '<tool_calls>[{"name": "get_weather", "arguments": {}}]</tool_calls>'
        )
        req = request(
            tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}}
        )
        _, calls, _ = await openai_compat.run_chat(req, cli)
        assert len(cli.prompts) == 2 and calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_json_mode_retries_then_fails(self):
        req = request(response_format={"type": "json_object"})
        _, _, _ = await openai_compat.run_chat(req, FakeCLI("nope", '{"ok": true}'))
        with pytest.raises(HTTPException) as e:
            await openai_compat.run_chat(req, FakeCLI("nope", "still nope"))
        assert e.value.status_code == 502

    @pytest.mark.asyncio
    async def test_cli_usage_limit_maps_to_429(self):
        class LimitedCLI(FakeCLI):
            async def run_completion(self, prompt, **_):
                yield {"is_error": True, "result": "Claude AI usage limit reached"}

        with pytest.raises(HTTPException) as e:
            await openai_compat.run_chat(request(), LimitedCLI())
        assert e.value.status_code == 429


class TestEmbeddings:
    def test_fit_pads_and_truncates(self):
        assert openai_compat._fit([0.6, 0.8], 4) == [0.6, 0.8, 0.0, 0.0]
        truncated = openai_compat._fit([3.0, 4.0, 1.0], 2)
        assert truncated == pytest.approx([0.6, 0.8])

    @pytest.mark.asyncio
    async def test_float_and_base64_output(self):
        class FakeEmbedder:
            def embed(self, inputs):
                import numpy as np

                return [np.array([1.0, 0.0]) for _ in inputs]

        async def fake_get():
            return FakeEmbedder()

        with patch.object(openai_compat, "_get_embedder", fake_get):
            out = await openai_compat.handle_embeddings({"input": "x", "dimensions": 3})
            assert out["data"][0]["embedding"] == [1.0, 0.0, 0.0]
            out = await openai_compat.handle_embeddings(
                {"input": ["x", "y"], "dimensions": 3, "encoding_format": "base64"}
            )
            raw = base64.b64decode(out["data"][1]["embedding"])
            assert struct.unpack("<3f", raw) == (1.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_rejects_token_arrays(self):
        with pytest.raises(HTTPException):
            await openai_compat.handle_embeddings({"input": [[1, 2, 3]]})
