"""End-to-end check of the bridge with the official openai SDK.

Usage: .venv/bin/python examples/smoke_test.py   (reads API_KEY/PORT from .env)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("BRIDGE_URL", f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1"),
    api_key=os.getenv("API_KEY", "none"),
    timeout=600,
)
MODEL = os.getenv("SMOKE_MODEL", "claude-haiku-5-5")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def check_models():
    ids = {m.id for m in client.models.list().data}
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} <= ids, ids


def check_chat():
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Reply with exactly: pong"}]
    )
    assert "pong" in r.choices[0].message.content.lower(), r


def check_stream():
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text, finish, usage = "", None, None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        for c in chunk.choices:
            text += c.delta.content or ""
            finish = c.finish_reason or finish
    assert "1" in text and "5" in text and finish == "stop" and usage, (text, finish, usage)


def check_embeddings():
    r = client.embeddings.create(
        model="text-embedding-3-small", input=["def add(a, b): return a + b", "sum two numbers", "the weather in Paris"]
    )
    vecs = [d.embedding for d in r.data]
    assert all(len(v) == EMBED_DIM for v in vecs), [len(v) for v in vecs]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(vecs[0], vecs[1]) > dot(vecs[0], vecs[2]), "code/query should be closer than code/weather"


def check_tools():
    messages = [{"role": "user", "content": "What's the weather in Paris right now?"}]
    r = client.chat.completions.create(model=MODEL, messages=messages, tools=[WEATHER_TOOL])
    choice = r.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.tool_calls, choice
    call = choice.message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments)["city"].lower().startswith("paris")

    messages += [
        choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 17, "sky": "overcast"}'},
    ]
    r = client.chat.completions.create(model=MODEL, messages=messages, tools=[WEATHER_TOOL])
    final = r.choices[0]
    assert final.finish_reason == "stop" and "17" in final.message.content, final


def check_tools_stream():
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Weather in Oslo?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=True,
    )
    calls, finish = [], None
    for chunk in stream:
        for c in chunk.choices:
            calls += c.delta.tool_calls or []
            finish = c.finish_reason or finish
    assert finish == "tool_calls" and calls[0].function.name == "get_weather", (calls, finish)


def check_json_mode():
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give the capital and population (int) of France."}],
        response_format={"type": "json_object"},
    )
    assert isinstance(json.loads(r.choices[0].message.content), dict)

    schema = {
        "type": "object",
        "properties": {"severity": {"type": "string", "enum": ["low", "high"]}, "reason": {"type": "string"}},
        "required": ["severity", "reason"],
    }
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Rate this diff: `- if user.is_admin:` `+ if True:`"}],
        response_format={"type": "json_schema", "json_schema": {"name": "rating", "schema": schema}},
    )
    out = json.loads(r.choices[0].message.content)
    assert out["severity"] in ("low", "high") and out["reason"], out


def check_concurrency():
    with ThreadPoolExecutor(6) as pool:
        replies = list(
            pool.map(
                lambda i: client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": f"Reply with only the number {i}"}]
                ).choices[0].message.content,
                range(6),
            )
        )
    assert all(str(i) in r for i, r in enumerate(replies)), replies


CHECKS = [check_models, check_chat, check_stream, check_embeddings, check_tools, check_tools_stream, check_json_mode, check_concurrency]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    failed = 0
    for check in CHECKS:
        if only and check.__name__.removeprefix("check_") not in only:
            continue
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {check.__name__}: {e!r}"[:2000])
    sys.exit(1 if failed else 0)
