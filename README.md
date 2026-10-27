# claude-bridge

An OpenAI-compatible API served by your logged-in [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI.
Point any OpenAI client at it (the official SDKs, LiteLLM, agents, review bots) and it talks to
Claude through your own Claude Code login.

Forked from [RichardAtCT/claude-code-openai-wrapper](https://github.com/RichardAtCT/claude-code-openai-wrapper).
This fork adds what that wrapper lacked for programmatic clients: OpenAI function calling, JSON
mode, embeddings, a hard sandbox and a concurrency cap.

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Streaming and non-streaming; `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` |
| `POST /v1/embeddings` | Local model, sized to match `text-embedding-3-small` (1536 dims) by default |
| `GET /v1/models` | Includes `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` |
| `POST /v1/messages` | Anthropic Messages format, inherited from upstream (text only) |
| `GET /health` | |

Usage counts against your Claude plan, exactly as if you ran `claude -p` yourself.

## Quick start

Needs Python 3.10+ and the `claude` CLI, logged in (`claude auth login`, check with `claude auth status`).

```bash
git clone https://github.com/alicin/claude-bridge && cd claude-bridge
uv venv .venv && uv pip install --python .venv -e . openai   # or: poetry install
cp .env.example .env        # then set API_KEY to any secret your clients will send
bin/run-bridge.sh           # serves on http://127.0.0.1:8000
```

Check it works end to end:

```bash
.venv/bin/python examples/smoke_test.py
```

This runs chat, streaming, embeddings, tool calls in two turns, streamed tool calls, both JSON
modes and 6 parallel requests through the official `openai` SDK.

`bin/run-bridge.sh` unsets inherited `CLAUDE_*` / `ANTHROPIC_*` variables before starting. If you
launch the bridge from inside a Claude Code session, those variables would otherwise make the
nested CLI ignore your login. If `ANTHROPIC_API_KEY` is set, the CLI bills that key instead of
your subscription.

## Use it

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<API_KEY from .env>")

reply = client.chat.completions.create(
    model="claude-sonnet-5-5",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }],
)
print(reply.choices[0].message.tool_calls)   # [ChatCompletionMessageToolCall(... get_weather {"city": "Paris"})]

vectors = client.embeddings.create(model="text-embedding-3-small", input=["def add(a, b): return a + b"])
```

Your client runs the tools, exactly as with OpenAI. Send the results back as `role: "tool"`
messages and the bridge carries the conversation on.

## How it works

Each request runs one `claude -p` process through the Claude Agent SDK.

- **Sandbox.** Every process runs with Claude Code's own tools, settings, `CLAUDE.md`, skills, hooks
  and MCP servers off, in an empty temp directory: `--tools ''`, `--setting-sources=`,
  `--safe-mode`, `--strict-mcp-config`, `--disable-slash-commands`,
  `--no-session-persistence`. A request can't make Claude read, write or run anything on the
  host, whatever it asks for. Only the model is exposed.
- **Concurrency.** At most `MAX_CONCURRENCY` processes run at once, and the rest wait in a queue.
  Streaming responses send SSE keepalive comments while a request waits, so proxies don't time out.
- **Function calling.** Tool schemas go into the system prompt. Claude is asked to end its reply
  with a `<tool_calls>[{"name": ..., "arguments": {...}}]</tool_calls>` block, which is parsed back
  into OpenAI `tool_calls` with `finish_reason: "tool_calls"`. Calls to unknown function names are
  dropped. A forced `tool_choice` that comes back without a call is retried once. Earlier tool
  calls and `role: "tool"` results are written into the prompt as a transcript.
- **JSON mode.** `json_object` and `json_schema` add an instruction (with the schema). The reply
  has code fences stripped and the JSON extracted; if it isn't valid JSON, the request is retried
  once, then fails with a 502. The schema is given to the model but not validated.
- **Embeddings.** [fastembed](https://github.com/qdrant/fastembed) runs `EMBEDDING_MODEL` locally;
  it's downloaded on first use, about 0.6 GB for the default code model. Vectors are zero-padded
  to `EMBEDDING_DIM`, which leaves dot products and cosine similarity unchanged, so a vector column
  sized for OpenAI embeddings works as-is. The request's `dimensions` and `encoding_format`
  (`float` / `base64`) are honoured.

## Configuration

Set in `.env` (see `.env.example` for everything).

| Variable | Default | |
| --- | --- | --- |
| `API_KEY` | prompt at startup | Bearer token clients must send |
| `PORT` / `CLAUDE_WRAPPER_HOST` | `8000` / `0.0.0.0` | Use `127.0.0.1` unless clients are on other machines |
| `CLAUDE_CLI_PATH` | CLI bundled with the SDK | e.g. `~/.local/bin/claude` to use your installed one |
| `DEFAULT_MODEL` | `claude-sonnet-5-5` | Used when a request names no model |
| `MAX_CONCURRENCY` | `3` | Parallel `claude` processes |
| `MAX_TIMEOUT` | `600000` | Per-request timeout, ms |
| `RATE_LIMIT_ENABLED` | `true` | Per-IP limits; turn off for a single client that fans out, and rely on `MAX_CONCURRENCY` |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Any fastembed text model |
| `EMBEDDING_DIM` | `1536` | Output width after padding |

## Limitations

- **Not real token streaming.** `claude -p` returns the whole reply at once, so `stream: true`
  sends it as one content chunk after the wait.
- **Sampling is not enforced.** `temperature`, `top_p`, `max_tokens`, `stop`, `seed` and penalties
  are accepted but not enforced. `n > 1` is rejected.
- **Estimated usage.** Token counts are estimated at roughly 4 characters per token.
- **Text only.** Images in messages are replaced with `[image omitted]`.
- **Tool calls depend on the model.** They work as long as the model follows the `<tool_calls>`
  format, which current Claude models do reliably in testing, but it isn't guaranteed the way
  native tool use is.
- **Embeddings come from a different model.** Don't mix them in one index with real OpenAI
  embeddings. Token-array inputs aren't supported; send strings.
- **Leftover upstream endpoints.** The session, tool-configuration and MCP endpoints are still
  present but have no effect on chat, which is always sandboxed and stateless. `enable_tools` and
  `session_id` are ignored.

## Development

```bash
uv pip install --python .venv pytest pytest-asyncio hypothesis requests black mypy
.venv/bin/python -m pytest tests/
.venv/bin/python -m tests.fake_cli_server       # the API on :8001 with a scripted fake CLI
BRIDGE_URL=http://127.0.0.1:8001/v1 .venv/bin/python examples/smoke_test.py
```

The fake-CLI server tests the OpenAI translation layer without using any Claude usage.

## License

MIT, see [LICENSE](LICENSE). Based on claude-code-openai-wrapper by Richard Atkinson, which
declares the MIT license in its `pyproject.toml`.
