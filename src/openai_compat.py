"""OpenAI chat/embeddings semantics the Claude CLI lacks: function calling, JSON mode, embeddings."""

import asyncio
import base64
import json
import logging
import os
import re
import struct
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.models import ChatCompletionRequest, Message

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")
# text-embedding-3-small's width, so vector columns sized for OpenAI embeddings work unchanged.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
KEEPALIVE_SECONDS = 15

TOOL_BLOCK = re.compile(r"<tool_calls>\s*(.*?)\s*(?:</tool_calls>|$)", re.S)
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _tool_instructions(req: ChatCompletionRequest) -> Optional[str]:
    if not req.tools or req.tool_choice == "none":
        return None
    functions = [t.get("function", t) for t in req.tools if t.get("type", "function") == "function"]
    lines = [
        "You can call the functions below, described as OpenAI function JSON schemas:",
        json.dumps(functions, indent=1),
        "",
        "To call functions, end your reply with exactly one block of this form and write nothing after it:",
        "<tool_calls>",
        '[{"name": "<function name>", "arguments": {<arguments as JSON>}}]',
        "</tool_calls>",
        "Arguments must be valid JSON matching the function's parameters schema. "
        "Function results come back in <tool_result> blocks. "
        "If no function is needed, answer normally without the block. "
        "Never claim to have called a function without emitting the block.",
    ]
    if req.tool_choice == "required":
        lines.append("You must call at least one function in this reply.")
    elif isinstance(req.tool_choice, dict):
        name = req.tool_choice.get("function", {}).get("name")
        lines.append(f"You must call the function `{name}` in this reply.")
    if req.parallel_tool_calls is False:
        lines.append("Call at most one function per reply.")
    return "\n".join(lines)


def _json_instructions(req: ChatCompletionRequest) -> Optional[str]:
    response_format = req.response_format or {}
    fmt = response_format.get("type")
    if fmt == "json_object":
        return "Respond with a single valid JSON object only: no prose, no markdown code fences."
    if fmt == "json_schema":
        schema = response_format.get("json_schema", {})
        return (
            "Respond with a single valid JSON value only, no prose and no markdown code fences, "
            "that conforms to this JSON schema:\n" + json.dumps(schema.get("schema", schema))
        )
    return None


def _text(msg: Message) -> str:
    # Message's validator flattens content parts to a string.
    return msg.content if isinstance(msg.content, str) else ""


def _render_assistant(msg: Message) -> str:
    parts = [_text(msg)] if msg.content else []
    if msg.tool_calls:
        calls = []
        for call in msg.tool_calls:
            fn = call.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                pass
            calls.append({"name": fn.get("name"), "arguments": args})
        parts.append("<tool_calls>\n" + json.dumps(calls) + "\n</tool_calls>")
    return "\n".join(parts)


def build_prompt(req: ChatCompletionRequest) -> Tuple[str, Optional[str]]:
    system_parts = [
        _text(m) for m in req.messages if m.role in ("system", "developer") and m.content
    ]
    system_parts += [s for s in (_tool_instructions(req), _json_instructions(req)) if s]
    system_prompt = "\n\n".join(system_parts) or None

    turns = [m for m in req.messages if m.role not in ("system", "developer")]
    if len(turns) == 1 and turns[0].role == "user":
        return _text(turns[0]), system_prompt

    call_names = {
        c.get("id"): c.get("function", {}).get("name") for m in turns for c in (m.tool_calls or [])
    }
    rendered = []
    for m in turns:
        if m.role == "user":
            rendered.append(f"<user>\n{_text(m)}\n</user>")
        elif m.role == "assistant":
            rendered.append(f"<assistant>\n{_render_assistant(m)}\n</assistant>")
        else:
            name = m.name or call_names.get(m.tool_call_id, "")
            rendered.append(
                f'<tool_result tool_call_id="{m.tool_call_id or ""}" name="{name}">\n'
                f"{_text(m)}\n</tool_result>"
            )
    prompt = (
        "The conversation so far is below. Write only the assistant's next message, "
        "without any <assistant> wrapper.\n\n" + "\n\n".join(rendered)
    )
    return prompt, system_prompt


def _strip_fences(text: str) -> str:
    m = FENCE.match(text.strip())
    return m.group(1) if m else text.strip()


def extract_json(text: str) -> Optional[str]:
    candidate = _strip_fences(text)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = candidate.rfind("}" if candidate[start] == "{" else "]")
    try:
        json.loads(candidate[start : end + 1])
        return candidate[start : end + 1]
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str, req: ChatCompletionRequest) -> Tuple[Optional[str], List[Dict]]:
    m = TOOL_BLOCK.search(text)
    if not m:
        return text, []
    raw = extract_json(m.group(1))
    if raw is None:
        logger.warning("Unparseable <tool_calls> block: %s", m.group(1)[:200])
        return text, []
    calls = json.loads(raw)
    if isinstance(calls, dict):
        calls = calls.get("tool_calls", [calls])
    known = {t.get("function", t).get("name") for t in req.tools or []}
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", call)
        name = fn.get("name")
        if name not in known:
            logger.warning("Dropping call to unknown function %r", name)
            continue
        args = fn.get("arguments", fn.get("parameters", {}))
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        )
    if req.parallel_tool_calls is False:
        out = out[:1]
    if not out:
        return text, []
    return text[: m.start()].strip() or None, out


def _result_error(chunks: List[Dict[str, Any]]) -> Optional[str]:
    for chunk in chunks:
        if chunk.get("is_error"):
            return chunk.get("error_message") or chunk.get("result") or "Claude CLI error"
    return None


async def _complete(claude_cli, prompt: str, system_prompt: Optional[str], model: str) -> str:
    chunks = []
    async for chunk in claude_cli.run_completion(
        prompt=prompt, system_prompt=system_prompt, model=model, max_turns=1, stream=False
    ):
        chunks.append(chunk)
    error = _result_error(chunks)
    if error:
        status = 429 if "limit" in error.lower() else 502
        raise HTTPException(status_code=status, detail=error)
    text = claude_cli.parse_claude_message(chunks)
    if not text:
        raise HTTPException(status_code=502, detail="No response from Claude CLI")
    return text


def _must_call(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and (req.tool_choice == "required" or isinstance(req.tool_choice, dict))


async def run_chat(req: ChatCompletionRequest, claude_cli) -> Tuple[Optional[str], List[Dict], str]:
    prompt, system_prompt = build_prompt(req)
    text = await _complete(claude_cli, prompt, system_prompt, req.model)
    content: Optional[str] = text
    tool_calls: List[Dict] = []
    if req.tools and req.tool_choice != "none":
        content, tool_calls = parse_tool_calls(text, req)
        if not tool_calls and _must_call(req):
            retry = (
                prompt
                + "\n\nYour previous reply did not contain a valid <tool_calls> block. Reply again and include one."
            )
            content, tool_calls = parse_tool_calls(
                await _complete(claude_cli, retry, system_prompt, req.model), req
            )
    if not tool_calls and req.response_format and req.response_format.get("type") != "text":
        parsed = extract_json(content or "")
        if parsed is None:
            retry = (
                prompt
                + "\n\nYour previous reply was not valid JSON:\n"
                + (content or "")[:2000]
                + "\n\nReply again with only the JSON."
            )
            parsed = extract_json(await _complete(claude_cli, retry, system_prompt, req.model))
        if parsed is None:
            raise HTTPException(status_code=502, detail="Model did not return valid JSON")
        content = parsed
    return content, tool_calls, prompt


def _usage(prompt: str, content: Optional[str], tool_calls: List[Dict]) -> Dict[str, int]:
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, (len(content or "") + len(json.dumps(tool_calls))) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def handle_chat(req: ChatCompletionRequest, claude_cli):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    if req.stream:
        return StreamingResponse(
            _stream(req, claude_cli, request_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    content, tool_calls, prompt = await run_chat(req, claude_cli)
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": _usage(prompt, content, tool_calls),
        }
    )


async def _stream(req, claude_cli, request_id: str, created: int) -> AsyncGenerator[str, None]:
    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None, **extra) -> str:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(body)}\n\n"

    task = asyncio.create_task(run_chat(req, claude_cli))
    # Comment lines keep proxies from timing out while the request waits for a CLI slot.
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
        if not done:
            yield ": keepalive\n\n"
    try:
        content, tool_calls, prompt = task.result()
    except HTTPException as e:
        yield f"data: {json.dumps({'error': {'message': str(e.detail), 'code': e.status_code}})}\n\n"
        return
    except Exception as e:
        logger.exception("Streaming chat failed")
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
        return

    yield chunk({"role": "assistant", "content": ""})
    if content:
        yield chunk({"content": content})
    if tool_calls:
        yield chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(tool_calls)]})
    yield chunk({}, "tool_calls" if tool_calls else "stop")
    if req.stream_options and req.stream_options.include_usage:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [],
            "usage": _usage(prompt, content, tool_calls),
        }
        yield f"data: {json.dumps(body)}\n\n"
    yield "data: [DONE]\n\n"


_embedder = None
_embedder_lock = asyncio.Lock()


async def _get_embedder():
    global _embedder
    async with _embedder_lock:
        if _embedder is None:
            from fastembed import TextEmbedding

            _embedder = await asyncio.to_thread(TextEmbedding, model_name=EMBEDDING_MODEL)
    return _embedder


def _fit(vector: List[float], dim: int) -> List[float]:
    # Zero-padding keeps dot products and cosine similarity unchanged.
    if len(vector) >= dim:
        vector = vector[:dim]
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]
    return vector + [0.0] * (dim - len(vector))


async def handle_embeddings(body: Dict[str, Any]) -> Dict[str, Any]:
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
        raise HTTPException(status_code=400, detail="input must be a string or list of strings")
    dim = int(body.get("dimensions") or EMBEDDING_DIM)
    embedder = await _get_embedder()
    vectors = await asyncio.to_thread(lambda: [v.tolist() for v in embedder.embed(inputs)])
    as_base64 = body.get("encoding_format") == "base64"
    data = []
    for i, vector in enumerate(vectors):
        vector = _fit(vector, dim)
        embedding: Any = (
            base64.b64encode(struct.pack(f"<{dim}f", *vector)).decode() if as_base64 else vector
        )
        data.append({"object": "embedding", "index": i, "embedding": embedding})
    tokens = sum(len(i) for i in inputs) // 4
    return {
        "object": "list",
        "data": data,
        "model": body.get("model", EMBEDDING_MODEL),
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
