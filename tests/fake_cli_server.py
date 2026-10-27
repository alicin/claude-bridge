"""Bridge with the Claude CLI swapped for a scripted fake, to exercise the OpenAI translation offline.

Usage: .venv/bin/python -m tests.fake_cli_server   then
       BRIDGE_URL=http://127.0.0.1:8001/v1 .venv/bin/python examples/smoke_test.py
"""

import json
import re

import uvicorn

from src import main


async def fake_run_completion(prompt, system_prompt=None, **_):
    system_prompt = system_prompt or ""
    if "<tool_result" in prompt:
        temp = re.search(r'"temp_c": (\d+)', prompt).group(1)
        text = f"It is {temp}°C and overcast in Paris."
    elif "<tool_calls>" in system_prompt:
        city = "Oslo" if "Oslo" in prompt else "Paris"
        call = [{"name": "get_weather", "arguments": {"city": city}}]
        text = f"Checking.\n<tool_calls>\n{json.dumps(call)}\n</tool_calls>"
    elif "JSON schema" in system_prompt:
        text = '{"severity": "high", "reason": "auth check removed"}'
    elif "JSON object" in system_prompt:
        text = '```json\n{"capital": "Paris", "population": 68000000}\n```'
    elif "pong" in prompt:
        text = "pong"
    elif "Count" in prompt:
        text = "1, 2, 3, 4, 5"
    else:
        text = re.search(r"number (\d+)", prompt).group(1)
    yield {"subtype": "success", "is_error": False, "result": text}


async def fake_verify_cli():
    return True


main.claude_cli.run_completion = fake_run_completion
main.claude_cli.verify_cli = fake_verify_cli

if __name__ == "__main__":
    uvicorn.run(main.app, host="127.0.0.1", port=8001)
