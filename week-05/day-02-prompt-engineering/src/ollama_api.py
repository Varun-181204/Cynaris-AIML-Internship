"""Ollama chat API client using the Python standard library."""

import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

API_URL = "http://localhost:11434/api/chat"
DEFAULT_TIMEOUT = 180
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class OllamaAPIError(RuntimeError):
    """Raised when Ollama cannot return a valid chat response."""


def chat(model, system_prompt, prompt, timeout=DEFAULT_TIMEOUT):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("System prompt must not be empty.")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise OllamaAPIError(
                f"Ollama response exceeded {MAX_RESPONSE_BYTES} bytes "
                f"for model {model}."
            )
        result = json.loads(raw.decode("utf-8"))
    except OllamaAPIError:
        raise
    except (URLError, TimeoutError, OSError, json.JSONDecodeError,
            UnicodeDecodeError) as exc:
        raise OllamaAPIError(
            f"Ollama request/response failed for {model}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise OllamaAPIError("Ollama returned an invalid JSON object.")

    if "error" in result:
        raise OllamaAPIError(f"Ollama API error for {model}: {result['error']}")

    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaAPIError(
            f"Ollama returned missing or empty content for model {model}."
        )

    return {
        "model": model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "response": content.strip(),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }
