"""Small standard-library client for the local Ollama chat API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI/ML learning assistant. "
    "Explain concepts accurately in simple language, using concise examples."
)


class OllamaClientError(RuntimeError):
    """Raised when the Ollama server returns an error or cannot be reached."""


class OllamaClient:
    def __init__(
        self,
        model="llama3.2:3b",
        base_url=DEFAULT_BASE_URL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        timeout=120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout = timeout

    def chat(self, prompt):
        """Send one prompt and return the assistant response text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaClientError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaClientError(
                "Ollama returned invalid JSON"
            ) from exc

        message = result.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaClientError("Ollama response has no message content")
        return content.strip()


if __name__ == "__main__":
    client = OllamaClient()
    question = input("Enter a prompt: ").strip()
    print("\n" + client.chat(question))
