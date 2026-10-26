"""Fast unit tests for the Ollama API client; no running Ollama required."""

import json
import sys
import unittest
from urllib.error import HTTPError
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ollama_api import (
    MAX_RESPONSE_BYTES,
    OllamaAPIError,
    chat,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data if isinstance(data, bytes) else json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.data[:size] if size >= 0 else self.data


class OllamaAPITests(unittest.TestCase):
    @patch("ollama_api.urlopen")
    def test_sends_model_system_and_user_prompt(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse({
            "message": {"content": "A clear answer"}
        })
        result = chat("llama3.2:3b", "Be concise.", "Explain ML.")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["messages"][0]["content"], "Be concise.")
        self.assertEqual(payload["messages"][1]["content"], "Explain ML.")
        self.assertEqual(result["response"], "A clear answer")
        self.assertGreaterEqual(result["latency_seconds"], 0)

    def test_rejects_empty_prompts(self):
        with self.assertRaises(ValueError):
            chat("llama3.2:3b", "System", " ")
        with self.assertRaises(ValueError):
            chat("llama3.2:3b", " ", "Question")

    @patch("ollama_api.urlopen")
    def test_rejects_missing_response_content(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse({"message": {}})
        with self.assertRaises(OllamaAPIError):
            chat("llama3.2:3b", "System", "Question")

    @patch("ollama_api.urlopen", side_effect=OSError("connection refused"))
    def test_wraps_connection_errors(self, mock_urlopen):
        with self.assertRaises(OllamaAPIError):
            chat("llama3.2:3b", "System", "Question")

    @patch("ollama_api.urlopen")
    def test_preserves_ollama_api_error_message(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse({"error": "model not found"})
        with self.assertRaisesRegex(OllamaAPIError, "model not found"):
            chat("missing-model", "System", "Question")

    @patch("ollama_api.urlopen")
    def test_reports_malformed_json(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(b'{"message":{"content":"abc')
        with self.assertRaisesRegex(OllamaAPIError, "failed"):
            chat("llama3.2:3b", "System", "Question")

    @patch("ollama_api.urlopen")
    def test_rejects_oversized_response(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(
            b"x" * (MAX_RESPONSE_BYTES + 1)
        )
        with self.assertRaisesRegex(OllamaAPIError, "exceeded"):
            chat("llama3.2:3b", "System", "Question")


    @patch("ollama_api.urlopen")
    def test_wraps_http_error_with_model_and_status(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "http://localhost:11434/api/chat", 404, "Not Found", None, None
        )
        with self.assertRaisesRegex(OllamaAPIError, r"missing-model.*404"):
            chat("missing-model", "System", "Question")

    @patch("ollama_api.urlopen")
    def test_wraps_invalid_utf8(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(bytes([255, 254, 0]))
        with self.assertRaisesRegex(OllamaAPIError, "failed"):
            chat("llama3.2:3b", "System", "Question")

    @patch("ollama_api.urlopen")
    def test_rejects_list_json_payload(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(["a", "b"])
        with self.assertRaisesRegex(OllamaAPIError, "invalid JSON object"):
            chat("llama3.2:3b", "System", "Question")


if __name__ == "__main__":
    unittest.main()
