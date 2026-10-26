"""Unit tests for the Ollama client; no running server is required."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ollama_client import OllamaClient, OllamaClientError


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class OllamaClientTests(unittest.TestCase):
    @patch("ollama_client.urlopen")
    def test_chat_sends_model_system_prompt_and_user_prompt(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(
            {"message": {"role": "assistant", "content": "  Hello!  "}}
        )
        client = OllamaClient(
            model="llama3.2:3b",
            system_prompt="Be concise.",
        )

        answer = client.chat("What is ML?")

        self.assertEqual(answer, "Hello!")
        request = mock_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "llama3.2:3b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][0]["content"], "Be concise.")
        self.assertEqual(body["messages"][1]["content"], "What is ML?")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            OllamaClient().chat("  ")

    @patch("ollama_client.urlopen")
    def test_missing_message_content_raises_clear_error(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse({"done": True})
        with self.assertRaises(OllamaClientError):
            OllamaClient().chat("Explain Python")

    @patch("ollama_client.urlopen")
    def test_connection_failure_is_wrapped(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("server unavailable")
        with self.assertRaises(OllamaClientError):
            OllamaClient().chat("Explain Python")


if __name__ == "__main__":
    unittest.main()
