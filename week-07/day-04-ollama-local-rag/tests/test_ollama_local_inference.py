import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_local_inference import (
    build_payload, parse_response, SYSTEM_PROMPT
)

class TestOllamaInference(unittest.TestCase):

    def test_payload_contains_model(self):
        self.assertEqual(
            build_payload("llama3.2:3b", "Explain RAG.")["model"],
            "llama3.2:3b"
        )

    def test_system_prompt(self):
        self.assertEqual(
            build_payload("llama3.2:3b", "Explain RAG.")["messages"][0]["content"],
            SYSTEM_PROMPT
        )

    def test_streaming_disabled(self):
        self.assertFalse(
            build_payload("qwen2.5:3b", "Explain embeddings.")["stream"]
        )

    def test_response_parsing(self):
        self.assertEqual(
            parse_response({"message": {"content": "  Hello  "}}),
            "Hello"
        )

    def test_invalid_response(self):
        with self.assertRaises(ValueError):
            parse_response({})

if __name__ == "__main__":
    unittest.main()
