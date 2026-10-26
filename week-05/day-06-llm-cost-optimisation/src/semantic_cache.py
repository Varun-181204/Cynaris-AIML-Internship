"""Embedding-based semantic cache with cosine similarity matching."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CacheEntry:
    question: str
    answer: str
    embedding: np.ndarray


class SemanticCache:
    """Return cached answers when question similarity meets a threshold."""

    def __init__(self, threshold: float = 0.92) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self._entries: list[CacheEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self, question: str, answer: str, embedding: Any
    ) -> None:
        vector = self._validate_embedding(embedding)
        if not question.strip():
            raise ValueError("question must not be empty")
        if not answer.strip():
            raise ValueError("answer must not be empty")

        # Replace an existing exact-question entry instead of duplicating it.
        for index, entry in enumerate(self._entries):
            if entry.question.casefold() == question.casefold():
                self._entries[index] = CacheEntry(
                    question, answer, vector
                )
                return

        self._entries.append(CacheEntry(question, answer, vector))

    def lookup(self, embedding: Any) -> dict[str, Any] | None:
        """Return the best match at or above threshold, otherwise None."""
        if not self._entries:
            return None

        query_vector = self._validate_embedding(embedding)
        matrix = np.vstack([entry.embedding for entry in self._entries])
        scores = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])

        if score < self.threshold:
            return None

        entry = self._entries[best_index]
        return {
            "question": entry.question,
            "answer": entry.answer,
            "similarity": score,
        }

    @staticmethod
    def _validate_embedding(embedding: Any) -> np.ndarray:
        vector = np.asarray(embedding, dtype=float)

        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding must contain only finite values")
        if np.linalg.norm(vector) == 0:
            raise ValueError("embedding must not be a zero vector")

        return vector
