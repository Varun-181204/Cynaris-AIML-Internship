from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from src import rag_pipeline


def test_load_documents_returns_source_documents():
    documents = rag_pipeline.load_documents()

    assert len(documents) >= 1
    assert all(isinstance(doc, Document) for doc in documents)
    assert all(doc.page_content.strip() for doc in documents)


def test_documents_have_source_metadata():
    documents = rag_pipeline.load_documents()

    for document in documents:
        assert "source" in document.metadata
        assert document.metadata["source"].endswith(".txt")


def test_document_chunking_respects_chunk_size():
    documents = [Document(page_content="A " * 1000, metadata={"source": "test.txt"})]

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_answer_question_raises_when_no_context():
    vector_store = MagicMock()
    vector_store.as_retriever.return_value.invoke.return_value = []

    with pytest.raises(RuntimeError, match="No relevant context retrieved"):
        rag_pipeline.answer_question(vector_store, "Test question")


def test_answer_question_returns_expected_structure():
    vector_store = MagicMock()
    retrieved_document = Document(
        page_content="RAG combines retrieval and generation.",
        metadata={"source": "rag_fundamentals.txt"},
    )
    vector_store.as_retriever.return_value.invoke.return_value = [retrieved_document]

    mock_response = MagicMock()
    mock_response.content = "RAG combines retrieval and generation."

    with patch.object(rag_pipeline, "llm"):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch.object(rag_pipeline, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            )

            result = rag_pipeline.answer_question(vector_store, "What is RAG?")

    assert result["question"] == "What is RAG?"
    assert result["answer"] == "RAG combines retrieval and generation."
    assert len(result["contexts"]) == 1
    assert result["sources"] == ["rag_fundamentals.txt"]
