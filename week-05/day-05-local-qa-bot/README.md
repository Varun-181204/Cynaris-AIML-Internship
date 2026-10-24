# W5D5 — Local Q&A Bot (Ollama + ChromaDB)

## 1. Objective

Build a local document question-answering bot that retrieves relevant passages from a PDF and uses a locally running language model to generate context-grounded answers.

## 2. Architecture

1. **PDF extraction:** `pypdf` extracts text from document pages.
2. **Chunking:** The extracted text is split into smaller, overlapping chunks.
3. **Embeddings:** Ollama's `nomic-embed-text` model converts text into vector embeddings.
4. **Vector storage and retrieval:** ChromaDB stores the vectors and metadata, then retrieves relevant chunks for a question.
5. **Answer generation:** Ollama's `llama3.2:3b` model answers using the retrieved context.
6. **Source traceability:** Results include the source filename, page number, and retrieval distance.

## 3. Requirements

- Python with a working virtual environment
- Ollama installed and running locally
- The following Ollama models:
  - `nomic-embed-text`
  - `llama3.2:3b`

Install the Python dependencies from the repository root:

```powershell
python -m pip install -r .\week-05\day-05-local-qa-bot\requirements.txt
```

Pull the required Ollama models if they are not already installed:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Ensure Ollama is available at `http://localhost:11434`.

## 4. Run the bot

From the repository root:

```powershell
python .\week-05\day-05-local-qa-bot\src\local_qa_bot.py
```

The demonstration indexes `sample_data/vector_databases_guide.pdf`, asks two questions, prints the answers and their retrieved sources, and saves the results to:

`outputs/qa_evidence.json`

The persistent ChromaDB database is stored under `data/chroma/`. This generated database is excluded from Git.

## 5. Run tests

Run all Week 5 tests from the repository root:

```powershell
python -m pytest --import-mode=importlib .\week-05 -q
```

The verified result for this implementation is **34 passed, 1 warning**. The warning is a ChromaDB telemetry dependency deprecation warning; it did not fail the tests.

Run only the W5D5 tests:

```powershell
python -m pytest --import-mode=importlib .\week-05\day-05-local-qa-bot\tests -v
```

## 6. Evidence

- `outputs/qa_evidence.json` — results from two real questions answered by the local models.
- `outputs/qa_evidence_initial.json` — preserved evidence from the initial single-question run.
- `tests/test_local_qa_bot.py` — automated tests for PDF processing, retrieval, answer generation, input validation, and error cases.

The recorded real run indexed three chunks and used `nomic-embed-text` for embeddings and `llama3.2:3b` for answer generation.

## 7. Limitations

- Answers depend on the quality and completeness of the indexed document and retrieved passages.
- The system prompt encourages grounded answers, but does not guarantee factual correctness.
- The model may fail to recognize that retrieved context is insufficient or may interpret it incorrectly.
- Retrieval distance is a similarity-related measure, not a confidence score or proof of correctness.
- The demonstration uses a small sample PDF; broader document coverage requires additional testing.

## 8. Project structure

```text
day-05-local-qa-bot/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── chroma/                 # Generated; excluded from Git
├── outputs/
│   ├── qa_evidence.json
│   └── qa_evidence_initial.json
├── sample_data/
│   └── vector_databases_guide.pdf
├── src/
│   └── local_qa_bot.py
└── tests/
    └── test_local_qa_bot.py
```

## 9. Responsible use

Use the bot as a document-assisted research tool, not as an unquestionable source of truth. Check the cited source pages before relying on important answers, and avoid indexing sensitive documents unless their handling and storage are appropriate.
