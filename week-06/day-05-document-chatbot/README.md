# W6D5 – Document Chatbot with LangChain

## Overview
A PDF-based document chatbot built using LangChain, Ollama and ChromaDB. It retrieves relevant document chunks, generates context-grounded answers, maintains conversation history, and demonstrates an agent with two tools.

## Features
- PDF loading using PyPDFLoader.
- Recursive text splitting (700 characters, 100 overlap).
- Persistent ChromaDB vector storage using cosine distance.
- Ollama embeddings using `nomic-embed-text`.
- Document question-answering using `llama3.2:3b`.
- PromptTemplate → ChatOllama → StrOutputParser chain.
- Five-turn ConversationBufferMemory demonstration.
- LangChain agent with calculator and local web-search stub.
- Safe arithmetic evaluation using AST validation.
- JSON execution evidence.

## Project Structure
- `src/document_chatbot.py` – Main chatbot implementation.
- `tests/` – Automated tests.
- `data/Module-3-PE.pdf` – Input PDF.
- `data/chroma_db/` – Persistent vector database.
- `outputs/chatbot_evidence.json` – Execution evidence.
- `docs/` – CIA interaction records.

## Requirements
- Python 3.14
- Ollama running locally.
- Ollama models:
  - `nomic-embed-text`
  - `llama3.2:3b`

## Execution

Run from the W6D5 directory:

```powershell
python -m src.document_chatbot
python -m pytest tests -v
```

## Validation Results

| Component | Result |
|---|---:|
| PDF pages loaded | 13 |
| PDF chunks indexed | 28 |
| Embedding dimensions | 768 |
| Document questions | 5/5 |
| Memory turns | 5/5 |
| Conversation inputs retained | 5/5 |
| Agent tasks | 3/3 |
| Expected tool calls verified | Yes |
| Automated tests | 10/10 |

## Known Limitations
- ConversationBufferMemory produces LangChain deprecation warnings but is retained to demonstrate the assignment requirement.
- The agent may generate inaccurate explanations after receiving correct tool results. Actual tool outputs are recorded separately from the agent's final response.
- The human-review answer did not include a page citation in this execution.
- The web-search tool is a local knowledge-base stub, not live internet search.

See `outputs/chatbot_evidence.json` for execution details.