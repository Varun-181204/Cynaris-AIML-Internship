# W6D1 — LangChain Fundamentals: Chains, Memory and Agents

## Objective

Explore LangChain fundamentals by building and testing:
1. An LLM chain using PromptTemplate, Ollama and StrOutputParser.
2. A conversational memory system using ConversationBufferMemory.
3. An AI agent with two tools: a calculator and a local web-search stub.

## Technology Stack

- Python 3.14
- LangChain 1.4.7
- LangChain Core 1.7.3
- LangChain Ollama 1.1.0
- LangChain Classic 1.0.8
- LangGraph 1.2.19
- Ollama
- llama3.2:3b
- pytest

## Project Structure

```text
day-01-langchain-fundamentals/
├── src/
│   ├── __init__.py
│   ├── basic_chain.py
│   ├── conversation_memory.py
│   └── agent_tools.py
├── tests/
│   └── test_langchain.py
├── outputs/
│   ├── chain_results.json
│   ├── memory_results.json
│   └── agent_results.json
├── docs/
└── README.md
```

## Practical Tasks

### Task 1 — Basic LLM Chain

Implemented a LangChain sequence:

PromptTemplate → ChatOllama → StrOutputParser

Tested five different topics and recorded model responses and execution times.

**Result:** 5/5 inputs processed successfully.

### Task 2 — Conversation Memory

Implemented ConversationBufferMemory to retain conversation history across five interactions.

Verified that all five user inputs and five assistant responses were retained.

**Result:** 5/5 turns completed and 10/10 messages retained.

Note: ConversationBufferMemory and InMemoryChatMessageHistory produce deprecation warnings in the installed LangChain versions. They are used here because the internship task explicitly requires ConversationBufferMemory. Modern LangChain applications should consider agent checkpointing or other supported memory approaches.

### Task 3 — Agent with Two Tools

Implemented an agent using LangChain's create_agent API.

**Tool 1: Calculator**
- Supports basic arithmetic operations.
- Uses Python AST parsing instead of eval().
- Rejects unsupported and unsafe expressions.
- Handles division-by-zero errors.

**Tool 2: Web Search Stub**
- Uses a predefined local knowledge base.
- Demonstrates agent tool calling.
- Does not access the live internet.
- Clearly identifies its results as local stub results.

Tested three tasks: calculator, search, and a combined task.

**Result:** 3/3 tasks completed with expected tool usage verified.

## Automated Testing

Run the tests from the repository root:

```powershell
python -m pytest week-06/day-01-langchain-fundamentals/tests/ -v
```

**Latest result:** 12 tests passed.

Tests cover calculator operations, invalid expressions, unsafe input rejection, and search-stub responses.

## Running the Experiments

Ensure Ollama is running and the `llama3.2:3b` model is available.

From the repository root:

```powershell
python week-06/day-01-langchain-fundamentals/src/basic_chain.py

python week-06/day-01-langchain-fundamentals/src/conversation_memory.py

python week-06/day-01-langchain-fundamentals/src/agent_tools.py
```

## Evidence

Execution results are saved in the `outputs/` directory:

- `chain_results.json`
- `memory_results.json`
- `agent_results.json`

## Learning Outcomes

- Understand LangChain prompt templates and sequential chains.
- Connect a local Ollama model to LangChain.
- Understand conversational memory and message retention.
- Build an agent using the modern create_agent API.
- Define tools and verify actual tool execution.
- Apply safe arithmetic parsing and automated testing.
