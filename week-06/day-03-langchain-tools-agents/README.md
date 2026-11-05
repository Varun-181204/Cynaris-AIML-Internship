# W6D3 – LangChain Tools & Agents

## Overview

This task demonstrates LangChain fundamentals using a local Ollama model. It covers LCEL chains, conversation memory, and tool-based agents.

**Model:** `llama3.2:3b`  
**Runtime:** Ollama  
**Python:** 3.14

## Project Structure

```text
day-03-langchain-tools-agents/
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
│   └── CIA_INTERACTIONS.md
├── requirements.txt
├── README.md
└── SELF_REVIEW.md
```

## 1. Basic LangChain Chain

Implemented an LCEL chain using:

`PromptTemplate → ChatOllama → StrOutputParser`

- Uses five distinct ML-related questions.
- Uses the local `llama3.2:3b` model.
- Records individual execution times and average latency.
- Saves execution evidence to `outputs/chain_results.json`.

**Result:** 5/5 inputs processed successfully.

## 2. Conversation Memory

Implemented `ConversationBufferMemory` using `langchain_classic.memory`.

- Maintains one memory instance across five conversation turns.
- Uses a SmartStudy project scenario.
- Loads previous conversation history before each model invocation.
- Saves each response back into the same memory buffer.
- Generates a final project summary using accumulated context.

**Result:** 5/5 successful turns and 10/10 messages retained.

**Compatibility note:** `ConversationBufferMemory` is deprecated in newer LangChain versions. It is retained because it is explicitly required by the assignment. Future maintenance should evaluate a supported replacement.

## 3. LangChain Tools and Agent

Implemented a LangChain agent using `langchain.agents.create_agent`.

### Tools

- **Calculator:** AST-based arithmetic evaluator with an operator whitelist. It does not use `eval()` or `exec()`.
- **Web Search Stub:** Searches a small local knowledge base. It does not access the live internet.

### Agent Tasks

1. Calculate `425 * 3 + 175`.
2. Search the local knowledge base for LangChain information.
3. Calculate 20% of 450 and search for RAG information.

**Result:** 3/3 tasks completed, with expected tool usage verified through `ToolMessage` evidence.

## 4. Testing

Run from the repository root:

```powershell
python -m pytest week-06/day-03-langchain-tools-agents/tests/ -v
```

**Latest result: 18 passed.**

Coverage includes:
- Five-input chain configuration.
- Model configuration and chain evidence.
- Five-turn conversation memory.
- Memory evidence and retained messages.
- Calculator arithmetic operations.
- Unsafe expressions and unsupported functions.
- Excessive exponent rejection.
- Malicious import execution rejection.
- Local knowledge-base searches.
- Agent execution evidence.

## 5. Installation and Execution

Activate the repository virtual environment and install the dependencies:

```powershell
pip install -r week-06/day-03-langchain-tools-agents/requirements.txt
```

Ensure Ollama is installed and the model is available:

```powershell
ollama pull llama3.2:3b
```

Run individual modules from the repository root:

```powershell
python -m src.basic_chain
python -m src.conversation_memory
python -m src.agent_tools
```

If module execution is used, run each command from its corresponding project directory or use the repository's established import configuration.

## 6. Evidence and Documentation

- `outputs/chain_results.json` – chain execution results and latency.
- `outputs/memory_results.json` – conversation memory results.
- `outputs/agent_results.json` – agent tasks and tool usage.
- `tests/test_langchain.py` – automated test suite.
- `docs/CIA_INTERACTIONS.md` – CIA mentor review and decisions.
- `SELF_REVIEW.md` – implementation review and learning outcomes.

## 7. Learning Outcomes

- Understand LangChain Expression Language (LCEL).
- Connect prompt templates, local LLMs and output parsers.
- Maintain context across multiple conversation turns.
- Build agents with custom tools.
- Apply AST-based restrictions to calculator inputs.
- Verify agent tool usage and implementation quality through automated tests.
