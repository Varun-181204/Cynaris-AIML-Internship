# W6D2 – LangChain Memory and Agent Tools

## Objective
Implement LangChain fundamentals using Ollama, conversation memory,
and a two-tool agent.

## Environment
- Python 3.14
- LangChain
- LangChain Classic
- LangChain Ollama
- Ollama
- Model: qwen2.5:3b
- Testing: pytest

## Implementation

### 1. Basic Chain
Implemented PromptTemplate → ChatOllama → StrOutputParser.

- Tested 5 questions.
- All 5 inputs completed successfully.
- Average response latency: approximately 7.81 seconds.
- Evidence: outputs/chain_results.json

### 2. Conversation Memory
Implemented ConversationBufferMemory using LangChain Classic.

- Completed 5 conversation turns.
- Retained 10/10 messages.
- Verified conversation history.
- Evidence: outputs/memory_results.json

Note: ConversationBufferMemory and InMemoryChatMessageHistory
produce deprecation warnings in the installed LangChain versions.
They are retained because ConversationBufferMemory is explicitly
required by the assignment.

### 3. Two-Tool Agent
Implemented an agent using langchain.agents.create_agent.

Tools:
- calculator: restricted AST-based arithmetic evaluator.
- web_search_stub: local knowledge-base search, not live internet.

Verification:
- Task 1: Calculator passed.
- Task 2: Local knowledge search passed.
- Task 3: Both tools passed.

Evidence: outputs/agent_results.json

### 4. Automated Testing
Executed:

python -m pytest week-06/day-02-langchain-memory/tests/ -v

Result: 12 passed.

Tests cover arithmetic, invalid expressions, calculator safety,
local knowledge retrieval, and search-stub limitations.

## Project Structure

- src/basic_chain.py
- src/conversation_memory.py
- src/agent_tools.py
- tests/test_langchain.py
- outputs/chain_results.json
- outputs/memory_results.json
- outputs/agent_results.json
- docs/CIA_INTERACTIONS.md
- README.md
- SELF_REVIEW.md
- requirements.txt

## Conclusion
W6D2 successfully demonstrates LangChain chains, conversation
memory, and tool-using agents with local Ollama inference.