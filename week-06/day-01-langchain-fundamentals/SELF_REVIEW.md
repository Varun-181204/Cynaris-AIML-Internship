# W6D1 Self-Review — LangChain Fundamentals

## 1. Work Completed

- [x] Implemented a PromptTemplate → Ollama → StrOutputParser chain.
- [x] Tested the chain with five different inputs.
- [x] Implemented ConversationBufferMemory.
- [x] Verified conversation history across five turns.
- [x] Created an agent using LangChain create_agent.
- [x] Implemented a calculator tool.
- [x] Implemented a local web-search stub.
- [x] Tested three agent tasks.
- [x] Verified expected tool calls.
- [x] Created automated unit tests.
- [x] Saved execution evidence in JSON format.

## 2. Test Results

| Component | Expected | Actual | Status |
|---|---:|---:|---|
| Basic chain | 5 inputs | 5 successful | PASS |
| Conversation memory | 5 turns | 5 successful | PASS |
| Memory retention | 10 messages | 10 retained | PASS |
| Agent tasks | 3 tasks | 3 successful | PASS |
| Agent tool verification | 3 tasks | 3 verified | PASS |
| Automated tests | 12 tests | 12 passed | PASS |

## 3. What I Learned

- PromptTemplate separates reusable instructions from input data.
- LangChain chains connect prompts, language models and output parsers.
- Conversation memory allows an application to retain previous interactions.
- Agents can select and invoke tools to complete tasks.
- Tool execution should be verified rather than assuming the model used a tool.
- Safe input validation is important when implementing calculator tools.

## 4. Challenges Faced

**Python import path issue**

Initially, pytest could not locate the `src` package. This was resolved by adding the practical task directory to `sys.path` in the test file.

**LangChain deprecation warnings**

ConversationBufferMemory and InMemoryChatMessageHistory generated deprecation warnings. They were retained because ConversationBufferMemory is explicitly required by the assignment.

**Local search limitation**

The search tool uses a predefined local knowledge base rather than a live search API. This limitation is documented in the implementation and its responses.

## 5. Improvements for Future Work

- Explore modern LangChain short-term memory and checkpointing.
- Integrate a real search API with appropriate configuration.
- Expand the calculator's supported operations and validation.
- Add more agent test cases and error-handling scenarios.
- Explore LangGraph workflows for more complex agent applications.

## 6. Overall Self-Assessment

All three practical tasks were implemented and executed successfully. The agent demonstrated actual use of both tools, and all 12 automated tests passed.

**Overall status:** Completed — pending final documentation review, CIA mentor review and Git submission.
