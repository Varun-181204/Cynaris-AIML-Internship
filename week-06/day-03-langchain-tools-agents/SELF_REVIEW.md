# W6D3 – Self Review

## 1. Work Completed

- Built a LangChain LCEL chain using PromptTemplate, ChatOllama and StrOutputParser.
- Successfully processed five distinct ML-related inputs.
- Implemented five-turn conversation memory using a single ConversationBufferMemory instance.
- Retained all ten user/assistant messages across the conversation.
- Built a LangChain agent using create_agent.
- Implemented a restricted AST-based calculator tool.
- Implemented a local web-search stub.
- Verified three agent tasks and expected tool usage.
- Added automated tests, including malicious-expression rejection.

## 2. Testing Results

| Component | Result |
|---|---:|
| Basic chain | 5/5 inputs |
| Conversation memory | 5/5 turns |
| Retained messages | 10/10 |
| Agent tasks | 3/3 |
| Automated tests | 18/18 passed |

## 3. Security Review

- Calculator uses Python AST parsing.
- Only explicitly permitted arithmetic operations are evaluated.
- Unsupported expressions and unsafe function calls are rejected.
- No eval() or exec() is used.
- Web-search functionality is a local stub, not a live internet integration.

## 4. CIA Mentor Review

Two CIA mentor reviews were completed.

### Review 1 – Chain and Memory

- Confirmed the chain construction and five distinct inputs.
- Verified that one ConversationBufferMemory instance is created outside the conversation loop.
- Rejected the incorrect recommendation to recreate memory per turn.
- Retained the assignment-required legacy memory class and documented its deprecation status.
- Preserved the existing manual history-loading and saving implementation.

### Review 2 – Agent and Tools

- Confirmed use of the LangChain 1.x create_agent API.
- Reviewed the AST-based calculator and local web-search stub.
- Retained the existing tool implementation and descriptions.
- Added a malicious import-execution rejection test.
- Avoided introducing an unnecessary external knowledge-base file or changing the working agent API.

## 5. Limitations

- The web-search tool uses a local knowledge base and does not perform live searches.
- ConversationBufferMemory is deprecated in newer LangChain versions.
- Model responses and execution latency depend on local hardware and Ollama configuration.

## 6. Final Assessment

W6D3 requirements are implemented and tested. The chain, conversation memory and agent workflows produced the expected results.

**Final automated test result: 18 passed.**

The implementation is ready for documentation review, Git commits and push to the existing W6 pull request.
