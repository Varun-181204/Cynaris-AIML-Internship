# W6D5 Self Review – Document Chatbot

## Implementation Checklist

- [x] Built PromptTemplate → Ollama LLM → StrOutputParser chain.
- [x] Tested five document questions.
- [x] Loaded and split the PDF into 28 chunks.
- [x] Generated 768-dimensional embeddings.
- [x] Stored document vectors persistently in ChromaDB.
- [x] Implemented five-turn ConversationBufferMemory.
- [x] Verified conversation inputs were retained.
- [x] Implemented calculator tool with AST-based expression validation.
- [x] Implemented local web-search stub.
- [x] Integrated both tools with a LangChain agent.
- [x] Generated JSON execution evidence.
- [x] Automated tests passed: 10/10.

## Execution Results

| Component | Result |
|---|---:|
| PDF chunks | 28 |
| Document questions | 5/5 |
| Memory turns | 5/5 |
| Retained conversation inputs | 5/5 |
| Agent tasks | 3/3 |
| Expected tool calls verified | Yes |
| Automated tests | 10/10 |

## Limitations Identified

1. The agent's final response can introduce information not present in its tool results. Actual tool outputs are separately recorded in the evidence file.
2. The human-review answer did not include a page citation in the recorded execution.
3. ConversationBufferMemory and some dependency integrations produce deprecation warnings.
4. The web-search implementation searches a local knowledge base and does not access the internet.

## Improvements for Future Work

- Validate generated answers against retrieved document context.
- Improve citation enforcement and source-page verification.
- Use a modern LangChain memory/checkpointing approach in a future version.
- Add more agent evaluation tests for tool-output grounding.

## Overall Assessment

The W6D5 implementation, execution pipeline, and automated tests completed successfully. The identified limitations are documented for future improvement.