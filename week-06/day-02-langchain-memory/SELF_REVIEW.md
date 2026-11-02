# W6D2 – Self Review

## 1. Tasks Completed

- [x] Implemented PromptTemplate → Ollama → StrOutputParser.
- [x] Tested the chain with five different inputs.
- [x] Implemented ConversationBufferMemory.
- [x] Verified memory retention across five conversation turns.
- [x] Implemented a two-tool LangChain agent.
- [x] Created calculator and web search stub tools.
- [x] Executed three agent tasks and verified tool usage.
- [x] Created automated tests.
- [x] Saved JSON execution evidence.

## 2. Test Results

| Component | Result |
|---|---:|
| Basic chain | 5/5 successful |
| Conversation memory | 5/5 turns |
| Retained messages | 10/10 |
| Agent tasks | 3/3 successful |
| Automated tests | 12/12 passed |

## 3. Technical Learnings

- Understood how LangChain connects prompts, LLMs, and output parsers.
- Learned how conversation memory maintains previous interactions.
- Implemented tools using LangChain's @tool decorator.
- Used create_agent from the modern LangChain 1.x API.
- Learned to verify tool execution using ToolMessage.
- Used Python AST parsing for safer arithmetic evaluation.
- Practised automated testing with pytest.

## 4. Challenges and Solutions

**Challenge 1:** LangChain memory classes generated deprecation warnings.

**Solution:** Retained ConversationBufferMemory because it is explicitly
required by the internship assignment and documented the warnings.

**Challenge 2:** Initial tests expected exceptions from the calculator.

**Solution:** Updated tests to validate the error messages returned by
the calculator tool without weakening its safety checks.

**Challenge 3:** Search-stub test expectations did not match its output.

**Solution:** Corrected assertions to match the actual documented
local knowledge-base responses.

## 5. Improvements

- Replace deprecated memory components when the assignment permits.
- Extend the local knowledge base with additional topics.
- Add more arithmetic edge-case tests.
- Explore persistent conversation storage.

## 6. Final Assessment

W6D2 objectives were completed successfully.

All three implementation tasks passed their execution checks,
and all 12 automated tests passed.

The implementation uses local Ollama inference with Qwen 2.5:3b.