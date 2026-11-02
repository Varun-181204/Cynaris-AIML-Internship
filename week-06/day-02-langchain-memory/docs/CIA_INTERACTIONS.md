# W6D2 – CIA Mentor Interactions

## Interaction 1: Chain and Conversation Memory

**Mode:** Full Stack Mentor Mode

**Topics reviewed:**
- PromptTemplate and runnable pipeline.
- ChatOllama integration.
- StrOutputParser.
- ConversationBufferMemory.
- Memory loading and saving.
- LangChain 1.x compatibility.

**CIA feedback:**
- Confirmed the PromptTemplate → LLM → StrOutputParser pipeline.
- Confirmed the memory load/invoke/save approach.
- Recommended explicit memory keys and ChatPromptTemplate.
- Suggested context-window management for longer conversations.

**Evaluation:**
The existing implementation successfully completed five turns
and retained all 10 messages.

The existing PromptTemplate chain works with ChatOllama.
ChatPromptTemplate is an optional improvement, not a required fix.

ConversationBufferMemory was retained because the assignment
explicitly requires it.

## Interaction 2: Agent and Tool Safety

**Mode:** Full Stack Mentor Mode

**Topics reviewed:**
- LangChain @tool decorator.
- Calculator implementation.
- Restricted AST evaluation.
- Local web search stub.
- create_agent compatibility.
- Automated tests.

**CIA feedback:**
- Confirmed tool registration and agent integration.
- Approved restricted AST evaluation instead of eval().
- Recommended clearer error handling and logging.
- Suggested additional resource safeguards.

**Evaluation:**
The implementation already includes:
- Type hints and tool descriptions.
- Restricted AST arithmetic.
- Expression-length and exponent limits.
- Local-only knowledge-base search.
- ToolMessage-based tool verification.

The suggested AgentExecutor and alternative invocation
approaches were not adopted because they do not match
the current LangChain 1.x implementation.

## Final Outcome

Both CIA interactions were completed and reviewed.

No unnecessary changes were made based on optional,
outdated, or technically inaccurate suggestions.

Automated tests: 12/12 passed.
Agent tasks: 3/3 passed.