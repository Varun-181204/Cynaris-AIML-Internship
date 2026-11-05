
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "llama3.2:3b"

CONVERSATION = [
    "I am building an AI project called SmartStudy.",
    "It uses Python and LangChain for its application logic.",
    "I selected ChromaDB as the vector database.",
    "The project uses Ollama with the Llama 3.2 model.",
    "Summarize my project name, programming language, framework, vector database, and LLM.",
]


def run_memory():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="langchain_classic",
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="response",
        return_messages=False,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant. Use the conversation history "
        "to answer accurately.\n\n"
        "Conversation history:\n{history}\n\n"
        "User: {input}\n"
        "Assistant:"
    )

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    chain = prompt | llm | StrOutputParser()

    results = []

    for turn, user_input in enumerate(CONVERSATION, start=1):
        start_time = time.perf_counter()

        try:
            memory_variables = memory.load_memory_variables({})
            response = chain.invoke({
                "history": memory_variables["history"],
                "input": user_input,
            })

            memory.save_context(
                {"input": user_input},
                {"response": response},
            )

            latency = round(time.perf_counter() - start_time, 3)

            results.append({
                "turn": turn,
                "user_input": user_input,
                "response": response,
                "success": bool(response.strip()),
                "history_length": len(
                    memory.load_memory_variables({})["history"]
                ),
                "latency_seconds": latency,
            })

            print(f"\nTurn {turn}: {user_input}")
            print(f"Response: {response}")
            print(f"Latency: {latency}s")

        except Exception as exc:
            results.append({
                "turn": turn,
                "user_input": user_input,
                "response": "",
                "success": False,
                "error": str(exc),
            })
            print(f"\nTurn {turn} failed: {exc}")

    final_history = memory.load_memory_variables({})["history"]

    all_inputs_retained = all(
        user_input in final_history
        for user_input in CONVERSATION
    )

    evidence = {
        "task": "W6D3 LangChain Conversation Memory",
        "model": MODEL_NAME,
        "memory_type": "ConversationBufferMemory",
        "total_turns": len(CONVERSATION),
        "successful_turns": sum(
            item["success"] for item in results
        ),
        "retained_messages": len(memory.chat_memory.messages),
        "expected_messages": len(CONVERSATION) * 2,
        "all_user_inputs_retained": all_inputs_retained,
        "history_verification": all_inputs_retained,
        "final_history": final_history,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    output_file = OUTPUT_DIR / "memory_results.json"
    output_file.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nMemory verification:")
    print(f"Successful turns: {evidence['successful_turns']}/5")
    print(
        f"Retained messages: {evidence['retained_messages']}/"
        f"{evidence['expected_messages']}"
    )
    print(f"All inputs retained: {all_inputs_retained}")
    print(f"Evidence saved: {output_file}")

    return evidence


if __name__ == "__main__":
    run_memory()
