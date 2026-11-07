
"""W6D2: Verify conversation history using ConversationBufferMemory."""

from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import time

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = ROOT / "week-06/day-02-langchain-memory/outputs/memory_results.json"

MODEL = "qwen2.5:3b"

CONVERSATION = [
    "I am developing a college project called CampusCart.",
    "The project uses Python and FastAPI for its backend.",
    "I have selected PostgreSQL as the database.",
    "I plan to deploy the application using Render.",
    "Summarize the project name, backend technology, database, and deployment platform.",
]


def build_chain():
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant with access to conversation history.

Previous conversation:
{history}

Current message:
{input}

Use the previous conversation when answering.
If the user asks about previously shared details, recall them accurately.
Keep responses clear and concise.
"""
    )

    llm = ChatOllama(
        model=MODEL,
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=180,
    )

    return prompt | llm | StrOutputParser()


def main():
    chain = build_chain()

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="response",
        return_messages=False,
    )

    results = []

    for turn, user_input in enumerate(CONVERSATION, start=1):
        start = time.perf_counter()

        try:
            memory_variables = memory.load_memory_variables({})
            history = memory_variables["history"]

            response = chain.invoke({
                "history": history,
                "input": user_input,
            })

            memory.save_context(
                {"input": user_input},
                {"response": response},
            )

            latency = round(time.perf_counter() - start, 3)

            results.append({
                "turn": turn,
                "user_input": user_input,
                "assistant_response": response,
                "history_before_turn": history,
                "latency_seconds": latency,
                "status": "success",
            })

            logger.info("Turn %d/%d completed (%.2fs)", turn, len(CONVERSATION), latency)

        except Exception as exc:
            logger.exception("Turn %d failed", turn)
            results.append({
                "turn": turn,
                "user_input": user_input,
                "error": str(exc),
                "status": "failed",
            })

    final_history = memory.load_memory_variables({})["history"]

    retained_inputs = all(
        user_input in final_history for user_input in CONVERSATION
    )

    message_count = len(memory.chat_memory.messages)
    expected_messages = len(CONVERSATION) * 2

    history_verified = (
        retained_inputs
        and message_count == expected_messages
        and all(item["status"] == "success" for item in results)
    )

    evidence = {
        "task": "W6D2 Conversation Memory",
        "model": MODEL,
        "memory_type": "ConversationBufferMemory",
        "total_turns": len(CONVERSATION),
        "successful_turns": sum(
            item["status"] == "success" for item in results
        ),
        "expected_messages": expected_messages,
        "retained_messages": message_count,
        "all_user_inputs_retained": retained_inputs,
        "history_verified": history_verified,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_history": final_history,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(evidence, indent=4), encoding="utf-8")

    logger.info("Retained messages: %d/%d", message_count, expected_messages)
    logger.info("All user inputs retained: %s", retained_inputs)
    logger.info("History verification: %s", history_verified)
    logger.info("Evidence saved to %s", OUTPUT_FILE)

    if not history_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
