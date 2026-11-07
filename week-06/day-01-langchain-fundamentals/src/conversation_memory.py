
"""W6D1: ConversationBufferMemory with Ollama."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "week-06" / "day-01-langchain-fundamentals" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "memory_results.json"

MODEL_NAME = "llama3.2:3b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CONVERSATION_TURNS = [
    "I am planning a beginner AI project called StudyBuddy.",
    "What is the name of the project I mentioned?",
    "I want to build it using Python.",
    "Which project and programming language did I choose?",
    "Summarize the project details I shared with you.",
]

PROMPT_TEMPLATE = """
You are a helpful conversational AI assistant.

Previous conversation:
{history}

Current user message:
{input}

Instructions:
- Use previous conversation when relevant.
- Remember the details shared by the user.
- Answer in simple English.
- Do not invent personal details.
- Keep your response concise.

Assistant response:
"""


def build_components():
    """Initialize the LLM, prompt, parser and conversation memory."""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=PROMPT_TEMPLATE,
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_predict=150,
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="response",
        return_messages=False,
    )

    return chain, memory


def run_conversation() -> dict:
    """Run five conversation turns and verify retained history."""

    chain, memory = build_components()

    results = []

    for turn_number, user_input in enumerate(
        CONVERSATION_TURNS, start=1
    ):
        logger.info(
            "Processing conversation turn %d/%d",
            turn_number,
            len(CONVERSATION_TURNS),
        )

        history_before = memory.load_memory_variables({})["history"]

        response = chain.invoke({
            "history": history_before,
            "input": user_input,
        })

        memory.save_context(
            {"input": user_input},
            {"response": response},
        )

        history_after = memory.load_memory_variables({})["history"]

        results.append({
            "turn": turn_number,
            "user_input": user_input,
            "assistant_response": response.strip(),
            "history_before_turn": history_before,
            "history_after_turn": history_after,
            "success": bool(response.strip()),
        })

        logger.info(
            "Turn %d completed. History length: %d characters",
            turn_number,
            len(history_after),
        )

    final_history = memory.load_memory_variables({})["history"]

    expected_messages = len(CONVERSATION_TURNS) * 2
    actual_messages = len(memory.chat_memory.messages)

    all_user_inputs_retained = all(
        user_input in final_history
        for user_input in CONVERSATION_TURNS
    )

    return {
        "experiment": "W6D1 ConversationBufferMemory",
        "model": MODEL_NAME,
        "total_turns": len(results),
        "successful_turns": sum(r["success"] for r in results),
        "expected_messages": expected_messages,
        "actual_messages": actual_messages,
        "all_user_inputs_retained": all_user_inputs_retained,
        "history_verified": (
            actual_messages == expected_messages
            and all_user_inputs_retained
        ),
        "final_conversation_history": final_history,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def main() -> None:
    """Execute and save the memory experiment."""

    logger.info("Starting ConversationBufferMemory experiment")

    evidence = run_conversation()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(evidence, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("W6D1 CONVERSATION MEMORY RESULTS")
    print("=" * 60)

    for result in evidence["results"]:
        print(f"\nTurn {result['turn']}")
        print(f"User: {result['user_input']}")
        print(f"Assistant: {result['assistant_response']}")

    print("\n" + "=" * 60)
    print(f"Successful turns: {evidence['successful_turns']}/5")
    print(
        f"Messages retained: "
        f"{evidence['actual_messages']}/"
        f"{evidence['expected_messages']}"
    )
    print(
        f"All user inputs retained: "
        f"{evidence['all_user_inputs_retained']}"
    )
    print(f"History verified: {evidence['history_verified']}")
    print(f"Evidence saved: {OUTPUT_FILE}")
    print("=" * 60)

    if not evidence["history_verified"]:
        raise SystemExit("Conversation memory verification failed.")


if __name__ == "__main__":
    main()
