
import ast
import json
import operator
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.messages import ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_DIR = DATA_DIR / "chroma_db"

PDF_PATH = DATA_DIR / "Module-3-PE.pdf"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
COLLECTION_NAME = "w6d5_document_chatbot"

QUESTIONS = [
    "What is generative AI?",
    "What are the four types of generative AI guardrails?",
    "Explain input guardrails.",
    "What is the purpose of output guardrails?",
    "Why is human review important in responsible AI?",
]

CONVERSATION = [
    "I am studying generative AI.",
    "My document discusses responsible AI and guardrails.",
    "It describes input and output guardrails.",
    "It also discusses retrieval and human review.",
    "Summarize the four guardrail types discussed in my document.",
]

AGENT_TASKS = [
    "Calculate 425 * 3 + 175.",
    "Search the local knowledge base for LangChain.",
    "Calculate 20 percent of 450 and search for RAG.",
]

KNOWLEDGE_BASE = {
    "langchain": (
        "LangChain is a framework for building LLM applications "
        "using prompts, chains, tools and agents."
    ),
    "rag": (
        "Retrieval-Augmented Generation retrieves relevant external "
        "context before an LLM generates an answer."
    ),
    "ollama": (
        "Ollama runs large language models locally, including "
        "Llama 3.2 and Qwen 2.5."
    ),
    "python": (
        "Python is a programming language widely used in AI, "
        "automation and data science."
    ),
}

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_evaluate(node):
    if isinstance(node, ast.Expression):
        return safe_evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        left = safe_evaluate(node.left)
        right = safe_evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent limit exceeded.")

        result = operation(left, right)

        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("Result exceeds the allowed limit.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(safe_evaluate(node.operand))

    raise ValueError("Unsupported expression.")


def build_vector_store():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(pages)

    documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "source": PDF_PATH.name,
                "page": str(chunk.metadata.get("page", "unknown")),
            },
        )
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"w6d5-pdf-{i:04d}" for i in range(len(documents))]
    vectors = embeddings.embed_documents(
        [document.page_content for document in documents]
    )

    collection.upsert(
        ids=ids,
        documents=[document.page_content for document in documents],
        metadatas=[document.metadata for document in documents],
        embeddings=vectors,
    )

    return collection, embeddings, len(documents)


def retrieve(collection, embeddings, question, top_k=3):
    vector = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for index, content in enumerate(results["documents"][0]):
        matches.append({
            "id": results["ids"][0][index],
            "document": content,
            "metadata": results["metadatas"][0][index],
            "cosine_distance": results["distances"][0][index],
        })

    return matches


def build_document_chain():
    prompt = PromptTemplate.from_template(
        "You are a document-grounded AI assistant. "
        "Use ONLY facts explicitly stated in the supplied document context. "
        "Never infer, guess, or add general knowledge. "
        "If the context does not directly answer the question, respond exactly: "
        "'The information was not found in the provided document context.' "
        "For every factual claim, cite its page using [Page N]. "
        "Never cite a page unless the supporting information appears in that "
        "page's context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def run_document_questions(collection, embeddings):
    chain = build_document_chain()
    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        start = time.perf_counter()
        matches = retrieve(collection, embeddings, question)

        context = "\n\n".join(
            f"[Page {item['metadata']['page']}]\n{item['document']}"
            for item in matches
        )

        try:
            answer = chain.invoke({
                "context": context,
                "question": question,
            })

            results.append({
                "test_id": index,
                "question": question,
                "answer": answer,
                "retrieved_chunks": len(matches),
                "source_pages": sorted({
                    item["metadata"]["page"] for item in matches
                }),
                "success": bool(answer.strip()) and bool(matches),
                "latency_seconds": round(time.perf_counter() - start, 3),
            })

            print(f"\nDocument question {index}: {question}")
            print(f"Answer: {answer}")
            print(f"Retrieved chunks: {len(matches)}")

        except Exception as exc:
            results.append({
                "test_id": index,
                "question": question,
                "answer": "",
                "success": False,
                "error": str(exc),
            })
            print(f"Question failed: {exc}")

    return results


def run_conversation_memory():
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="response",
        return_messages=False,
    )

    prompt = PromptTemplate.from_template(
        "You are a document study assistant. "
        "Use the conversation history to remember earlier information.\n\n"
        "History:\n{history}\n\n"
        "User: {input}\nAssistant:"
    )

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    results = []

    for turn, user_input in enumerate(CONVERSATION, start=1):
        history = memory.load_memory_variables({})["history"]

        try:
            response = chain.invoke({
                "history": history,
                "input": user_input,
            })

            memory.save_context(
                {"input": user_input},
                {"response": response},
            )

            results.append({
                "turn": turn,
                "input": user_input,
                "response": response,
                "success": bool(response.strip()),
            })

            print(f"\nMemory turn {turn}: {user_input}")
            print(f"Response: {response}")

        except Exception as exc:
            results.append({
                "turn": turn,
                "input": user_input,
                "response": "",
                "success": False,
                "error": str(exc),
            })

    history = memory.load_memory_variables({})["history"]
    retained = all(item in history for item in CONVERSATION)

    return {
        "total_turns": len(CONVERSATION),
        "successful_turns": sum(item["success"] for item in results),
        "retained_messages": len(memory.chat_memory.messages),
        "expected_messages": len(CONVERSATION) * 2,
        "all_inputs_retained": retained,
        "final_history": history,
        "results": results,
    }


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    if not isinstance(expression, str) or len(expression) > 100:
        return "Calculation error: invalid expression."

    try:
        return str(safe_evaluate(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def web_search_stub(query: str) -> str:
    """Search a local knowledge base; this is not live internet search."""
    normalized = re.sub(r"[^a-z0-9 ]", " ", query.lower())

    matches = [
        value
        for topic, value in KNOWLEDGE_BASE.items()
        if topic in normalized
    ]

    if not matches:
        return "No matching topic found in the local knowledge base."

    return "\n".join(matches)


def run_agent():
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    agent = create_agent(
        model=llm,
        tools=[calculator, web_search_stub],
        system_prompt=(
            "You are a tool-grounded assistant. For arithmetic, call calculator. "
            "For knowledge searches, call web_search_stub. "
            "Treat the tool response as the only source of truth. "
            "Do not replace, reinterpret, or contradict its result with outside knowledge. "
            "If the tool reports no matching topic, say so. "
            "The search tool accesses only a local knowledge base, never the internet."
        ),
    )

    results = []

    for index, task in enumerate(AGENT_TASKS, start=1):
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": task}]},
                config={"recursion_limit": 12},
            )

            messages = response["messages"]

            tool_messages = [
                message
                for message in messages
                if isinstance(message, ToolMessage)
            ]

            tool_calls = [
                message.name for message in tool_messages
            ]

            tool_results = [
                str(message.content) for message in tool_messages
            ]

            expected = (
                ["calculator"] if index == 1
                else ["web_search_stub"] if index == 2
                else ["calculator", "web_search_stub"]
            )

            verified = set(expected).issubset(set(tool_calls))
            agent_final_answer = str(messages[-1].content)
            answer = "\n".join(tool_results)

            success = (
                verified
                and bool(tool_results)
                and all(result.strip() for result in tool_results)
            )

            results.append({
                "task_id": index,
                "task": task,
                "answer": answer,
                "agent_final_answer": agent_final_answer,
                "tool_results": tool_results,
                "tool_calls": tool_calls,
                "verified": verified,
                "success": success,
            })

            print(f"\nAgent task {index}: {task}")
            print(f"Tools used: {tool_calls}")
            print(f"Tool results: {tool_results}")
            print(f"Agent final answer: {agent_final_answer}")

        except Exception as exc:
            results.append({
                "task_id": index,
                "task": task,
                "success": False,
                "error": str(exc),
            })
            print(f"Agent task failed: {exc}")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building W6D5 document vector store...")
    collection, embeddings, chunk_count = build_vector_store()

    print(f"PDF chunks indexed: {chunk_count}")
    print(f"Collection count: {collection.count()}")

    print("\nRunning five document questions...")
    document_results = run_document_questions(collection, embeddings)

    print("\nRunning five conversation memory turns...")
    memory_results = run_conversation_memory()

    print("\nRunning three agent tasks...")
    agent_results = run_agent()

    evidence = {
        "task": "W6D5 Document Chatbot with LangChain",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "embedding": EMBED_MODEL,
            "chat": CHAT_MODEL,
        },
        "pdf": PDF_PATH.name,
        "pdf_pages": len(PyPDFLoader(str(PDF_PATH)).load()),
        "collection": COLLECTION_NAME,
        "distance_metric": "cosine",
        "embedding_dimensions": len(
            embeddings.embed_query("dimension check")
        ),
        "pdf_chunks_indexed": chunk_count,
        "collection_count": collection.count(),
        "document_questions": {
            "total": len(document_results),
            "passed": sum(x["success"] for x in document_results),
            "results": document_results,
        },
        "conversation_memory": memory_results,
        "agent": {
            "total_tasks": len(agent_results),
            "passed": sum(x["success"] for x in agent_results),
            "all_expected_tools_verified": all(
                x.get("verified", False) for x in agent_results
            ),
            "results": agent_results,
        },
    }

    evidence_path = OUTPUT_DIR / "chatbot_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nW6D5 SUMMARY")
    print(
        "Document questions:",
        evidence["document_questions"]["passed"],
        "/ 5",
    )
    print("Memory turns:", memory_results["successful_turns"], "/ 5")
    print("Memory retained:", memory_results["all_inputs_retained"])
    print("Agent tasks:", evidence["agent"]["passed"], "/ 3")
    print(
        "All expected tools verified:",
        evidence["agent"]["all_expected_tools_verified"],
    )
    print("Evidence saved:", evidence_path)


if __name__ == "__main__":
    main()
