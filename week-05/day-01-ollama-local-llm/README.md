# W5D1 — Running LLMs Locally with Ollama

## Objective
Run a local language model, call Ollama's HTTP API from Python, test a custom system prompt, and compare two models on identical questions.

## Environment
- Runtime: Python 3.14+
- Local inference server: Ollama
- Default model: `llama3.2:3b`
- Python client uses only the standard library; no cloud API key is required.

## Setup
1. Install Ollama and ensure its local service is running.
2. Pull the model: `ollama pull llama3.2:3b`
3. Verify the API: `Invoke-RestMethod http://localhost:11434/api/tags`
4. Run unit tests from the repository root:
   `python -m pytest week-05/day-01-ollama-local-llm/tests -v`
5. Run the interactive client:
   `python week-05/day-01-ollama-local-llm/src/ollama_client.py`

## API usage
The client sends a system message and a user message to `/api/chat` with streaming disabled. Model choice and system prompt are configurable. The client reports connection/API failures with a clear exception.

## Practical evidence
Run the five prompts and the three-question model comparison using the workflow supplied for this task. Save actual responses in `outputs/inference_results.json` and record observed differences in `outputs/model_comparison.md`. Do not substitute invented or expected responses for measured output.

## Viva notes
- **Model parameter vs prompt parameter:** model parameters are learned weights; a prompt is input instructions/context supplied at inference time.
- **Local model vs cloud API:** local inference can improve data control and avoid per-request cloud charges, but depends on local hardware and may have lower capability or speed.
- **Quantization:** representing weights with fewer bits reduces memory/storage needs and can improve inference efficiency, often with some quality trade-off.
