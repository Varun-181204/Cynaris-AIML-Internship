# W7D4 – Ollama Local LLM Inference

## Objective
Run local LLM inference using Ollama, apply a custom system prompt,
test five prompts with Llama 3.2 3B, and compare Llama 3.2 3B
with Qwen 2.5 3B on three identical questions.

## Models
- llama3.2:3b
- qwen2.5:3b

## Setup
Install Ollama and pull the models:

    ollama pull llama3.2:3b
    ollama pull qwen2.5:3b

Run from this project directory:

    python src/ollama_local_inference.py

Run tests:

    python -m pytest tests -v

## Implementation
The Python script calls Ollama's local REST chat API.
A custom system prompt is sent using the structured system message.
Five prompts are tested with Llama 3.2 3B.
Three of those questions are reused to compare both models.

The experiment records complete responses and request latency
in outputs/model_comparison.json.

## Observed results

| Question | Llama 3.2 latency | Qwen 2.5 latency |
|---|---:|---:|
| RAG | 64,931.48 ms | 91,162.52 ms |
| Vector databases | 79,814.94 ms | 77,971.71 ms |
| Embeddings vs tokens | 71,495.34 ms | 68,464.56 ms |

Llama was faster on the first question. Qwen was faster on the
second and third questions. These are single-run observations,
not statistically significant benchmark results.

One observed quality issue: for the fifth Llama prompt, the model
responded that it could not find information about Ollama. This
was not an accurate answer to the question.

## Evaluation criteria
Responses should be reviewed for:
- Factual accuracy
- Relevance
- Clarity
- Completeness
- Response latency

## Limitations and risks
- The experiment uses small local models with finite knowledge
  and context capacity.
- Only three questions are used for model comparison.
- Each comparison is a single run and is not statistically robust.
- Latency depends on local hardware, memory availability and
  background workload.
- The custom system prompt may influence response style and content.
- Model responses can contain incorrect or incomplete information.
- Retrieved or generated text should be manually checked before use.

## Testing
Five unit tests passed, covering request payload construction,
system prompt inclusion, disabled streaming, response parsing
and invalid response handling.

## CIA review
See docs/CIA_INTERACTIONS.md for the two mentor review interactions.
