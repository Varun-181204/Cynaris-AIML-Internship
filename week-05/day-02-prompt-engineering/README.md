# W5D2: Prompt Engineering & System Prompts with Ollama

## Objective
Explore how system prompts and prompt-engineering techniques influence local LLM responses, and compare Llama 3.2 3B with Qwen 2.5 3B.

## Implementation
- `src/ollama_api.py`: Standard-library Ollama chat API client with validation, error handling, and latency measurement.
- `src/prompt_experiments.py`: Five experiments covering role prompting, output constraints, few-shot prompting, structured reasoning, and audience adaptation.
- `src/compare_models.py`: Compares two local models using the same three questions and system prompt.
- `tests/test_ollama_api.py`: Unit tests using mocked HTTP responses.

## Evidence
- `outputs/prompt_experiments.json`
- `outputs/prompt_experiments.md`
- `outputs/model_comparison_results.json`
- `outputs/model_comparison.md`

## Observations
Both models produced the correct 75% accuracy calculation. Their explanations and formatting differed. Qwen had higher observed latency on two of the three comparison questions in this run; timings are environment-dependent and are not a general performance benchmark.

Review the saved outputs for the full responses and measured timings.

