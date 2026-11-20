# CIA Mentor Interactions – W7D4

## Interaction 1 – Final submission checklist
CIA reviewed the W7D4 submission requirements and highlighted:
- Identical prompts and system prompts for model comparison.
- Correct latency measurement and evidence.
- Quality observations supported by actual model responses.
- Honest limitations and complete deliverables.
- Descriptive commits and weekly PR updates.

## Interaction 2 – Implementation quick-check
CIA provided implementation review guidance covering:
- Ollama REST API configuration, timeout and error handling.
- Structured system and user messages.
- Five-prompt Llama 3.2 experiment.
- Fair comparison settings for both models.
- Latency evidence and reproducibility.
- Python code quality and unit testing.

## Actions and verification
- Both models received the same three comparison questions.
- Both models use the same system prompt and temperature setting.
- API requests use a 300-second timeout.
- Latency is measured using time.perf_counter().
- Five unit tests passed.
- Further response-quality observations are documented separately.
