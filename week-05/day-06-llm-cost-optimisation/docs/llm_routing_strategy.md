# LLM Cost Optimisation: Routing Strategy

## Objective

Route approximately 70% of requests to a low-cost local model and reserve
the remaining 30% for a hosted model when higher capability is needed.

## Proposed routing policy

1. Default to a local model (`llama3.2:3b` or `qwen2.5:3b`) for routine
   questions, extraction, classification, and short summaries.
2. Use a hosted model for complex multi-step reasoning, difficult synthesis,
   or requests where local output fails quality checks.
3. Apply a quality gate: if the local answer is incomplete or fails validation,
   retry with the hosted model when permitted.
4. Measure actual traffic. A 70/30 target is a routing objective, not a
   guaranteed outcome for every arbitrary batch of ten requests.
5. Log model name, token usage, latency, estimated cost, cache hit/miss,
   and quality outcome for every request.

## Evaluation plan

Compare the baseline (all requests sent to the hosted baseline model) with
the routed approach using the same representative evaluation set.

Report:
- Percentage of requests served locally and by hosted models.
- Estimated USD and INR token costs using the assignment's illustrative rates.
- Latency and task-quality changes.
- Cache hit rate and avoided hosted requests.
- Fallback frequency and failure cases.

## Pricing assumptions and limitations

The cost audit uses the assignment rates: GPT-4o input $5/M and output
$15/M; Claude Sonnet 4 input $3/M and output $15/M; Gemini 2.0 Flash
input $0.10/M and output $0.40/M; GPT-4o mini input $0.15/M and output
$0.60/M. USD/INR is fixed at ₹88 for this exercise, not a live FX rate.
Local hosted-token charges are modelled as zero; hardware, electricity,
maintenance, and operational costs are excluded.

Do not claim savings or preserved quality until the baseline and routed
systems have been measured on the same workload.
