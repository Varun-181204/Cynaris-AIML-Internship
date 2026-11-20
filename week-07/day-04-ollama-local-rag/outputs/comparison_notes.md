# W7D4 Model Comparison Notes

## Question 1 – RAG
Llama 3.2: 64,931.48 ms
Qwen 2.5: 91,162.52 ms

## Question 2 – Vector databases
Llama 3.2: 79,814.94 ms
Qwen 2.5: 77,971.71 ms

## Question 3 – Embeddings vs tokens
Llama 3.2: 71,495.34 ms
Qwen 2.5: 68,464.56 ms

## Quality observation
The Llama response to the fifth experiment prompt incorrectly
claimed it could not find information about Ollama. This is an
observed response-quality limitation.

The full model responses are available in model_comparison.json.
The three comparison answers should be manually reviewed before
making broader claims about which model is better.
