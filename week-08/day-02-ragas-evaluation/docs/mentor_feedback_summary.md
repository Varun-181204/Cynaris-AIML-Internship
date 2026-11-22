# CIA Mentor Feedback Summary

Two mentor interactions were received for Week 8 Day 2.

The architecture checklist highlighted source metadata, document chunking, ChromaDB persistence, retrieval configuration and evaluation dataset quality.

The methodology review recognised the measurable faithfulness improvement from 0.5258 to 0.5750 after reducing top_k from 3 to 2.

Limitations identified:
- Evaluation dataset contains 10 questions.
- Only one retrieval parameter was optimised.
- Optimised faithfulness was measured separately; the other three metrics were not rerun.
- Retrieved chunk-level error analysis remains future work.

These limitations are documented rather than presented as completed work.
