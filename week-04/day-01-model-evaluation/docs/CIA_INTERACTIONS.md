# CIA Interaction Log — W4D1

## Interaction 1 — Senior AI/ML Mentor Code Review

- Mode: Full Stack Mentor Mode
- Scope: Evaluation correctness, leakage risks, low recall, learning curves,
  and test coverage.
- Feedback received:
  - The pipeline and stratified train/test split are appropriate.
  - Keep raw training features as input to cross-validation so preprocessing
    is fitted independently inside each fold.
  - Investigate low recall using validation-based threshold selection or
    class weighting.
  - Consider learning-curve variability and add threshold-related tests.
- Changes made:
  - Added standard-deviation bands to the learning curve.
  - Added a test that lowering the threshold does not reduce recall.
- Verification: Pending final rerun.

## Interaction 2 — Evaluation and Viva Review

- Mode: Full Stack Mentor Mode
- Scope: Precision-recall trade-off, threshold selection, learning-curve
  interpretation, improvements, and viva preparation.
- Feedback received:
  - Tune the decision threshold on validation data, not on the held-out test
    set.
  - Select the optimization metric according to the problem's needs.
  - Use training-versus-validation curves to investigate overfitting and
    underfitting.
  - Consider class weighting and feature improvements as further experiments.
- Changes made:
  - Documented low recall and threshold-selection considerations.
  - Added a threshold monotonicity test and learning-curve variability bands.
- Verification: Pending final rerun.

## Limitations
- The experiment uses synthetic data and is not a production deployment.
- The threshold test verifies monotonic recall behavior; it does not select
  an optimal threshold.
- No CrewAI, LangGraph, MLflow, or Ragas integration is claimed.
