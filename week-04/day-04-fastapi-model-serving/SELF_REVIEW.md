# W4D4 Self-Review Checklist

## Implementation
- [x] FastAPI root, health, and prediction endpoints implemented.
- [x] Reuse W4D3 serialized regression model.
- [x] Validate exactly ten finite numeric feature values.
- [x] Handle model loading failure and return HTTP 503 on prediction.
- [x] Add structured application logging and typed response model.
- [x] Document API contract and error responses.

## Tests and evidence
- [ ] All automated tests pass.
- [ ] Invalid counts, non-numeric values, NaN, and infinities are rejected.
- [ ] Missing-model behaviour returns HTTP 503.
- [ ] Evidence regenerated from the final code.
- [ ] Live Uvicorn endpoints verified.

## Documentation and mentor review
- [x] README documents setup, server startup, request schema, and error codes.
- [x] First CIA review findings recorded.
- [x] Second CIA submission-readiness review recorded.
- [ ] Final verification status updated after testing.

## Git workflow
- [ ] First meaningful W4D4 implementation commit.
- [ ] Second meaningful W4D4 evidence/documentation commit.
- [ ] Push existing W4 branch.
- [ ] Verify combined W4D3 + W4D4 PR.
