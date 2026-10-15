# W4D3 — CIA Mentor Interactions

## Interaction 1 — Serialization code review

**Topic:** Correctness, security, error handling, and testing.

**Mentor feedback:**
- The joblib and pickle round-trip approach and prediction comparisons are appropriate.
- Create parent directories before saving artifacts.
- Add contextual error handling for serialization and deserialization failures.
- Avoid bare runtime assertions as the only validation mechanism, because Python optimization can disable them.
- Keep pickle loading restricted to trusted artifacts.
- A unit test for serialization equivalence is valuable.

**Planned response:**
Inspect the existing implementation and tests first. Add only improvements that fit the current design, and rerun the test suite.

## Interaction 2 — Follow-up safety review

**Topic:** Validation after loading and artifact integrity.

**Mentor feedback:**
- Prediction equivalence is a useful baseline check.
- Add a reproducibility or missing/corrupted-artifact test.
- A SHA-256 checksum can detect artifact changes when compared with a trusted, previously recorded expected hash.
- The example hash provided by the mentor was a placeholder and must not be used as a real checksum.

**Planned response:**
Check existing error handling and test coverage. Prefer a practical missing/corrupted-file test. If checksum verification is added, calculate and store the real digest in a manifest and compare it against a trusted expected value. A checksum alone does not prove that an artifact is safe to deserialize.

## Verification status

- First CIA interaction: feedback received.
- Second CIA interaction: feedback received.
- Code changes based on feedback: implemented and verified by the four-test suite.
- Final test run after changes: all four tests passed; prediction equivalence and experiment evidence verified.

