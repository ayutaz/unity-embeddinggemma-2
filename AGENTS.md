# Development rules

- Never push directly to `main`. Create a task branch and submit a pull request. Do not merge unless requested.
- Implement all behavior with TDD: write the test, observe the intended failure, implement, then verify green.
- Use `tools/` and uv for all Python work (`uv sync`, `uv add`, `uv run`). Do not use pip install.
- Prefer GitHub Actions for tests, model downloads/conversion, Unity execution, and benchmarks. Keep local execution to small necessary checks.
- Do not commit downloaded weights, generated model files, or large validation artifacts. Do not use Git LFS for models.
- M1 requires actual Sentis Editor results against the pinned Python reference. Offline/tiny-model tests do not prove real-model or GPU compatibility.
- Record failures and missing execution separately from verified passing results. Never count a skipped GPU test as a GPU pass.
