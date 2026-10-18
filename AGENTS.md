# Development rules

- Never push directly to `main`. Create a task branch and submit a pull request. Do not merge unless requested.
- Implement all behavior with TDD: write the test, observe the intended failure, implement, then verify green.
- Use `tools/` and uv for all Python work (`uv sync`, `uv add`, `uv run`). Do not use pip install.
- Prefer GitHub Actions for tests, model downloads/conversion, Unity execution, and benchmarks. Keep local execution to small necessary checks.
- Do not commit downloaded weights, generated model files, or large validation artifacts. Do not use Git LFS for models.
- M1 requires actual Sentis Editor results against the pinned Python reference. Offline/tiny-model tests do not prove real-model or GPU compatibility.
- Record failures and missing execution separately from verified passing results. Never count a skipped GPU test as a GPU pass.

## Public-data privacy

- Read `.agents/skills/project-privacy/SKILL.md` before publishing code, documentation, screenshots, PR bodies, releases or validation artifacts.
- Install the versioned Git hooks using `tools/install_privacy_hooks.py` via uv. Do not bypass pre-commit, commit-msg or pre-push privacy checks.
- Publish only relative/environment-based paths, public GitHub/noreply commit identities, and image bytes reviewed visually and with OCR. Keep original local evidence private; never regenerate an evidence screenshot to hide a leak.
- A clean HEAD does not make old ancestors safe. Scan outgoing history and publication payloads, not just current source files. Never echo detected private values in reports or comments.
