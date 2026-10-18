---
name: project-privacy
description: Check this public repository before commits, pushes, PR descriptions, screenshots, or validation artifact publication; detect personal paths, private identities, credentials, and unchecked images.
---

Read `docs/privacy.md` and run the privacy checker from `tools/` with uv. Keep raw findings and original screenshots in ignored local storage; print only rule, repository-relative filename, and line number.

- Before committing, run `uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy staged --root ..`. Scan the Git index, not only working files. Use the public GitHub handle and GitHub noreply email for commit metadata; check commit messages and co-author trailers too.
- Before pushing, scan all ancestors of every outgoing commit. A clean current file does not make an old private ancestor safe. Never use `--no-verify`, an empty hooksPath, or an allowlist to conceal a finding.
- For `gh pr merge`, explicitly pass `--author-email` with the public GitHub noreply identity. Do not rely on the account's default squash author email. The commit name must match the public handle in its noreply address.
- Before a PR, issue, release, or artifact upload, inspect the actual outgoing body/file separately. It may be untracked and outside the commit scan. Use repository-relative or environment-variable paths. Do not copy terminal commands, stack traces, editor title bars, or absolute file-picker paths into public records.
- Keep each workflow artifact upload behind its matching privacy guard. A passing text/filename check does not certify numerical/model binary internals. Reject or privately retain unsafe diagnostics; do not weaken checks to publish a failing test log.
- Inspect every proposed screenshot visually and with OCR, including title bars, panels, filenames and metadata. Prefer synthetic paths and input data when capturing. Unreviewed images fail closed. Add a hash-bound asset review only after both checks, and repeat them whenever image bytes change. Do not alter validation evidence into a fabricated screenshot.
- Original numerical fixture binaries have exact SHA256 reviews. Do not generalize those reviews to logs, archives, model weights, recordings, or arbitrary binaries.
- If a leak is found, stop publication and automated merges, preserve work privately, inspect all refs and commit identities, and remove the data from current files and history. History rewrite/force push requires user authorization; ordinary privacy checking does not grant it. Preserve branch protection, guard remote heads with leases, verify the remote result, and account for GitHub PR/cache copies separately.
- Before deleting hosted workflow runs or artifacts, map all pinned artifact IDs and verify private backups of every required immutable byte, including model graphs. A metadata-only snapshot is not a model/reference backup. Run deletion also removes artifacts and can break acceptance CI. If exact backup verification fails, stop deletion and prepare a provider-support request; never substitute a regenerated model and call it the original.

Git hooks and CI are the enforcement layers. The skill and Codex tool hook supplement them; do not claim hooks are active before installation/runtime verification.
