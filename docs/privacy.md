# Public data and privacy checks

Public files, images, commit identities/messages and publication payloads must not contain personal home-directory paths, personal email addresses, credentials, or unreviewed screenshots. Keep original local receipts and screenshots outside Git. Use relative paths or environment placeholders in shareable reports.

Run from `tools/`:

```sh
uv run --no-project --python 3.13 python ../tools/install_privacy_hooks.py
uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy staged --root ..
uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy history --root ..
```

The Git pre-commit hook scans staged bytes and configured identity. The commit-msg hook scans messages and trailers. The pre-push hook scans the complete ancestry of every outgoing commit, including ancestors that an older checkout could accidentally reintroduce. Failures print rule/file/line only. Missing tools, malformed policies, and scan errors fail closed. Do not bypass these checks.

Images are blocked until visual and OCR review are recorded in `.privacy-reviewed-assets.json`, bound to the exact SHA256. A changed image requires a new review. The checker cannot infer private names from pixels: the absence of text findings does not replace visual/OCR review. Existing reviewed numerical `.bin` fixtures are narrowly hash-bound; arbitrary binaries remain blocked.

Codex configuration is in `.codex/hooks.json`; its publication checks and the `project-privacy` skill supplement Git hooks and CI. Codex requires review/trust of new hook definitions through `/hooks` before it runs them. Until that review is complete, Git hooks and CI still enforce repository publication checks. See [official hook documentation](https://learn.chatgpt.com/docs/hooks).

During privacy cleanup, screenshots containing local paths were removed rather than regenerated or edited into numerical evidence. Shareable receipts may have redacted path fields; their numerical/model hashes retain their original meanings, while original raw report hashes do not prove the bytes of a redacted report. Historical source SHA references change during a history rewrite and must be re-qualified before use as installation pins.

Force pushing cleaned branches does not purge GitHub's read-only PR refs, cached commit views, forks or third-party clones. GitHub Support handles inaccessible PR/cache copies; record that cleanup separately from verification of writable branches. See [GitHub sensitive-data removal](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
