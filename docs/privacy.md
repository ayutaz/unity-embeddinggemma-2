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

The six small procedural RGB input fixtures have [visual and English/Japanese OCR reviews](results/privacy-synthetic-vision-images-review-20261018.json). Their original reviewed PNG bytes are versioned under `tools/embeddinggemma_tools/fixtures/vision/`. Vision preparation reuses them only when decoded mode, dimensions and every pixel match the generated input; different input pixels retain their own encoding. This avoids codec-dependent PNG bytes without changing model inputs. Artifact review requires the exact selected path, SHA256 and both review checks. It does not approve screenshots or new images by directory, filename pattern or file type. Original validation screenshots are not regenerated or altered.

Codex configuration is in `.codex/hooks.json`; its publication checks and the `project-privacy` skill supplement Git hooks and CI. Codex requires review/trust of new hook definitions through `/hooks` before it runs them. Until that review is complete, Git hooks and CI still enforce repository publication checks. See [official hook documentation](https://learn.chatgpt.com/docs/hooks).

Publication checks read untracked PR/API body files and release upload files too. File writes/edits inspect the actual new content. A missing, dynamic or unreadable publication input is rejected. The shell hook supplements the Git checks; arbitrary scripts and external clients cannot be guaranteed to pass through a tool-name matcher.

Every workflow artifact upload requires the adjacent artifact privacy guard to succeed for the same selected paths. It checks filenames, text, private configuration and unreviewed images without modifying evidence. Missing optional artifacts remain optional. Generated numerical/model payloads (`.onnx`, `.pt2`, `.data`, `.bin`, `.npy`, `.npz`, `.safetensors`) have separate producer validation: this text guard does not certify their internal bytes. Git commits and release uploads still reject arbitrary unreviewed binary content. Package consumer reports publish repository-relative paths or a generic placeholder instead of a developer's home directory.

GitHub rejected repository-wide commit-email metadata rules with HTTP 422 for this repository. No such server-side ruleset is active; do not claim otherwise. Local Git hooks, required CI and an explicitly supplied noreply squash author remain necessary. GitHub documents metadata restrictions for Enterprise organizations in its [rules reference](https://docs.github.com/en/enterprise-cloud%40latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

PR CI scans the explicit source head and all its ancestors. GitHub's synthetic PR test-merge commit can inherit account identity settings outside the source branch; it is not falsely reported as source-controlled history. The default manual history audit still scans all refs. Account email visibility and existing provider-generated PR/cache metadata require separate remediation.

During privacy cleanup, screenshots containing local paths were removed rather than regenerated or edited into numerical evidence. Shareable receipts may have redacted path fields; their numerical/model hashes retain their original meanings, while original raw report hashes do not prove the bytes of a redacted report. Historical source SHA references change during a history rewrite and must be re-qualified before use as installation pins.

Force pushing cleaned branches does not purge GitHub's read-only PR refs, cached commit views, forks or third-party clones. GitHub Support handles inaccessible PR/cache copies; record that cleanup separately from verification of writable branches. See [GitHub sensitive-data removal](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
