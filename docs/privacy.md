# Public data and privacy checks

Public files, images, commit identities/messages and publication payloads must not contain personal home-directory paths, personal email addresses, credentials, or unreviewed screenshots. Keep original local receipts and screenshots outside Git. Use relative paths or environment placeholders in shareable reports.

Run from `tools/`:

```sh
uv run --no-project --python 3.13 python ../tools/install_privacy_hooks.py
uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy staged --root ..
uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy history --root ..
```

The Git pre-commit hook scans staged bytes and configured identity. The commit-msg hook scans messages and trailers. The pre-push hook scans the complete ancestry of every outgoing commit, including ancestors that an older checkout could accidentally reintroduce. Failures print rule/file/line only. Missing tools, malformed policies, and scan errors fail closed. Do not bypass these checks.

GitHub-generated PR test-merge commits can copy the account's display name even when local commits and noreply emails are safe. Before GitHub publication, run `uv run --no-project --python 3.13 python -m embeddinggemma_tools.privacy github-profile` from `tools/`. The check reads the authenticated GitHub profile through `gh api`; it never changes account settings or prints private fields. A nonempty display name must equal the public login or an explicitly user-selected public name bound to that exact account. The owner instructed that account `ayutaz` retain the public display name `yousan`; this exact pair is accepted. Other accounts, other names, whitespace/case variants and lookup errors still fail closed. This profile rule grants no exception for personal paths, private email, credentials or source commit identities. Git pushes to GitHub and Codex `gh pr create/edit/merge` commands require the profile check; local Git remotes and read-only commands remain independent. Existing PR/cache copies with private paths or email still need separate cleanup.

Images are blocked until visual and OCR review are recorded in `.privacy-reviewed-assets.json`, bound to the exact SHA256. A changed image requires a new review. The checker cannot infer private names from pixels: the absence of text findings does not replace visual/OCR review. Existing reviewed numerical `.bin` fixtures are narrowly hash-bound; arbitrary binaries remain blocked.

Actions also inspect PT2 archive/member names and textual metadata, plus ONNX
protobuf strings with the pinned parser. Models are not blanket binary exemptions:
unparseable files, missing parsers and private metadata fail closed. Tensor buffers
remain opaque and require their separate source/numerical validation. New PT2
exports omit optional node stack traces before first serialization; retained
original evidence is not rewritten. See the
[model artifact metadata checks](model-artifact-metadata.md).

Relative input fields alone do not make a sample screenshot safe: result metadata and exception messages can expand them to absolute paths. The media sample renders result scores/names and a general recovery message instead of full source paths or raw diagnostics. Full input metadata and `LastError` remain local diagnostic data and require a separate publication review. Inspect results and error states, preserve any failing original privately, and record later fixed-source execution as new evidence.

GitHub squash authors also use the account display name. After #89, the actual public main commit `ca2ff1a567825cd9be6986e10ec727b1908a8430` has the owner-selected public name `yousan` and the explicitly supplied `41669061+ayutaz@users.noreply.github.com` address; GitHub's API reports its signature verified. History classification recognizes only this exact public name/address pair as an author in commit objects with the GitHub/noreply committer and PGP signature header format. The offline checker does not authenticate that signature or trust code because of it. Other names, account IDs, private email, taggers, committers, unsigned metadata and locally configured identities retain the strict handle rule. Every reachable object, message and path is still scanned. This corrects a false positive for an explicitly public alias; it does not approve private identities or provider cache copies. See the [TDD/readback checkpoint](results/provider-public-author-tdd-20261018.json).

Model release uploads have a separate, narrow guard. `model_release_privacy` reads `.privacy-reviewed-model-releases.json` and its canonical-hash-bound review receipt from committed HEAD, never an untracked/staged policy override. It binds the exact descriptor and inventory, reviewed producer, all object names/sizes/hashes and source/module correspondence. Each proposed asset is stream-hashed again. Sentis model-description UTF-8 string candidates and tokenizer JSON are checked for private metadata; coefficient buffers are allowed only as exact reviewed producer bytes. A registered hash does not excuse a private description. Generic images, archives, missing/changed proofs and unreviewed binary objects fail closed. Index/history checks still reject model weights in Git.

The two [FP32](results/six-module-fp32-model-distribution-privacy-20261018.json) and [Float16](results/six-module-float16-model-distribution-privacy-20261018.json) composition reviews cover 820 exact objects and 553 Sentis models each. Every destination/source object and selected module binding was independently checked. Their numerical buffers are hash-bound copies of the reviewed producers, not scanned as arbitrary text or generally certified as harmless binary data. These records do not prove public acquisition, model execution, platform compatibility or formal release readiness. Composition inventories retain their original `privacy_review_completed=false` preparation checkpoint; the separate committed review is the later publication authority.

The six small procedural RGB input fixtures have [visual and English/Japanese OCR reviews](results/privacy-synthetic-vision-images-review-20261018.json). Their original reviewed PNG bytes are versioned under `tools/embeddinggemma_tools/fixtures/vision/`. Vision preparation reuses them only when decoded mode, dimensions and every pixel match the generated input; different input pixels retain their own encoding. This avoids codec-dependent PNG bytes without changing model inputs. Artifact review requires the exact selected path, SHA256 and both review checks. It does not approve screenshots or new images by directory, filename pattern or file type. Original validation screenshots are not regenerated or altered.

Codex configuration is in `.codex/hooks.json`; its publication checks and the `project-privacy` skill supplement Git hooks and CI. Codex requires review/trust of new hook definitions through `/hooks` before it runs them. Until that review is complete, Git hooks and CI still enforce repository publication checks. See [official hook documentation](https://learn.chatgpt.com/docs/hooks).

Publication checks read untracked PR/API body files and release upload files too. File writes/edits inspect the actual new content. A missing, dynamic or unreadable publication input is rejected. The shell hook supplements the Git checks; arbitrary scripts and external clients cannot be guaranteed to pass through a tool-name matcher.

The [original image reference route](image-reference-distribution.md) has a narrow
JSON-only release guard for three immutable synthetic reference/audit files.
`image_reference_release_privacy` reads the exact hash-bound review from committed
HEAD and rechecks the complete proposed bytes, full text privacy and producer /
model revision. Its bounded 32 MiB handling covers the original 23,780,437-byte
encoder JSON; it does not raise the ordinary 16 MiB limit, approve large notes,
allow arbitrary numerical/binary data or replace image visual/OCR checks.
Untracked/staged reviews, changed files and private text fail closed. These
metadata assets carry no graph, model execution or CPU/GPU qualification.

Every workflow artifact upload requires the adjacent artifact privacy guard to succeed for the same selected paths. It checks filenames, text, private configuration and unreviewed images without modifying evidence. Missing optional artifacts remain optional. Generated numerical/model payloads (`.onnx`, `.pt2`, `.data`, `.bin`, `.npy`, `.npz`, `.safetensors`) have separate producer validation: this text guard does not certify their internal bytes. Git commits and release uploads still reject arbitrary unreviewed binary content. Package consumer reports publish repository-relative paths or a generic placeholder instead of a developer's home directory.

Python CI loads `embeddinggemma_tools.pytest_privacy` so JUnit skip locations use repository-relative paths, or an explicit external-location placeholder. Test outcomes, skip counts, line numbers and reasons remain unchanged; personal data inside a skip reason still blocks publication. This reporting change never turns a missing device, model or GPU run into a pass.

GitHub rejected repository-wide commit-email metadata rules with HTTP 422 for this repository. No such server-side ruleset is active; do not claim otherwise. Local Git hooks, required CI and an explicitly supplied noreply squash author remain necessary. GitHub documents metadata restrictions for Enterprise organizations in its [rules reference](https://docs.github.com/en/enterprise-cloud%40latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

PR CI scans the explicit source head and all its ancestors. GitHub's synthetic PR test-merge commit can inherit account identity settings outside the source branch; it is not falsely reported as source-controlled history. The default manual history audit still scans all refs. Account email visibility and existing provider-generated PR/cache metadata require separate remediation.

During privacy cleanup, screenshots containing local paths were removed rather than regenerated or edited into numerical evidence. Shareable receipts may have redacted path fields; their numerical/model hashes retain their original meanings, while original raw report hashes do not prove the bytes of a redacted report. Historical source SHA references change during a history rewrite and must be re-qualified before use as installation pins.

Release creation and later uploads both check explicit asset bytes, including the `gh release new` alias and inherited repository options. Use a literal tag and file paths; unknown flags, missing files, dynamic paths and unexpanded patterns fail closed. Model assets still require the exact committed producer/hash review. That review never approves binary release notes, images, arbitrary archives or models in Git. The command hook supplements explicit payload review; it is not a complete shell interpreter or a substitute for reviewing other publication routes.

Native regression data uses its own committed HEAD policy, `.privacy-reviewed-native-regressions.json`, and [exact inventory](results/native-regression-release-inventory-20261019.json) / [review](results/native-regression-exact-asset-review-20261019.json). The [TDD checkpoint](results/native-regression-payload-guard-tdd-20261019.json) covers the narrowly registered two PT2 graphs, four reference/audit JSON files and six original input buffers. The [actual twelve-asset check](results/native-regression-real-asset-guard-20261019.json) passed using committed HEAD authority without review substitution; that checkpoint uploaded no files. Publication rechecks full file hashes, canonical source mappings, every graph member/CRC, semantic and non-model-member hashes, and textual metadata without importing Torch. The exact source-bound numerical buffers are not certified by text scanning. This adds no generic archive/binary exception and no model permission in Git. The later [data preview and durable-route implementation](native-regression-distribution.md) retain original numerical gates and distinguish actual executed graph hashes from preserved source identity. The required workflow uses that explicit route; the later [Windows v0.1.0 acceptance](results/windows-v010-release-20261021.json) records the completed six-case numerical audit. Earlier guard-only checkpoints remain distinct from model execution.

Force pushing cleaned branches does not purge GitHub's read-only PR refs, cached commit views, forks or third-party clones. GitHub Support handles inaccessible PR/cache copies; record that cleanup separately from verification of writable branches. See [GitHub sensitive-data removal](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
