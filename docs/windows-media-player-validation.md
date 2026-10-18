# Windows image/audio Player validation

Status: private implementation candidate. Editor contract tests and original-fixture loading are separate from an actual Release Player build or real-model GPU acceptance. Neither a new model generation nor the final multimodal product is qualified by this harness alone.

`ValidationPlayerBuild.BuildWindowsMedia(outputExe, sourceCommit)` prepares a Windows x64 IL2CPP / Release / High-stripping build. It uses an empty generated scene and the unique `EmbeddingGemmaMediaValidationBuild` resource marker. Existing text validation and ordinary Players do not receive this marker. Build settings and scenes are restored on success and failure, and an injected build cannot produce a successful build receipt.

## Inputs and invocation

The SDK declares `com.unity.modules.unitywebrequest@1.0.0` as a direct Runtime
dependency because media-cache loading uses `UnityEngine.Networking`. Its UPM
source is `builtin`; Sentis and Newtonsoft remain pinned registry dependencies.
Installing the SDK without samples must resolve that module transitively.
The [minimal fixed-Git build attempt](results/media-player-minimal-network-dependency-20261018.json)
at `3dc2737` resolved the requested SDK but failed compilation with CS0103/CS1069
before Player building because this declaration was missing. Two regression
tests failed before the fix, then 53 package/consumer tests passed. Actual
compilation and Release Player execution after the fix remain to be recorded.

Use an empty output directory, a full reviewed source commit, the resolved SDK, and the repository's validation assemblies. Keep the prepared model bundle and original inputs outside Git. Build receipts and model-generation qualification must identify the same source/model candidate before a run can contribute to final acceptance.

The fixture root contains these original files:

| Relative file | Purpose |
| --- | --- |
| `image-embedding-reference.json` | Original final document vectors, pinned SHA256 `90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac` |
| `vision/vision-reference.json` | Original image reference, hash checked without materializing its large intermediate arrays |
| `vision/vision-validation.json` | Pinned original image-file hashes and sizes |
| `vision/{square,landscape,portrait}/{0,1}.png` | Six original input PNGs; bytes are not regenerated |
| `audio-embedding-reference.json` | Original final document vectors, pinned SHA256 `281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001` |
| `audio/audio-validation.json` | Pinned original PCM/mask hashes and sizes |
| `roles/image/query/image-embedding-reference.json` | Original Query vectors, SHA256 `dc4ad0a8e43e391a251e7ff3fb4f569b6bddb7442852f8721b1b7657ea91f82d` |
| `roles/image/raw/image-embedding-reference.json` | Original Raw vectors, SHA256 `120faf6aea190466b625f7c606a4c14e38315101a265c7dd457af13694512e9c` |
| `roles/audio/query/audio-embedding-reference.json` | Original Query vectors, SHA256 `515a32124c94437b372f81b3be4eb605717b1c4ae9293bf2c3b851450f35565b` |
| `roles/audio/raw/audio-embedding-reference.json` | Original Raw vectors, SHA256 `ea4298313a536aa3b0c1f5eac7934240abaf3cb68170be36deb64c214fbb92a9` |
| `audio/{case}-pcm-fp32.bin`, `audio/{case}-samples-int32.bin` | Complete original 179200-sample arrays for all six audio cases |

Image-source/audit and audio-audit hashes are fixed in `PlayerMediaFixture`. Input paths must be canonical and relative. PCM masks must be a binary real prefix; padding remains exactly zero. Wrong profiles, amplitudes outside the SDK range and nonfinite values are rejected rather than truncated or clipped.

Launch the built executable with a compute graphics device. Do not use `-nographics` for real GPU execution. For example, with environment variables identifying privately retained inputs:

```powershell
& $env:EG_MEDIA_PLAYER_EXE -force-d3d11 `
  --embeddinggemma-media-bundle $env:EG_MEDIA_BUNDLE `
  --embeddinggemma-media-fixtures $env:EG_MEDIA_FIXTURES `
  --embeddinggemma-media-module image-square `
  --embeddinggemma-media-precision fp32 `
  --embeddinggemma-media-descriptor $env:EG_MEDIA_DESCRIPTOR_SHA256 `
  --embeddinggemma-media-output $env:EG_MEDIA_RESULT `
  --embeddinggemma-media-run-id $env:EG_MEDIA_RUN_ID
```

The fixture loader requires all three original roles; absent Query/Raw files cannot reduce the run to Document coverage. Input IDs repeat across roles, so captures identify both ID and role. Each original Texture/PCM input is shared across its three role cases and retained until fixture disposal.

Every argument is explicit. The descriptor digest is 64 lowercase hex digits; the unique run ID is 32. The output must not already exist. `float16` selects the existing 0.99 vector-cosine threshold; `fp32` selects 0.999. An independent qualification receipt must tie this precision to the actual pinned bundle. Supplying a descriptor hash does not qualify a model generation.

Run each fixed module in a separate bounded process:

| Module | Fixed cases across Document / Query / Raw |
| --- | ---: |
| `image-square` | 6 |
| `image-landscape` | 6 |
| `image-portrait` | 6 |
| `audio-1119` | 15 |
| `audio-839` | 3 |

## Evidence and remaining acceptance

The runner checks every input/reference hash before loading models, verifies all live Sentis Worker assignments before and after execution, captures final 768-dimensional vectors, and applies the existing finite/unit-length and cosine gates. It times the first successful inference before cancellation probes, tests a pre-cancelled and a pending API call, then checks three successful warm repetitions for the first input of each role and release. Receipt-write failures still attempt session release.

Pending-call cancellation occurs at the first asynchronous yield; it does not certify interruption during a submitted GPU kernel. Timings include preprocessing, scheduling, fences and final vector readback. Managed-heap samples are not process peak or VRAM measurements. Raw exception logs stay private; shareable receipts use exception types and relative fixture names rather than local paths. All outgoing receipts still require the project privacy scan. No screenshot is produced by this runner.

Injected contract execution always has `gpu_verified=false`. Only successful execution through the sealed prepared-module adapter inside the actual Windows Player can set the per-module GPU flag. The receipt always keeps `full_multimodal_acceptance=false` and `model_generation_qualified=false`: final acceptance additionally needs the existing text suite, all required roles, mixed search ranks/scores, sample UI, switching, cancellation during real work, independently bounded cold/warm processes, process/VRAM measurement, source/model audits and successful required CI. Preserve all unsuccessful or missing runs separately.

Actual Release compilation, IL2CPP reflection preservation and real-model execution remain unverified for this candidate until those runs occur. Contract tests, fixture parsing and build-option injection do not prove them.

The [2026-10-18 local audit](results/media-player-contracts-20261018.json) records 60 versioned Editor tests and five private checks loading the original fixtures: 65 passed, zero failed, zero skipped. Runtime/build/test source files were compared against the tested consumer snapshot. The initial unimplemented behavior and the receipt-write/auxiliary-Worker regressions were observed failing before their fixes. Raw logs, fixture bytes and NUnit XML remain private.

The [all-role extension](results/media-player-all-role-contracts-20261018.json) observed 19 failing checks before implementation, then 79 passes / zero failures / zero skips and Editor exit0, including five private checks loading all 36 pinned role vectors and the original inputs. The original audio Document reference predates `metadata.role`: only its exact pinned file is admitted with Document explicitly selected, and every row must still declare Document. Generic reference selection rejects absent or mixed role metadata. This is contract/input evidence; actual all-role Player GPU execution remains unperformed.
