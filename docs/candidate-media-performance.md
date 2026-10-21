# Repeated-input performance for the reviewed distribution

Windows v0.1.0 is published. See [current status](status.md) and the [measured Windows performance and memory scope](windows-media-memory-results.md). The checkpoints below describe their recorded sources; they are not the current release task list.

Historical checkpoint (2026-10-20): Windows functional stages 1–3 are accepted within the [source-qualified record](results/windows-first-stages-1-to-3-20261020.json). Performance was the next release stage at that checkpoint. The [four-condition audio GPU result](results/audio-gpu-repeat-four-conditions-20261020.json) covers 16 calls; its partial memory sampling does not establish a guaranteed peak, matched CPU speedup or minimum hardware. The contract and earlier execution notes below retain their own sources and scopes.

The [persistent Windows allocation observer](windows-gpu-memory-observation.md)
adds a read-only native PDH query with process-lifetime and clock checks. Its
small contract and CPU-only collector probe do not fill historical sampling
gaps or qualify real-model memory, performance or minimum hardware. New model
observations must retain their own numerical gates, actual intervals and gaps.

The later [ten-condition native allocation readback](windows-media-memory-results.md)
rechecks 40 actual GPU calls and samples in all 180 measured intervals, including
release. It records 12 query errors and unavailable samples separately. These new
observations retain their own sources; they do not repair historical gaps or
establish guaranteed peaks, matched CPU speedup or minimum hardware.

The candidate fixture uses the published FP32 / Float16-weight descriptors while
the existing retained-bundle fixture keeps its original pins. Both execute the
same four rounds: `first`, `warm`, `repeat-1`, `repeat-2`, with one identical
Texture/PCM input. Capture precision, module, descriptor and cosine threshold
before execution. Missing/unknown candidate precision or a non-media module fails
before loading; later environment edits cannot select a different threshold.

The [contract checkpoint](results/candidate-media-repeated-policy-tdd-20261019.json)
records 34 intended failures followed by 68 passing Editor checks, zero skips and
normal exit. A test-attribute compatibility failure and a missing import during
implementation are retained separately. This local validation overlay uses the
fixed SDK package bytes and does not execute models or prove GPU timings.

The explicit model-running entry is:

```text
EmbeddingGemma.Validation.Tests.CandidateMediaRepeatedInputPerformanceTests.ReviewedDistributionFixedInputFirstWarmRepeatedCallsKeepReferenceAndMeasureStages
```

Provide `EG_MEDIA_PRECISION` (`fp32` or `float16-weights`),
`EG_MEDIA_BUNDLE_MODULE` (one of three image or two audio profiles),
`EG_MEDIA_BUNDLE_OUTPUT`, a new `EG_MEDIA_EXECUTION_REPORT`, and the fixed
`EG_ACCEPTANCE_SOURCE`. Descriptor bytes and original Texture/PCM/reference
hashes remain pinned. FP32 cosine stays at 0.999 and Float16 weights at 0.99.
The existing host allowance remains 12 GiB unless a separately checked explicit
allowance is supplied.

`EG_ENCODER_GPU_DISPATCH_TRACE=1` retains the existing encoder and
preprocessor/backbone GPU submission observations. All eight submission scopes
must have positive GPU timestamps; missing scopes stay null and fail that gate.
These intervals include commanded copies/queue effects and do not establish pure
kernel or SDK-exclusive load/release time. Whole-process CPU, wall and native
memory keep their existing scopes. Optional frame allocation observations include
the surrounding Editor work and I/O; they are not SDK-exclusive allocation.

Reports bind `bundle_policy`, `composed_candidate_distribution` and the expected
descriptor to the actual descriptor hash. Original first/warm definitions,
stage boundaries, captures, four repeated calls, numerical checks, cleanup and
normal-exit requirements are retained. Keep raw reports private because input
diagnostics can contain local paths; publish only independently checked summaries.

The earlier image checkpoint below retains its own source and sampling gaps.
The [later GPU media readback](windows-media-memory-results.md) covers both audio
profiles and all image profiles. The [native CPU audio readback](native-audio-CPU-results.md)
adds eight original-gate-passing actual CPU carries for the same audio inputs.
The [explicit image CPU conditions](image-CPU-fixed-conditions.md) add twelve
original-gate-passing calls with per-condition statistics, load/release and
capture audits. A uniform sixteen-lane landscape failure remains recorded.
CI integration, minimum hardware, privacy cleanup and new-tag installation remain pending. Other platforms follow
the Windows-first plan. Contract tests alone do not complete those release gates.


## 実画像の反復測定

[固定sourceの測定記録](results/candidate-image-repeated-2cc7d94-20261019.json)は、公開取得した両精度の画像3形状で各4回、合計24回の参照一致・GPU計時区間・解放と正常終了を確認しています。SDKは固定Git、検証コードはimmutableなローカルGit archiveで、既存の所有Libraryを再使用しました。新規Git consumerの導入検証とは分けます。

| profile / weights | load 秒 | first 秒 | warm 3回 秒（範囲） | host peak GiB | sampled dedicated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| image-square / float16-weights | 46.342 | 46.161 | 4.988–5.215 | 7.650 | 4.277 |
| image-landscape / float16-weights | 42.672 | 7.044 | 4.736–4.964 | 7.751 | 4.720 |
| image-portrait / float16-weights | 37.398 | 7.013 | 4.707–4.873 | 7.736 | 4.783 |
| image-square / fp32 | 40.755 | 9.536 | 4.719–5.227 | 8.886 | 3.272 |
| image-landscape / fp32 | 38.223 | 7.020 | 4.673–4.941 | 8.984 | 3.840 |
| image-portrait / fp32 | 36.488 | 6.191 | 4.595–4.861 | 8.720 | 4.283 |

CPU値はEditor全体のCPU時間で、CPU backendとの比較ではありません。GPU値はqueueや転送を含むmarker区間で、kernel専有時間ではありません。GPUメモリは観測最大値で、保証されたpeakや最低要件ではなく、短いrelease区間などの欠測はnullです。各起動で明示16 GiB host allowanceを使用し、SDKの既定12 GiBは変更していません。

この旧checkpointでは、他のdesktop GPU processが動作する条件で起動前のGPU空き容量を記録し、音声4条件は空き約10 GiBに対する保守的な14 GiB起動基準を満たさず未実行でした。後続の音声実行は上記の別記録に保持します。この起動基準は最低hardware要件ではありません。専有GPU/VRAM・最低要件・他環境の必須端末を完了扱いにしません。
