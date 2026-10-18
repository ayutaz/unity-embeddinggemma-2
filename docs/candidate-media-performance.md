# Repeated-input performance for the reviewed distribution

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

Real candidate runs, media CPU comparisons, final stage/load/release/VRAM evidence,
minimum hardware and required devices remain pending. The contract tests and
historical retained-model observations do not complete those release gates.
