# Composed candidate sample validation

The new six-module distribution has separate, fixed gates. Retained-model
`GpuMediaExecutionPolicy` descriptor pins, precision thresholds and memory
limits remain unchanged. `CandidateMediaBundlePolicy` requires the exact FP32
or Float16 distribution descriptor, its base/audio producer descriptors and
all six module declarations. It has no default precision or arbitrary hash
override.

Use the following distinct candidate fixtures after installing the SDK and
validation package from the reviewed fixed Git source and importing the
matching MultimodalSearch sample:

- `CandidateMultimodalSearchSampleGpuTests.ReviewedDistributionSampleKeepsOriginalCrossModalReferences`
- `CandidateMultimodalSearchSampleGpuTests.ReviewedDistributionSampleRendersOriginalSearchResultsInPlayMode`

The rendered entry point requires actual PlayMode and a new private screenshot
prefix. Both entry points reuse the original PCM, image/document/query pins,
vector gates, full rank/score comparison, cancellation, module switching,
cache lease and Worker/kernel/tensor release checks. They require this complete
media cohort before model loading; the shared fixture then executes all four
pinned text queries:

```text
image-square:square-0,image-landscape:landscape-0,image-portrait:portrait-0,audio-1119:silence,audio-839:padded_noise
```

Set `EG_MEDIA_SAMPLE_CASES` to that cohort, `EG_MEDIA_PRECISION` explicitly to
`fp32` or `float16-weights`, `EG_MEDIA_BUNDLE_OUTPUT` to the matching reviewed
distribution, and new private `EG_MEDIA_SAMPLE_CACHE` / `EG_MEDIA_SAMPLE_REPORT`
locations. Set `EG_ACCEPTANCE_SOURCE` to the fixed execution source. Rendered
execution additionally uses `EG_MEDIA_SAMPLE_SCREENSHOT`. Source, package lock,
imported sample and original fixture bytes must be verified before execution.

The existing 12 GiB host guard remains the default; an explicit allowance up to
16 GiB requires the existing physical-memory preflight. Both candidate fixtures
keep the existing 900-second test cap. Never turn a skipped run, contract test
or file preparation into an actual GPU/UI pass. These sample gates are not
microphone/playback, independent cold/warm, resource attribution, all-platform
or full-release acceptance.

The [TDD/compile checkpoint](results/candidate-media-sample-policy-tdd-20261018.json)
observed ten missing-policy failures and seven missing-cohort-checker failures,
then passed 34 candidate/retained policy controls with no skips and process
exit 0. This used a local-file, no-graphics metadata consumer with official
embedded Test Framework 1.6.0. It compiled both candidate GPU fixtures but ran
no real model, PlayMode rendering or GPU arithmetic. Actual fixed-Git candidate
sample execution remains a separate checkpoint.

Keep original screenshots/captures and raw reports private. Any image proposed
for publication needs visual and OCR review of the exact bytes and a hash-bound
review. Do not regenerate or alter an evidence screenshot to hide a leak.
