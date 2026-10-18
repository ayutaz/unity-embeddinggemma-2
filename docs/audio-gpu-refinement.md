# Saved-candidate audio GPU preparation

This prepares an **unqualified** candidate. It does not execute the full encoder,
validate GPU arithmetic, or change the original numerical reference. The seed is
pinned to source `0cee4c7b77c89410895d52e4420816a5e66a8845` and Actions run
`39117103059`; both profile artifacts, receipt bytes and model bytes are checked
before ONNX deserialization. A fetched ZIP is also hashed in full before extraction.

From `tools/`, in a new workspace with separate input and output directories:

```sh
uv sync --locked
uv run --locked python -m embeddinggemma_tools.audio_gpu_refinement \
  --fetch --input ../artifacts/candidate-input \
  --case silence --source "$GITHUB_SHA" \
  --output ../artifacts/candidate-gpu-refinement
```

The execution source must be a full lowercase commit SHA. Use `padded_noise` for
the second profile. Existing output directories are rejected; original evidence
is never overwritten. For CI, dispatch **Frozen audio native normalization replay**
with `prepare_gpu_candidate=true` and `preparation_stage=refined-partitions`.
The default stage remains `seed` for the existing seed workflow.

Preparation proves fixed Pad geometry from bounded integer expressions, audits
both LayerNorm scopes, all 109 RMS paths, the 12 transposed outer RMS paths and
the second stem convolution. It applies explicit FP32 inverse, mean, moment and
convolution refinements while retaining every original initializer byte. The
single biased Gemm keeps its original bias in a following Add; this explicit
product-then-add still requires original-reference numerical acceptance.

The planner cuts 135 fixed projections, 36 dynamic matrix products, 36 SiLU
operations and 12 each of query scale, Tanh, scalar division, Softmax and GLU:
267 operations and 268 segments. Original coefficient views may include transpose
and broadcast. External coefficient files are written in audited K-by-N order;
source orientation and both hashes remain in the plan. Closed Slice-based GLU
halves are normalized to Split before cutting, without changing coefficients.
Every segment boundary must have explicit positive FP32 dimensions.

`partition-plan.json` binds files by size and SHA256 and records each refinement.
`audio-gpu-refinement.json` records the source, phase and result; failures record
only the exception type, without raw diagnostics. Each saved segment receives a
textual metadata privacy audit. Workflow uploads require the matching privacy
guard. Failed runs upload only the safe status receipt, never partial models.

Successful preparation leaves `numeric_acceptance`, `gpu_validated` and
`generation_qualified` false. Adoption requires actual Sentis GPU execution for
all six original signals at the original elementwise tolerances, the required
roles and final consumer/Player gates. Older retained-bundle GPU results do not
qualify this candidate.

The first actual two-profile refinement run at `b6ede71` completed successfully:
[Actions 39120142716](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39120142716).
The [complete uploaded-artifact readback](results/audio-candidate-gpu-partitions-readback-b6ede71-20261018.json)
verified both ZIP hashes, every member CRC, all 415 plan-bound payload sizes/hashes
per profile and two exact private archive copies. Each archive has 417 members,
including the plan and status receipt. These are preparation and preservation
results; they do not themselves qualify full-model numerical or GPU execution.

The subsequent [actual encoder GPU gate](results/audio-candidate-direct-encoder-gpu-a164cba-20261018.json)
at fixed Git source `a164cba` passed both profiles and all six original signals.
Independent comparison of 824,320 captured values found zero original-tolerance
violations, and six corrupted-output controls were rejected. Both processes
exited 0. The SDK and validation package source readback checked 510 files: 508
with text line-ending normalization or exact binary hashes, and the two package
manifests with only the installed UPM fingerprint added and all original fields
unchanged. This qualifies the candidate encoder gate, not its final roles,
Player, other platforms or release generation. The original native PT2 bytes
remain unrecovered.

## Separate actual GPU gate

`CandidateAudioGpuReferenceTests` in the opt-in validation package keeps the old
retained-model gates intact. Both SDK and validation package must resolve from
the same fixed Git commit. It accepts only the two complete preparation plans
and status receipts pinned to the independently read-back bytes above, verifies
all 415 payloads and keeps preparation qualification flags false.

Place already hash-verified extracted payloads under
`Assets/CandidateGpuAudio/frames-1119-valid-1119` or
`Assets/CandidateGpuAudio/frames-1119-valid-839` before starting Unity. Verify the
copied bytes before the Editor imports ONNX assets. Run one profile at a time
with actual GPUCompute, Unity 6000.3.21f1 and Sentis 2.6.1:

- `FullProfileMustMatchAllFiveOriginalSignalsOnGpu`
- `PaddedProfileMustMatchOriginalSignalOnGpu`

Set `EG_ACCEPTANCE_SOURCE` to the fixed execution commit,
`EG_ORIGINAL_AUDIO_ROOT` to the separately preserved original reference/input
directory and `EG_CANDIDATE_GPU_REPORT` to a new private result file. The tests
reuse the original strict elementwise gate and its immutable reference/audit
hashes. Workers must reject CPU computation and intermediate readback. Raw
Editor logs, NUnit results and original vector captures remain private.

The [provenance guard TDD](results/candidate-audio-plan-guard-tdd-20261018.json)
observed 13 intended failures before implementation, then 13 passes with no
skips and process exit 0. The actual GPU fixture compiled in the final run.
This local-file/no-graphics test loaded no model; it proves neither a fixed Git
consumer nor GPU arithmetic. Startup, framework and initial fixture compile
failures are retained separately.

The five additional `PcmFull*MustMatchAllOriginalRolesOnGpu` tests
and `PcmPaddedNoiseMustMatchAllOriginalRolesOnGpu` fixtures reuse the original
PCM, preprocessor, backbone, tokenizer and Query/Document/Raw reference hashes.
Run each signal's three roles in its own process; all six signal results and 18
vectors remain required. Set `EG_CANDIDATE_GPU_REPORT` to a new private base file;
the fixture adds the selected signal to its parent filename and gives its child
report and three captures unique names. The signal must finish before its final
role flag can pass. Existing invalid-input, real-stage-failure and disposal/reuse
controls remain required. Their initial
[compile checkpoint](results/candidate-audio-role-fixture-compile-20261018.json)
passed the 13 provenance controls; actual role execution is a separate gate.

The initial [five-signal grouped role attempt](results/candidate-audio-grouped-role-timeout-b4fbf2e-20261018.json)
failed the default 180000 ms Unity test limit after one signal's three vectors.
The parent later recorded a MissingReferenceException; its causal relation is
not proven. This is a failed overall attempt, not a role acceptance or a runtime
stability improvement. The revised per-signal protocol preserves the default
test timeout, all original reference thresholds and the full 18-condition scope.

The subsequent [per-signal actual PCM role gate](results/audio-candidate-real-pcm-all-roles-gpu-c1e9aa8-20261018.json)
at fixed Git source `c1e9aa8` passed all six signals and all 18 original role
vectors. All six owned processes exited 0; no test skipped. Independent capture
comparison found a minimum cosine of `0.9999997097692651`, against the unchanged
FP32 gate of `0.999`, and rejected 18 corrupted-cosine controls. The impulse
invalid-input, actual stage-failure disposal and borrowed-stage reuse controls
also passed. Original captures and receipts were copied and hash-verified in a
separate private backup. This audio API result leaves bundle serialization,
Player, actual microphone, other platforms and required CI acceptance open.

## Candidate bundle serialization

`CandidateAudioBundlePreparation.Run` serializes both pinned audio profiles
using the same segment transforms, aliases and operation coefficients as the
GPU gate. It reuses the original hash-checked preprocessor, backbone and
tokenizer. Both SDK and validation packages must resolve from the fixed
`EG_ACCEPTANCE_SOURCE`; set `EG_CANDIDATE_BUNDLE_OUTPUT` to a new directory.
Run the entry point with Unity batch mode and `-nographics` after the verified
models have been imported. It performs no model inference. The safe phase
receipt is `candidate-audio-preparation.json`; existing output is rejected.

The [preparation TDD](results/candidate-audio-bundle-preparation-tdd-20261018.json)
observed three intended missing-implementation failures, then passed 22
destination, provenance and writer checks with no skips and process exit 0.
This proves preflight/serialization controls, not a real candidate bundle pass.
Generated bundle bytes require independent readback, privacy validation and
actual saved-bundle GPU/Player checks before adoption.

The subsequent [actual saved-bundle and Player checkpoint](results/audio-candidate-saved-bundle-and-player-39be89d-20261018.json)
serialized both profiles at `39be89d`, verified all 434 descriptor objects
(285 unique Sentis models), their original operation/coefficient bindings and
a second private copy. Model-description UTF-8 strings and tokenizer JSON
passed a separate privacy audit; numerical weight buffers were hash-bound,
not interpreted as text or generally approved for arbitrary binary uploads.
The saved Editor GPU Document gate passed all six signals with the unchanged
12 GiB host-native guard, process exit 0 and six rejected corrupted controls.

The same new FP32 bundle then passed the actual Windows IL2CPP / Release /
High stripping Player for both audio profiles. Independent original-reference
recalculation verified 18 base and 18 warm vectors, both before/after audits
of all 270 Workers, cancellation recovery, release and process exit 0. All
24 corrupted vector/role/backend/flush/identity/process/coverage controls
were rejected. Warm vectors are in-process repetitions, not independent
warm launches. Pending-call cancellation was tested at the first yield, not
while a GPU kernel was in flight.

The binary was built at `89b1ab9`; the bundle was produced at `39be89d`.
Their SDK Runtime Git trees are identical, and the executable hash and
previously verified 62 compiled source files were checked. No new build is
claimed. Float16 derivation/acceptance, generation adoption, required CI,
empty-cache acquisition, final integrated UX/performance and other-platform
acceptance remain open.
