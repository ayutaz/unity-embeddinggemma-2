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
results; full-model numerical and GPU qualification remain unperformed.

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
