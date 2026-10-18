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
