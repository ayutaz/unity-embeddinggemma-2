# Regenerated audio candidate qualification

This is an opt-in numerical qualification route for the fixed, **unqualified**
generation `audio-e9de158-20261018`. It does not change the mandatory original
graph gate or adopt a new model generation. GPU, consumer, Player and platform
acceptance remain separate requirements in the [release plan](roadmap.md).

## Direct saved-candidate GPU preparation seed

`audio_candidate_gpu_seed` provides a separate preparation route from the exact
hash-pinned candidate PT2 graph. It copies graph routing while borrowing every
original tensor attribute at the same binding. It replaces only the two audited
native LayerNorm calls (widths 32 and 128, no bias, epsilon 1e-6, unused statistics)
with the existing FP32 Welford8 arithmetic. No checkpoint weights are downloaded
again, no new learned parameters are created, and the candidate graph and original
numerical references remain unchanged. Source tensor hashes are checked again
after ONNX export tracing.

The ordinary derived GraphModule returns the saved program's single feature
Tensor; the export program's internal one-element output tuple is not treated as
an additional model output. Live native statistics, different normalization
contracts, computed scales and changed tensor bindings fail before export.

The existing registered normalization workflow has an explicit, false-by-default
preparation mode. It covers both profiles independently and retains the original
normalization replay mode separately:

```sh
gh workflow run audio-native-norm-replay.yml --ref fix/privacy-guard \
  -f prepare_gpu_candidate=true
```

Only a successful producer and the matching privacy guard may upload the seed
ONNX and its JSON receipt, for 90 days. Source PT2 files, raw inputs and original
references are excluded. The producer audits all textual protobuf fields, rejects
external or string tensor payloads, and removes only optional `doc_string` and
`metadata_props`. It compares every other protobuf field against the pre-removal
model, preserving tensor bytes and computation. Opaque numerical payloads are
preserved, not certified as arbitrary-binary privacy-safe data.

The [preparation checkpoint](results/audio-candidate-gpu-seed-preparation-20261018.json)
records 8 initial failing checks, 3 workflow failures and 2 additional CLI/receipt
failures before their fixes. The related 40 checks and actionlint then passed.
This is small-model/export/workflow evidence. An actual real-model CI execution,
further GPU refinements, original encoder tolerances, all six signals, real GPU
execution and final consumer/Player acceptance remain required. A seed export
alone never sets numerical, GPU or generation qualification to true.

The first actual two-profile run at `410c9e5`,
[39116468876](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39116468876),
failed both exports with `UnsupportedOperatorError`; neither model was uploaded.
The saved graph contains an explicit `aten::_to_copy` which the ordinary module
trace did not expose in the earlier tests. A saved-copy export reproducer and
four dtype/storage checks failed before a narrow export bridge was added. The
bridge uses `Tensor.to(copy=True)` with the same dtype/device semantics, keeping
values and distinct storage while exposing a supported legacy ONNX spelling.
The source graph is unchanged; this equivalent copy is recorded separately from
the two normalization arithmetic changes. The related 45 tests then passed.
The [failed-run checkpoint](results/audio-candidate-gpu-seed-export-failed-410c9e5-20261018.json)
does not claim that a subsequent actual export or any GPU acceptance succeeded.

The copy-corrected actual run at `bb72c30`,
[39116795317](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39116795317),
also failed both exports before upload. The failed graph dump no longer contains
the unsupported saved-copy operation; the installed opset-20 exporter registry
identifies the remaining internal `aten::_softmax` spelling as unsupported.
A saved internal-Softmax export test and three axis-specific bit controls failed
before an FP32 `torch.softmax` bridge was added. It rejects half promotion and
records this equivalent export spelling separately from normalization changes.
The related 49 checks passed. Both failed actual runs remain failures; successful
seed generation and subsequent numeric/GPU acceptance must still be observed.

The next actual run at `0cee4c7`,
[39117103059](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39117103059),
succeeded for both profiles, including the textual-metadata and artifact privacy
checks. The [complete uploaded-archive readback](results/audio-candidate-gpu-seed-readback-0cee4c7-20261018.json)
verified both entire ZIP SHA256 values, every member CRC, both model SHA256 values
and two exact private archive copies. The normal and padded models are
1,231,902,161 and 1,231,902,858 bytes. Each retains all 1,067 original used Tensor
bindings and their post-export hashes, creates no new learned parameters, and
records two LayerNorm replacements, one equivalent copy and twelve equivalent
Softmax spellings. There were no optional metadata fields to remove in either
actual ONNX file; every textual protobuf field passed the producer audit.

The artifacts `12132399063` and `12132404118` expire at
`2027-01-16T09:53:22Z`. They preserve preparation seeds, not a permanent model
distribution service. Both earlier failed runs remain failures. GPU refinements,
partitioning, all-six original encoder tolerances, actual GPU execution, generation
qualification and final consumer/Player correspondence remain required.

## Independent inputs and provenance

- The [candidate contract](../tools/references/audio-candidate-e9de158.json)
  pins producer source/run, artifact digest, both native graphs, the producer's
  reference/audit bytes, source tree and dependency lock. The loader verifies the
  contract's own SHA256 before reading a model.
- The original reference JSON and audit are downloaded separately from the
  [verified recovered golden](results/original-audio-golden-ci-readback-20261018.json).
  Their immutable SHA256 values, model revision, official input/weight hashes,
  Torch version and element tolerances remain unchanged.
- Every graph/input is size- and SHA256-checked before deserialization. Candidate
  outputs never supply the expected values. The original loader continues to
  require the original graph bytes.
- Reports identify the candidate generation and graph producer separately from
  the original reference producer. The ordinary original-graph auditor rejects
  candidate provenance.

## Execution

Run the existing workflow from the reviewed branch, with the explicit candidate
flag. Keep both recovery modes off. Heavy downloads and execution run in Actions.

```sh
gh workflow run audio-runtime-reference.yml --ref fix/privacy-guard \
  -f qualify_candidate_generation=true \
  -f recover_golden_only=false -f recover_original_bytes=false
```

Six jobs cover `silence`, `impulse`, `sine`, `nyquist`, `noise` and `padded_noise`
across both profiles. Each downloads one fixed graph and its original raw feature
input, then executes the existing five-part `ordered_original_midpoint_bmm`
pipeline alongside its native ONNX baseline. The actual previous ONNX output is
the next part's input; all inputs, outputs and carries are captured.

The native whole/cut control keeps `MKL_CBWR=COMPATIBLE`, `MKL_DYNAMIC=FALSE` and
four threads. The independent candidate auditor rehashes the captured control
and compares its bits to the captured whole output. It also recomputes the
existing metrics and every carry hash against the original reference. Fresh
native differences from the original reference remain visible diagnostics.

The aggregate audit requires exactly six distinct cases, both profiles,
824,320 original-reference values at one execution source, no per-element
tolerance failures and the existing cosine threshold. Missing, partial, mixed
generation/source, changed-control and forged-metric records fail closed.
Each execution has a 40-minute cap; the aggregate audit has a 10-minute cap.
Artifact uploads retain their actual failure status and require the adjacent
privacy check for exactly the selected payload. No new images are produced.

## Candidate preservation before artifact expiry

The candidate producer artifact has short retention. Preserve its exact source
bytes privately before expiry. Distribution copies use a separate SHA256 and
remove only optional node `metadata.stack_trace` fields. This is not recovery of
the original graph bytes. Original reference JSON and source model bytes are
never edited or regenerated by this operation.

The explicit manual preservation mode downloads both fixed candidate graphs and
all six audited raw inputs. It independently compares the parsed computation
structure, streams and hashes every other archive member, and rejects remaining
private paths, email or credential patterns in textual metadata and member names.
Opaque tensor payload contents are preserved by hash; this metadata audit does
not certify arbitrary binary contents as private-data-free.

Source and cleaned graphs then execute all six native inputs under the same
four-thread MKL controls, one profile at a time. Every output bit must agree.
Only successful comparisons write the distribution manifest; only a successful
job and matching artifact privacy guard permit the selected distribution upload.
Source graphs, original goldens and raw diagnostics are excluded from that
upload. Its requested retention is 90 days, which is preservation for subsequent
qualification, not a permanent public model acquisition service.

```sh
gh workflow run audio-runtime-reference.yml --ref fix/privacy-guard \
  -f preserve_candidate_distribution=true \
  -f qualify_candidate_generation=false \
  -f recover_golden_only=false -f recover_original_bytes=false
```

All generation and GPU qualification flags remain false. The ordinary mandatory
original-graph workflow and its pinned requirements remain unchanged. The small
tests and full-profile archive check are recorded in the
[preparation receipt](results/audio-candidate-distribution-preparation-20261018.json).
That preparation receipt predates the actual execution. Subsequently,
[preservation run 39109531607](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39109531607)
completed successfully: both profiles and all six actual native outputs were bit
identical before and after the debug-metadata removal. The uploaded artifact is
`12129783259`, with expiration `2027-01-16T07:56:21Z`.
[Independent readback](results/audio-candidate-distribution-ci-readback-20261018.json)
verified the uploaded manifest CRC and both nested model metadata structures and
privacy checks. It did not independently download and hash the entire uploaded
2.45 GB archive; that limit remains explicit in the receipt. No GPU acceptance
is inferred from native CPU comparisons.

The subsequent [full uploaded-archive readback](results/audio-candidate-distribution-full-readback-20261018.json)
completed independently. Its complete ZIP SHA256, every ZIP member CRC and both
uploaded model SHA256 values matched, with two exact private archive copies.
Both distribution graph hashes also match local derivations from the preserved
producer source graphs. The earlier metadata-only receipt retains its original
scope; it was not rewritten into a full-archive verification claim.

Large local transfers encountered HTTP 403 after the signed redirect's lifetime.
The artifact reader can now renew that same fixed artifact's redirect and retry
only the failed byte range. Renewal is bounded to one retry per range, concurrent
requests share an updated URL, and other HTTP failures remain failures. Exact
206, Content-Range, ZIP CRC and pinned model SHA checks remain in force; GitHub
credentials never accompany requests to the signed blob URL. The renewal and
existing artifact reader checks passed 26 tests, including a parallel expiry
case and an end-to-end mocked redirect/auth separation check. This is distinct
from the preservation run, which used the earlier reader implementation.

## Preserve the original reference and raw inputs separately

The metadata-clean graph artifact intentionally contains no numerical goldens or
raw feature inputs. Their earlier artifacts have independent short retention.
The manual input-preservation mode copies the exact original reference/audit JSON
and all six raw feature files into a separate, 90-day artifact. No model graph is
downloaded or deserialized, and neither references nor input features are
regenerated. Each selected input must match the unchanged original audit SHA256;
changed references, missing/changed inputs, wrong artifact provenance and an
occupied destination fail before a publishable manifest is completed.

```sh
gh workflow run audio-runtime-reference.yml --ref fix/privacy-guard \
  -f preserve_original_inputs=true \
  -f preserve_candidate_distribution=false -f qualify_candidate_generation=false \
  -f recover_golden_only=false -f recover_original_bytes=false
```

The mode is exclusive of model preservation, numerical qualification and
recovery jobs. A matching artifact privacy guard and successful execution are
required before uploading exactly `artifacts/original-audio-inputs/**`. The input
and distribution tests passed 32 checks; workflow/recovery checks passed 32 and
actionlint passed. These small checks do not claim an actual preservation upload
or GPU execution. Fixed acquisition pins must be updated only after the new
artifact's exact bytes and provenance have been independently read back.

[Input preservation run 39110353490](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39110353490)
succeeded at source `2971d748fe824ea4bcc580a2ec583f2f734b7420`. Artifact
`12130281114` expires at `2027-01-16T08:09:23Z`. Its
[independent full readback](results/original-audio-input-preservation-readback-20261018.json)
verified the entire uploaded ZIP SHA256 and every CRC, both original JSON hashes,
all six raw input hashes and two exact private ZIP copies. No model was executed.

The golden loader and input-only acquisition now pin that artifact's exact digest,
run and source rather than the earlier short-retention copies. Original JSON
bytes, source producer and tolerances remain unchanged. Native diagnostics retain
the same explicit original JSON SHA checks. The acquisition change passed 67
selected distribution/input/loader tests and 25 workflow/golden checks with
actionlint. The required original native-graph gate remains independent and is
not made passing by saving numerical inputs.

The committed loader at `0d13a3d` also completed an
[actual acquisition of both original JSON files and all six raw inputs](results/original-audio-preserved-acquisition-0d13a3d-20261018.json)
from the new pinned artifact. All eight output hashes matched, without any model
download or model execution. This verifies numerical-input acquisition only;
the complete six-module fresh consumer remains a separate release requirement.

## Relating the GPU preparation to retained original ONNX

Both original native ONNX files remain available with their original audit hashes
(`7770a8af...` and `af2e97b7...`); this does not recover the missing original PT2
files. Their [computation-identity baseline](results/original-native-onnx-computation-identity-20261018.json)
binds every embedded tensor, operator, edge, shape, type, IR version and opset,
excluding only `doc_string` and `metadata_props` throughout nested protobuf
messages. External tensor references are rejected. Source bytes remain private
and unchanged. The helper source hash and library versions are recorded.

Ten tests verified metadata-only identity, changed weights/operators/edges/shapes/
opsets/IR rejection, nested graphs, source-hash checks and external-data rejection.
The baseline was then computed on both exact original ONNX files. Comparing new
candidate exports to it and performing real GPU acceptance remain unexecuted.
A matching metadata-normalized identity would support reuse of preparation
arithmetic; it would not replace actual numeric/GPU or fresh-consumer tests.
The candidate producer audit declares different unsplit ONNX hashes
(`288adeda...` and `db9904af...`). Those declarations do not prove the exact
uploaded model contents or a metadata-only difference. Compare the actual
candidate computation before reusing any preparation on that basis.

## Actual native ONNX identity diagnostic

The [component baseline](results/original-native-onnx-computation-components-20261018.json)
adds independent hashes for model headers, the graph without dense initializers,
initializer values, named values and initializer order, plus operator counts.
Whole computation identity is still required for a match: equal weight values
alone, renamed edges or reordered initializer bytes do not silently pass it.
The component metadata is bound by a canonical JSON hash; original model bytes
remain unchanged and no model inference was used to compute this baseline.

An explicit manual diagnostic fetches each actual candidate native ONNX file and
the producer audit, checks fixed artifact provenance and audit/model SHA256,
then compares whole identity and components for the correct profile. A mismatch
returns a failing job and preserves an honest completed diagnostic; input or
parsing failures remain incomplete reports. Only the report JSON is selected
for the matching privacy guard and 90-day upload. Models and raw logs stay out
of that publication. Related computation/diagnostic tests passed 27 checks,
workflow/recovery checks passed 33 and actionlint passed; actual execution is
pending until its run is recorded separately.

```sh
gh workflow run audio-runtime-reference.yml --ref fix/privacy-guard \
  -f diagnose_candidate_onnx_identity=true \
  -f preserve_original_inputs=false -f preserve_candidate_distribution=false \
  -f qualify_candidate_generation=false \
  -f recover_golden_only=false -f recover_original_bytes=false
```

This diagnostic does not change the required original graph gate, numerical
tolerances or any GPU/model-adoption flag.

The [actual two-profile diagnostic and independent archive readback](results/audio-candidate-native-onnx-identity-readback-20261018.json)
completed at source `ef066bb` in run `39111610873`. Both jobs failed their whole
identity comparison, and both reports completed: each candidate has 587 dense
initializers rather than 586, and the 36 old `Swish` nodes are represented by
36 `Neg` / `Exp` / added `Add` / added `Div` chains. Model headers match; whole
graphs and initializer signatures do not. Both actual candidate file hashes
match the [previous division-SiLU export](results/m4-division-silu-real-onnx-failed-20261014.json),
which remained a failed unsplit numerical baseline. This is an identified
historical export change, not evidence of native identity or GPU acceptance.

The full-profile [retained GPU-preparation initializer comparison](results/audio-candidate-gpu-preparation-initializers-20261018.json)
also found the same multiset of all 587 typed initializer values in the candidate
and retained Welford8 model; named-value signatures differ. The later scalar-mean
GPU input contains additional transformation initializers. Neither the value
multiset nor historical file hashes prove complete preparation correspondence.
Trace the transformations and perform actual GPU/consumer/Player tests before
using either finding for generation adoption.

## Remaining release conditions

Even a successful numerical audit leaves `generation_qualified=false` and
`gpu_validated=false`. Before adoption, the affected artifacts need actual
Sentis GPU encoder and final-vector verification, all required roles, mixed
search, cancellation/recovery, switching/release, fresh consumer and Release
Player evidence. All required platforms and final-candidate correspondence
remain mandatory. The original JSON, tolerances and failed baseline results are
retained; a separately adopted generation must keep its own identity.
