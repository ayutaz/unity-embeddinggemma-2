# Regenerated audio candidate qualification

This is an opt-in numerical qualification route for the fixed, **unqualified**
generation `audio-e9de158-20261018`. It does not change the mandatory original
graph gate or adopt a new model generation. GPU, consumer, Player and platform
acceptance remain separate requirements in the [release plan](roadmap.md).

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

## Remaining release conditions

Even a successful numerical audit leaves `generation_qualified=false` and
`gpu_validated=false`. Before adoption, the affected artifacts need actual
Sentis GPU encoder and final-vector verification, all required roles, mixed
search, cancellation/recovery, switching/release, fresh consumer and Release
Player evidence. All required platforms and final-candidate correspondence
remain mandatory. The original JSON, tolerances and failed baseline results are
retained; a separately adopted generation must keep its own identity.
