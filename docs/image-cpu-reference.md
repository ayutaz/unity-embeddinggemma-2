# Complete image CPU reference measurement

This harness connects the official CPU image processor, actual vision encoder /
projector and actual language backbone for four calls per fixed image profile.
It is a Python comparison process. Unity SDK inference continues to use Sentis
GPU workers; this adds no product CPU backend or automatic fallback.

## First real-model result

[Source 8c60111 / run 39297313285](results/image-cpu-reference-8c60111-actual-20261020.json)
completed all three four-round measurements. Landscape and portrait passed;
square failed with 600 original encoder tolerance violations in each round.
All 12 final vectors and original pixels / positions matched. Three complete
archives, all CRCs and two private copies were verified; all 48 captures were
independently rechecked. Original FP32 pointwise and producer nominal FP64
comparison counts agreed on these actual captures.

Square and landscape both reported AVX512, while portrait reported AVX2. The
physical CPU model and complete math context were not captured in this first
source. ISA labels alone do not explain the difference. The cause remains
unproven and the whole CPU comparison remains unqualified. This result is kept
under its original source, not relabeled as a later-source execution.

## Inputs and source

Use the [unchanged original image data](image-reference-distribution.md). The
encoder target is the existing GPU fixture pin `3dfccd1a…`; the Document final
target remains `90692ad3…`. The distinct historical Document encoder generation
`7c183c52…` is recorded separately. Prior AVX2 and cross-generation failures are
preserved and do not become passes by running this harness.

Before loading weights, the harness verifies source HEAD and the executed Python
files / lockfile against Git, all six original snapshot file hashes, and the
reviewed procedural PNG's original size/hash. Model acquisition uses the fixed
model revision. The three measured inputs are `square-0`, `landscape-0` and
`portrait-0`, matching the fixed GPU performance inputs. This is not an arbitrary
image or all-environment compatibility claim.

## Four actual carries and retained evidence

Each first / warm / repeat-1 / repeat-2 call executes preprocessing → encoder →
backbone. Actual processor positions and pixels go to the encoder, then its
actual features go to the backbone. Expected intermediate features are used only
for inspection. Original input state is checked before and after each call.

The optional `CpuReferenceCase.after_stage` observer snapshots actual output
tensors after each stage timer stops and before the next consumer can mutate
them. Its return value never replaces the actual downstream input. Snapshot wall
and process CPU times are recorded separately and excluded from reported stage /
chain intervals. Other protocol checks remain in the chain interval; loading,
input signatures, vector inspection, capture disk writes and artifact handling
are outside it. Peak memory includes instrumentation.

Each profile retains four pixels, four positions, four encoder outputs and four
final vectors, plus source, timing and audit JSON. Int64 positions must fit Int32
losslessly before encoding the original `<i4` representation. Floating captures
retain actual FP32 bytes. Intermediate inspection checks original shapes and
pixel/position SHA256, encoder `atol=1e-5 / rtol=1e-4` and minimum row cosine
0.999999. Pixel/position hash checks are stricter than an element tolerance.
Final vectors keep the original finite 768-value, squared unit norm tolerance
0.002 and cosine 0.999 gates. No expectation is normalized or regenerated.

The separate auditor recomputes these gates from raw captures, rechecks their
hashes and fixed filenames, and binds the source journal, case, module, original
references and timing-stage sequence. A reported success, updated capture hash
or final-vector match cannot conceal an invalid intermediate.

Model loading / capture failures preserve safe phase and error-class records.
Numerical failures retain their captures and return a failing process exit.
Output must be empty; an old report or failed attempt is never overwritten.
Linux peak RSS is recorded for the whole process under an explicit 16 GiB
ceiling; other hosts report that metric unavailable. This is not SDK-exclusive
memory, VRAM or a validated minimum hardware requirement.

## Recorded CPU context and controlled comparison

The context-aware report format is schema 2. Before model loading and after
execution it records selected CPU model/vendor fields, the full Torch runtime
version, selected math-library version lines, matmul precision, MKL / oneDNN
availability and selected numerical environment variables. Changes during a
measurement fail qualification. The independent auditor binds both snapshots to
the source journal. Older schema-1 evidence remains under its original source.
It does not acquire fields that were never observed.

Only selected hardware fields are returned; machine names, serials, user fields
and file paths are excluded. Selected values still undergo text privacy checks.
Missing hardware data and unobserved denormal state remain unknown. Library
availability and requested environment values do not prove a particular operator
dispatch or executed instruction set.

For a controlled comparison, select both `image_cpu_measurement=true` and
`image_cpu_host_comparison=true`. Each of two hosts runs all three shapes in
separate processes, first with `MKL_CBWR` unset, then with `MKL_CBWR=COMPATIBLE`.
The same fixed model, inputs, four actual carries and original gates apply.
Every result is retained. Any failed measurement or audit makes the job fail;
the next case is still attempted so a negative baseline does not hide a later
comparison. Models and snapshots stay outside the upload directory.

This is an experiment, not an assumed repair. Intel documents conditional
reproducibility controls and limits non-Intel processors to `AUTO` / `COMPATIBLE`
choices; this does not promise equivalence to these original goldens.
[Intel oneMKL CNR](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/getting-started-with-conditional-numerical.html).
PyTorch likewise does not guarantee bitwise identity across platforms, so all
original numerical gates are still applied to actual captures.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

## CI execution

The [paired-host actual result](results/image-cpu-paired-hosts-1789f0c-actual-20261020.json)
records run `39304317802` at source `1789f0c`. Both host samples reported AMD
EPYC 9V45, Torch `2.14.1+cu130`, oneMKL 2024.2 / MKL-DNN 3.12.0, `highest`
matmul precision and four intra-op / two inter-op threads. Context snapshots
were stable within each host; only the requested `MKL_CBWR` setting changed.

| Math context | Square violations per round | Landscape | Portrait | Actual chains |
| --- | ---: | ---: | ---: | ---: |
| Native, both host samples | 0 | 0 | 0 | 24 passing |
| COMPATIBLE, both host samples | 361 | 95 | 128 | 24 failing |

The complete two ZIPs, provider digests, CRCs and two private copies were
verified. Independent recomputation covered all 192 tensor captures. All 48
final vectors and every pixels / positions capture matched their original
gates. Native encoder values were bitwise equal to the pinned original; the
COMPATIBLE intermediate failures remain failures. The workflow and both jobs
therefore concluded `failure` and are never counted as passing executions.

Native chain wall time ranged from 9.065 to 9.691 seconds on first execution
and 8.832 to 9.641 seconds over subsequent executions. Whole-process peak RSS
was 3,629,277,184 to 3,854,102,528 bytes. These values exclude loading and
observer copies and do not establish SDK-exclusive memory, minimum hardware
or a speedup against a different GPU host. The older square result with 600
violations lacks the physical CPU context; its cause remains unproven.
This result establishes only the recorded native CPU reference cases, not
arbitrary CPU compatibility, Windows CPU comparison or complete performance.

Use the existing **Frozen native CPU context diagnostics** workflow with only
`image_cpu_measurement=true`. Legacy selections must retain their defaults.
Three independent Ubuntu jobs use FP32, four intra-op threads and two inter-op
threads. They fetch original data, download / verify the pinned snapshot, measure,
then independently audit. Failures remain failures. A matching artifact privacy
guard precedes upload of only numerical output and reports, including safe
partial evidence; model weights and snapshots are excluded.

From `tools/`, equivalent explicit entries are:

```sh
uv run --locked python -m embeddinggemma_tools.image_reference_distribution --output ../artifacts/original-image-references
uv run --locked python -m embeddinggemma_tools.image_cpu_reference measure --references ../artifacts/original-image-references --profile square --output ../artifacts/image-cpu-square --source "$GITHUB_SHA"
uv run --locked python -m embeddinggemma_tools.image_cpu_reference audit --references ../artifacts/original-image-references --output ../artifacts/image-cpu-square --source "$GITHUB_SHA"
```

`GITHUB_SHA` must be the actual checked-out source commit. Prefer CI for real
weights and measurement. [TDD evidence](results/image-cpu-reference-tdd-20261020.json)
is small synthetic validation, not real-model acceptance. Real execution and
independent artifact readback must be recorded separately. All reports retain
`CPU_comparison_qualified=false` and `complete_performance_accepted=false`;
required environments, matching GPU measurements and remaining release gates
still need their own evidence.
