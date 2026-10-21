# Opt-in real-weight validation

Updated 2026-10-21. Windows SDK v0.1.0 has completed fresh-tag Editor / Player acceptance; see [current status](../../docs/status.md). The numerical diagnostic notes below retain their original 2026-10-14 execution scope. This development-only package is separate from the runtime UPM package and weights-free contract tests. Pin runtime and validation to the same immutable Git SHA, add validation via `?path=/validation/com.ayutaz.embeddinggemma.validation#SHA`, and put both packages in `testables`. The runtime package never depends on this package. No weights or generated model files are included.

## Fixture entry points and historical execution scope

| Fixture | Scope | Backend policy |
| --- | --- | --- |
| `GpuExecutionProfileTests` | Tiny and one audited real square image; seven workload phases, host memory / CPU time | GPUCompute only; two tests, not numeric encoder acceptance |
| `FrozenAudioProjectionTests` | Fixed native input / weights; linear1, linear2, post norm only | Historical CPU/GPU component evidence; not full encoder |
| `CompensatedVisionReferenceTests` | Audited original vision FP32 models, 3 profiles, 2 images each | GPUCompute only; 3 conditions / 6 numeric cases |
| `CompensatedNormAudioEncoderReferenceTests` | Audited original audio features, 6 signals / 2 profiles, norm1 / projection64 / dynamic16 candidate | GPUCompute only; 1 condition / 6 numeric cases |

The compensated fixtures reject a CPU backend and unavailable compute shaders, verify output GPU residency, and reject CPU computation fallback / readback flush. Reports declare `execution_backend=GPUCompute` and `cpu_validated=false`. Their GPU-only changes compiled in fixed consumer `b796e504367fb9165a4992ff036a205fb2e7d429`; the profiling fixture passed both tiny / real image tests with no skips and process exit 0. This does not rerun every compensated numeric condition. Other historical fixtures still contain CPU cases; do not run this entire assembly as a GPU-only workload.

The user authorized plan steps 1 and 2. GPU phase measurements passed and the subsequent real Python ONNX SiLU validation executed all six original signals across both profiles, with all six failing the unchanged tolerance. No step 3 numerical fix or full-chain acceptance was performed. The historical interrupted vision run and previous 12 failed CPU/GPU audio conditions remain separate evidence. See the [new measurements](../../docs/gpu-execution-and-silu-validation.md), [status](../../docs/status.md), [plan](../../docs/roadmap.md), and [ledger](../../docs/steps-1-3-acceptance.md).

## Artifacts and load cost

Fixtures reuse audited sibling `artifacts/k/artifacts/vision-encoder` and `artifacts/k/artifacts/audio-real-chain` data. Projection component fixtures also use `audio-native-feed-forward-replay`. Missing files or SHA mismatches fail before numeric comparison. Keep source / graph / input / reference hashes unchanged and store each run separately.

Transforms are diagnostic Editor/test candidates, not runtime optimizations. The observed vision candidate expanded 1823 to 25265 layers; audio dynamic16 reached 45623 layers. Graph transformation and loading still use CPU resources before GPU scheduling. Do not infer speed or memory improvements from GPU residency or small tests. Profile individual phases and use the bounded one-input restart plan before attempting all conditions.

No fixture in this package alone proves Texture/PCM preprocessing, final 768-dimensional API, all roles, Player, device, lifecycle or release acceptance. Numeric failure, missing execution, interruption, timeout and GPU skips are separate outcomes.
