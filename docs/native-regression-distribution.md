# Pinned native audio regression distribution

This acquisition route preserves the selected generation's original numerical
gate without depending on an Actions artifact retention period. The candidate
workflow uses it explicitly; a fresh complete six-case execution remains pending
at this implementation checkpoint.

The [data preview](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/native-regression-preview-e9de158)
contains two metadata-clean PT2 graphs, four original/candidate reference and
audit JSON files, six unchanged synthetic feature inputs, the inventory and
three license/attribution files. [All 16 remote sizes/digests, notes and the tag](results/native-regression-preview-publication-20261019.json)
were read back. This is separate from the formal SDK Release and the Sentis model
previews. Models remain outside Git and LFS.

The fixed [upstream model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md)
identifies Google DeepMind and Apache-2.0. Keep `LICENSE.txt`, `NOTICE.txt` and
`MODIFICATIONS.txt` with the downloaded data. The client fetches them for both
original-reference and candidate directories.

## Exact source and executed bytes

The [release contract](../tools/references/native-regression-release.json) pins
release/asset IDs, byte counts, SHA256 and publication source. Its exact SHA256 is
`947430cb584490077a900afed6d1fcc1415c08493ee7efea3e2727fa46fd6fb8`.
The [distribution inventory](../tools/references/native-regression-distribution.json)
has SHA256 `e4da37c53cc9b97d421b17caf56eacb6003fdca9160ac6e8f6a86a5314299d5e`.
Both files retain exact bytes across operating systems.

The candidate contract, selected-generation manifest and its seven proofs stay
unchanged. Original reference SHA256 remains
`ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9`;
original audit SHA256 remains
`68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1`.
Candidate reference/audit JSON still describe the preserved source generation.
They are not rewritten to label a distribution graph as original bytes.

| Report field | Meaning in the explicit distribution route |
| --- | --- |
| `graph_sha256`, `executed_graph_sha256` | Actual metadata-clean PT2 bytes verified before loading |
| `source_graph_sha256` | Preserved generation's raw graph identity |
| `graph_transport_contract_sha256` | Exact pinned release contract |
| `graph_distribution_inventory_sha256` | Exact reviewed distribution inventory |

Only optional node stack traces differ. Each graph's other 799 members and
computational JSON were compared against its preserved raw source. The two raw
source and distribution hashes are distinct. Every load rechecks file bytes,
graph metadata and the explicit source marker before Torch deserialization.
This does not recover the lost earlier native graphs or certify opaque tensor
buffers through text scanning.

## Explicit required-CI path

Prefer the required GitHub Actions workflow for model acquisition and execution.
It uses these separate entries:

1. `selected_audio_generation fetch-distribution` checks the reviewed selection,
   fetches exact original goldens and only the requested profile/input.
2. `frozen_audio_chained_onnx --candidate-distribution-directory` requires raw
   input mode and records actual executed graph identity in job, input and target
   provenance. It is mutually exclusive with `--candidate-directory`.
3. `selected_audio_generation fetch-original-distribution` obtains immutable
   goldens for the independent auditor.
4. `selected_audio_generation audit-distribution` requires all six distinct
   signals, both profiles, five connected parts, one current execution source,
   824,320 values and zero original-tolerance violations. Native full/cut bit
   checks, four-thread controls and the original numerical gates are retained.

The existing raw loader and raw-report auditor keep their strict source-byte
contract. They reject distribution graphs/reports; there is no automatic fallback
between routes. A failed fetch cannot overwrite evidence. Candidate and original
output directories must be disjoint. Downloads use anonymous HTTPS, exact size
and hash checks, bounded temporary files and checked redirects.

## Checkpoint and remaining work

The [implementation/TDD checkpoint](results/native-regression-durable-route-tdd-20261019.json)
records the intended failures, 284 related passing tests and actionlint. Tiny
actual Torch graphs cover the new input route; original captures independently
exercise the new audit contract. Anonymous public HTTP acquisition of the two
original JSON files and four metadata files passed exact-size/hash checks.
That precommit HTTP boundary check is not a fresh fixed-Git consumer or the
complete real-model regression.

The capture protocol still does not preserve intermediate ONNX graphs for the
independent auditor; `graphs_independently_rehashed=false` remains visible. No
CPU reference execution is counted as a Unity GPU pass. Fresh complete CI,
remaining performance/devices, provider cleanup and the formal SDK Release stay
separate gates. Before retiring an older hosted artifact, inventory every pinned
dependency and verify exact private backups of all required original bytes.
