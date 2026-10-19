# Original image reference data transport

The CPU comparison needs the same immutable encoder baseline as the existing
GPU fixture, plus the original Document final baseline. Regenerating JSON with
equal current outputs would change the source identity. This route transports
the original files without relying on an Actions retention period.

The [data preview](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/image-reference-preview-6644eea)
was published from `19fa109ad171b50a27530b299c961578a9792e69`.
[Remote readback and anonymous acquisition](results/image-reference-preview-publication-20261020.json)
verified all six assets, exact sizes/digests, notes and the target commit. Every
downloaded byte equals both reviewed private copies. The strict loader read
the six original cases. This is data availability, not new model execution or
the formal SDK Release.

## Exact data and attribution

The [distribution contract](../tools/references/image-reference-distribution.json)
has SHA256 `4d9acf6cf78d36e7667fb291829c12c4fa84140b82e443c58cd210cfa3143b37`.
It names six flat files: the original vision reference and audit, original
Document final reference, `LICENSE.txt`, `NOTICE.txt` and `MODIFICATIONS.txt`.
The original three JSON files total 24,085,651 bytes; all six files total
24,098,590 bytes. Each file has verified private complete copies.

The encoder baseline remains `3dfccd1a…`, audit `ab1aa775…`, Document final
`90692ad3…`. The Document producer's different encoder generation `7c183c52…`
is preserved as provenance, not substituted for the GPU baseline. No model,
graph, PNG, input buffer or recording is distributed by this route. The JSON
contains source-bound synthetic numerical outputs; this is not a general
privacy certification for numerical data.

## Publication enforcement

The ordinary publication hook retains its 16 MiB cap. The original encoder JSON
is larger, so only the exact three files registered in the
[committed review](../tools/references/image-reference-release-review.json)
use `image_reference_release_privacy`. Its review SHA256 is
`6644eeabc7ac9721f241d381139148b412e73f4c145601560559d98cfec21864`.

The guard reads review authority from Git HEAD, checks that exact review hash,
then verifies the entire proposed file's size/hash, complete text privacy and
producer/revision metadata. It does not use a working-tree or staged approval,
truncate a file or skip its numerical/metadata sections during text inspection.
Changed or unknown assets fail closed. The 32 MiB guard bound does not extend to
unreviewed JSON, arbitrary binary data, images or release notes. Notes and the
three attribution files retain ordinary text checks.

## Consumer command

Run from `tools/` using the source that includes this acquisition module:

```sh
uv run --locked python -m embeddinggemma_tools.image_reference_distribution --output ../artifacts/original-image-references
```

Use an empty owned directory. Downloads use anonymous HTTPS, checked redirects,
exact sizes and SHA256, bounded temporary files and exclusive publication. A
failure cannot overwrite existing evidence and does not create a success receipt.
All downloaded files undergo full text privacy checks, then the strict cohort
loader checks the original metadata and all six cases. Keep attribution files
with the data. The receipt uses relative filenames and explicitly grants no
CPU, GPU, performance or formal release qualification.

The next step is the complete measured preprocessing → actual encoder → actual
backbone CPU chain, followed by independent numerical acceptance under the
unchanged references/tolerances. Historical local AVX2 and cross-generation
failures remain separate unresolved evidence.
