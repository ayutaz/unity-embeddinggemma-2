# Model artifact metadata privacy

Actual fresh CI PT2 model JSON contained absolute build-environment paths in
optional node stack traces. The previous generic artifact guard explicitly left
model binary contents to separate producer validation; that boundary did not
prevent these metadata fields from being uploaded. Original metadata and source
evidence are retained privately. No detected path or identity is reproduced here.

The artifact guard now reads PT2 archive/member names, model JSON and other
textual members, and walks ONNX protobuf strings. ONNX string attributes are
checked without decoding numerical tensor buffers or loading external weights.
Unparseable models, missing parsers and unsafe metadata fail closed. Diagnostics
use fixed rule/path labels. The pinned ONNX parser is supplied to the actual
Actions guard command, rather than assumed present in an isolated Python runtime.

New PT2 exports remove optional node stack traces before their first
serialization. Graph operations, arguments, parameters, buffers and numerical
acceptance gates are retained. Tiny export tests check exact output/input values;
they are not real-model or Unity GPU acceptance. Retained files are inspected
read-only, not sanitized in place or relabeled under an original source hash.

The [TDD and actual-file checkpoint](results/model-artifact-metadata-privacy-20261019.json)
records intended failures, passing checks and the Action-equivalent boundary on
two original raw and two reviewed distribution PT2 files. Original raw files
must fail; reviewed distribution files must pass. Neither result certifies opaque
tensor data through text scanning or proves a model's numerical correctness.

The previous candidate CI was cancelled after the metadata finding and is not a
pass. [Repaired-source acceptance](results/native-distribution-integration-344f07a-20261019.json)
passed all fifteen jobs and independently audited all seven numerical archives
before PR #97 was integrated. Older
hosted artifacts and provider copies remain a separate cleanup task: inventory
all dependencies and verify complete immutable private backups before deletion.
