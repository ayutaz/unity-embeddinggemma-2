# Opt-in real-weight validation

This development-only package is separate from the runtime UPM package and its weights-free regression tests. Add it from the same immutable Git SHA with `?path=/validation/com.ayutaz.embeddinggemma.validation#SHA` and include both packages in `testables` only in a validation consumer. The runtime package never depends on this package.

`FrozenAudioProjectionTests` uses the existing audited `artifacts/k/artifacts/audio-real-chain` and `audio-native-feed-forward-replay` fixtures relative to the sibling consumer. Missing files or SHA mismatches fail before numeric comparisons. No weights, generated models or binaries are included. The tests reuse previously TDD-validated coefficient mapping and norm isolation unchanged.

Scope: fixed native inputs, saved FP32 weights, linear1 / linear2 / post norm only. This is not full-encoder, final API, image, all-signal or Player acceptance.
