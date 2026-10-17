# 音声encoderへのTensor接続

2026-10-17：`PartitionedAudioFeatureEncoder`はmain統合済みです。元6信号・両profileの実GPU比較は全824,320値で元`atol=1e-5 / rtol=1e-4`の超過0、計算 / 入力CPU fallback0。Windows Editor / Unity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定sourceでの結果です。後続の必須CIも合格しています。正式リリースや他端末の合格は意味しません。[検証状況と固定証拠](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)を参照してください。

`PartitionedAudioFeatureEncoder`は`IAsyncAudioFeatureEncoder`を実装するGPUCompute専用の低レベルAPIです。固定profileと係数を監査した分割Model、operation接続表、`IGpuTensorOperation[]`を受け取ります。モデル準備・自動選択・取得は行いません。最終768次元APIは[AudioEmbedder](audio-embedder.md)です。

```csharp
// Unityメインスレッドから呼ぶasyncメソッド内。全引数は同じ固定profileで監査済み。
var pcm = new AudioPcmInput(samples, samplingRate: 16000, channels: 1);
var profile = new AudioEncoderProfile(pcm.ValidFrameCount);
using var preprocessor = new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute);
using var encoder = new PartitionedAudioFeatureEncoder(
    segments, operationInputs, operationOutputs, operations, profile);
using var processed = preprocessor.Process(pcm);
using var encoded = await encoder.EncodeAsync(processed, cancellationToken);
Tensor<float> borrowedSoftTokens = encoded.SoftTokens;
// encodedを保持したままbackboneへ渡す。operationsは借用物なので別途Disposeする。
```

入力はGPU上のfloat `[1,1119,128]` featureとint32 `[1,1119]` maskを所有する`AudioPreprocessingResult`です。有効prefixはprofileと厳密に一致させます。出力はGPU上のfloat `[ceil(validFrames/4),512]`。検証済み固定profileは1119 / 839 frames、280 / 210 soft tokensです。同じsoft token数でも有効frame数が異なる入力は拒否します。CPU前処理からの暗黙uploadや中間readbackは行いません。

Model、operation、入力結果は借用します。encoderは分割Workerを所有し、`AudioEncodingResult`は独立した出力を所有します。`EncodeAsync`はGPU境界を待ちながら完了済みキャッシュを解放します。キャンセル / 例外時も、await終了まで入力と借用operationを保持してください。結果のDispose後に`SoftTokens`を参照できません。encoderのDisposeと、呼び出し側によるoperationのDisposeは別です。

単一Modelを受け取る同期`AudioFeatureEncoder`も互換APIとして残っていますが、上記の分割実モデルの受け入れ経路ではありません。`AudioEmbedder.EmbedAsync`に非同期encoderを渡さない場合、前処理を始める前に`NotSupportedException`となり、同期 / CPUへの自動切替はしません。

[元参照との厳密比較・テキスト回帰](../../../docs/results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json)と[実PCM最終API全role](../../../docs/results/m4-real-pcm-all-roles-staged-gpu-20261016.json)を参照してください。分割Model配布、長期メモリ・速度、Player・他端末は後続工程です。
