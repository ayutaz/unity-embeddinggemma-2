# 音声専用の埋め込みAPI

更新: 2026-10-21。Windows版[v0.1.0は正式公開済み](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)です。公開tagの空consumerで、Unity6000.3.21f1 / Sentis2.6.1、FP32 / Float16、全PCM profile・roleと混在検索・回復・解放を実Editor / Windows IL2CPP Playerで確認しました。[正式版の確認記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/results/windows-v010-release-20261021.json)を参照してください。

16f1での元API検証・18条件の履歴は[従来の確認記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/results/goal4-current-completion-audit-20261016.json)に保持しています。以下の古いsourceの数値は、新tagの実行へ読み替えません。

実モデルの検証済み経路は`PartitionedAudioFeatureEncoder`と`AudioEmbedder.EmbedAsync`です。`AudioEmbedder`は`IAsyncAudioEmbedder`を実装します。同期`Embed`もありますが、連続同期実行ではGPUキャッシュ保持により12 GiBのホストメモリ上限を超えた失敗があり、同じ受け入れ結果として扱えません。

```csharp
// Unityメインスレッドから呼ぶasyncメソッド内。
// 各Model、分割接続表、operation、tokenizerは同revision / 固定profileで監査済み。
var pcm = new AudioPcmInput(samples, samplingRate: 16000, channels: 1);
var encoderProfile = new AudioEncoderProfile(pcm.ValidFrameCount);
// backboneProfileはImageFeatureCount=0、AudioFeatureCount=encoderProfile.FeatureCount。
using var preprocessor = new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute);
using var encoder = new PartitionedAudioFeatureEncoder(
    segments, operationInputs, operationOutputs, operations, encoderProfile);
using var backbone = new MultimodalBackbone(backboneModel, backboneProfile, BackendType.GPUCompute);
var prompts = new MultimodalPromptBuilder(tokenizerJson, tokenizerConfigurationJson, backboneProfile);
using var audio = new AudioEmbedder(preprocessor, encoder, backbone, prompts);
float[] vector = await audio.EmbedAsync(
    pcm, TextRole.Document, cancellationToken: cancellationToken);
// operationsは借用物。await完了後、上の所有stageと別に呼び出し側でDisposeする。
```

入力は16 kHz mono、有限な振幅[-1,1]の161〜179200 samplesです。検証済みencoder profileは有効1119 / 839 frames、280 / 210 soft tokensです。モデルの取得・変換・profile選択は呼び出し側の責任です。再利用時は同じstageとAPIを保持し、前の呼び出しをawaitしてから次を開始します。

`EmbedAsync`は前処理とbackboneをメインスレッドで同期実行し、encoderの分割境界でGPU完了を待ち、完了済みキャッシュを解放します。全処理がバックグラウンドになるわけではありません。`Task.Run`へ移さないでください。中間feature / mask / soft tokenはGPU上で渡し、最終768次元だけをCPUへ取得します。GPU非対応端末でCPU推論へフォールバックしません。

backend、encoder feature数、backbone / prompt profileは構築時に照合します。PCM有効frame数と公式promptはGPU処理前に検証します。maskはPCMの不変metadataから組み立てます。無効role・予約tokenを含むtitle・異なるprefixは拒否し、返却vectorの有限値・次元・単位長を検査します。不正なvectorを再正規化して合格にする処理はありません。

APIはstageとprompt builderを借用し、呼び出しごとの前処理結果・encoder結果を所有します。成功・例外・キャンセル時に中間結果を解放し、返す配列は独立コピーです。重複呼び出しは拒否します。キャンセルはGPU分割境界で反映され、実行中のGPU演算を即時中断する保証はありません。`AudioEmbedder.Dispose()`は待機中の呼び出しをキャンセルしますが、借用stageを破棄しません。完了または例外までawaitしてからstageとoperationを解放してください。

[実PCM全18条件](../../../docs/results/m4-real-pcm-all-roles-staged-gpu-20261016.json)ではCPU推論fallback・中間readback / flushが0、最大ホストピークは約7.13 GiBでした。これはVRAMや長期安定性の測定ではありません。実GPU encoder後のbackbone例外からの再利用と、API破棄後の借用stage再利用も確認済みです。[元encoder全6信号の厳密比較](../../../docs/results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json)は824,320値で元`atol=1e-5 / rtol=1e-4`を維持しています。

最終vectorは元cosine基準で18条件合格していますが、追加の要素別診断には差が残ります。Query / Raw参照の固定encoder境界による生成、旧同期実行の失敗、sourceごとの検証範囲は固定証拠に保持しています。分割モデル準備の配布、検索 / 再生 / 収録sample、連続性能、Player・他OS / 実機は後続工程です。
