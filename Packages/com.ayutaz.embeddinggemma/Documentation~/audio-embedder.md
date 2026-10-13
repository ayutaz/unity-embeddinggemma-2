# 音声専用の埋め込みAPI

`AudioEmbedder` / `IAudioEmbedder`は、固定profileのPCM前処理 → 音声encoder / projector → 共通backboneをメインスレッドで同期実行し、独立した768次元のCPU配列を返します。3つの実行stageと公式prompt builderは借用します。stageのModelとWorkerは呼び出し側で管理します。

```csharp
var pcm = new AudioPcmInput(samples, samplingRate: 16000, channels: 1);
var encoderProfile = new AudioEncoderProfile(pcm.ValidFrameCount);
// backboneProfileはimage count=0、audio count=encoderProfile.FeatureCount。
// 取得・監査済みの各Modelとtokenizer設定は、同じ固定profile / revisionに合わせます。
using var preprocessor = new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute);
using var encoder = new AudioFeatureEncoder(encoderModel, BackendType.GPUCompute, encoderProfile);
using var backbone = new MultimodalBackbone(backboneModel, backboneProfile, BackendType.GPUCompute);
var prompts = new MultimodalPromptBuilder(tokenizerJson, tokenizerConfigurationJson, backboneProfile);
using var audio = new AudioEmbedder(preprocessor, encoder, backbone, prompts);
float[] vector = audio.Embed(pcm, TextRole.Document, title: "fixed audio fixture");
```

backend、encoder feature数、backbone / promptの各profile値を構築時に照合します。実行時はPCMの有効frame数をencoderへ厳密に合わせ、GPU処理前にpromptを検証します。`BuildAudio`はtokenizer設定のaudio markerとPCMのprefix metadataからmaskを作るため、GPU maskをCPUへダウンロードしません。Query / Document / Rawとtitleの規則は既存の公式text prefixを使います。

中間feature / soft tokenはGPU上で渡します。画像feature数0のTensorを共通backboneへ供給し、最終vectorだけをCPUで受け取ります。前処理とencoderの結果は呼び出しごとの所有物として成功・例外の両方で解放します。`AudioEmbedder.Dispose()`は借用stageを破棄しません。stageはAPI利用後に呼び出し側が解放してください。

返却vectorは有限値・768次元・単位長を確認し、独立した配列へコピーします。不正なvectorを再正規化して受け入れる処理は行いません。対応外prefixの切り捨て、暗黙のCPU fallback、モデルの自動取得は行いません。

今回の受け入れは小さいgraphによるAPI合成・公式marker / mask・所有権・CPU / 実GPUの検証です。実音声encoderの固定数値gate、独立した最終音声Python参照との全chain一致、検索 / 再生 / 収録sample、Player・他OS / 実機は未完了です。これらを満たしてから正式リリースの受け入れへ進みます。
