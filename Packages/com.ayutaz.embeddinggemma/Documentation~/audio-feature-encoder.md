# 音声encoderへのTensor接続

> 2026-10-14: このencoder APIはDraft実装です。合成契約合格と実encoderの数値合格は別で、最新実候補は不合格。運用経路はGPUComputeを明示し、前処理・backboneとbackendを揃えます。

`AudioFeatureEncoder`は`AudioPreprocessor`の結果を受け取り、投影済みsoft tokenを返す低レベルAPIです。メインスレッドで同期実行します。最終768次元の埋め込みAPIではありません。

CPU / GPUComputeを明示し、前処理とencoderを同じbackendで構成します。GPUではfeature / maskをCPUへダウンロードせず、既存のfeature Tensorをそのままencoderへ渡します。PCMのuploadは前処理の入力境界です。

```csharp
// 両Modelは呼び出し側で取得・監査・ロードしたもの。
// encoderModelはpcm.ValidFrameCountと同じprefixでexportしたモデルを選びます。
var pcm = new AudioPcmInput(samples, samplingRate: 16000, channels: 1);
var profile = new AudioEncoderProfile(pcm.ValidFrameCount);
using var preprocessor = new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute);
using var encoder = new AudioFeatureEncoder(encoderModel, BackendType.GPUCompute, profile);
using var processed = preprocessor.Process(pcm);
using var encoded = encoder.Encode(processed);
Tensor<float> borrowedSoftTokens = encoded.SoftTokens;
// encodedを保持したまま、共通backboneへborrowedSoftTokensを渡します。
```

入力はfloat `[1,1119,128]`の`input`一つ、出力はfloat `[ceil(validFrames/4),512]`です。有効frame数は1〜1119。既存export候補のfull / partial profileは1119 / 839 frames、280 / 210 soft tokenです。APIはモデルを生成・自動選択しません。shape確認に加え、モデル内の固定maskとprofileの出自照合は取得・変換段階の監査で行います。

`Encode`は有効frame数を厳密に照合します。2 framesと3 framesはどちらも1 soft tokenですが、別の固定maskとして扱い、異なるprefixを拒否します。CPU前処理の結果をGPU encoderへ渡す暗黙uploadも行いません。

Modelは借用、Workerはencoderが所有します。前処理結果は呼び出し側が所有し、encoderは破棄しません。`AudioEncodingResult`は出力のdevice copyを所有するため、worker再利用・解放後も保持できます。結果のDispose後は`SoftTokens`を使用できません。中間Tensorは最終処理が終わってから解放してください。

今回の小さいCPU / 実GPUテストはprofile、入力拒否、所有権、GPU受け渡しの確認です。実音声encoderの固定数値条件、最終埋め込み・検索sample・Player・他OSの受け入れは未完了です。

前処理・encoder・共通backboneを合成する開発中のAPIは[AudioEmbedder](audio-embedder.md)を参照してください。最終768次元APIの小さい合成検証と、実モデル数値の受け入れを区別します。
