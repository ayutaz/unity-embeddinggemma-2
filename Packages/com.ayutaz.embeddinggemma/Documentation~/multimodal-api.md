# マルチモーダルの低レベルAPI

更新: 2026-10-17。独自のstageを接続する場合の入力・所有権・前処理の契約です。まず[導入・API](index.md)を確認してください。通常の利用は[Image and Audio Search](../Samples~/MultimodalSearch/README.md)と[音声embedder](audio-embedder.md)から始めます。実行sourceと未検証環境は[検証状況](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)を参照してください。

## 共通本体API

画像・音声の最終APIはmainへ統合済みです。各前処理の出力はencoder入力で、最終768次元embeddingとは異なります。

`MultimodalBackboneProfile`と`MultimodalBackbone`は、encoder / projectorが生成した512幅のsoft-token Tensorを共通本体へ渡すAPIです。画像Texture・音声PCMからの前処理やprompt生成を含みません。Windowsの固定sourceで実画像・PCMから最終embeddingまでのGPU検証が合格しています。Windows版v0.1.0では画像・音声のWindows Playerも固定profileで受け入れ済みです。zero-feature profileの個別受け入れと他OSは別の検証対象です。

profileの長さ・placeholder ID・feature数・vocabulary / pad IDは、監査済みモデルと公式processorの参照から設定します。対応する保存graphは4入力の名前・型・static shapeが一致する必要があります。

```csharp
using var backbone = new MultimodalBackbone(model, profile, BackendType.GPUCompute);
float[] embedding = backbone.Embed(inputIds, attentionMask, imageFeatures, audioFeatures);
```

`model`・feature Tensorは呼び出し側が所有します。GPUComputeでは非空featureを既にGPUへ置いてください。backboneはfeatureをCPUへdownload・Disposeせず、最終768次元だけを独立したCPU配列へコピーします。featureの全値に対するCPU検査は行わないため、入力側でもfinite条件を保証してください。長さ / mask / slot数・feature shape・dataの生存・backendの不一致は拒否します。backboneをDisposeした後も、借用Tensorの解放は呼び出し側が行います。

### 公式promptの構築

`MultimodalPromptBuilder`は固定revisionのtokenizer JSON・tokenizer設定とbackbone profileを使い、単一画像 / 単一音声または両方のpromptを作ります。画像feature数は実encoder profileの値（square 256、landscape / portrait 252等）を渡します。音声は前処理のbinary feature maskを2段stride-2で縮約し、有効なsoft token数を求めます。開始 / 終了marker、token ID、feature数は設定・profileに一致する必要があります。

```csharp
var prompts = new MultimodalPromptBuilder(tokenizerJson, tokenizerConfigurationJson, profile);
var prompt = prompts.Build("<|image|>", imageFeatureCount: 256, role: TextRole.Document);
// 対応するencoder / backbone graphとGPU Tensorを準備してから渡します。
float[] embedding = backbone.Embed(prompt.InputIds, prompt.AttentionMask, imageFeatures, audioFeatures);
```

これは低レベルAPIの接続例です。実encoderから最終embeddingまでの固定source検証と、この例に任意のgraphを渡した場合の互換性は別です。音声なら`Build("<|audio|>", audioFeatureMask: mask)`、両方ならtemplate内に各placeholderを1つずつ置きます。query / document prefixとtitleは`TextPrompts`の契約を維持します。入力templateに開始 / 終了markerを事前展開しないでください。結果のIDs / maskは独立した配列copyです。

text用tokenizerの既存truncation / paddingを解除し、完全なtoken化の後にprofile長へ右paddingします。上限超過は例外となり、切り捨てません。固定profileごとにbuilderを作って再利用してください。promptはCPU側の入力境界で、GPU featureの読み出しやencoder実行は行いません。zero-featureのbackbone graphや全end-to-endの検証は別です。

## 画像前処理Runtime

```csharp
using var preprocessor = new ImagePreprocessor();
using var prepared = preprocessor.Process(texture);
// prepared.Pixels: GPUCompute float [1,2520,768]、画像encoderへの借用入力。
// prepared.Positions: row-majorの(x,y)、末尾[-1,-1]のreadonly metadata。
// prepared.ResizedWidth / ResizedHeight / SoftTokenCount: 公式profileの形状。
```

batch 1、8bit RGB / RGBAのTexture2D・作成済みsingle-sample 2D RenderTextureに対応します。各辺1〜8192px、sRGB / linearの保存値をRGB入力として扱い、alphaを無視します。浮動小数・HDR・圧縮formatは明示拒否します。公式bicubic antialiasのresize・丸め、16px patch・最大2520 patchesへのzero paddingをGPUで実行し、中間pixelをCPUへdownloadしません。

入力Textureはcallerが所有します。結果は出力・作業Tensor / 補間bufferを所有し、processorの再利用・Dispose後も独立して生存します。次段GPU処理が完了するまで結果を保持し、その後Disposeしてください。Unity main threadのみ。公開版v0.1.0ではTexture / RenderTextureの固定条件を両重み形式の実Editor / Windows Playerで確認しています。EditorとPlayerでは確認したshapeの範囲が異なります。具体的な範囲・性能・対応機器と他環境の未検証項目は[検証状況](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)を参照してください。

## PCM入力境界

`AudioPcmInput`は固定16kHz mono PCMの入力準備です。対応長は161〜179200 samples、振幅はfiniteな[-1,1]。対応外入力はArgumentException（nullはArgumentNullException）で拒否します。

```csharp
var input = new AudioPcmInput(pcm, samplingRate: 16000, channels: 1);
// Waveform / SampleMask: owned dataを参照するReadOnlySpan、各179200要素。
// SourceSampleCount / ValidFrameCount: 実入力長と公式semicausal maskの有効frame数。
```

callerのPCM配列から独立したsnapshotを作り、右側zero paddingとprefix maskを保持します。caller配列や公開コピーを変更しても準備済み入力は変わりません。これはCPU入力境界のAPIであり、前処理Workerは次のAPIへ渡します。音声encoder・最終embedding推論は別のAPIで行います。

## PCM前処理Runtime

`AudioPreprocessor`は監査済みの固定前処理Modelを借用し、Workerと入力upload Tensorを所有します。対応graphはfloat `waveform` / int `sample_mask`の[1,179200]入力と、順序付き2出力（float特徴量[1,1119,128]、int frame mask[1,1119]）です。backendはCPUまたはGPUComputeを明示し、対応外backendや固定shape / type不一致は拒否します。

```csharp
using var preprocessor = new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute);
// Unityメインスレッドから呼ぶasyncメソッド内。
using var prepared = await preprocessor.ProcessAsync(new AudioPcmInput(pcm));
// prepared.Features / prepared.FrameMaskを後続Workerへ直接渡します。
// prepared.ValidFrameCountは入力prefixから決めた有効frame数です。
```

`AudioPreprocessingResult`は両出力Tensorを所有します。GPUではdevice copyで保持し、中間downloadを行いません。Workerの再実行 / Dispose後も結果は独立して生存し、結果をDisposeすると両Tensorを解放します。後続推論への貸し出しが終わるまで結果を保持してください。同期 / 非同期APIともUnity main threadで呼びます。非同期前処理はGPU fence完了後にcacheを解放します。返されたtaskが完了するまで借用stageを破棄しないでください。最終音声embeddingはAudioEmbedder.EmbedAsyncで実行します。
