# 導入とRuntime API

テキスト・画像・音声の埋め込みと検索をUnityへ組み込むWindows向け`0.1.0`の入口です。公開状況は[GitHub Releases](https://github.com/ayutaz/unity-embeddinggemma-2/releases)、source別の受け入れと他環境の未検証範囲は[検証状況](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)を参照してください。

## 1. パッケージを追加する

基準はUnity **6000.3.21f1** / Sentis **2.6.1**です。Package Managerの **Add package from git URL** へ、公開済み正式tag `v0.1.0`のURLを入力します。公開・導入の結果は[Release notes](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)を参照してください。

```text
https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/com.ayutaz.embeddinggemma#v0.1.0
```

clone済みなら **Add package from disk** でこのパッケージの`package.json`を指定することもできます。Package Managerの表示名は現在 **EmbeddingGemma** です。SentisとNewtonsoft JSONは依存から解決されます。URP・uloop・Pythonは利用側のRuntimeには不要です。

固定tagのURLと実際のresolved SHAを照合してください。固定sourceの導入、同一SDKの既存Player結果の適用、新tagでのEditor / Player実行は、それぞれsourceを付けた別の記録として扱います。

## 2. モデルを用意する

**UPMにモデルは含まれません。** [モデル準備](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/model-preparation.md)に従って取得・監査・変換してください。

- テキスト：`.sentis`と`tokenizer.json`。batch1 / sequence128、int32 `input_ids` / `attention_mask`、float `[1,768]`出力。
- 画像・音声：固定profileのモデル・係数・tokenizerと`bundle.json` / `objects/`。全6 moduleのFP32 / Float16重みは[公開モデルpreviewの取得手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/media-model-preparation.md)から明示取得できます。元checkpointからの再生成と、変換済みモデルの取得は別の経路です。
- モデルは`google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`へ固定しています。producer・graph・ファイルhashを確認し、異なるrevisionを混ぜないでください。

## 3. サンプルから試す

Package ManagerのSamplesからImportします。

| サンプル | 入口・条件 |
| --- | --- |
| [Text Search](../Samples~/TextSearch/README.md) | `TextSearch.unity`。準備済みmodel / tokenizerで固定6文書を検索 |
| [Image and Audio Search](../Samples~/MultimodalSearch/README.md) | `MultimodalSearch.unity`。準備済みbundle、対応する画像 / mono PCMのprofileが必要 |

## テキストAPI

次はUnityメインスレッドから呼ぶ例です。`modelAsset`は準備したSentis `ModelAsset`、`tokenizerJson`は同じrevisionのtokenizer JSON文字列です。

```csharp
using EmbeddingGemma;
using Unity.InferenceEngine;

var model = ModelLoader.Load(modelAsset);
using var embedder = new TextEmbedder(model, tokenizerJson, BackendType.GPUCompute);
float[] query = embedder.EmbedQuery("猫はどこで眠っていますか？");
float[] document = embedder.EmbedDocument("猫が窓辺で眠っています。", "猫の様子");
float[] raw = embedder.EmbedRaw("前置きなしの入力");
```

同期処理なので、毎フレームmodel / Workerを作らず再利用します。同時呼び出しは扱いません。戻り値は独立したCPU `float[768]`です。`TextSearchIndex`は文書を先に埋め込み、検索時にqueryだけを推論します。借用したembedderは呼び出し側で解放します。`TextSearchSession`は内部embedderを所有・解放します。

空文字の埋め込みは有効ですが、検索sampleは空queryを拒否します。null textや非有限・非単位長の結果は拒否し、暗黙に補正しません。

## 画像・音声API

`ImageEmbedder.EmbedAsync`と`AudioEmbedder.EmbedAsync`は、対応profile・監査済みstageを使い、前処理→encoder→共通backboneをGPUで実行します。画像はTexture / RenderTexture、音声は`AudioPcmInput`（16kHz mono）を受け取ります。最終768次元のみCPUへ取得します。

高水準の利用例は[Image and Audio Search](../Samples~/MultimodalSearch/README.md)、stageの組み立て・prompt・前処理の契約は[低レベルAPI](multimodal-api.md)、[音声embedder](audio-embedder.md)と[音声encoder](audio-feature-encoder.md)を参照してください。[API全体の補足](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/runtime-api.md)もあります。

- 非同期呼び出しもUnityメインスレッドから行い、`Task.Run`へ移しません。
- 借用Texture・model・stage・中間tensorは、そのGPU処理とtaskの完了まで生存させます。
- `AudioPreprocessor.ProcessAsync`はGPU fence後に完了したcacheを解放します。反復処理には`AudioEmbedder.EmbedAsync`を使用してください。
- `GpuMediaSession`は選択moduleを所有し、切替・`CloseAsync`で処理完了と解放を待ちます。最終vectorを混在索引へ保持できます。
- GPU非対応時にCPUへ自動切り替えしません。固定profileを外れる長さ・shape・形式は拒否します。

## モデルの保存

```csharp
TextModelFile.Save(model, writablePath); // .sentis
TextModelFile.Save(model, writableFloat16Path, float16: true);
var restored = TextModelFile.Load(writableFloat16Path);
```

保存・量子化は大きいメモリとI/Oを使う準備処理です。毎推論時に実行せず、書き込み可能な場所を指定してください。Float16重み保存は元modelを変更せず、演算全体のFP16化・高速化を保証しません。保存APIの過去Windows実測を、他OS・全Playerへ一般化しません。

## テスト・問題の切り分け

Runtime利用だけならTest Frameworkは不要です。package契約テストを実行するときは、consumerに公式 `com.unity.test-framework` **1.6.0**と`"testables": ["com.ayutaz.embeddinggemma"]`を設定します。対象assemblyは`EmbeddingGemma.Package.Editor.Tests`です。

Unity6000.3.21f1のテスト環境構成は[公式frameworkの導入記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/m2-embedded-framework-validation.md)を参照してください。vendorのPackageCacheを手で書き換えることを導入手順にしません。

モデル欠落・hash不一致・対応外GPU・入力profile不一致は、それぞれエラーとして確認します。小さい契約テストやPython CIの成功は実モデルGPUの合格と分けます。[現状](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md) / [ロードマップ](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/roadmap.md)。
