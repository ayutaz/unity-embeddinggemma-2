# 導入とテキストAPI

更新日: 2026-10-09。開発版 `0.1.0-pre.1` は [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6)にあり、未マージ・未リリースです。
ローカルフォルダ依存で別Unityプロジェクトへの導入・compile・契約29件成功を確認しました。Git URLでのEditor導入はまだ実行していません。

## 導入

Unity 6000.3.16f1で、Package Managerの「Add package from disk」からこのパッケージの `package.json` を指定します。
Git URLの指定形式は [Unity公式手順](https://docs.unity3d.com/6000.3/Documentation/Manual/upm-git.html)に沿っています。
`<commit-sha>` はパッケージを含む40桁commitに置き換えます。検証済みソースの例は `8a585bc634a210e5ffd9ceaa5d67b85f1c9316a3` です。
このcommitの指定と、Git URL導入をEditorで実行済みであることは区別します。
公開tagはまだありません。

```text
https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/com.ayutaz.embeddinggemma#<commit-sha>
```

SentisとNewtonsoft JSONはpackage manifestから解決します。URP・uloopは利用側に不要です。
Editor契約テストを実行する場合は、利用側のmanifestへ `com.unity.test-framework` 1.6.0と
`"testables": ["com.ayutaz.embeddinggemma"]` を追加し、`EmbeddingGemma.Package.Editor.Tests` を実行します。
テストframeworkは配布Runtimeの必須依存ではありません。

## モデル準備

モデルファイルは同梱しません。[リポジトリのモデル変換・監査手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/automation.md)で、
同一revisionから生成したモデルと `tokenizer.json` を準備してください。
対象は `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、
batch 1 / sequence 128、int32の `input_ids` / `attention_mask`、L2正規化済みfloat `[1,768]` 出力です。
生成artifactのsource・hashを監査し、モデルとtokenizerを別revisionで混在させないでください。

## 推論

```csharp
using EmbeddingGemma;
using Unity.InferenceEngine;

var model = ModelLoader.Load(modelAsset); // Sentis ModelAsset
using var embedder = new TextEmbedder(model, tokenizerJson, BackendType.GPUCompute);
float[] query = embedder.EmbedQuery("猫はどこで眠っていますか？");
float[] document = embedder.EmbedDocument("猫が窓辺で眠っています。", "猫の様子");
float[] raw = embedder.EmbedRaw("前置きなしの入力");
```

メインスレッドで同期的に使用します。同時呼び出しは扱いません。Workerを再利用し、Disposeで解放します。
戻り値は独立した `float[768]`。GPU指定でも最終ベクトルはCPUへ取得します。
GPU非対応時のCPUへの自動切り替えはありません。
空文字は有効、null textは拒否。documentのnullタイトルだけ `none` とし、空タイトル・空白・Unicodeは保持します。
不適合なモデル入出力、非有限・非単位長の結果は例外とし、APIで補正しません。

## 保存

```csharp
TextModelFile.Save(model, writablePath); // .sentis拡張子
TextModelFile.Save(model, writableFloat16Path, float16: true);
var restored = TextModelFile.Load(writableFloat16Path);
```

Float16重み保存は元のModelを変えません。保存と量子化は大きいメモリ・I/Oを使用するため、毎推論時に呼びません。
Windows Editorではfp32 / Float16重みの保存・ロードと両backend各15件を検証済みです。UPM移行後の全保存・量子化ベンチマークは再実行していません。
読み取り専用の配置先へ保存しないでください。Player・macOS / iOS / Androidの保存API互換性は後続検証の対象です。
Float16重みは全演算のfp16化や速度改善を保証しません。
