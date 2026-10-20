# 推論API

> 更新: 2026-10-20。Windows候補の全入力・全role・横断検索は、[実Editorと同一SDKの既存Player結果](results/windows-first-stages-1-to-3-20261020.json)で工程1〜3を受け入れました。性能・最低要件・新tag導入・正式公開は未完了です。[導入の入口](../Packages/com.ayutaz.embeddinggemma/Documentation~/index.md) / [検証状況](status.md) / [現在の実行計画](next-release-plan.md)。

## マルチモーダルAPIとGPU設定

mainへ統合済みの`PartitionedImageFeatureEncoder` / `ImageEmbedder.EmbedAsync`と`PartitionedAudioFeatureEncoder` / `AudioEmbedder.EmbedAsync`は、実Texture・PCMから前処理→encoder→backbone→正規化済み768次元をGPUComputeで実行する固定source検証済み経路です。画像・音声は各6ケースとQuery / Document / Rawの18条件、計算 / 入力CPU fallback・中間readback0。元encoderの画像全3 / 音声全2 profileの厳密数値gateと必須CIも合格しています。[元API検証](results/goal4-current-completion-audit-20261016.json)と[現在のサンプル検証](multimodal-search-sample.md)を区別してください。全端末・正式配布の合格は表しません。

統合済みの`IAsyncAudioPreprocessor` / `AudioPreprocessor.ProcessAsync`は、GPU fence完了までPCM入力・出力copyを保持し、完了した前処理のGPU cacheを解放します。`AudioEmbedder.EmbedAsync`は対応stageがあればこの経路を待ちます。反復処理には非同期APIを使用し、キャンセル・Dispose後も返されたtaskの完了前に借用stageを破棄しないでください。`GpuMediaSession.CloseAsync`はその完了と所有stageの解放を待ちます。

両APIはstageとprompt builderを借用し、呼び出しごとの中間結果を所有します。返す768次元CPU配列は独立コピーです。非同期encoderはGPU分割境界を待ち、音声では完了済みキャッシュを解放します。GPU処理のdispatchはメインスレッドで行い、`Task.Run`へ移しません。音声前処理は非同期fence待機に対応しますが、backbone呼び出しは同期処理です。呼び出しをawaitしてから借用stage / operationをDisposeしてください。CPU APIの選択肢と、ここで検証したGPU専用経路を区別します。

使用例・prefix / mask・キャンセルと所有権は[音声API](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md)と[音声encoder](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-feature-encoder.md)を参照してください。分割Model・係数・operation接続表は対象profileを監査したものを呼び出し側で準備します。自動ダウンロード・profile選択は行いません。単一Modelの同期encoder APIを、この実モデル受け入れ経路と同一視しません。

`GpuBlockedMatmul`の加算blockは係数・形状と独立参照に照合して指定します。局所patchのblock384成功を他の行列へ一般化せず、採用済み全encoderの計算順を保持します。モデル準備の再利用、長期メモリ / 速度、sample / Player、他端末は後続工程です。

## 既存テキストAPI

更新日: 2026-10-10。[現在の状態](status.md)でmain / PRと検証範囲を確認する。

推論APIは [PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3)、保存APIは [PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)、検索APIは [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10)、検索サンプルはPR #11で統合済み。文書の確認基準mainは `4ee0cbb`。M1実測の対応は当時のmain `8146107`を維持する。
**Windows Editorの実モデルCPU / GPUCompute全15ケースと再推論、保存・量子化を含むM1受け入れ検証が合格**。
API導入当時は単体契約24件、M1完了時は保存・測定を含む30件が合格。PR #6では29件をパッケージへ移し、測定契約1件を元プロジェクトに残して両方の合格を確認した。[M1完了検証](m1-completion-validation.md) / [UPM検証](m2-package-validation.md)を参照。

`EmbeddingGemma.Runtime` は UnityEditor を参照しない assembly。
Unity 6000.3.16f1 / Sentis 2.6.1 の、batch 1 / length 128 / fp32 / 768次元の export を対象とする。
モデル・tokenizerは [CI成果物を検証して配置](automation.md)する。
PR #6でRuntimeを `Packages/com.ayutaz.embeddinggemma/Runtime/` へ移行し、元のソース・assembly名・GUIDを維持した。
このUPM構成はPR #6でmainへ反映済み。実行依存はSentis 2.6.1とNewtonsoft JSON 3.2.2。検索APIはPR #10、検索シーンはPR #11で追加した。画像・音声APIもmainへ統合済みで、上のマルチモーダルAPI節を参照。
別プロジェクトへの導入は [パッケージ文書](../Packages/com.ayutaz.embeddinggemma/Documentation~/index.md)、結果は [UPM検証](m2-package-validation.md)を参照。

```csharp
using EmbeddingGemma;
using Unity.InferenceEngine;

// modelAsset は Sentis ModelAsset、tokenizerJson は同じ成果物の JSON 文字列。
var model = ModelLoader.Load(modelAsset);
using var embedder = new TextEmbedder(model, tokenizerJson, BackendType.GPUCompute);
float[] query = embedder.EmbedQuery("猫はどこで眠っていますか？");
float[] document = embedder.EmbedDocument("猫が窓辺で眠っています。", title: "猫の様子");
float[] raw = embedder.EmbedRaw("前置きなしの入力");
```

| API | 入力の規則 |
| --- | --- |
| `EmbedQuery(text)` | `task: search result \| query: ` を前置 |
| `EmbedDocument(text, title = null)` | `title: <title> \| text: ` を前置。nullタイトルは `none`、空タイトルは空のまま |
| `EmbedRaw(text)` | 前置なし。空白・改行・Unicode をそのまま渡す |
| `Embed(text, TextRole, title = null)` | 同じ処理をrole指定で呼ぶ。query / raw でtitle指定は例外 |
| `Dispose()` | Worker を解放。二重呼び出し可能。破棄後の推論は `ObjectDisposedException` |

空文字は有効、null の text は引数例外。長文の切り詰めと padding は固定条件を保存した tokenizer JSON に従う。
`TextPrompts.Format` で参照と同じ書式だけを取得できる。文字列の trim / Unicode 正規化はAPI側で追加しない。
Sentis 2.6.1が空文字を拒否するため、`TextTokenizer` が空シーケンスへ同じ特殊トークン / padding 設定を適用する。
固定モデルの設定用であり、任意のモデルのtokenizer互換性を保証するAPIではない。

Unity のメインスレッドで同期的に呼ぶ。インスタンスの同時呼び出しは扱わない。
Worker は再利用し、入力 Tensor は各呼び出し後に解放する。出力 Tensor は Worker の所有で、呼び出し元には独立した `float[768]` を返す。
GPU 指定時は最終ベクトルを CPU に取得する。projection・mean pooling・L2 正規化は export 内に含まれる。

CPU / GPUCompute を明示する。GPU 非対応をCPUに切り替えない。
モデルは int32 `[1,128]` の `input_ids` / `attention_mask` と出力1件を必要とする。
出力が float `[1,768]` でない、非有限値を含む、単位長でない場合は `InvalidOperationException`。
不適合出力をAPIで補正して合格扱いにはしない。モデルとtokenizerのrevision / hash監査は配置時に行う。

## 保存・再読み込み

[PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)でmainへ統合済み。契約5件と実モデルのfp32 / Float16重み・両backend全15件が合格。[検証記録](m1-completion-validation.md)を参照。

```csharp
TextModelFile.Save(model, "artifacts/m1-completion/fp32.sentis");
TextModelFile.Save(model, "artifacts/m1-completion/fp16.sentis", float16: true);
var restored = TextModelFile.Load("artifacts/m1-completion/fp16.sentis");
using var restoredEmbedder = new TextEmbedder(restored, tokenizerJson, BackendType.GPUCompute);
```

`.sentis`拡張子を要求し、親ディレクトリを作成。一時ファイルを保存・ロード確認後、既存先を置換する。失敗時は一時ファイルを削除する。
fp16保存はディスク経由でモデルをコピーしてから重み量子化し、入力Modelを変更しない。書き出しは大きいメモリとI/Oを必要とするため、毎推論時に行わない。
macOS / iOS / Android、UPMリリースは [M2計画](m2-plan.md)の対象。

## 文書を事前埋め込みして検索する

検索APIはPR #10でmainへ統合した。小さい文書群を全件比較する同期APIで、ベクトルDBは使用しない。

```csharp
var documents = new[] {
    new SearchDocument("cat-ja", "猫が窓辺で眠っています。", "猫の様子"),
    new SearchDocument("space-en", "Spacecraft observe planets and moons.", "Space exploration")
};
using var search = new TextSearchSession();
search.Prepare(documents, () => new TextEmbedder(model, tokenizerJson, BackendType.GPUCompute));
SearchHit[] hits = search.Search("猫はどこにいますか？", limit: 2);
foreach (var hit in hits)
    UnityEngine.Debug.Log($"{hit.Document.Id}: {hit.Score:F4}");
```

`TextSearchSession`はfactoryから受け取った`ITextEmbedder`を所有する。再準備の前、準備失敗時、Disposeで解放する。破棄後の操作は拒否する。直接`TextSearchIndex`を使う場合は呼び出し側が推論器を所有する。

文書は準備時に一度だけ埋め込み、ベクトルをコピーして保持する。検索はqueryの埋め込みだけを追加する。768次元の有限・正のnormを要求し、cosine降順、同点はIDのordinal順で並べる。文書IDは小文字ASCIIの`[a-z0-9][a-z0-9_-]*`、重複不可。空の文書・query、空corpus、1未満のlimitは拒否する。nullタイトルは既存document APIと同じ扱いを維持する。

[Text Searchサンプル](../Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md)と[モデル準備手順](model-preparation.md)に画面からの利用方法を示す。実モデルの参照順位・backend・実行環境は[検索検証記録](m2-search-validation.md)を参照する。
