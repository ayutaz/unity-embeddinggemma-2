# テキスト推論 API

更新日: 2026-10-09。[現在の状態](status.md)でmain / PRと検証範囲を確認する。

推論APIは [PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3)、保存APIは [PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)で統合済み。現在の基準mainは `8146107`。
**Windows Editorの実モデルCPU / GPUCompute全15ケースと再推論、保存・量子化を含むM1受け入れ検証が合格**。
API導入当時は単体契約24件、M1完了時は保存・測定を含む30件が合格。PR #6では29件をパッケージへ移し、測定契約1件を元プロジェクトに残して両方の合格を確認した。[M1完了検証](m1-completion-validation.md) / [UPM検証](m2-package-validation.md)を参照。

`EmbeddingGemma.Runtime` は UnityEditor を参照しない assembly。
Unity 6000.3.16f1 / Sentis 2.6.1 の、batch 1 / length 128 / fp32 / 768次元の export を対象とする。
モデル・tokenizerは [CI成果物を検証して配置](automation.md)する。
PR #6でRuntimeを `Packages/com.ayutaz.embeddinggemma/Runtime/` へ移行し、元のソース・assembly名・GUIDを維持した。
このUPM構成はmainへの反映待ち。実行依存はSentis 2.6.1とNewtonsoft JSON 3.2.2で、画像・音声APIや検索シーンはまだない。
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
