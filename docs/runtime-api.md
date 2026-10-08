# テキスト推論 API

実装ブランチ: `feat/m1-sentis-runtime`、[PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3) は未マージ。
**Windows Editor の実モデル CPU / GPUCompute 全15ケースと再推論、単体契約24件が合格**。[実行記録](m1-runtime-validation.md) を参照。

`EmbeddingGemma.Runtime` は UnityEditor を参照しない assembly。
Unity 6000.3.16f1 / Sentis 2.6.1 の、batch 1 / length 128 / fp32 / 768次元の export を対象とする。
モデル・tokenizer は [CI成果物を検証して配置](automation.md) する。配布用 UPM 整理は M2 で行う。

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

`.sentis` 保存・量子化・性能測定、macOS / iOS / Android、UPM リリースは未検証で次段階。
