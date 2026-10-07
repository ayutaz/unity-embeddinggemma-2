# EmbeddingGemma 2 の Unity 対応に関する新規性調査

- 調査日: 2026-10-07
- 対象: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)

## 結論

現時点(2026-10-07)では新規性がある。ただし、テキストの埋め込みだけを Unity で動かす部分は、近いうちに既存のライブラリで実現される可能性が高い。

新規性を主張しやすいのは次の 2 点。

1. Unity 純正の Inference Engine(旧 Sentis)だけで動かすこと
2. 画像・音声を含むマルチモーダルの埋め込みを Unity で扱うこと

## モデルの概要

| 項目 | 内容 |
| --- | --- |
| 公開日 | 2026-10-06 |
| パラメータ数 | 740M(テキストだけなら 270M 分の読み込みで動く) |
| モダリティ | テキスト・画像・音声・動画を同じ 768 次元の空間に埋め込む |
| 次元の短縮 | Matryoshka 表現により 128〜768 次元に短縮できる |
| ライセンス | Apache 2.0 |
| 公式に挙げられた実行環境 | LiteRT、MediaPipe、llama.cpp(GGUF)、transformers.js、MLX、Ollama など |

Google の発表では、Unity・C#・.NET への言及はない。

> 補足: 最初に開いたモデルカードの要約に「2025年初頭リリース」とあったが、公式ブログと報道はすべて 2026-10-06 としているため、誤りとして扱った。

## Unity での既存事例

### 直接の事例

- GitHub で「embeddinggemma」と「unity」または「sentis」を組み合わせて検索した結果、該当リポジトリは 0 件だった。初代 EmbeddingGemma(300M)も同じだった。
- Hugging Face の派生モデルには GGUF・MLX・ONNX・LiteRT・CoreML 版があるが、Unity や Sentis 向けの版はない。

### 近い先行事例

| 事例 | 内容 | 本件との関係 |
| --- | --- | --- |
| [unity/inference-engine-minilm-v6](https://huggingface.co/unity/inference-engine-minilm-v6) | Unity 公式。Inference Engine でテキストの埋め込み(MiniLM v6)を動かすサンプル | 「Unity で埋め込みを動かすこと」自体は新しくない |
| [LLMUnity](https://github.com/undreamai/LLMUnity) | llama.cpp のラッパー。GGUF の埋め込みモデルで RAG ができる | `ggml-org/embeddinggemma-2-GGUF` がすでにあるため、llama.cpp が新しいモデル構造に対応すれば、テキストだけの対応はほぼそのまま動く可能性が高い。最新リリースは v3.0.3(2026年3月)。対応済みかどうかは未確認 |
| EmbeddingGemma.NET | 初代 EmbeddingGemma を ONNX Runtime 経由で C# から使う実装 | ONNX Runtime の Unity 用プラグインと組み合わせれば、Unity でも動く見込み |

## Inference Engine での対応状況

[onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) のグラフを取得して演算子を集計し、[Inference Engine の対応演算子一覧](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-operators.html) と照らし合わせた。ONNX 版はテキスト・画像・音声のモデルに分かれている。

| ファイル | Inference Engine で動かない演算子 |
| --- | --- |
| テキスト用 `model.onnx`(fp32) | `RotaryEmbedding`(非対応と明記)、`SkipSimplifiedLayerNormalization`、`SimplifiedLayerNormalization` |
| テキスト用 q4 / q8 | 上の 3 つに加えて `MatMulNBits`、`GatherBlockQuantized` |
| 画像用 `vision_encoder.onnx` | `MultiHeadAttention`。また `NonZero` は CPU でしか動かない |
| 音声用 `audio_encoder.onnx` | `SimplifiedLayerNormalization` |

公開されている ONNX は、ONNX Runtime の独自演算子(`com.microsoft` ドメイン)を前提にしている。Inference Engine で動かすには、少なくとも次のどれかが必要になる。

- 独自演算子を使わず、標準の演算子だけで ONNX を書き出し直す
- 独自演算子をグラフ上で標準の演算子に分解する
- 足りない演算子を自前で実装する

この作業がそのまま技術的な貢献になり、新規性の中心になる。量子化版(int4 / int8)をモバイルの GPU で動かせれば、さらに価値が上がる。

## アプローチ別の新規性

| アプローチ | 新規性 | 理由 |
| --- | --- | --- |
| テキストだけを llama.cpp(LLMUnity など)で動かす | 低い | 既存ライブラリの更新で実現しそう |
| ONNX Runtime の Unity プラグインで動かす | 低〜中 | 公開済みの ONNX がそのまま使える。.NET での先行例もある |
| Inference Engine だけでテキストを動かす(iOS / Android、GPU) | 中〜高 | 演算子の置き換えが必要で、先行例がない |
| 画像・音声も Unity で扱う(例: スクリーンショットや声でゲーム内を検索) | 高い | Unity でのマルチモーダル埋め込みの先行例が見つからない |

## 注意点

- 公開から 1 日しか経っていないため、数週間で状況が変わる可能性が高い。早く出すことが新規性を保つ条件になる。
- Unity Asset Store と論文(arXiv など)はまだ調べていない。研究として新規性を主張するなら、ここも確認が必要。
- LLMUnity が EmbeddingGemma 2 に対応済みかどうかは確かめていない。

## 次のステップ

テキスト用モデルを標準の演算子だけで書き出し直し、Inference Engine で読み込めるかを試す。これが一番手早く検証できる。

## 参考資料

- [EmbeddingGemma 2 – Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
- [EmbeddingGemma 2: The Developer Guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
- [google/embeddinggemma-2 – Hugging Face](https://huggingface.co/google/embeddinggemma-2)
- [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
- [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
- [Unity Inference Engine 対応演算子](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-operators.html)
- [unity/inference-engine-minilm-v6](https://huggingface.co/unity/inference-engine-minilm-v6)
- [LLMUnity(undreamai)](https://github.com/undreamai/LLMUnity)
- SiliconANGLE(2026-10-06)
- Unite.AI
- EmbeddingGemma.NET
- UniChat
