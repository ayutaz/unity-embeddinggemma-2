# 技術調査: Sentis で EmbeddingGemma 2 を動かす方法

- 調査日: 2026-10-07
- 関連: [ゴール](goal.md) / [新規性調査](embeddinggemma-2-unity-novelty.md)

## 要約

- Sentis の最新版は **2.6.1**(2026-04-02 公開、パッケージ名 `com.unity.ai.inference`)。このプロジェクトの Unity 6000.3 で使える。
- EmbeddingGemma 2 は「画像・音声のエンコーダの出力を、テキスト用モデルの入力列に差し込む」構造になっている。そのため Sentis 側は **3 つのエンコーダ + 差し込み処理** として組む。
- モデルの持ち込みは、**PyTorch(`.pt2`)の直接読み込みを第一候補**、**標準の演算子だけで書き出した ONNX を予備**とする。LiteRT 版は Sentis が int4 に対応していないため使えない。
- 前処理(画像のパッチ分割、音声のメルスペクトログラム)と差し込み処理は、Sentis の Functional API で GPU 上のグラフとして作る。
- 最大のリスクはメモリ。Sentis の量子化は MatMul などの重みにしか効かず、約 134M パラメータある埋め込み表が fp32 のまま残る可能性がある。

## 1. Sentis の最新版

### バージョン

| バージョン | 公開日 | 主な変更 |
| --- | --- | --- |
| **2.6.1** | 2026-04-02 | ドキュメントの修正(最新) |
| 2.6.0 | 2026-03-30 | ONNX opset 25 に対応。RMSNorm・Swish の Functional メソッド。PyTorch 読み込みで Buffer に対応。トークナイザの切り詰め機能 |
| 2.5.0 | 2026-01-29 | **PyTorch(`.pt2`)の直接読み込み**。**Hugging Face の `tokenizer.json` の読み込み** |
| 2.4.0 | 2025-10-30 | 名前を Inference Engine から Sentis に戻す。**トークナイザ API**。**LiteRT の読み込み**。**STFT などの音声向け演算子** |

- 動作条件は Unity 6000.0 以降。依存パッケージは Burst・Collections・App UI・Newtonsoft Json・Image Conversion。
- 名前空間は `Unity.InferenceEngine` のまま(表示名だけが Sentis に戻った)。
- このプロジェクトの `Packages/manifest.json` には、まだ Sentis が入っていない。

### 読み込める形式

| 形式 | 対応範囲 | EmbeddingGemma 2 での評価 |
| --- | --- | --- |
| PyTorch(`.pt2`、ExportedProgram) | Core ATen IR(約 180 演算子)まで分解したもの | **第一候補**。transformers から直接書き出せる |
| ONNX | opset 7〜25 | **予備**。独自演算子を含まない形で書き出し直す必要がある |
| LiteRT(`.tflite`) | 標準の演算子のみ | **使えない**。int4 の定数と StableHLO の演算子に非対応。公式の LiteRT 版は int4 で、`.litertlm` 形式の入れ物に入っている |

外部の重みファイル(`.onnx_data` など)は、同じフォルダに置けば自動で読み込まれる。

### この案件に関係する機能

| 機能 | API | 使いどころ |
| --- | --- | --- |
| テクスチャ → テンソル | `TextureConverter.ToTensor(Texture, Tensor, TextureTransform)` | `RenderTexture` を GPU 上のままテンソルにする。サイズが違えば線形補間で拡大・縮小する |
| グラフの編集・作成 | `FunctionalGraph`、`Functional.*`、`graph.Compile()` | 前処理・差し込み・後処理を GPU のグラフとして足す。`Compile` は重いので事前に行い、保存しておく |
| 音声向け演算子 | STFT、DFT、MelWeightMatrix | メルスペクトログラムを GPU 上で計算する |
| 量子化 | `ModelQuantizer.QuantizeWeights(QuantizationType.Float16 / Uint8, ref model)` | 重みを fp16 または uint8 で持つ。**MatMul・Dense・Conv の重みだけが対象** |
| 保存形式 | `ModelWriter.Save`、`ModelLoader.Load(path)`(`.sentis`、FlatBuffers) | 大きなモデルは `.sentis` にして StreamingAssets から読む |
| トークナイザ | `HuggingFaceParser.GetDefault().Parse(json)` | Gemma の `tokenizer.json` を読む。足りない部品は自作して登録できる |
| 実行 | `Worker(model, BackendType.GPUCompute)` | GPU のコンピュートシェーダで実行。`ScheduleIterable` で複数フレームに分けられる |
| 結果の取得 | `ReadbackAndCloneAsync`、`ReadbackRequest` | 非同期で CPU に読み戻す。検索に使うベクトルは最後に 1 回だけ読み戻す |

### 演算子の対応状況(この案件に関係するもの)

| 演算子 | ONNX 読み込み | PyTorch 読み込み | 対処 |
| --- | --- | --- | --- |
| RMSNormalization | 対応 | 分解されて対応 | 不要 |
| RotaryEmbedding / Attention(ONNX の標準演算子) | **非対応** | 分解されて対応 | ONNX では基本的な演算子に分解して書き出す |
| `masked_scatter` | — | **非対応(明記)** | 書き出し用のラッパーで `where` や `index_put` に書き換える |
| NonZero | CPU のみ | 制限あり | 入力の形を固定して、使わないようにする |
| STFT / MelWeightMatrix / DFT | 対応(全バックエンド) | — | 音声の前処理に使う |
| MatMulInteger / DequantizeLinear など量子化系 | **非対応** | — | Sentis 側の量子化を使う |

### Web(WebGPU)について

- Unity の WebGPU 対応は、6000.3(このプロジェクト)では試験的な扱い。6.6(2026-09)で正式対応になった。
- Sentis の GPUCompute が WebGPU で動くかは、まだ確かめていない。Web 対応は「できれば」の扱いのままとする。

## 2. EmbeddingGemma 2 の構造

### 全体の流れ

```
テキスト ─ トークナイザ ─ input_ids ─┐
                                    ├─ 埋め込み表を引く ─ 差し込み ─ テキスト用の本体(24層)─ 平均プーリング ─ 正規化 ─ 768次元
画像 ─ パッチ分割 ─ 画像エンコーダ ───┤        (<|image|> などの位置を特徴量で置き換える)
音声 ─ メルスペクトログラム ─ 音声エンコーダ ─┘
```

- 画像と音声は **別々のベクトルにはならない**。エンコーダが出す 512 次元の「ソフトトークン」を、テキスト中の `<|image|>` や `<|audio|>` の位置に差し込み、テキスト用の本体に通す。
- テキストと画像を混ぜた入力(例: 「防水シューズ <|image|>」)も、1 つのベクトルになる。
- sentence-transformers の構成は「本体 → 平均プーリング(プロンプト部分も含める)→ 正規化」の 3 段。初代にあった Dense 層はない。

### 各部品

| 部品 | 規模 | 構成(`config.json` より) | 入出力(onnx-community 版) |
| --- | --- | --- | --- |
| テキスト用の本体 | 270M(本体 130M + 埋め込み表 140M) | 24 層、hidden 512。スライディング窓(512)と全体注意を 5:1 で繰り返す。全体注意の層は head_dim 512。語彙 262,144 | 入力: `input_ids`、`attention_mask`、`image_features [N,512]`、`audio_features [N,512]`。出力: `sentence_embedding [B,768]` |
| 画像エンコーダ | 170M | 16 層、hidden 768、パッチ 16px。3×3 のプーリングで 1 画像あたり 280 トークン(70〜1120 に変更可能) | 入力: `pixel_values [B, パッチ数, 768]`、`pixel_position_ids [B, パッチ数, 2]`。出力: `[トークン数, 512]` |
| 音声エンコーダ | 300M | 12 層、hidden 1024。Conformer 系。40ms で 1 トークン(1 秒あたり 25 トークン) | 入力: `input_features [B, フレーム数, 128]` とマスク。出力: `[トークン数, 512]` |

### 前処理の仕様

| 対象 | 仕様 |
| --- | --- |
| テキスト | Gemma のトークナイザ(BPE、`byte_fallback` あり、空白を `▁` に置換、`<bos>` … `<eos>` で囲む)。タスクごとの前置き文を付ける(例: 検索の質問は `task: search result \| query: `、文書は `title: none \| text: `)。画像と音声には付けない |
| 画像 | RGB にし、0〜1 に変換(255 で割るだけで、平均・分散による正規化はなし)。リサイズは bicubic。16px のパッチに分割する |
| 音声 | モノラル 16kHz。窓 320(20ms)、ずらし幅 160(10ms)、FFT 512、メル 128 本(0〜8000Hz)、`mel_floor` は 0.001 |
| 出力 | Matryoshka により先頭の 512 / 256 / 128 次元に切り詰められる。切り詰めたあとは再び正規化する |

## 3. モデルを持ち込む方法の比較

| 方法 | 手順 | 長所 | 短所 | 評価 |
| --- | --- | --- | --- | --- |
| **A. PyTorch の直接読み込み** | transformers で読み込む → 書き出し用ラッパーで入出力を整える → `torch.export` → Core ATen に分解 → `.pt2` | 入出力の形と差し込み処理を Python で自由に決められる。注意や RoPE は自動で分解される | 機能が新しい(2.5〜)。動作確認された PyTorch は 2.9.1 で、最新の 2.14 との相性は未確認 | **第一候補** |
| **B. ONNX の書き出し直し** | A と同じラッパー → `torch.onnx.export` で独自演算子なしに書き出す | 読み込み機能が成熟している。モデルの中身を Netron などで確認しやすい | opset 23 以降で Attention や RotaryEmbedding にまとめられないよう注意が必要 | **予備** |
| C. 公開済み ONNX の手直し | onnx-community 版の独自演算子を、標準の演算子の組み合わせに置き換える | 差し込み処理を含む入出力の設計をそのまま使える | グラフの手術が壊れやすい。q4 / q8 版は使えない | 参考にとどめる |
| D. LiteRT の読み込み | 公式の LiteRT 版を読み込む | 小さい(165MB) | int4・StableHLO に非対応 | **不可** |
| E. Functional API で全部作る | safetensors の重みを読み、C# でモデルを組み立てる | 完全に制御できる | 作業量が大きい | 最後の手段。前処理と後処理には使う |

方法 A・B で共通のラッパーを Python 側に作り、出力形式を切り替えられるようにしておく。

## 4. Unity 側の設計(案)

### グラフの分け方

| グラフ | 入力 | 出力 | 作り方 |
| --- | --- | --- | --- |
| 画像の前処理 + 画像エンコーダ | 画像テンソル(`TextureConverter` で作成)`[1,3,H,W]` | `[T,512]` | Functional API で、リサイズ → パッチ分割(reshape / transpose)→ エンコーダ。解像度を固定し、位置 ID は定数にする |
| 音声の前処理 + 音声エンコーダ | 波形 `[1, サンプル数]` | `[T,512]` | Functional API で、STFT → 振幅 → メル → log → エンコーダ |
| テキスト用の本体 | `input_ids`、マスク、差し込む特徴量 | `[1,768]` | 書き出したモデルに、差し込み・平均プーリング・正規化・切り詰めを足す |

- 画像と音声の特徴量は GPU 上に置いたまま、テキスト用の本体に渡す。CPU に読み戻すのは最後の 768 次元だけにする。
- 長さが変わる入力(テキストの長さ、音声の長さ)は、上限を決めて 0 で埋め、マスクで扱う。形が固定されていると Sentis の最適化が効く。

### メモリ対策の案

- Sentis の量子化は MatMul・Dense・Conv の重みだけが対象なので、埋め込み表(262,144 × 512、約 134M パラメータ)は fp32 のまま残り、約 537MB になるおそれがある。
- 対策の候補:
  - 埋め込み表を MatMul として扱える形に変える
  - 埋め込み表をグラフの外に出し、fp16 または int8 の表を C# 側で引いてから GPU に渡す
- 実際にどれだけ残るかは、量子化したあとのモデルで測ってから判断する。

### 注意点

- **色空間**: このプロジェクトは URP(リニア色空間)。`TextureConverter` は sRGB からリニアへ変換するが、参照実装(PIL)は sRGB の値を 255 で割るだけ。変換の有無で値がずれるので、最初に確かめる。
- **リサイズ方式**: 参照実装は bicubic だが、`TextureConverter` は線形補間。ずれる場合は、Functional API の Interpolate(cubic には制限あり)を使うか、参照側の画像を先に同じ大きさにしておく。
- **トークナイザ**: Sentis の Hugging Face 用パーサが、Gemma の構成(BPE + `byte_fallback`、Replace による正規化、`MergedWithPrevious` の Split、TemplateProcessing)にすべて対応しているかは未確認。足りない部品は自作して登録する。`tokenizer.json` は 32MB あるので、モバイルでの読み込み時間も測る。
- **画像の解像度**: 参照実装は縦横比を保ってトークン数の上限まで拡大・縮小する。Unity 側では解像度を固定する(例: 768×768 → 48×48 パッチ → 256 トークン)。比較するときは、参照側にも同じ解像度の画像を渡す。

## 5. 正しさの検証方法

1. Python で参照データを作る(固定のテキスト・画像・音声のセット)。
   - 最終の埋め込みだけでなく、途中の値(トークン ID、画像のパッチ、メルスペクトログラム、各エンコーダの出力)も保存する。
2. Unity の EditMode テストで、段階ごとに比べる。
   - トークン ID は完全一致。
   - 途中の値は誤差の範囲内。
   - 最終の埋め込みはコサイン類似度で判定する(fp32 で 0.999 以上、量子化版で 0.99 以上)。
3. 段階ごとに比べることで、ずれの原因(トークナイザ、前処理、モデル本体)をすぐ特定できるようにする。

## 6. 進め方(M1: テキスト用モデルを動かすまで)

| # | 作業 | 完了の条件 |
| --- | --- | --- |
| 1 | 環境の準備: Sentis 2.6.1 を追加する。Python 環境は `tools/` の uv プロジェクトを使う(準備済み) | Unity と Python の両方で、モデルを読み込めるようになっている |
| 2 | 参照データの作成: sentence-transformers でテキストの埋め込みを作り、途中の値と一緒に保存する | テキストの参照データが揃っている |
| 3 | トークナイザの検証: Sentis の `HuggingFaceParser` で `tokenizer.json` を読み、トークン ID を比べる | 日本語・英語・記号・絵文字を含むテストで、トークン ID が完全一致する |
| 4 | テキスト用モデルの書き出し: ラッパーを作り、`.pt2` で書き出して Sentis に読み込む(だめなら ONNX) | Sentis で読み込めて、CPU と GPU の両方で動く |
| 5 | 一致の確認: 参照データと比べる | コサイン類似度 0.999 以上 |
| 6 | 量子化と保存: fp16 で量子化し、`.sentis` として保存する。メモリを測る | 0.99 以上を保ったまま、メモリ使用量を把握できている |

M3(画像)・M4(音声)では、同じ流れで、エンコーダと Functional API による前処理を足していく。

### リポジトリでの扱い

- モデルファイル(`.pt2`、`.onnx`、`.sentis`、safetensors)はコミットしない。`.gitignore` に入れ、取得と変換はスクリプトで行う。
- 変換用の Python スクリプトは `tools/` にまとめ、Unity の `Assets/` とは分ける。

### Python 環境(uv)

Python は uv で管理する。`tools/` が uv のプロジェクトで、依存は `pyproject.toml` と `uv.lock` に記録する。

- 依存の追加は `uv add <パッケージ>` で行う。`pip install` や、`pyproject.toml` の手書きはしない。
- スクリプトは `uv run python <スクリプト>` で実行する(`tools/` で実行する)。
- 別の環境で揃えるときは `uv sync` を使う。

| 項目 | 内容 |
| --- | --- |
| Python | 3.13 以上 |
| 主な依存 | torch 2.14.1、transformers 5.19.0、sentence-transformers 6.1.0、onnx 1.23.2 |
| 確認済み | transformers 5.19.0 に `EmbeddingGemma2Model`・`EmbeddingGemma2TextModel`・`EmbeddingGemma2Processor` が含まれている |

## 未確認の事項(実際に試して確かめる)

- Sentis の PyTorch 読み込みが、torch 2.14 で書き出した EmbeddingGemma 2 を読めるか
- Sentis のトークナイザが、Gemma の `tokenizer.json` を正しく扱えるか
- テキスト用の本体の中で、埋め込み表がどう使われているか(画像・音声のトークン位置の扱いや、層ごとの入力の作り方)。transformers のモデル実装で確かめる
- 埋め込み表が、量子化のあとも fp32 のまま残るか
- 画像の色空間とリサイズによる差が、どの程度出るか
- Sentis の GPUCompute が WebGPU で動くか

## 参考資料

- [Sentis 2.6 マニュアル](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html)
- [Sentis 2.6 What's new](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/whats-new.html)
- [Sentis 2.6 CHANGELOG](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/changelog/CHANGELOG.html)
- [Sentis: PyTorch 形式への書き出し](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/export-convert-torch.html)
- [Sentis: 対応する PyTorch 演算子](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-torch-export-operators.html)
- [Sentis: 対応する ONNX 演算子](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-operators.html)
- [Sentis: 対応する LiteRT 演算子](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-litert-operators.html)
- [Sentis: 量子化](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/quantize-a-model.html)
- [Sentis: テクスチャからテンソルへの変換](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/convert-texture-to-tensor.html)
- [Sentis: モデルの編集(Functional API)](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/edit-a-model.html)
- [Sentis: トークナイザ](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/tokenizer.html)
- [Unity パッケージレジストリ(com.unity.ai.inference)](https://packages.unity.com/com.unity.ai.inference)
- [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)(`config.json`、`processor_config.json`、`config_sentence_transformers.json`、`modules.json`、`tokenizer.json`)
- [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)(グラフの入出力と演算子を集計)
- [litert-community/embeddinggemma-2-text-270m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm)
- [Unity 6.6 での WebGPU の正式対応(cinevva)](https://app.cinevva.com/news/2026-09-01-unity-6-6-webgpu-production)
