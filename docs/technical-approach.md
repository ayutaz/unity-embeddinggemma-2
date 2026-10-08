# 技術調査: Sentis で EmbeddingGemma 2 を動かす方法

- 調査日: 2026-10-07
- 実装・計画の更新日: 2026-10-08
- 関連: [ゴール](goal.md) / [新規性調査](embeddinggemma-2-unity-novelty.md)

> Python の text-only 参照生成と Core ATen export は実装済みで、小さいモデルのオフラインテスト 33 件が合格。
> 実モデルのPython参照生成 / 保存済みexportの全15ケース照合はCIで成功（修正後runの最小cosine 0.9999997616）。
> Sentis import / tokenizer / CPU / GPUCompute、`.sentis` 保存・量子化は未検証。
> 詳細計画は [M1 計画](m1-plan.md)、実行済みの証拠は [検証記録](m1-validation.md)を参照。

## 要約

- 採用する Sentis は **2.6.1**(2026-04-02 公開、パッケージ名 `com.unity.ai.inference`)。manifest に追加済みだが、指定 Unity 6000.3.19f1 での依存解決と実動作は未検証。
- EmbeddingGemma 2 は「画像・音声のエンコーダの出力を、共通のテキスト本体の入力列に差し込む」構造。Sentis 側は **テキスト本体 + 画像 / 音声エンコーダ + 差し込み処理** として設計する。M1 はテキスト経路のみ。
- モデルの持ち込みは、**PyTorch(`.pt2`)の直接読み込みを第一候補**、**標準の演算子だけで書き出した ONNX を予備**とする。LiteRT 版は Sentis が int4 に対応していないため使えない。
- 前処理(画像のパッチ分割、音声のメルスペクトログラム)と差し込み処理は、Sentis の Functional API で GPU 上のグラフとして作る。
- 最大のリスクはメモリ。Sentis の量子化は MatMul などの重みにしか効かず、約 134M パラメータある埋め込み表が fp32 のまま残る可能性がある。

## 1. 採用する Sentis の仕様

### バージョン

| バージョン | 公開日 | 主な変更 |
| --- | --- | --- |
| **2.6.1** | 2026-04-02 | ドキュメントの修正。本プロジェクトの採用版 |
| 2.6.0 | 2026-03-20 | ONNX opset 25 に対応。RMSNorm・Swish の Functional メソッド。PyTorch 読み込みで Buffer に対応。トークナイザの切り詰め機能 |
| 2.5.0 | 2026-01-23 | **PyTorch(`.pt2`)の直接読み込み**。Hugging Face 用 tokenizer parser の機能追加 |
| 2.4.0 | 2025-10-22 | 名前を Inference Engine から Sentis に戻す。**トークナイザ API**。**LiteRT の読み込み**。**STFT などの音声向け演算子** |

公開日は [公式 CHANGELOG](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/changelog/CHANGELOG.html) で再確認した。

- 動作条件は Unity 6000.0 以降。依存パッケージは Burst・Collections・App UI・Newtonsoft Json・Image Conversion。
- 名前空間は `Unity.InferenceEngine` のまま(表示名だけが Sentis に戻った)。
- `Packages/manifest.json` に Sentis 2.6.1 を追加済み。`Packages/packages-lock.json` は Editor での依存解決後に更新・確認する（現時点では Sentis 未記録）。

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

- このプロジェクトの Unity 6000.3.19f1 / Sentis 2.6.1 で WebGPU を使う実動作は未検証。
- 他の Unity バージョンの対応状況を根拠に、今回の組み合わせで動くと判断しない。Web 対応は「できれば」の扱いのままとする。

## 2. EmbeddingGemma 2 の構造

### 全体の流れ

```
テキスト ─ トークナイザ ─ input_ids ─┐
                                    ├─ 埋め込み表を引く ─ 差し込み ─ テキスト本体(24層、内部512→768 projection)─ 平均プーリング ─ 正規化 ─ 768次元
画像 ─ パッチ分割 ─ 画像エンコーダ ───┤        (<|image|> などの位置を特徴量で置き換える)
音声 ─ メルスペクトログラム ─ 音声エンコーダ ─┘
```

- 画像と音声は **別々のベクトルにはならない**。エンコーダが出す 512 次元の「ソフトトークン」を、テキスト中の `<|image|>` や `<|audio|>` の位置に差し込み、テキスト用の本体に通す。
- テキストと画像を混ぜた入力(例: 「防水シューズ <|image|>」)も、1 つのベクトルになる。
- sentence-transformers の構成は「本体 → 平均プーリング(プロンプト部分も含める)→ 正規化」の 3 段。独立した Dense モジュールはないが、テキスト本体内部の `embedding_projection`（512→768）は保持する。
- `sliding_window=512` は双方向局所注意の片側の半径（左右を合わせて約1024）。固定長 128 の M1 実モデル検証だけでは窓境界をまたぐ長さの実モデル互換性は証明できない。小さいモデルの単体テストでは半径 2 / 長さ 6 で局所・全体注意を照合する。

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
| **A. PyTorch の直接読み込み** | transformers で読み込む → 書き出し用ラッパーで入出力を整える → `torch.export` → Core ATen に分解 → `.pt2` | 入出力の形と差し込み処理を Python で自由に決められる。注意や RoPE は自動で分解される | 採用した torch 2.14.1 の export と Sentis importer の組み合わせは未検証 | **第一候補** |
| **B. ONNX の書き出し直し** | A と同じラッパー → `torch.onnx.export` で独自演算子なしに書き出す | 読み込み機能が成熟している。モデルの中身を Netron などで確認しやすい | opset 23 以降で Attention や RotaryEmbedding にまとめられないよう注意が必要 | **予備** |
| C. 公開済み ONNX の手直し | onnx-community 版の独自演算子を、標準の演算子の組み合わせに置き換える | 差し込み処理を含む入出力の設計をそのまま使える | グラフの手術が壊れやすい。q4 / q8 版は使えない | 参考にとどめる |
| D. LiteRT の読み込み | 公式の LiteRT 版を読み込む | 小さい(165MB) | int4・StableHLO に非対応 | **不可** |
| E. Functional API で全部作る | safetensors の重みを読み、C# でモデルを組み立てる | 完全に制御できる | 作業量が大きい | 最後の手段。前処理と後処理には使う |

方法 A の Python ラッパーと `.pt2` export は実装済み。方法 B の ONNX export は未実装で、
実モデルの Sentis import の失敗を確認してから、同じ重み・入出力を使って追加する。

### 現在の Python 実装

| ファイル（`tools/embeddinggemma_tools/`） | 実装した処理 | 確認できた範囲 |
| --- | --- | --- |
| `model.py` | 公式 text config を使い、safetensors の `language_model.` 配下だけを strict load。fp32 / eager / eval | 小さい公式モデルの重み抽出 / 欠落検出と、CIでの実モデルtext-only load |
| `text.py` / `reference.py` | query / document / raw、公式 sentence-transformers pooling / L2 正規化、JSON 参照生成 | 小さいモデルと実モデル15ケースの参照生成 |
| `export.py` | 固定長注意 mask、Core ATen 分解、int32 scalar 制限、metadata-only assertion 除去、`.pt2` 保存 | 実モデルの保存・再読み込み照合成功。Sentis import は未実行 |
| `prepare.py` / `__main__.py` | pinned snapshot 取得、参照 / tokenizer / export、再読み込み全件照合、SHA-256、CLI | 小さいsnapshotのTDDと、CI実モデル15ケースの照合成功。run / artifact digestは検証記録を参照 |

M1 の既定は batch 1 / length 128 / fp32 / 768 次元。生成物は `reference.json`、
`tokenizer.json`、`model.pt2`、`export-validation.json`。Python の照合閾値は 0.999999、
Sentis の fp32 は 0.999、量子化版は 0.99 と分けて扱う。
CLI では入力長を変更できるが、既定以外の長さの実モデル・Sentis 互換性は未検証。

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

以下は段階の要約。進捗・依存条件・具体的な再開順序は [M1 詳細計画](m1-plan.md)を基準とする。

| # | 作業 | 完了の条件 |
| --- | --- | --- |
| 1 | 環境の準備: Sentis 2.6.1 の manifest 指定と Python uv 環境は準備済み。CI で Unity の依存解決を行う | Unity と Python の両方で、モデルを読み込めるようになっている |
| 2 | 参照データの作成: 実装済み CLI を CI で実モデルに実行し、入力・ID・mask・最終埋め込みを保存する | 固定 15 ケースの参照データが揃っている |
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
- 別の環境で lockfile と揃えるときは `uv sync --locked`、検証には `uv run --locked` を使う。

| 項目 | 内容 |
| --- | --- |
| Python | 3.13 以上 |
| 主な依存 | torch 2.14.1、transformers 5.19.0、sentence-transformers 6.1.0、onnx 1.23.2 |
| 確認済み | transformers 5.19.0 に `EmbeddingGemma2Model`・`EmbeddingGemma2TextModel`・`EmbeddingGemma2Processor` が含まれている |

## 未確認の事項(実際に試して確かめる)

- Sentis の PyTorch 読み込みが、torch 2.14 で書き出した EmbeddingGemma 2 を読めるか
- Sentis のトークナイザが、Gemma の `tokenizer.json` を正しく扱えるか
- text-only 経路はPythonの小さいモデル / 実モデルで照合済みだが、実モデルの Sentis 互換性と、画像 / 音声特徴量の差し込み・GPU 上の受け渡しは未検証
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
