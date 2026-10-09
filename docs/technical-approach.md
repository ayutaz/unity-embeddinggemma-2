# 技術調査: Sentis で EmbeddingGemma 2 を動かす方法

- 調査日: 2026-10-07
- 実装・計画の更新日: 2026-10-09
- 関連: [ゴール](goal.md) / [新規性調査](embeddinggemma-2-unity-novelty.md)

> PR #1〜#4はmain `8146107`へ統合済み。mainのPython CIは4環境各64件合格。
> 最終PR #4のPython参照生成 / 保存済みexport全15件照合も成功（run `37801226486`、最小cosine 0.9999998808）。
> Windows Editorで`.sentis`保存・再読み込み、Float16重み量子化、CPU / GPUCompute精度と測定までM1完了。[M1完了検証](m1-completion-validation.md)を参照。
> PR #6でCI整備・UPM化も実装・検証済み。確認したhead `8a585bc` のPython CIは4環境各102件・全8チェック成功、main統合待ち。[現在の状態](status.md) / [UPM検証](m2-package-validation.md)を参照。
> 次は [M2計画](m2-plan.md)。外部仕様は2026-10-07の調査を基礎とし、2026-10-09に採用版の公式export / 量子化 / UPM手順とインストール済み2.6.1の実装を再確認した。画像・音声の設計案は未実装。

## 要約

- 採用する Sentis は **2.6.1**(2026-04-02 公開、パッケージ名 `com.unity.ai.inference`)。指定 Unity 6000.3.16f1 の Windows Editorで実モデルfp32 CPU / GPUComputeを全15件検証済み。
- EmbeddingGemma 2 は「画像・音声のエンコーダの出力を、共通のテキスト本体の入力列に差し込む」構造。Sentis 側は **テキスト本体 + 画像 / 音声エンコーダ + 差し込み処理** として設計する。M1 はテキスト経路のみ。
- テキストでは**PyTorch(`.pt2`)の直接読み込みを採用し、Windows CPU / GPUComputeで実モデル検証済み**。標準演算子だけのONNXは互換性問題が生じた場合の予備で、未実装。LiteRT版のint4経路は現行Sentisの対象外。
- 画像のパッチ分割、音声のメルスペクトログラム、特徴量の差し込みはM3 / M4でSentisのGPUグラフとして実装する計画。現行APIはテキストのみ。
- メモリはM2の実機でも検証する。現行SentisのFloat16重み量子化で保存サイズは約49.45%減ったが、Editorの段階別メモリをモデル専有量や他端末の必要量とは解釈しない。

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
- root manifest / packages-lockでSentis 2.6.1の依存解決済み。PR #6のUPM manifestはSentis 2.6.1 / Newtonsoft JSON 3.2.2を直接宣言し、root lockにembedded packageを記録。別consumerでも解決・compile・29契約テストが成功した。

### 読み込める形式

| 形式 | 対応範囲 | EmbeddingGemma 2 での評価 |
| --- | --- | --- |
| PyTorch(`.pt2`、ExportedProgram) | Core ATen IR(約 180 演算子)まで分解したもの | **テキストで採用・検証済み**。transformersから書き出し、int32制限などのimporter対応を適用 |
| ONNX | opset 7〜25 | **予備**。独自演算子を含まない形で書き出し直す必要がある |
| LiteRT(`.tflite`) | 標準の演算子のみ | **使えない**。int4 の定数と StableHLO の演算子に非対応。公式の LiteRT 版は int4 で、`.litertlm` 形式の入れ物に入っている |

外部の重みファイル(`.onnx_data` など)は、同じフォルダに置けば自動で読み込まれる。

### この案件に関係する機能

| 機能 | API | 使いどころ |
| --- | --- | --- |
| テクスチャ → テンソル | `TextureConverter.ToTensor(Texture, Tensor, TextureTransform)` | `RenderTexture` を GPU 上のままテンソルにする。サイズが違えば線形補間で拡大・縮小する |
| グラフの編集・作成 | `FunctionalGraph`、`Functional.*`、`graph.Compile()` | 前処理・差し込み・後処理を GPU のグラフとして足す。`Compile` は重いので事前に行い、保存しておく |
| 音声向け演算子 | STFT、DFT、MelWeightMatrix | メルスペクトログラムを GPU 上で計算する |
| 量子化 | `ModelQuantizer.QuantizeWeights(QuantizationType.Float16 / Uint8, ref model)` | 重みの保存形式を変更。採用版の実装はConv / ConvTranspose / Gather / Dense / MatMul / MatMul2Dのfloat定数入力を対象にする。Float16のみ実モデル検証済み |
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

- このプロジェクトの Unity 6000.3.16f1 / Sentis 2.6.1 で WebGPU を使う実動作は未検証。
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
| **A. PyTorch の直接読み込み** | transformersで読み込む → ラッパー → `torch.export` → Core ATen分解 → `.pt2` | テキストのprojection・pooling・正規化を一体で書き出せる | 条件変更時は再照合。画像 / 音声のimportは未検証 | **テキストで採用済み**。torch 2.14.1の固定exportとSentis 2.6.1で全15件合格 |
| **B. ONNX の書き出し直し** | A と同じラッパー → `torch.onnx.export` で独自演算子なしに書き出す | 読み込み機能が成熟している。モデルの中身を Netron などで確認しやすい | opset 23 以降で Attention や RotaryEmbedding にまとめられないよう注意が必要 | **予備** |
| C. 公開済み ONNX の手直し | onnx-community 版の独自演算子を、標準の演算子の組み合わせに置き換える | 差し込み処理を含む入出力の設計をそのまま使える | グラフの手術が壊れやすい。q4 / q8 版は使えない | 参考にとどめる |
| D. LiteRT の読み込み | 公式の LiteRT 版を読み込む | 小さい(165MB) | int4・StableHLO に非対応 | **不可** |
| E. Functional API で全部作る | safetensors の重みを読み、C# でモデルを組み立てる | 完全に制御できる | 作業量が大きい | 最後の手段。前処理と後処理には使う |

方法AはPython exportだけでなく、Windows Sentisのimport・推論・保存・Float16重みの照合まで完了した。
方法BのONNX exportは未実装。条件変更時に実際の非互換が確認された場合に、同じ重み・入出力で追加を検討する。

### 現在の Python 実装

| ファイル（`tools/embeddinggemma_tools/`） | 実装した処理 | 確認できた範囲 |
| --- | --- | --- |
| `model.py` | 公式 text config を使い、safetensors の `language_model.` 配下だけを strict load。fp32 / eager / eval | 小さい公式モデルの重み抽出 / 欠落検出と、CIでの実モデルtext-only load |
| `text.py` / `reference.py` | query / document / raw、公式 sentence-transformers pooling / L2 正規化、JSON 参照生成 | 小さいモデルと実モデル15ケースの参照生成 |
| `export.py` | 固定長注意 mask、Core ATen 分解、int32 scalar 制限、metadata-only assertion 除去、`.pt2` 保存 | 実モデルのPython再読み込みとWindows Sentis import / fp32 CPU / GPUCompute照合が成功 |
| `prepare.py` / `__main__.py` | pinned snapshot 取得、参照 / tokenizer / export、再読み込み全件照合、SHA-256、CLI | 小さいsnapshotのTDDと、CI実モデル15ケースの照合成功。run / artifact digestは検証記録を参照 |

M1 の既定は batch 1 / length 128 / fp32 / 768 次元。生成物は `reference.json`、
`tokenizer.json`、`model.pt2`、`export-validation.json`。Python の照合閾値は 0.999999、
Sentis の fp32 は 0.999、量子化版は 0.99 と分けて扱う。
CLI では入力長を変更できるが、既定以外の長さの実モデル・Sentis 互換性は未検証。

## 4. Unity側の実装と画像 / 音声の設計案

テキストの同期APIは [API手順](runtime-api.md)の `TextEmbedder` / `TextTokenizer` / `TextPrompts` / `TextModelFile` として実装済み。
GPU指定でも最後の768次元はCPUへ取得する。batch 1 / length 128に固定し、backendを自動で切り替えない。
以下の画像 / 音声グラフ、可変長対応、特徴量の差し込みは未実装。現行テキストexportには画像 / 音声の入力を含めない。

### グラフの分け方

| グラフ | 入力 | 出力 | 作り方 |
| --- | --- | --- | --- |
| 画像の前処理 + 画像エンコーダ | 画像テンソル(`TextureConverter` で作成)`[1,3,H,W]` | `[T,512]` | Functional API で、リサイズ → パッチ分割(reshape / transpose)→ エンコーダ。解像度を固定し、位置 ID は定数にする |
| 音声の前処理 + 音声エンコーダ | 波形 `[1, サンプル数]` | `[T,512]` | Functional API で、STFT → 振幅 → メル → log → エンコーダ |
| テキスト用の本体 | `input_ids`、マスク、差し込む特徴量 | `[1,768]` | 書き出したモデルに、差し込み・平均プーリング・正規化・切り詰めを足す |

- 画像と音声の特徴量は GPU 上に置いたまま、テキスト用の本体に渡す。CPU に読み戻すのは最後の 768 次元だけにする。
- 長さが変わる入力(テキストの長さ、音声の長さ)は、上限を決めて 0 で埋め、マスクで扱う。形が固定されていると Sentis の最適化が効く。

### 重み量子化とメモリの確認

採用したSentis 2.6.1の `Runtime/Core/Quantization/QuantizeConstantsPass.cs` を再確認した。
対象layer集合にはGatherもあり、埋め込み表が必ずfp32のまま残るという当初の仮説は採用しない。
Float16では定数をhalfで保持してCastを挿入するため、保存形式と実行中のfloatテンソルを区別する。
これは配布package revision `ee0fb239bfee8fc968d32e221d392e46b20bf5bc` の実装確認であり、特定定数の実測一覧や専有VRAM測定ではない。

M1の保存サイズはfp32 `1095598044` bytes、Float16重み `553775500` bytes（約49.45%削減）。
GPU定常中央値は50.26ms / 58.20msで、速度改善は確認されなかった。
Editor全体の段階別メモリは [完了検証](m1-completion-validation.md)を参照し、モバイルの必要量は各実機で測る。
Uint8、int4、画像 / 音声の量子化、埋め込み表を別管理する変更は未実装・未検証。

### 注意点

- **色空間**: 検証プロジェクトはURPのリニア色空間だが、UPM RuntimeにURP依存はない。画像対応ではtexture import設定とGPU読み取り時のsRGB変換を確認し、参照実装と同じ値になるか中間テンソルで照合する。現段階では前処理の一致を未検証。
- **リサイズ方式**: 参照実装は bicubic だが、`TextureConverter` は線形補間。ずれる場合は、Functional API の Interpolate(cubic には制限あり)を使うか、参照側の画像を先に同じ大きさにしておく。
- **トークナイザ**: 固定tokenizerで全15件のID / mask一致を検証済み。空文字はSentis parserの例外に対し `TextTokenizer` が同じpost processor / truncation / paddingを適用する。任意のtokenizer設定の互換性は保証しない。JSONは約32MBで、モバイルの準備時間は未測定。
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

以下は完了したM1の実施順序と合格条件。各段階はWindows Editorで完了し、実測は [M1詳細計画](m1-plan.md) / [完了検証](m1-completion-validation.md)に記録した。今後の作業順序は [M2計画](m2-plan.md)を基準とする。

| # | 作業 | 完了の条件 |
| --- | --- | --- |
| 1 | Sentis 2.6.1 / uv環境を準備。Windows Editorと別UPM consumerで依存解決 | Unity / Pythonでモデルを読み込める。クラウドUnityの実行とは区別 |
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
| 主な依存 | lockfileのtorch 2.14.1、transformers 5.19.0、sentence-transformers 6.1.0、onnx 1.23.2。devはpytest / PyYAML。最新公開版ではなく検証に固定した版 |
| 確認済み | transformers 5.19.0 に `EmbeddingGemma2Model`・`EmbeddingGemma2TextModel`・`EmbeddingGemma2Processor` が含まれている |

## 未確認の事項(実際に試して確かめる)

- Windows Editorの固定fp32 `.pt2` import、fp32 / Float16重みの保存・ロードと両backendの照合は成功。他プラットフォーム・Playerの互換性は未検証
- Sentis 2.6.1のtokenizerは空文字を拒否したため、同じ特殊トークン / paddingを適用する `TextTokenizer` を追加。固定15件のID / mask完全一致を確認。他の入力条件への拡張時は再検証する
- text-only の実モデルSentis経路はWindows CPU / GPUComputeで照合済み。画像 / 音声特徴量の差し込み・GPU上の受け渡しは未検証
- Float16保存後の各定数の形式と実行時のメモリ内訳、Uint8の精度・速度・対応範囲
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
