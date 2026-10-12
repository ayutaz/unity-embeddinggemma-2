# プロジェクトのゴール

- 設定日: 2026-10-07
- 更新日: 2026-10-13
- 背景: [新規性調査](embeddinggemma-2-unity-novelty.md)

リポジトリはpublicのOSSとして開発中。**正式リリースはM3画像・M4音声を含む全対応と必須検証の完了後に行う**。[全体TDD計画](multimodal-tdd-plan.md) / [M3](m3-plan.md) / [M4](m4-plan.md)を現在の実行順とする。Windowsの既存実測は[現在の状態](status.md)と[M2基盤計画](m2-release-plan.md)を参照する。標準環境の再現、必須他環境、残る安定性、最終候補 / tag導入・正式Releaseは未完了。
以下の導入時の基準はPR #12時点の履歴として保持する。確認基準main `4ee0cbb`にはPR #1〜#4・#6・#8〜#12を統合済み。
保存・Float16重み量子化・両backend精度・性能 / メモリ測定までWindows EditorのM1受け入れ条件が合格した。
確認基準mainの [CI run 37975720534](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975720534)は全8 job成功。Python4環境・実モデルPython照合・lint・パッケージ監査を確認。Sentisの実測はWindowsの検証記録を参照する。
PR #6のCI整備・UPM移行はmainへ反映済み、GitHub側のmain保護も設定・再確認済み。
文書のみのPR #7も全8 jobが成功し、未マージで閉じた。PR #5の文書変更はPR #6へ含め、#5は閉じた。
検索サンプルとモデル準備手順はWindows consumer再現を含め確認し、PR #11で統合済み。次は [M2計画](m2-plan.md)の他環境検証、Git URL導入、配布を進める。M2全体の受け入れは未完了。
最新のmain / PRの区別は [現在の状態](status.md)、導入と回帰の証拠は [パッケージ検証](m2-package-validation.md)を参照。
M1の証拠は [M1計画](m1-plan.md)、[完了検証記録](m1-completion-validation.md)を参照。
リポジトリの公開と、全モダリティ対応後のUPMパッケージ / サンプルの正式リリースは別の段階として扱う。

## ゴール

**Unity の Sentis だけで、EmbeddingGemma 2 のテキスト・画像・音声の埋め込みを GPU 上で作れるようにする。**

次の 3 つをすべて満たすことをゴールとする。

1. **Sentis だけで動かす**
   - llama.cpp・ONNX Runtime・LiteRT などのネイティブプラグインを使わない。
   - 推論はSentis(`com.unity.ai.inference`)を使う。現行RuntimeはUnity提供のNewtonsoft JSONも使用する。配布時に直接依存を明示する。
2. **テキスト・画像・音声の 3 つに対応する**
   - テキスト用の本体(270M)、画像エンコーダ(170M)、音声エンコーダ(300M)を Sentis で動かす。
   - 画像 / 音声エンコーダの特徴量は共通のテキスト本体へ渡し、768 次元の埋め込みを得る。モダリティをまたいだ検索(例: テキストで画像を探す)ができる。
3. **入力を GPU 上で処理する**
   - 画像: `Texture` / `RenderTexture` を CPU に読み戻さず、GPU 上でリサイズ・正規化してモデルに渡す。
   - 音声: 波形からメルスペクトログラムへの変換を、Sentis の演算子(STFT など)で GPU 上で行う。
   - テキスト: トークナイズは CPU で行ってよい。それ以降は GPU で処理する。

## 達成の基準

| 項目 | 基準 |
| --- | --- |
| 正しさ(fp32) | 同じ入力に対し、Python の参照実装の埋め込みとのコサイン類似度が 0.999 以上 |
| 正しさ(量子化版) | 同じくコサイン類似度が 0.99 以上 |
| モダリティ横断 | 固定のテストセットで、テキスト→画像、テキスト→音声の検索順位が参照実装と一致する |
| GPU 処理 | 画像と音声の前処理で、CPU への読み戻しが発生しない |
| ネイティブ依存 | パッケージにネイティブプラグインが含まれない |
| 動作環境(必須) | Unity Editor(macOS / Windows)、iOS、Android |
| 動作環境(できれば) | Web(WebGPU) |

参照実装は transformers / sentence-transformers を使う。プロンプト、本体内部の 512→768 projection、
プロンプトを含む mean pooling、L2 正規化、Matryoshka による次元の切り詰めの扱いを揃える。
独立した sentence-transformers Dense モジュールがないことを、projection 不要と解釈しない。

## 成果物

- Unity パッケージ(UPM 形式): モデルの読み込み、前処理、推論、後処理の API
- モデルを変換する Python スクリプト(Sentis で読める形式への書き出し)
- サンプルシーン: テキスト検索、画像検索、音声検索
- 各プラットフォームでの速度とメモリの測定結果

モデルファイルはリポジトリにコミットしない(Git LFS は使わない)。Hugging Face などから取得する手順やスクリプトを用意する。

## マイルストーン

| # | 内容 | 完了の条件 | 状態 |
| --- | --- | --- | --- |
| M1 | テキスト用モデルを Sentis で動かす | Windows Editor の CPU / GPUCompute で全 15 ケースの token ID / mask が完全一致、fp32 cosine >= 0.999。`.sentis` 保存・再読み込み、fp16 量子化版 cosine >= 0.99、時間・メモリ測定 | Windows Editor受け入れ検証完了 |
| M2 | テキストUPM・サンプル・検証基盤を安定化する | 標準consumer / 固定Git導入、API・検索・Player・解放、macOS / Windows / iOS / Androidの精度・速度 / メモリ。正式公開は最終段階へ移す | Windows実測あり。標準21f1再現・新candidate・性能・他環境は未完了 |
| M3 | 画像用モデルに対応する | [画像TDD計画](m3-plan.md)。GPU前処理 / encoder / 共通本体、実モデル精度、テキスト→画像順位、sample・Player | 画像encoder Python合格。PR #41のSentis Float16 CPU / GPU全6条件合格、FP32全6条件は要素単位許容差不合格。画像API / sample未実装、GPU前処理・最終768次元未検証 |
| M4 | 音声用モデルに対応する | [音声TDD計画](m4-plan.md)。GPU前処理 / encoder / 共通本体、実モデル精度、テキスト→音声順位、sample・収録 / 再生・Player | 音声前処理Python合格・統合済み。音声encoderはPR #40に実装済みだが保存graphの実モデル照合不合格・未統合。API / sample未実装、Unity未検証 |
| 最終公開 | 全対応後の正式リリース | M2〜M4と全必須環境、混在検索、最終candidate / tag固定導入、文書・ライセンス・GitHub Release | 未着手 |

M2の再現可能な開発基盤 → 共通特徴量 / 長さ拡張 → M3 → M4 → 全必須環境の統合検証 → 正式公開の順とする。環境確保と実行可能な他OS検証は並行し、M2の全OS完了やtext-only ReleaseをM3・M4の着手条件にしない。必須macOS / Windows / iOS / Androidは維持し、Windows M1合格で他環境を成功扱いにしない。

## 開発と受け入れの規則

- 全実装を TDD で進める。テストを書き、意図した失敗を確認してから実装・合格確認を行う。
- Python は `tools/` の uv に統一する。
- モデル取得、変換、Unity 実行、測定は GitHub Actions を優先し、ローカル実行は小さい必要最小限の確認に限る。
- main への直接 push は禁止。作業ブランチと PR を使い、merge は依頼があるまで行わない。
- 小さいモデルのテスト合格、CI の静的検査、GPU テストの skip は実モデルの Sentis / GPU 合格に含めない。

## 対象外

- 動画の埋め込み(画像の対応後に、余力があれば検討する)
- llama.cpp・ONNX Runtime などを使う方式
- モデルの追加学習(ファインチューニング)
- 本格的なベクトルデータベース(サンプルでは単純な全件比較による検索にとどめる)

## 確認が必要なこと

- tokenizerとCore ATen `.pt2` importはWindows M1の固定15件で合格。モデル・条件・プラットフォームを変える際は再照合する
- モバイルで動かせるメモリ量に収まるか。重みの概算だけで判断せず、埋め込み表、実行時テンソル、backend を含めて測る
- 話し言葉の音声が、テキストと同じ意味の空間に入るか
