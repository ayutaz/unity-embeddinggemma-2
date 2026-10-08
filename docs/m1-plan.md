# M1 詳細計画

更新日: 2026-10-08

## 目的と完了条件

EmbeddingGemma 2 のテキスト入力を Unity Editor の Sentis で実行する。
固定テストセットの全件で Python 参照実装に対する fp32 出力のコサイン類似度が
0.999 以上であることを CPU / GPUCompute の両方で確認する。
トークン ID と attention mask は完全一致させる。出力は有限な 768 次元の単位ベクトルとする。
検索用 / 文書用プロンプト、空文字、日英混在、空白、改行、記号、絵文字、切り詰めを検証する。

M1 は Windows Editor での検証を対象とする。macOS / iOS / Android、公開、画像、音声は後続のマイルストーン。
量子化とメモリ測定も既存の M1 作業計画に従って実施し、fp32 の合格と分けて記録する。
量子化版の閾値は 0.99。未実行の項目を成功扱いにしない。

## 現在地（2026-10-08 確認）

**M1 は未完了。Python の実装と小さいモデルの検証が済み、draft PR #1 の CI の結果を確認中。実モデルの参照生成を開始済み。**

| 対象 | 確認できた状態 | 残っていること |
| --- | --- | --- |
| Python | uv 管理の参照生成・text-only loader・Core ATen export・CLI を実装。オフライン 33 件合格 | Ubuntu / Windows × Python 3.13 / 3.14 の CI、実モデル 15 ケースの照合 |
| Unity | 6000.3.19f1 を指定。Sentis 2.6.1 を manifest に追加。Editor acceptance tests を作成 | Editor での依存解決、packages-lock 更新、テストのコンパイルと red / green |
| 推論・保存 | Python の小さいモデルで保存済み `.pt2` を再読み込みして別入力を検証 | 実モデルの Sentis import、CPU / GPUCompute、C# ランタイム API、`.sentis`、量子化、測定 |
| GitHub | public。[draft PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1)を提出し、4 workflow の初回PR runを起動済み | runの結果確認と失敗の切り分け。Unity Secrets / Variables は未登録 |
| 開発環境 | `feat/m1-text-tdd` の実装コミット `4fec060` を push 済み。main への commit / push / merge は未実施 | PR の CI 結果に基づく検証と記録 |

GitHub の既存 Dependency Graph 2 件の成功と、新規runの起動は、M1 の CI 合格ではない。
GitHub 側の main ブランチ保護と repository rulesets は未設定。
現在の直接 push 禁止は開発規則であり、サーバーで強制できている状態ではない。
実行ログと確認方法は [検証記録](m1-validation.md)、workflow の準備条件は [CI 手順](ci.md) を参照。

## 開発規則

- リポジトリは public の OSS。ユーザー指定により CI/CD を最大限利用する。モデル取得・参照生成・export・Unity テスト・性能測定は原則 GitHub Actions で行う。
- ローカルは編集、静的確認、必要最小限の軽い red / green 確認に限る。ローカルへの Unity インストールや実モデルの重い実行は進めない。
- 全実装を TDD で進める。仕様を検証するテストを書く → 意図した理由の失敗を実行で確認する → 最小実装 → 合格 → 必要な整理。
- Python は `tools/` の uv 環境に統一する。`uv sync` / `uv add` / `uv run` を使用する。
- Python 単体テストは小さいランダム初期化モデルなどでオフライン実行できるようにする。
- 実モデルの参照結果との照合は独立した統合テストとし、小さいモデルの合格で代替しない。
- モデルは Hugging Face のコミット SHA を固定し、参照データにライブラリ、入力条件、モデルの情報を記録する。
- ランタイムは Unity / Sentis のみ。ONNX Runtime などを Unity に導入しない。
- モデルと再生成可能な大きい成果物は Git 管理から除外し、生成コマンドと検証要約を管理する。

## 実施順序

| 段階 | 現在の状態 | 次の実施内容と成果物 | 合格条件 / 次の判断 |
| --- | --- | --- | --- |
| P0 環境 | Python 準備済み、Unity 未検証 | CI で指定 Editor / Sentis を起動し、依存解決ログと生成された packages-lock を確認 | Unity 6000.3.19f1 と Sentis 2.6.1 でテストをコンパイルできる |
| P1 参照仕様 | 固定入力と Python テスト実装済み | 実モデルの tokenizer で 15 ケースの ID / mask と実際の切り詰めを確認 | 特殊トークン、padding、truncation、プロンプト、出力構造が記録される |
| P2 参照生成 | CLI 実装済み、実モデル未実行 | pinned revision の重みから `reference.json` / `tokenizer.json` を CI 生成 | 全 15 ケースで有限な正規化済み 768 次元出力。revision / source SHA / 条件を記録 |
| P3 tokenizer | Editor acceptance test 作成済み、未コンパイル | Sentis tokenizer と Python を照合し、失敗した処理だけ TDD で修正 | 全 15 ケースの ID / mask が完全一致 |
| P4 export | 小さいモデルの `.pt2` 保存・再読み込み合格 | 実モデルの batch 1 / length 128 / fp32 export と Unity import。`.pt2` が非互換なら ONNX | Python eager 対保存済み export の全件 cosine >= 0.999999、Sentis import 成功 |
| P5 inference | Editor acceptance test 作成済み、推論 API 未実装 | CPU / GPUCompute の CI で red を確認し、C# ランタイム API を TDD で実装 | 両 backend の全 15 ケースで cosine >= 0.999、有限・768 次元・単位長 |
| P6 保存・測定 | 未着手 | `.sentis` 保存・再読み込み、fp16 量子化、精度・時間・メモリ測定を TDD で実装 | fp32 保存前後一致、量子化版の全件 cosine >= 0.99。条件と測定値を記録 |

P2 / P4 の Python 処理は同じ `prepare` コマンドで行う。P3 と P4 の Unity import の
切り分けは実際の失敗から進め、P5 の成立後に P6 へ進む。

## 固定する入力と参照条件

- モデル: `google/embeddinggemma-2`
- revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- 入力: [tools/cases/text.json](../tools/cases/text.json) の 15 ケース。日本語 / 英語の query・document、タイトル、空文字、空白、改行、混在、絵文字、記号、Unicode、長文の切り詰めを含む。
- batch 1 / sequence length 128 / fp32。入力は `input_ids` と `attention_mask`。本体の 512→768 projection を保持する。
- 公式 sentence-transformers の mean pooling（プロンプトを含む）と L2 正規化を Python 参照とする。
- query は `task: search result | query: `、document は `title: none | text: `（タイトルがあれば置換）、raw は前置きなし。
- テストの成功だけでなく、15 件すべての case ID と出力を確認する。長文の切り詰めが実モデルの tokenizer で発生することも P1 で確かめる。

## 互換性の判断手順

1. `.pt2` が失敗したら、export 側と importer 側を分け、例外と未対応演算子を記録する。
2. 同じ入出力と重みの wrapper を標準 ONNX 演算子で export する。
3. ONNX でも失敗する場合は該当演算子を基本演算へ分解し、小さい入力のテストで一致を確認する。
4. 参照実装との比較を省略したり、別モデルに変更したりして合格扱いにしない。

## 検証の証拠

各段階で red / green のコマンドと結果を記録する。Python の実モデル照合、Unity の NUnit XML、
モデル revision、バックエンド、入力長、最小 cosine、失敗件数、測定条件を残す。
export と最終埋め込みがずれた場合は、同一 ID / mask を渡してモデル本体と後処理を切り分ける。

## 次の作業と受け渡し

| 順序 | 作業 | 開始条件 | 完了を示す証拠 |
| --- | --- | --- | --- |
| 1（済） | `feat/m1-text-tdd` を作成し、変更を commit / push、draft PR 提出 | 承認モード変更後、ブランチ作成成功 | [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1)、初回実装コミット `4fec060`。モデルや artifacts は commit していない |
| 2 | Python matrix・workflow lint・実モデル参照生成を PR で実行 | PR と対象 workflow がリモートに存在する | 4 環境の pytest XML、actionlint、15 件の export report / digest。いずれも同じソース revision |
| 3 | Unity CI を開始 | Unity Secrets 登録。step 2 の実モデル export 成功 | Editor ログ、依存解決後の packages-lock、NUnit XML。失敗は red の理由を記録 |
| 4 | importer / tokenizer / C# API を TDD で修正 | step 3 の再現可能な失敗 | P3〜P5 の全件合格。ONNX に変更する場合は同じ入力・重みで照合し、ステージングとテストも更新 |
| 5 | GPUCompute の実行環境を確定 | GPU job の graphics / compute ログ | CPU 代替や skip でない GPUCompute の全件合格。hosted runner が非対応なら実行可能な GPU runner を用意 |
| 6 | 保存・量子化・測定と M1 完了監査 | P5 合格 | P6 のテスト、保存前後と量子化版の全ケース結果、メモリ・時間・測定環境 |

Unity Secrets の準備は step 2 と並行できる。GPU runner の利用に追加の費用・設定が必要なら
具体的な選択肢を確認する。ローカルに実モデルや Unity を導入して進める前提にはしない。
main 保護のサーバー設定も未完了事項として追跡し、必須 CI の check 名が確定した段階で設定する。
PR の merge、UPM リリース、モデルの再配布はこの計画の実行に含めない。

## M1 完了時に残す証拠

- PR の head SHA と実際に検証した checkout SHA（PR イベントでは merge SHA）、モデル revision、Unity / Sentis / Python ライブラリのバージョン。`metadata.source_commit` は checkout SHA と照合する。
- 全 15 ケースの token ID / mask 一致、CPU / GPUCompute それぞれの cosine、有限性、形、L2 norm。
- `.sentis` 保存前後の照合と fp16 量子化版の cosine（全件 >= 0.99）。
- import / tokenizer 準備 / 初回推論 / 定常推論の時間、保存サイズ、メモリ。OS、CPU、GPU、graphics API、入力長、反復回数、warmup を記録。
- CI run URL と NUnit XML / ログ、成果物 SHA-256。artifact の期限が切れても要約と再生成手順は docs に残す。
- macOS / iOS / Android は後続で検証することを明記し、Windows M1 の結果を流用して成功扱いにしない。

## 一次資料

- [公式モデル](https://huggingface.co/google/embeddinggemma-2)
- [Sentis PyTorch export](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/export-convert-torch.html)
- [既存技術方針](technical-approach.md)
