# M1 詳細計画

更新日: 2026-10-08

計画の基準: [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1) 統合後の `main`、`60f906d9b8c91a95593898571963d0b3b385a053`。
後続の実装・実測: [PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3)（未マージ）、C#内容は `6da634a`。

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

**M1 は未完了。基盤は PR #1 で main に統合済み。後続 PR #3 のブランチで成果物の監査・配置と空文字互換修正を行い、Windows Editor の tokenizer / fp32 CPU / GPUCompute 全15件と C# API経由の照合・再推論が合格した。C# 単体契約24件も合格。保存・量子化・時間 / メモリ測定が残る。**

| 対象 | 確認できた状態 | 残っていること |
| --- | --- | --- |
| Python | `60f906d` の main CI 4環境で各47件合格。最終 PR head `673cb9a` の実モデル15ケースも合格、最小 cosine `0.9999998807907104` | Sentisとの照合 |
| Unity | 6000.3.16f1 / Sentis 2.6.1 / Unity CLI Loop 3.14.0。実モデルimportと全15件の ID / mask、CPU / GPUCompute fp32照合が合格 | 保存・量子化・測定 |
| 推論・保存 | Python exportとSentis fp32、公開C# APIを照合済み。CPU最小cosine `0.99999999999923483`、GPU `0.99999999999970735` | `.sentis` 保存・再読み込み、fp16量子化、時間 / メモリ測定 |
| GitHub | public。PR #1 は Squash マージ済み。main CI 成功。後続PR #2 / #3 は未マージ | 最新実装PRのCI確認とmainの保護設定。Unity Secrets / Variables は未登録で、任意のクラウド検証は未実行 |
| 開発環境 | Unity 6000.3.16f1 / Sentis 2.6.1 / uloop のハーネスを統合済み。最新 main への同期を確認 | 次の実装ごとに作業ブランチ・PRを作成。実モデルのローカル検証を継続 |

main の Python / lint 成功と、後続ブランチの実際のSentis検証を分けて記録する。fp32合格だけではM1全体を完了にしない。
GitHub 側の main ブランチ保護と repository rulesets は未設定。
現在の直接 push 禁止は開発規則であり、サーバーで強制できている状態ではない。
実行ログと確認方法は [検証記録](m1-validation.md)、workflow の準備条件は [CI 手順](ci.md) を参照。
実モデル・C# APIの詳細は [作業計画](m1-runtime-plan.md)、[実行記録](m1-runtime-validation.md)、[API手順](runtime-api.md) を参照。

## 開発規則

- リポジトリは public の OSS。ユーザー指定により CI/CD を最大限利用する。モデル取得・参照生成・export・Unity テスト・性能測定は原則 GitHub Actions で行う。
- Python のモデル取得・変換は引き続き CI を使う。ユーザー指定により、インストール済み Unity 6000.3.16f1 でのローカル起動・検証を進める。
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
| P0 環境 | ローカルで指定 Editor / Sentis / uloop を解決し、コンパイル合格。main 統合済み | 依存変更時にコンパイルを再確認 | Unity 6000.3.16f1 と Sentis 2.6.1 でテストをコンパイルできる（確認済み） |
| P1 参照仕様 | 固定15入力・特殊トークン・padding・切り詰めを実データで照合済み | 入力条件を拡張する際は参照を更新して再照合 | 入出力条件を確認し、Sentis に同じ条件を渡す（現条件で合格） |
| P2 参照生成 | run `37755304407` 成功。成果物のrevision・source・SHA-256監査と配置が完了 | 失効または生成コード・依存・条件の変更時はCI再生成 | 成功runの全15件・生成条件・ファイル整合性を確認済み |
| P3 tokenizer | Sentis空文字例外を再現し、設定を維持した互換処理を追加。全15件合格 | 現条件の回帰テストを維持 | 全15件のID / mask完全一致（合格） |
| P4 export | Python eager / 保存済み `.pt2` の全15件とSentis importが合格 | 現行 `.pt2` を保存・量子化の元に使う。ONNX fallbackは現状不要 | Python cosine >= 0.999999、Sentis import成功（合格） |
| P5 inference | Windows CPU / GPUComputeとC# API経由で全15件・再推論が合格。単体契約24件合格 | PRの最新CI確認と受け渡し後、P6へ | 両backend全15件でcosine >= 0.999、有限・768次元・単位長（合格） |
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
| 0（済） | 基盤 PR #1 を統合し、最新 main に同期 | 最終 PR CI 成功とユーザーのマージ指示 | main `60f906d`、統合後も Python 4環境各47件・lint 成功。証拠は [検証記録](m1-validation.md) |
| 1（済） | CI 参照成果物を取得・監査・配置 | 成功runと未失効artifact | 3ファイルのSHA-256、model revision、固定15件、sourceを監査。`artifacts/m1-stage.json` |
| 2（済） | 既存ハーネスで実モデルの失敗を特定 | 参照配置済み、認証済みローカルEditor | `.pt2` import成功、空文字のtokenizer例外で3件失敗。XMLとログを回収 |
| 3（済） | tokenizer の非互換を TDD で修正 | 空文字例外の再現 | 特殊トークン / padding設定を維持するSentis経路を追加し、P3 / P4合格 |
| 4（ローカル合格） | CPU / GPUCompute 照合と C# ランタイム API | tokenizer / import合格 | P5全15件とAPI再推論が両backendで合格。RTX 4070 Ti SUPER / D3D12、skip=0。最新PR CIを確認中 |
| 5（次） | `.sentis` 保存・再読み込み、fp16 量子化 | fp32 の両 backend と API が合格、実装PRの受け渡し | 保存前後の一致、量子化版の全15件 cosine >= 0.99。保存・量子化の失敗と未実行を分ける |
| 6 | 時間・メモリ測定、M1 完了監査 | step 5 完了 | import / 初回 / 定常推論、保存サイズ・メモリ、測定条件と全証拠。完了後に M2 の詳細計画へ進む |

step 1 の取得候補は run `37755304407` の
`m1-reference-b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d`（約1.1 GB）。
2026-10-08 の確認時点で未失効、期限は **2026-10-11 18:37:45 JST**。
PR head `673cb9a`、参照生成の merge checkout `b3b0d76`、統合済み main `60f906d` は
commit SHA が異なるが Git tree は一致することを確認した。取得後は JSON の `metadata.source_commit` も照合する。
PR #2 の計画更新時は未取得だったが、PR #3 の作業で取得・監査・配置が完了した。
失効、モデル生成コード・依存・入力条件の変更時は最新 main の
`model-reference.yml` を手動実行し、成功した新しい run / source SHA を記録する。

ユーザー指定により Unity のローカル自動操作を採用する。実モデルの変換は CI、
成果物の取得・Sentis検証は既存のローカル Editor で行う。[ハーネスの手順](automation.md)を参照。
クラウド Unity CI は Secrets が必要な Linux CPU の手動補助検証に限定した。未実行であり、Windows / GPUの証拠にはしない。
main 保護のサーバー設定も未完了事項として追跡する。自動チェックには paths filter があるため、
全 PR に要求する check と起動条件を揃え、文書のみの PR が未起動 check 待ちにならない構成を検討してから設定する。
各実装は TDD とブランチ・PR運用を継続し、merge はその都度依頼があるまで行わない。
UPM リリースとモデルの再配布は M1 完了後の別作業とする。

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
