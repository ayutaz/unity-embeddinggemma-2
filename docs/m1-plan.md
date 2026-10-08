# M1 詳細計画

更新日: 2026-10-08

計画の基準: [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1) 統合後の `main`、`60f906d9b8c91a95593898571963d0b3b385a053`。
`git fetch origin --prune` と `git pull --ff-only origin main` で同期済み。後続の [PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3) は未マージであり、以下では main の状態とブランチ上の検証を分ける。

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

**M1 は未完了。基盤は PR #1 で main に統合済み。未マージの PR #3 では参照成果物の配置、空文字のtokenizer互換修正、Windows Editor の実モデル fp32 CPU / GPUCompute、C# API の照合まで合格した。次はPRの統合準備、`.sentis` 保存・再読み込み、fp16量子化、時間・メモリ測定を進める。**

| 対象 | 確認できた状態 | 残っていること |
| --- | --- | --- |
| Python | main `60f906d` は4環境各47件合格。PR #3 head `805d7e7` は4環境各60件と最新モデル生成CIの実モデル15件が合格。参照に使ったrun `37755304407` も15件合格 | base取り込みなどでheadが変わる際は最新runを確認 |
| Unity | PR #3で6000.3.16f1 / Sentis 2.6.1、token ID / mask全15件一致、実モデルCPU / GPUCompute各15件合格。単体契約24件も合格 | 保存・量子化・測定、統合後の変更内容に応じた回帰確認 |
| 推論・保存 | PR #3で公開C# APIも各backend全15件・Worker再利用後の再推論が合格。CPU最小cosine `0.99999999999923483`、GPU `0.99999999999970735` | `.sentis` 保存・再読み込み、fp16量子化、性能・メモリは未実行 |
| GitHub | public。PR #1は統合済み。PR #2は計画、PR #3はPR #2をbaseとする実装で、両方未マージ | 最新CIと依存・競合確認。main保護は未設定。任意のクラウドUnity検証は未実行 |
| 開発環境 | Unity 6000.3.16f1 / Sentis 2.6.1 / uloop のハーネスを統合済み。最新 main への同期を確認 | 次の実装ごとに作業ブランチ・PRを作成。実モデルのローカル検証を継続 |

main の Python / lint 成功と、PR #3のSentis実測を区別する。PR #3のfp32合格だけではM1全体を完了にしない。
実測したC#内容は `6da634a2c745cb1bfb226c1712666139e1366442`。証拠は
[固定commitの実行記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/805d7e75990716d41dc8b752c3b6de506595d08a/docs/m1-runtime-validation.md) と
[数値レポート](https://github.com/ayutaz/unity-embeddinggemma-2/blob/805d7e75990716d41dc8b752c3b6de506595d08a/docs/results/m1-windows-fp32-20261008.json)。
Pythonの60件×4環境は [run 37769319201](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37769319201) のログで確認した。
モデル生成 [run 37769319055](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37769319055) も成功し、15件・最小cosine `0.9999998807907104` をログで確認した。
GitHub 側の main ブランチ保護と repository rulesets は未設定。
現在の直接 push 禁止は開発規則であり、サーバーで強制できている状態ではない。
実行ログと確認方法は [検証記録](m1-validation.md)、workflow の準備条件は [CI 手順](ci.md) を参照。

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
| P1 参照仕様 | PR #3で特殊トークン・padding・切り詰めを全15件照合済み | 条件変更時に参照を再生成して再照合 | ID / mask完全一致 |
| P2 参照生成 | run `37755304407` のrevision・source・固定15件・SHA-256を監査してPR #3で配置済み | 失効・生成コード・依存・入力条件変更時はCI再生成 | 成功runと配置先の整合性。配置成功はSentis合格と別判定 |
| P3 tokenizer | PR #3で空文字例外を再現・修正後、全15件合格 | 空文字を含む回帰テストを維持 | 全15件のID / mask完全一致、ケース除外なし |
| P4 export | Pythonの保存済み `.pt2` 照合とPR #3のSentis import成功 | 現モデルを保存・量子化の元に使用 | 現状ONNX fallbackは不要 |
| P5 inference | PR #3でCPU / GPUComputeと公開APIの全15件、再推論、単体契約24件合格 | 最新CI・レビュー・統合準備を完了 | fp32 cosine >= 0.999、有限・768次元・単位長。skipをGPU合格にしない |
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
| 1（ブランチ上で済） | CI参照成果物の監査・配置、互換性修正、CPU / GPUCompute・公開API照合 | 成功runの固定成果物 | PR #3の実測記録。初回空文字例外3 failedと修正後3 passed / API 2 passedを分けて保存 |
| 2（進行中） | PR #2 / #3の統合準備 | 最新headのCI、差分・レビュー・競合・検証SHA確認 | PR #2の `7d1964b` をPR #3へ取り込み、文書競合を解消。最新headのCI確認後レビューへ渡す。マージ依頼を受けた時だけ #2 → #3 の順に統合 |
| 3 | `.sentis` 保存・再読み込み | PR #3の検証済みコードを含む基準を確定。統合前に進める場合は依存を明記したPR | 保存・ロードの契約テストでred確認後に実装。CPU / GPUCompute各15件で保存前後を比較し、Python参照fp32 cosine >= 0.999を維持 |
| 4 | fp16量子化 | fp32保存・再読み込み合格 | 量子化の契約をTDDで実装。別ファイルとして保存・ロードし、両backend各15件でPython参照cosine >= 0.99。shape・有限・単位長も確認 |
| 5 | 時間・メモリ測定 | fp32 / fp16の数値合格 | import・tokenizer準備・初回・定常推論、ファイルサイズ、取得可能なCPU / GPUメモリを同条件で記録。未取得の指標は未測定と明記 |
| 6 | M1完了監査、M2計画 | step 3〜5の証拠と最新コードの対応が取れる | 全15ケース、両backend、保存・量子化・測定・CIを監査。M2でUPM・サンプル・macOS / iOS / Androidを検証 |

step 1 でPR #3が使用した成果物は run `37755304407` の
`m1-reference-b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d`（約1.1 GB）。
2026-10-08 の確認時点で未失効、期限は **2026-10-11 18:37:45 JST**。
PR head `673cb9a`、参照生成の merge checkout `b3b0d76`、統合済み main `60f906d` は
commit SHA が異なるが Git tree は一致することを確認した。PR #3では JSON の `metadata.source_commit` と各ファイルhashも監査済み。
この計画更新ではモデルを取得していない。失効、モデル生成コード・依存・入力条件の変更時は最新 main の
`model-reference.yml` を手動実行し、成功した新しい run / source SHA を記録する。

保存・量子化・測定はそれぞれ小さい契約テストのred / greenと実モデル照合を分ける。
モデル生成・変換はActions、GPU実測は利用可能なローカルWindows Editorとuloopハーネスを使う。
測定は同じ入力順・warmup・反復回数・backend・OS / CPU / GPU・graphics APIを記録し、精度合格と速度評価を区別する。
重みサイズから実行時GPUメモリを推定して測定済みにしない。モデルと大きいログはGit管理外、数値要約・再生成手順だけを残す。

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
