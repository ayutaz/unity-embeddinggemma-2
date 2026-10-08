# 実モデル検証・ランタイム API の実行記録

日付: 2026-10-08。ブランチ: `feat/m1-sentis-runtime`。詳細は [計画](m1-runtime-plan.md)。

## 現在の結果

**依頼範囲のローカル検証は合格。成果物を配置し、tokenizer ID / mask 全15件、実モデル fp32 CPU / GPUCompute 全15件ずつ、公開 C# API 経由の全15件ずつと再推論が合格。C# 単体契約24件も合格。M1全体は保存・量子化・測定が残る。**

| 対象 | Red の実行 | Green の実行 / 限界 |
| --- | --- | --- |
| 成果物の検証・配置 | `uv run --locked pytest -q tests/test_stage.py`: `embeddinggemma_tools.stage` 未実装で collection error | 8 passed。SHA-256破損、revision / source不一致、不足ケース、cosine不合格、欠落ファイル、NaN を拒否。以前の成功レポートも無効化 |
| 固定スイートの監査 | 同じ15件でも異なるcase IDを含む成果物を受理してしまい、`wrong_suite` が1 failed | 固定 `tools/cases/text.json` と順序を含め照合するよう修正後、配置処理全体9 passed |
| C# prompting | `--suite compile`、`artifacts/unity-harness/runtime-api-red/`: `TextRole` 未実装による15 compile errors | `TextPromptsTests` 13 passed。query / document / raw、空文字・空タイトル・Unicode・null / 不正role / title を確認 |
| C# 推論 API | `--suite compile`、`artifacts/unity-harness/embedder-api-red/`: `TextEmbedder` 未実装で compile failure | 小さい Sentis graph の `TextEmbedderTests` 8 passed。Worker再利用、独立した出力配列、Dispose、入力契約、不正shape / norm / NaN、padding違反を確認。実モデルの互換性の証明ではない |
| M1 / API のハーネス分離 | `tests/test_unity.py`: assembly filter の違いと runtime scope 未実装で5 failed / 13 passed | 配置テストと合わせ26 passed。M1固定3件とAPIのbackend2件をclass filterで分離し、欠落・失敗・skipを拒否 |
| C# 契約テスト全体 | `uloop run-tests --filter-type regex --filter-value '^EmbeddingGemma\.Tests\.(TextPromptsTests\|TextEmbedderTests)\.' --unsaved-changes fail` | 21 passed、failed=0、skipped=0、inconclusive=0。`artifacts/unity-harness/runtime-unit-green.json` |
| 空文字の互換処理 | 実モデル初回は3件とも `AddedVocabulary.Split` の `ArgumentNullException`。`TextTokenizer` 未実装の compile red も確認 | 設定の特殊トークン / padding を維持する Sentis の空シーケンス経路を追加。小さいfixtureのテンプレート設定ミスも修正し、専用3件が合格。全単体契約は修正後24件合格、`runtime-unit-final.json` |

実行環境: ローカル Windows / Unity 6000.3.16f1 / Sentis 2.6.1、uloop dispatcher 3.8.1 / runner 3.8.0。
Python は `tools/` の uv を使用。モデル生成・変換はローカルで行っていない。
`artifacts/` と生成モデルは Git 管理外。元の基盤 CI と参照成果物の由来は [M1 検証記録](m1-validation.md) を参照。

## 取得・監査・配置の完了

- 成功 run `37755304407` の artifact を `gh run download` で取得。`model.pt2` は `1093341201` bytes、tokenizer は `32170775` bytes、reference は `408191` bytes。
- `uv run --locked python -m embeddinggemma_tools.stage --source-commit b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d` が成功。`artifacts/m1-stage.json` を保存し、`Assets/M1Generated/model.pt2` に配置。
- model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、source commit、batch 1 / length 128 / fp32 / 768次元、固定15ケースを監査。Python export の最小cosine `0.9999998807907104`。
- `model.pt2` SHA-256: `18018897e12a72af996253ffd4c8bb28aa8825e614f2813e298815851188d25f`。
- `reference.json` SHA-256: `637c45b2fa3be84d16b47a3456000e5ab27d1e50e37424288bf42b318d799cb8`。
- `tokenizer.json` SHA-256: `25975a936201b81402a3b867d5c462cf10808bb96759f97fe9aefecbb11c49df`。
- 生成コード（prepare / export / model / reference / text）、uv.lock、固定入力は main `60f906d` から変更なし。元の生成 checkout と main は tree一致を確認済みで、今回の C# / staging / harness 変更と区別する。
- 配置合格では `m1_reference_passed=false`。Sentis での数値結果は次の実行で別途記録する。

## 実モデルの Sentis / API 検証

| 実行 | 結果 | 最小cosine / 証拠 |
| --- | --- | --- |
| 初回 `real-model-first` | import・compile成功。先頭5ケースはCPU/GPUで照合できたが、空文字のtokenizationで3テスト失敗、skip=0 | `artifacts/unity-harness/real-model-first/test-results.xml`。未対応入力を除外せず修正へ進めた |
| 修正後 `real-model-empty-fix` | tokenizer / CPU / GPUCompute の3テスト合格、failed / skipped / inconclusive=0。全15件の ID / mask が完全一致 | CPU `0.99999999999923483`、GPU `0.99999999999970735`。`summary.json` / `run-tests.json` / `get-logs.json` |
| 公開API `real-runtime-first` | CPU / GPUCompute の2テスト合格、failed / skipped / inconclusive=0。query / document / raw APIで全15件、別入力後の再推論も合格 | CPU `0.99999999999923483`、GPU `0.99999999999970735`。同形式のハーネス証拠 |

最終実行のC#内容は `6da634a2c745cb1bfb226c1712666139e1366442` にコミット。
M1 / API 実行はコミット前から開始したが、テスト後のC#内容に差分はない。
[小さい数値レポート](results/m1-windows-fp32-20261008.json) に各backend・各caseのcosine、実行時刻、コードのLF正規化SHA-256、成果物hashと元の生成SHAを保存した。
Sentis出力は全件 `[1,768]`、有限、L2 normが1±0.001。GPUはRTX 4070 Ti SUPER / Direct3D12 / compute=Trueで実行し、Workerのbackendを検査した。
成功時はuloopがNUnit XMLを出力しないため、CLIの構造化結果と全caseログを証拠とする。失敗時XMLは別途保持する。

空文字の問題は Sentis 2.6.1 `Runtime/Tokenization/AddedVocabulary.cs:77` の実コードとXMLで特定。
`TextTokenizer` は非空入力を元のSentis pipelineに渡し、空入力は同じpost processor / truncation / padding設定を使うtoken-free pipelineで処理する。
固定モデルのnormalizerは空文字を空のままにするReplace設定。特殊トークンIDのハードコード、参照の変更、空文字ケースの除外、package cacheの修正は行っていない。

## 残っている確認と次段階

- head `805d7e7` の [Python PR run 37769319201](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37769319201) は4環境各60件、[実モデル run 37769319055](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37769319055) は15件・最小cosine `0.9999998807907104` で成功。以前のcode commit `6da634a` のPR runは後続commitでcancelledとなったため、成功に含めない（push runは成功）。
- 計画PR #2の `7d1964b` を取り込み、文書競合を解消。実測したC#内容、Pythonのコード・lock・固定条件には変更なし。取り込み後の最終結果は [PR #3 のChecks](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3/checks) とPR本文で対象headに対応するrunを確認する。
- 保存・再読み込み、fp16量子化、時間 / メモリ測定は今回の範囲外の次段階。

今回の範囲が合格しても、`.sentis` 保存・量子化・測定を終えるまで M1 全体は未完了。
