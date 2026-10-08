# 実モデル検証・ランタイム API の実行記録

日付: 2026-10-08。ブランチ: `feat/m1-sentis-runtime`。詳細は [計画](m1-runtime-plan.md)。

## 現在の結果

**進行中。C# API の単体契約21件と、Python の成果物配置 / ハーネス26件は合格。実モデル artifact は取得中で、Sentis CPU / GPUCompute と API の実モデル照合はまだ合格していない。**

| 対象 | Red の実行 | Green の実行 / 限界 |
| --- | --- | --- |
| 成果物の検証・配置 | `uv run --locked pytest -q tests/test_stage.py`: `embeddinggemma_tools.stage` 未実装で collection error | 8 passed。SHA-256破損、revision / source不一致、不足ケース、cosine不合格、欠落ファイル、NaN を拒否。以前の成功レポートも無効化 |
| C# prompting | `--suite compile`、`artifacts/unity-harness/runtime-api-red/`: `TextRole` 未実装による15 compile errors | `TextPromptsTests` 13 passed。query / document / raw、空文字・空タイトル・Unicode・null / 不正role / title を確認 |
| C# 推論 API | `--suite compile`、`artifacts/unity-harness/embedder-api-red/`: `TextEmbedder` 未実装で compile failure | 小さい Sentis graph の `TextEmbedderTests` 8 passed。Worker再利用、独立した出力配列、Dispose、入力契約、不正shape / norm / NaN、padding違反を確認。実モデルの互換性の証明ではない |
| M1 / API のハーネス分離 | `tests/test_unity.py`: assembly filter の違いと runtime scope 未実装で5 failed / 13 passed | 配置テストと合わせ26 passed。M1固定3件とAPIのbackend2件をclass filterで分離し、欠落・失敗・skipを拒否 |
| C# 契約テスト全体 | `uloop run-tests --filter-type regex --filter-value '^EmbeddingGemma\.Tests\.(TextPromptsTests\|TextEmbedderTests)\.' --unsaved-changes fail` | 21 passed、failed=0、skipped=0、inconclusive=0。`artifacts/unity-harness/runtime-unit-green.json` |

実行環境: ローカル Windows / Unity 6000.3.16f1 / Sentis 2.6.1、uloop dispatcher 3.8.1 / runner 3.8.0。
Python は `tools/` の uv を使用。モデル生成・変換はローカルで行っていない。
`artifacts/` と生成モデルは Git 管理外。元の基盤 CI と参照成果物の由来は [M1 検証記録](m1-validation.md) を参照。

## 残っている検証

- 取得 artifact の JSON / ファイル SHA-256 監査と Editor への配置。
- 既存 `M1ReferenceTests` の tokenizer / CPU / GPUCompute 3件。
- `TextEmbedderReferenceTests` の API 経由15ケース×2 backend、複数入力後の再推論。
- PR の Python 全体・実モデル Python export・workflow の結果と、実際の検証 source SHA の記録。

今回の範囲が合格しても、`.sentis` 保存・量子化・測定を終えるまで M1 全体は未完了。
