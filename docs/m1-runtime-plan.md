# M1 実モデル検証と C# ランタイム API の詳細計画

実行日: 2026-10-08。履歴の位置づけを2026-10-09に更新。実装ブランチ: `feat/m1-sentis-runtime`。
以下はmain `60f906d` と計画更新PR #2の `84b9dcd` を基点とした当初の計画と実行記録。
PR #3はmain `1f0e580`へ統合済み。その後PR #4も統合され、現在はmain `8146107`でM1完了。
最新の証拠は [M1完了検証](m1-completion-validation.md)、次の残作業は [M2計画](m2-plan.md)を参照。
当時は後続の計画更新 `7d1964b` を取り込み、C# / Python / 固定参照条件を変えず文書競合を解消した。

## 当時の範囲と完了条件

PR #3の範囲は参照成果物の配置、実モデルのSentis検証・互換性修正、C#ランタイムAPIまでだった。
`.sentis` 保存・fp16 量子化・性能 / メモリ測定は次の段階で、今回の完了を M1 全体の完了とはしない。

1. 固定 revision `914f7f89142e33e77833254d9c9b90c3cef7303b` の成功 CI 成果物を取得し、source commit / tree、全15ケース、生成条件、ファイル SHA-256 を照合して配置する。
2. Unity 6000.3.16f1 / Sentis 2.6.1 の Windows Editor で tokenizer ID / mask が全15件完全一致。CPU と GPUCompute はそれぞれ全15件で fp32 cosine >= 0.999、有限、768次元、単位長。GPU の skip / CPU 代替を認めない。
3. UnityEditor に依存しない C# API を実装し、query / document / raw、タイトル、入力長128、768次元の正規化済み出力を扱う。実モデルを使い、API 経由でも両 backend の全ケースを照合する。リソース破棄・破棄後の使用・不正入力は小さいテストで検証する。

## 実行順序と判断

| 段階 | 作業と TDD | 終了条件 / 証拠 |
| --- | --- | --- |
| A 成果物 | run `37755304407` の未失効 artifact を取得。生成コード・依存・入力条件が同じことを確認し、JSON / ファイル SHA-256 を照合。監査・配置をコード化する場合は正常系と破損・revision不一致・部分成果物の拒否を先にテスト | run・metadata・ファイル hash・配置先を小さい監査レポートに記録。モデルは Git 管理外 |
| B 実モデル red | 既存の3件を uloop ハーネスで実行。import、tokenizer、CPU、GPU のどこで失敗したかをログ / XML で特定 | 参照未配置以外の実際の結果を回収。未実行を合格にしない |
| C 互換性 | 実際に見つかった importer / 演算子 / tokenizer の差異に対するテストを先に追加して修正。`.pt2` 不成立なら同じ重み・入出力の ONNX export を CI で生成し、Python 全15件照合後に Editor へ配置 | tokenizer と両 backend が既存 acceptance tests で合格。元の失敗と修正後の結果を別々に記録 |
| D API red | `EmbeddingGemma` ランタイム assembly と API の契約をテストから定義。モデル・tokenizer を受け取り、同じ prompting / tokenization / inference を再利用する。query / document / raw、タイトル、null / 不正モード、破棄を検証 | 未実装 API によるコンパイル失敗または仕様違反の red を記録 |
| E API green | 最小実装。Worker / Tensor の所有権を明確にし IDisposable で解放。全15件を query / document / raw API 経由で CPU / GPUCompute 照合。再利用時の別入力も確認 | 単体契約テストと API 実モデルテストが合格。768次元・有限・単位長・cosine を全件記録 |
| F ハーネス / CI | 増えた API テストの成功だけで既存 M1 を誤判定しないよう検証範囲を明示。変更時は Python テストで red / green。関連 Python / export の重い検証は Actions、C# compile / tests はローカル Editor | 対象テストの不足・skip・タイムアウトが失敗になる。PR のCIとローカルの実行証拠を記録 |
| G 受け渡し | 計画・検証記録・API の使用例を更新して PR を提出。PR #2 と実装 PR の依存を記載 | 上記1〜3を証拠で監査。依頼があるまで merge しない |

## API の初期設計

- 名前空間 `EmbeddingGemma`、`TextEmbedder` が model と tokenizer JSON、CPU / GPUCompute backend を受け取る。
- `EmbedQuery(text)`、`EmbedDocument(text, title)`、`EmbedRaw(text)` で `float[768]` を返す。同期 API を第一段階とし、最終ベクトルだけ CPU へ読む。モデル内部の projection・pooling・正規化は export と同じものを使用する。
- 空文字は固定参照と同じく有効。null は引数例外。文書タイトルは null だけ `none` とし、空のタイトルは空のまま保持する（Python と一致）。
- 同じインスタンスで複数入力を順に処理し、Worker は再利用、入力 Tensor は呼び出し単位で解放。Dispose は二重呼び出し可能で、破棄後の推論を拒否する。
- backend を暗黙に切り替えない。GPU 非対応・モデル入出力不一致・非有限出力は明示的に失敗させる。

## PR #3実行時の証拠と制約

今回のA〜Eはローカル検証まで完了。成果物の監査・配置、空文字互換修正後のM1 3件、公開APIの2件が実モデルで合格し、単体契約24件も合格した。
最小cosineはCPU `0.99999999999923483`、GPUCompute `0.99999999999970735`。各backendで固定15ケース、APIではWorker再利用後の再推論も確認。
F / Gも完了しPR #3は統合済み。当時のhead `805d7e7` のCIはPython60件×4環境、実モデル15件が成功。
統合前の最終headのCIは [PR #3のChecks](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3/checks) とPR本文に記録。[実行記録](m1-runtime-validation.md)、[API手順](runtime-api.md)、[当時の数値レポート](results/m1-windows-fp32-20261008.json)を参照。
基盤の Python 47件×4環境・実モデル Python 15件から、今回の追加を含むCI結果は分けて記録する。
artifact は約1.1GBで保持3日。失効時や生成コード等の変更時は CI で再生成する。
ローカル Editor の未保存 Scene / Prefab を自動保存・破棄しない。モデル・ログ・XML は `artifacts/` / `.uloop/` / `Assets/M1Generated/` に保存し、要約だけ Git に残す。
