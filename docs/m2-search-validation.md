# 検索サンプル実装の検証記録

開始日: 2026-10-10。実装計画は [検索サンプルとモデル準備](m2-search-plan.md)。この記録は作業中で、検索サンプル全体・実モデル・新規checkout再現の合格はまだ確定していない。

## TDDの証拠

| 対象 | Red | Green / 状態 |
| --- | --- | --- |
| Python順位・固定入力・参照監査 | `tests/test_search.py`を先に追加、`ModuleNotFoundError: embeddinggemma_tools.search` | 同じ15件が合格。有限・非zero norm、cosine正規化、同点ID順、入力・順位・score・source不一致を確認 |
| Unity検索と所有権 | テストを先に追加、Unity compileはSearchDocument / ITextEmbedder未定義でCS0246 2件 | `EmbeddingGemma.Tests.TextSearchTests` 6件合格、failed / skipped / inconclusive = 0。実モデルを用いない契約 |
| prepare / stage接続 | 検索生成1件・配置監査5件を先に追加、未実装引数で6 failed | 検索・既存配置回帰と合わせ30件合格。検索参照と保存済みexportを照合し、hash・順位・欠落・export不一致では既存モデルを保護 |

Pythonのローカル確認は追加範囲だけuvで実行。全環境のテストと固定モデルの検索参照生成はActionsで行う。

## 未完了の確認

- サンプルUI・モデル準備と読み込み・エラー表示・終了時解放の操作。
- 固定6文書 / 4queryの実モデル順位・埋め込み精度をfp32 / Float16重み × CPU / GPUComputeで照合。
- Actions検索参照のsource SHA / hash監査と、新規checkout / 空のconsumerでの導入・起動再現。
- 小さい結果要約とUI画像の保存、利用者向けモデル取得・変換・配置・更新手順。

M1の過去実測・UPM移行の過去回帰は [現在の状態](status.md)の証拠を維持する。この検索サンプルの実モデル合格の代わりには数えない。
