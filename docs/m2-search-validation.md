# 検索サンプル実装の検証記録

開始日: 2026-10-10。実装計画は [検索サンプルとモデル準備](m2-search-plan.md)。この記録は作業中で、検索サンプル全体・実モデル・新規checkout再現の合格はまだ確定していない。

## TDDの証拠

| 対象 | Red | Green / 状態 |
| --- | --- | --- |
| Python順位・固定入力・参照監査 | `tests/test_search.py`を先に追加、`ModuleNotFoundError: embeddinggemma_tools.search` | 同じ15件が合格。有限・非zero norm、cosine正規化、同点ID順、入力・順位・score・source不一致を確認 |
| Unity検索と所有権 | テストを先に追加、Unity compileはSearchDocument / ITextEmbedder未定義でCS0246 2件 | `EmbeddingGemma.Tests.TextSearchTests` 6件合格、failed / skipped / inconclusive = 0。実モデルを用いない契約 |
| prepare / stage接続 | 検索生成1件・配置監査5件を先に追加、未実装引数で6 failed | 検索・既存配置回帰と合わせ30件合格。検索参照と保存済みexportを照合し、hash・順位・欠落・export不一致では既存モデルを保護 |
| サンプル登録・consumer導入 | packageのsamples未登録でKeyError | Pythonのpackage監査・新規consumer導入19件合格 |
| サンプルUI契約 | スタブに必要な操作・状態がなくUnity compile 30 errors | EditModeの5件合格。準備・検索・不正入力・推論失敗・backend不一致・明示解放・モデル保存とhashを確認 |
| 実際の無効化 | 最初のEditModeテストは無効化イベントの前提が不適切で1 failed / 4 passed | EditModeは明示解放の契約に修正。別のPlayModeテストで実GameObject無効化時の解放が1 passed / skip 0 |
| 準備の失敗記録 | tokenizer不正時に以前のsuccessが残り1 failed | 開始時にsuccessを無効化し、同じ準備テスト2件合格。既存モデルを維持 |
| 文書IDの一致 | 末尾改行をC#の`$`が受け入れ1 failed / 2 passed | 文字列全体の一致へ修正。パッケージ全契約38件合格 / skip 0 |
| 検索ハーネス | searchスコープ未実装で2 failed | 4条件必須・GPU skip禁止の2件合格。これは実Unity検索結果ではない |
| 軽量CI成果物 | 検索参照だけのartifact未登録で1 failed | 検索参照とexport報告だけを同梱するCI契約1件合格 |
| モデルキャッシュ | 別生成元参照の監査引数がなく5 failed | 同条件の別生成元だけを受け入れ、依存・tokenizer hash・source・順位の違いを拒否。関連する新規Python契約10件合格 |
| 実モデル検索 | 4条件のテストを先に実行し、検索参照が未配置で4 failed / skip 0 | 参照取得・監査・Sentis準備後に同じ4件で全ベクトルと全順位を照合する。未準備を合格にしない |

キャッシュ経路の追加後、stageの全19件も合格した。モデル生成元と検索参照生成元は独立して記録する。

基盤PRは [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10)。head `0b0dd394b11f8599a379eb6f2349792609e605f1`の [CI run 37955666558](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37955666558)は全8 job成功。
Python4環境各123件、M1の15入力と検索10入力の保存済みexport照合が合格。検索最小cosine `0.9999999403953552`。
実際のcheckoutは`2edb272f17a38c7af210474345c35f391fd6955c`。このPython照合をSentis検索の成功に数えない。

PlayModeテスト後にUnityが自動変更したEditor設定・define・App UI設定を読み戻して元へ戻した。次のcompileでは一時的な応答停止警告が出たが、同じ実行を継続観測し、status Readyと38件合格を確認した。再起動や未保存sceneの破棄はしていない。

大きいartifactの取得は開始後も0バイトのzipと接続中プロセスを確認したため、軽量参照と互換モデルキャッシュの監査経路を追加中。取得待ちを取得成功とは扱わない。

Pythonのローカル確認は追加範囲だけuvで実行。全環境のテストと固定モデルの検索参照生成はActionsで行う。

## 未完了の確認

- サンプルUI・モデル準備と読み込み・エラー表示・終了時解放の操作。
- 固定6文書 / 4queryの実モデル順位・埋め込み精度をfp32 / Float16重み × CPU / GPUComputeで照合。
- Actions検索参照のsource SHA / hash監査と、新規checkout / 空のconsumerでの導入・起動再現。
- 小さい結果要約とUI画像の保存、利用者向けモデル取得・変換・配置・更新手順。

M1の過去実測・UPM移行の過去回帰は [現在の状態](status.md)の証拠を維持する。この検索サンプルの実モデル合格の代わりには数えない。
