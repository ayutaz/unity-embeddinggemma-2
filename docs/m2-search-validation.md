# 検索サンプル実装の検証記録

開始日 / 更新日: 2026-10-10。実装計画は [検索サンプルとモデル準備](m2-search-plan.md)。Windowsの実モデル4条件は合格した。画面操作と新規consumerの実モデル再現は進行中で、一覧1 / 2全体はまだ完了していない。

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
| 実モデル検索 | 4条件のテストを先に実行し、検索参照が未配置で4 failed / skip 0 | 参照取得・監査・Sentis準備後に同じ4件が合格、failed / skipped / inconclusive = 0。全ベクトルと全順位を照合 |

キャッシュ経路の追加後、stageの全19件も合格した。モデル生成元と検索参照生成元は独立して記録する。

基盤PRは [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10)。head `0b0dd394b11f8599a379eb6f2349792609e605f1`の [CI run 37955666558](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37955666558)は全8 job成功。
Python4環境各123件、M1の15入力と検索10入力の保存済みexport照合が合格。検索最小cosine `0.9999999403953552`。
実際のcheckoutは`2edb272f17a38c7af210474345c35f391fd6955c`。このPython照合をSentis検索の成功に数えない。

PlayModeテスト後にUnityが自動変更したEditor設定・define・App UI設定を読み戻して元へ戻した。次のcompileでは一時的な応答停止警告が出たが、同じ実行を継続観測し、status Readyと38件合格を確認した。再起動や未保存sceneの破棄はしていない。

大きいartifactの取得中は0バイトのzipと接続中プロセスを確認した。同じ取得処理を継続観測し、最終的に取得成功を確認した。待機を失敗や成功へ置き換えていない。軽量参照と互換モデルキャッシュの監査経路も追加した。

Pythonのローカル確認は追加範囲だけuvで実行。全環境のテストと固定モデルの検索参照生成はActionsで行う。

## PR #11のCIとWindows実モデル

[PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11)のhead `618a93293cc89de6f19777e41a2c86cea3506896`の[CI run 37959339466](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37959339466)は全8 job成功。Python4環境各132件、M1の15入力と検索10入力の保存済みexport照合が合格した。実際のcheckout SHAは`13f194266162d2a728b2b2a0da9e9e58ae314ff6`。検索最小cosineは`0.9999999403953552`。

元のWindows検証プロジェクトではPR #10の成果物（source `2edb272f17a38c7af210474345c35f391fd6955c`）を監査・配置し、Editorメニューでfp32 / Float16重みの`.sentis`を保存した。実行コードはPR #11の上記head。Unity 6000.3.16f1 / Sentis 2.6.1、Windows 11、RTX 4070 Ti SUPER、Direct3D12で実行した。

| 保存形式 | backend | 参照との最小cosine | 全4query × 全6順位 |
| --- | --- | --- | --- |
| fp32 | CPU | 0.9999999999981735 | 一致 |
| fp32 | GPUCompute | 0.999999999999512 | 一致 |
| Float16重み | CPU | 0.9999999999982625 | 一致 |
| Float16重み | GPUCompute | 0.9999999999994984 | 一致 |

`TextSearchReferenceTests`は4 passed / failed 0 / skipped 0 / inconclusive 0。各条件で6文書 + 4queryの全10ベクトルを照合し、文書の埋め込み回数は6回。同一文書2件の同点時ID順も一致した。[小さい実測結果](results/m2-search-root-windows-20261010.json)に全順位を保存した。Float16重み保存は演算全体のfp16化や速度改善を意味しない。

新規checkoutはPR #11のheadから作成し、重み・Libraryを含まない空consumerへサンプルを導入した。モデルはPR #10の監査済みartifactを再利用し、検索参照はPR #11の軽量artifactへ更新した。依存バージョン・tokenizer hash・固定revision・shape・正規化が一致することをCLIで確認し、model sourceとsearch sourceを別々に記録した。Pythonのlocked環境・Unityパッケージキャッシュ・取得済みCI artifactを再利用しているため、ネットワークを含む完全な空キャッシュ再取得の検証とはしない。

新規consumerの`uloop launch`はEditorを起動したが、600秒の接続待ちが`UNITY_STARTUP_TIMEOUT`で失敗した。ハーネスの要約はstdoutのJSON解析失敗を記録しており、保存したstderrに上記の具体的な原因がある。同じEditorの`status`はReadyを確認できたため、再起動せず別のcompileで継続確認する。launch成功へ書き換えない。

サンプルsceneをPlayで起動し、日本語の入力欄とボタンが描画されることを画像で確認した。GameViewへの合成MouseDown / MouseUpだけでは準備状態へ遷移した証拠を得られなかったため、ボタン操作成功とは数えない。Playを停止し、GameViewの最大化を解除した。画面操作の受け入れは引き続き未完了。

## 未完了の確認

- サンプルUI・モデル準備と読み込み・エラー表示・終了時解放の操作。
- 新規checkout / 空のconsumer内での変換・実モデル検索・画面操作による再現。導入とsource SHA / hash監査・配置は成功済み。
- UI画像と操作記録の保存、[利用者向け手順](model-preparation.md)を再現結果と照合。

M1の過去実測・UPM移行の過去回帰は [現在の状態](status.md)の証拠を維持する。この検索サンプルの実モデル合格の代わりには数えない。
