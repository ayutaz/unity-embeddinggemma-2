# テキスト検索サンプルとモデル準備の実装計画

開始日 / 完了確認日: 2026-10-10。対象は利用者向け一覧の1（検索サンプル）と2（モデル取得・変換・配置手順）。[M2計画](m2-plan.md)のA / B / Cは実装・検証・統合済み。macOS・モバイル・公開リリースは後続であり、M2全体の完了とはしない。

## 2026-10-10の現在地

確認基準mainは`4ee0cbbde07fc99ca87d33ed52a44e6100ca0b0c`。PR #9（計画）・#10（検索基盤）・#11（サンプル・モデル準備）・#12（統合記録）は統合済み。PR #11の最終head CI run 37974713580、統合後main CI run 37975026704、PR #12統合後CI run 37975720534はすべて全8 job成功。実測のsource SHAは結果JSONを維持する。

| 段階 | 現在の状態 | 残る受け入れ確認 |
| --- | --- | --- |
| A | mainへ統合済み。固定6文書 / 4queryの参照生成・全順位・同点ID順・hash監査を実装 | 完了。統合後CI run 37964228515成功 |
| B | PR #11で統合。Windows実モデル4条件は4 passed / failed 0 / skip 0、全順位一致。Game View入力で準備・日英検索・空入力・解放・モデル欠落を確認し画像を保存 | 完了。最終headと統合後CI全8 job成功 |
| C | 別checkoutから短いパスの空consumerへ導入。依存解決・契約38件・sample 18件・モデル変換とhash監査、改善後の実モデル4 passed / skip 0・全順位一致・CLI完了応答、CPUの日英検索・空入力・解放を確認 | 完了。PR #11統合済み。従来の中断履歴と未解決ログを維持 |

実モデルの数値は[検証記録](m2-search-validation.md)と[Windows結果](results/m2-search-root-windows-20261010.json)を参照。fp32 / Float16重みの両方をCPU / GPUComputeで保存済みモデルから読み、各条件で6文書 + 4queryをPython参照へ照合した。

一覧1 / 2の実装・consumer再現・手順を確認済み。検証済みモデルの再利用をTDDで追加し、再import・再変換・全体Refreshと外部Accelerator待機を削減した。準備203秒 → 175秒は単回観測で、domain reloadの原因は未確定。以前の中断履歴を保持し、改善後の成功を別記した。次はmacOS / iOS / Androidの環境確保と実測、Git URL導入、リリース。hash確認の負荷と未解決allocation / fontログも追跡する。

以下は完了したA / B / Cの受け入れ条件。後続Dの順序と未検証環境は[M2計画](m2-plan.md)を参照する。

| 段階 | 実装 / 成果物 | 合格の証拠 |
| --- | --- | --- |
| A | 固定の日本語 / 英語文書・query、cosine順位計算と同点時ID順、検索のPython参照生成 | Python / C#のTDD red → green。Actionsで固定モデルrevision・prompt・tokenizer・128長の参照ベクトルと全順位を生成。出力のsource SHA・SHA-256を監査 |
| B | UPMのTextSearchサンプル、事前埋め込み・query入力・順位表示・CPU / GPU選択・モデル未準備 / 不正入力 / 実行失敗表示・終了時Dispose | 所有リソースとエラーの契約テスト。Windows Editorでfp32 / Float16重み × CPU / GPUComputeの全query順位をPython参照と照合、skip 0。UIの準備 → 入力 → 表示 → 終了を操作し画像を残す |
| C | Actions成果物の取得、監査、Sentis変換・配置・更新、サンプルの導入手順 | 重みを含まない新規checkout / 空のUnity consumerへサンプルを導入し、CI参照を配置して起動・検索を再現。revision・生成commit・hash、必要操作、失効時再生成を文書化 |

各挙動は先にテストを書き、意図した失敗を確認してから実装する。重いモデル参照生成・変換と全Python検証はActionsへ寄せ、ローカルでは新規契約の小さい確認と、クラウドで代替できない必要最小限のUnity CPU / GPU操作を行う。Pythonはtools/とuvのみ。

既存TextEmbedderとTextModelFileを使用し、同期・メインスレッド・batch 1 / length 128 / 768次元を維持する。最終ベクトルだけCPUへ戻して小さい文書群を全件比較する。GPU非対応時にCPUへ自動で切り替えず、利用者がbackendを選択する。

モデルは自動ダウンロードしない。利用者が明示的に用意した`.sentis`とtokenizerを読み、各検索で再保存・再量子化しない。重み・大きい生成結果はGit / LFSへ含めず、小さい実行要約と再現手順を記録する。

実装ブランチは`feat/m2-text-search-sample`（PR #11）で、mainへの直接pushを行わず、CI・差分・競合を確認して依頼に基づき統合した。A / B / Cの証拠は[検証記録](m2-search-validation.md)を参照。未実行・失敗・GPU skipは合格と分け、後続Dの合格とは扱わない。
