# テキスト検索サンプルとモデル準備の実装計画

開始日: 2026-10-10。対象は利用者向け一覧の1（検索サンプル）と2（モデル取得・変換・配置手順）。[M2計画](m2-plan.md)のA / B / Cを進める。macOS・モバイル・公開リリースは後続であり、この作業でM2全体の完了とはしない。

## 2026-10-10の現在地

mainは`be791e00a1b344c5453f70037d7116184169c17f`。PR #9（計画）と#10（検索基盤）は統合済みで、統合後CIも成功。PR #11（サンプル・モデル準備）はmainをbaseとする未統合PR。

| 段階 | 現在の状態 | 残る受け入れ確認 |
| --- | --- | --- |
| A | mainへ統合済み。固定6文書 / 4queryの参照生成・全順位・同点ID順・hash監査を実装 | 完了。統合後CI run 37964228515成功 |
| B | PR #11で実装。Windows実モデル4条件は4 passed / failed 0 / skip 0、全順位一致。実際のGame View入力イベントで準備・日本語 / 英語検索・空入力・解放・モデル欠落を確認し画像を保存 | PR #11の残変更に対するCIと統合後CI |
| C | 新規checkoutから短いパスの空consumerを作成し、依存解決・契約38件・サンプル5件・モデル変換が成功。実モデル4条件の数値と全順位を保存 | テスト後のdomain reload中に停止されたためCLI完了応答は未取得。consumer画面操作・高速化の確認・手順確定が残る |

実モデルの数値は[検証記録](m2-search-validation.md)と[Windows結果](results/m2-search-root-windows-20261010.json)を参照。fp32 / Float16重みの両方をCPU / GPUComputeで保存済みモデルから読み、各条件で6文書 + 4queryをPython参照へ照合した。

次は不要なモデル再import・再変換・全体Refreshを減らす改善をTDDで検証する。検証用consumerでは、到達できない外部Acceleratorの待機を避けるためプロジェクト単位でキャッシュを無効にする。domain reloadの原因は断定していない。停止前の数値結果とCLIの未完了を別記し、改善後のconsumer画面操作・実行完了を確認する。証拠・文書を整合させ、PR #11と統合後CIを確認するまで一覧1 / 2は進行中。

その後、macOS / iOS / Androidの環境確保と実モデル検証、commit / tag固定のGit URL導入、リリースへ進む。

| 段階 | 実装 / 成果物 | 合格の証拠 |
| --- | --- | --- |
| A | 固定の日本語 / 英語文書・query、cosine順位計算と同点時ID順、検索のPython参照生成 | Python / C#のTDD red → green。Actionsで固定モデルrevision・prompt・tokenizer・128長の参照ベクトルと全順位を生成。出力のsource SHA・SHA-256を監査 |
| B | UPMのTextSearchサンプル、事前埋め込み・query入力・順位表示・CPU / GPU選択・モデル未準備 / 不正入力 / 実行失敗表示・終了時Dispose | 所有リソースとエラーの契約テスト。Windows Editorでfp32 / Float16重み × CPU / GPUComputeの全query順位をPython参照と照合、skip 0。UIの準備 → 入力 → 表示 → 終了を操作し画像を残す |
| C | Actions成果物の取得、監査、Sentis変換・配置・更新、サンプルの導入手順 | 重みを含まない新規checkout / 空のUnity consumerへサンプルを導入し、CI参照を配置して起動・検索を再現。revision・生成commit・hash、必要操作、失効時再生成を文書化 |

各挙動は先にテストを書き、意図した失敗を確認してから実装する。重いモデル参照生成・変換と全Python検証はActionsへ寄せ、ローカルでは新規契約の小さい確認と、クラウドで代替できない必要最小限のUnity CPU / GPU操作を行う。Pythonはtools/とuvのみ。

既存TextEmbedderとTextModelFileを使用し、同期・メインスレッド・batch 1 / length 128 / 768次元を維持する。最終ベクトルだけCPUへ戻して小さい文書群を全件比較する。GPU非対応時にCPUへ自動で切り替えず、利用者がbackendを選択する。

モデルは自動ダウンロードしない。利用者が明示的に用意した`.sentis`とtokenizerを読み、各検索で再保存・再量子化しない。重み・大きい生成結果はGit / LFSへ含めず、小さい実行要約と再現手順を記録する。

作業ブランチは`feat/m2-text-search-sample`（PR #11）、baseはmain。mainへ直接pushせず、マージは依頼された範囲でCI・差分・競合を確認して行う。未実行・失敗・GPU skipは合格と分け、全段階の証拠が揃うまでこの作業を完了扱いにしない。
