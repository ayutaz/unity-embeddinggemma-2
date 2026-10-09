# テキスト検索サンプルとモデル準備の実装計画

開始日: 2026-10-10。対象は利用者向け一覧の1（検索サンプル）と2（モデル取得・変換・配置手順）。[M2計画](m2-plan.md)のA / B / Cを進める。macOS・モバイル・公開リリースは後続であり、この作業でM2全体の完了とはしない。

## 2026-10-10の現在地

mainは`c7d11897e43d374fbb87f3761f65fb3faec60c7c`。PR #9 → #10 → #11は依存順の未統合PRであり、以下の検索実装はPR側の成果である。

| 段階 | 現在の状態 | 残る受け入れ確認 |
| --- | --- | --- |
| A | 実装・CI検証済み。PR #10の全8 job成功、Python4環境各123件。固定6文書 / 4queryの参照生成・全順位・同点ID順・hash監査を実装 | PRの統合と統合後CI |
| B | PR #11でサンプル・モデル準備・リソース解放を実装。CI全8 job成功、Python4環境各132件。Windowsの実モデル4条件は4 passed / failed 0 / skip 0、全queryの全6順位が一致 | 実際の画面入力・準備・検索・エラー表示・終了の操作記録と画像。API契約の合格だけで画面操作を完了扱いにしない |
| C | 取得・変換・配置・更新手順と空consumer導入CLIを実装。新規checkoutからのサンプル導入とCI成果物のhash監査・配置まで成功 | 新規consumer内のモデル変換・実モデル検索・画面操作による再現確認 |

実モデルの数値は[検証記録](m2-search-validation.md)と[Windows結果](results/m2-search-root-windows-20261010.json)を参照。fp32 / Float16重みの両方をCPU / GPUComputeで保存済みモデルから読み、各条件で6文書 + 4queryをPython参照へ照合した。

次はBの画面操作、Cの新規consumer再現、証拠・文書の整合確認、PR依存順の統合と統合後CIを進める。その後、macOS / iOS / Androidの環境確保と実モデル検証、commit / tag固定のGit URL導入、リリースへ進む。画面操作・consumer再現が終わるまで一覧1 / 2は進行中とする。

| 段階 | 実装 / 成果物 | 合格の証拠 |
| --- | --- | --- |
| A | 固定の日本語 / 英語文書・query、cosine順位計算と同点時ID順、検索のPython参照生成 | Python / C#のTDD red → green。Actionsで固定モデルrevision・prompt・tokenizer・128長の参照ベクトルと全順位を生成。出力のsource SHA・SHA-256を監査 |
| B | UPMのTextSearchサンプル、事前埋め込み・query入力・順位表示・CPU / GPU選択・モデル未準備 / 不正入力 / 実行失敗表示・終了時Dispose | 所有リソースとエラーの契約テスト。Windows Editorでfp32 / Float16重み × CPU / GPUComputeの全query順位をPython参照と照合、skip 0。UIの準備 → 入力 → 表示 → 終了を操作し画像を残す |
| C | Actions成果物の取得、監査、Sentis変換・配置・更新、サンプルの導入手順 | 重みを含まない新規checkout / 空のUnity consumerへサンプルを導入し、CI参照を配置して起動・検索を再現。revision・生成commit・hash、必要操作、失効時再生成を文書化 |

各挙動は先にテストを書き、意図した失敗を確認してから実装する。重いモデル参照生成・変換と全Python検証はActionsへ寄せ、ローカルでは新規契約の小さい確認と、クラウドで代替できない必要最小限のUnity CPU / GPU操作を行う。Pythonはtools/とuvのみ。

既存TextEmbedderとTextModelFileを使用し、同期・メインスレッド・batch 1 / length 128 / 768次元を維持する。最終ベクトルだけCPUへ戻して小さい文書群を全件比較する。GPU非対応時にCPUへ自動で切り替えず、利用者がbackendを選択する。

モデルは自動ダウンロードしない。利用者が明示的に用意した`.sentis`とtokenizerを読み、各検索で再保存・再量子化しない。重み・大きい生成結果はGit / LFSへ含めず、小さい実行要約と再現手順を記録する。

実装ブランチは`feat/m2-text-search`（PR #10）と`feat/m2-text-search-sample`（PR #11）。計画更新PR #9を基礎にする。mainへ直接pushせず、マージは依頼された範囲でCI・差分・競合を確認して行う。未実行・失敗・GPU skipは合格と分け、全段階の証拠が揃うまでこの作業を完了扱いにしない。
