# 現在の状態と残タスク

確認日: 2026-10-10。GitHubのマージ結果・統合後CI・main保護、package manifest、既存のUnity実行結果を照合した。
最新の検索実装・計画更新は `feat/m2-text-search-sample` ブランチ。M1の履歴を維持し、今回のWindows検索実測を別の証拠として追加した。mainと未統合PRを区別する。

## mainと作業ブランチ

| 対象 | 確認した状態 |
| --- | --- |
| main | `be791e00a1b344c5453f70037d7116184169c17f`。PR #1〜#4・#6・#8〜#10統合済み。Windows EditorのM1、CI・UPM化・検索基盤を反映済み |
| [PR #9](https://github.com/ayutaz/unity-embeddinggemma-2/pull/9) | 計画更新を2026-10-10 02:05:30 JSTにsquash merge。commit `de82a29fc737bca0478c87d06ffcc4d94fb07e64`、統合後CI run 37963795561成功 |
| [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10) | 検索基盤を2026-10-10 02:09:08 JSTにsquash merge。commit `be791e00a1b344c5453f70037d7116184169c17f`、統合後CI run 37964228515成功 |
| [PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11) | mainをbaseとするサンプル・モデル準備手順。OPEN。base更新後のhead `bc2721d36f13a4df50720bf86d37c3b7f839e830`でCI全8 job成功。domain reload高速化を検証中 |
| [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6) | CI整備・UPM化を2026-10-10 00:07:13 JSTにsquash merge。commit `5ae4e8c29a3a4c8639bd94f7848e3683d063f4d2` |
| [PR #8](https://github.com/ayutaz/unity-embeddinggemma-2/pull/8) | 文書更新を最新mainへ更新し、全8チェック成功後に2026-10-10 00:11:01 JSTにsquash merge。commit `c7d11897e43d374fbb87f3761f65fb3faec60c7c` |
| main保護 | サーバー設定済み。PR必須、strict Required CI（GitHub Actions App 15368）、管理者適用、force push / 削除禁止、linear history・会話解決。人手承認数0 |
| [PR #5](https://github.com/ayutaz/unity-embeddinggemma-2/pull/5) | 文書変更をPR #6へ含め、未マージでクローズ |
| [PR #7](https://github.com/ayutaz/unity-embeddinggemma-2/pull/7) | 文書のみPRのCI確認用。検証後に未マージでクローズ |

全PRを対象とするCI入口、UPMコードと検索APIはmainへ統合済み。サンプルとモデル準備手順はPR #11側で未統合。mainへの直接pushや保護の迂回は行っていない。[開発規則](../AGENTS.md)に従い、マージは依頼された範囲でCI・差分・競合を確認して行う。

## 検証済みの範囲

| 対象 | 証拠・結果 | 限界 |
| --- | --- | --- |
| M1 | [完了検証](m1-completion-validation.md)。tokenizer全15件一致、fp32 / Float16重みのCPU / GPUCompute各15件、保存・再読み込み・時間 / メモリ測定合格 | Windows Editor 6000.3.16f1 / Sentis 2.6.1、固定モデル・batch 1 / length 128 / 768次元 |
| 統合後mainのCI | [run 37964228515](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37964228515)。main `be791e0`の全8 job成功、Python4環境各123件、実モデルPython M1 15件 + 検索10件、lint・パッケージ監査・Required CI成功 | Python照合・パッケージ静的監査であり、新たなUnity実行ではない |
| PR #6のCI | [run 37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)。Ubuntu / Windows × Python 3.13 / 3.14各102件、lint、実モデルPython15件、パッケージ監査、Required CIの全8 job成功 | 実際のcheckoutは `9ccff937ae9fc675c12aee8e1533d08cc0502e9a`。確認したPR headとtree一致。Python照合はSentis実行ではない |
| 文書のみPR | [run 37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)。PR #7はMarkdown 1ファイル・6行追加だけで全8 job成功 | baseは `feat/ci-upm-package`、headは `9633a1ca8d5f3a01eff427599187329c2447b68c`。古いmainへの統合済みという意味ではない |
| UPM導入 | [パッケージ検証](m2-package-validation.md)。別の空のUnityプロジェクトでローカルフォルダ依存解決・compile・契約29件成功、URP依存なし | 新規consumerのテストはモデル不要。Git URL導入とconsumer内の実モデルGPUは未実行 |
| UPM移行後の回帰 | 元プロジェクトで契約29件・測定契約1件、実モデルM1 3件・API 2件成功、failed / skipped / inconclusive = 0 | 保存・量子化・ベンチマーク全体を移行後に再実行した結果ではない。元M1実測を維持 |
| 検索サンプルのCI | [run 37959339466](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37959339466)。PR #11の実装headで全8 job成功、Python4環境各132件、M1 15入力 + 検索10入力のexport照合合格 | 実checkout `13f194266162d2a728b2b2a0da9e9e58ae314ff6`。Python照合をSentis GPUの成功に数えない |
| Windows検索実モデル | [検索検証記録](m2-search-validation.md) / [全順位と数値](results/m2-search-root-windows-20261010.json)。fp32 / Float16重み × CPU / GPUComputeの4 passed、failed / skipped / inconclusive = 0。固定6文書 / 4queryの全順位一致 | PR側のサンプル、元の検証プロジェクト。他環境や任意入力の品質保証ではない |
| 検索画面操作 | [操作記録](results/m2-search-ui-windows-20261010.json)。Game View入力イベントで準備・日本語 / 英語検索・空入力・解放・モデル欠落を確認。画面表示の不具合をred → greenで修正 | 元の検証プロジェクト。新規consumerの画面操作は残る |
| 新規consumer | 別checkoutから空の `artifacts/c` へ導入。compile・パッケージ契約38件・サンプル5件・モデル変換成功。実モデル4条件の数値と全順位も保存 | テスト後のdomain reload中に停止、CLI完了応答なし。ハーネス全体は失敗として保持し、NUnit件数は未取得 |

パッケージは `com.ayutaz.embeddinggemma` / `0.1.0-pre.1`、未リリース。
Runtimeと契約テストの19ファイルは移行前とGit blob一致、assembly名・GUIDを維持。
直接依存はSentis 2.6.1 / Newtonsoft JSON 3.2.2。モデル・ネイティブプラグイン・URP・uloop・Pythonを配布Runtimeへ含めない。

## 残タスクの順序

1. モデル準備の検証済みキャッシュ再利用・不要なimport / 全体Refreshの削減、到達できないAcceleratorの除外を検証する。domain reloadの原因は未確定。元プロジェクトの画面操作と実モデル4条件は確認済み。
2. 新規consumerで改善後の実行完了と画面操作を確認し、[モデル取得 / 変換 / 配置 / 更新手順](model-preparation.md)を確定する。停止前の数値照合と未取得のCLI完了応答を区別する。完了後にPR #11を扱い、統合後CIを確認する。PR #9 / #10は統合済み。
3. macOS Editor・iOS・Androidのrunner / toolchain / 実機を確保し、精度・backend・速度・メモリを実測する。環境の確保は1 / 2と並行して進める。
4. Git URLのcommit / tag固定導入とサンプル起動、文書・ライセンス・CHANGELOGを確認し、テキスト版をリリースする。ここまででM2完了。
5. M3で画像エンコーダ・GPU画像前処理・テキスト→画像検索を実装・照合する。
6. M4で音声エンコーダ・GPUメル前処理・テキスト→音声検索を実装・照合する。

詳細な依存・受け入れ条件は [M2計画](m2-plan.md)、全体のゴールは [ゴール](goal.md)。

## 未実行と失敗の扱い

- macOS / Windows Player / iOS / Android / WebGPUは未検証。Editorや小さいモデルの合格で置き換えない。
- クラウドUnityは任意手動Linux CPU検証。Secrets / Variablesは未登録、Editor job未実行。必要なのはこのクラウド経路の利用時で、検索サンプルなどの開発を止める条件ではない。
- 新規consumerのuloop launch readinessはタイムアウト。後続run-testsのcompile・29 passedとは分けて記録し、launch成功へ書き換えない。
- GPU skipやCPU代替をGPU成功として扱わない。モデルと大きいartifactはGit管理外、LFSを使わずActionsで再生成する。

## 文書の読み方

README、ゴール、M2計画、CI、API、automationは現在の利用・作業手順。
M1の各plan / validationは実行した段階の履歴で、当時のcommit・数値・失敗を維持する。
[技術調査](technical-approach.md)は採用済みのテキスト経路と未実装の画像 / 音声設計を区別する。
[新規性調査](embeddinggemma-2-unity-novelty.md)は2026-10-07の調査に2026-10-09の限定検索を追記したもので、先行実装の不存在を保証しない。
package内の文書は導入 / API / 未リリースと、PR側で実装したサンプルの使い方を説明する。AGENTS.mdとApacheライセンスは規則・条文を維持する。
