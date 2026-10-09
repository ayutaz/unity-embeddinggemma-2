# 現在の状態と残タスク

確認日: 2026-10-09。GitHubのPR・Checks・main保護、package manifest、既存のUnity実行結果を照合した。
文書更新は `docs/refresh-verified-ci-upm` ブランチ。文書の更新をUnityの再実行やmain統合の証拠として扱わない。

## mainと作業ブランチ

| 対象 | 確認した状態 |
| --- | --- |
| main | `8146107aa77904050c6235866c0cf79ecc80034c`。PR #1〜#4統合済み、Windows EditorのM1完了 |
| [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6) | CI整備・UPM化は実装・検証済み。確認したheadは `8a585bc634a210e5ffd9ceaa5d67b85f1c9316a3`、全8チェック成功・競合なし、未マージ |
| main保護 | サーバー設定済み。PR必須、strict Required CI（GitHub Actions App 15368）、管理者適用、force push / 削除禁止、linear history・会話解決。人手承認数0 |
| [PR #5](https://github.com/ayutaz/unity-embeddinggemma-2/pull/5) | 文書変更をPR #6へ含め、未マージでクローズ |
| [PR #7](https://github.com/ayutaz/unity-embeddinggemma-2/pull/7) | 文書のみPRのCI確認用。検証後に未マージでクローズ |

main保護の設定と、mainのworkflow / package更新は別である。全PRの新CIとUPMコードはPR #6にあり、mainへの反映はマージ待ち。
[開発規則](../AGENTS.md)に従い、マージは明示的な依頼があるまで行わない。

## 検証済みの範囲

| 対象 | 証拠・結果 | 限界 |
| --- | --- | --- |
| M1 | [完了検証](m1-completion-validation.md)。tokenizer全15件一致、fp32 / Float16重みのCPU / GPUCompute各15件、保存・再読み込み・時間 / メモリ測定合格 | Windows Editor 6000.3.16f1 / Sentis 2.6.1、固定モデル・batch 1 / length 128 / 768次元 |
| PR #6のCI | [run 37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)。Ubuntu / Windows × Python 3.13 / 3.14各102件、lint、実モデルPython15件、パッケージ監査、Required CIの全8 job成功 | 実際のcheckoutは `9ccff937ae9fc675c12aee8e1533d08cc0502e9a`。確認したPR headとtree一致。Python照合はSentis実行ではない |
| 文書のみPR | [run 37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)。PR #7はMarkdown 1ファイル・6行追加だけで全8 job成功 | baseは `feat/ci-upm-package`、headは `9633a1ca8d5f3a01eff427599187329c2447b68c`。古いmainへの統合済みという意味ではない |
| UPM導入 | [パッケージ検証](m2-package-validation.md)。別の空のUnityプロジェクトでローカルフォルダ依存解決・compile・契約29件成功、URP依存なし | 新規consumerのテストはモデル不要。Git URL導入とconsumer内の実モデルGPUは未実行 |
| UPM移行後の回帰 | 元プロジェクトで契約29件・測定契約1件、実モデルM1 3件・API 2件成功、failed / skipped / inconclusive = 0 | 保存・量子化・ベンチマーク全体を移行後に再実行した結果ではない。元M1実測を維持 |

パッケージは `com.ayutaz.embeddinggemma` / `0.1.0-pre.1`、未リリース。
Runtimeと契約テストの19ファイルは移行前とGit blob一致、assembly名・GUIDを維持。
直接依存はSentis 2.6.1 / Newtonsoft JSON 3.2.2。モデル・ネイティブプラグイン・URP・uloop・Pythonを配布Runtimeへ含めない。

## 残タスクの順序

1. PR #6のマージ依頼後に統合し、mainのCI成功とUPMコードの反映を確認する。
2. テキスト検索サンプルをTDDで実装。固定文書・queryの順位をPython参照と照合し、UI操作・エラー表示・リソース解放を確認する。
3. 固定revision・hashに基づくモデル取得 / 変換 / 配置 / 更新の利用者向け手順を整え、新規checkoutで再現する。既存の変換・監査CLIを基礎にする。
4. macOS Editor・iOS・Androidのrunner / toolchain / 実機を確保し、精度・backend・速度・メモリを実測する。環境の確保は2 / 3と並行して進める。
5. Git URLのcommit / tag固定導入とサンプル起動、文書・ライセンス・CHANGELOGを確認し、テキスト版をリリースする。ここまででM2完了。
6. M3で画像エンコーダ・GPU画像前処理・テキスト→画像検索を実装・照合する。
7. M4で音声エンコーダ・GPUメル前処理・テキスト→音声検索を実装・照合する。

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
package内の文書は導入 / API / 未リリースとサンプル未実装を説明する。AGENTS.mdとApacheライセンスは内容を確認し、規則・条文を維持する。
