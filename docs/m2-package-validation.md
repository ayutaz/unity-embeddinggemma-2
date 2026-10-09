# CI・main保護・UPM移行の検証

更新日: 2026-10-09。PR #6、ブランチ `feat/ci-upm-package`。
実測したコード: `b74180ec8dcdce99faf88dc129a58904c50a7946`。基準main: `8146107`。
確認済みPR head: `8a585bc634a210e5ffd9ceaa5d67b85f1c9316a3`。文書更新の新しいCIとこの過去実測を区別する。[現在の状態](status.md)を参照。
対象は利用者向け残タスク一覧の1（CI / main保護）と2（UPM化）。
**実装とWindows Editorの導入検証は完了。mainのworkflow / packageへの反映はPR merge待ち。**
検索サンプル、他環境、公開リリースは今回の範囲に含めない。

## 実装と結果

| 対象 | 確認済みの結果 |
| --- | --- |
| 全PR CI | paths filterなしの入口からPython・lint・実モデル参照・パッケージ監査を実行。集約Required CIは全4 jobのsuccessを要求。PR #6の全8チェック、文書のみPR #7の全8 job成功 |
| main保護 | GitHub APIでPR必須、Actions App 15368のRequired CI必須、strict、管理者適用、force push / 削除禁止、会話解決・linear historyを読み戻し |
| UPM | `com.ayutaz.embeddinggemma` / `0.1.0-pre.1`。Sentis 2.6.1 / Newtonsoft 3.2.2を直接依存として宣言 |
| 既存ソース | Runtime・契約テストとmetaの19ファイルを元mainのGit blobと比較し一致。元のAssets側Runtimeは削除、assembly名・GUID維持 |
| 元プロジェクト | compile合格、パッケージ契約29件と測定契約1件が合格 |
| 新規プロジェクト | 空のAssetsからUPMをローカルフォルダ依存で導入。依存解決・compile・パッケージ契約29件合格。URP依存なし |
| 実モデル回帰 | 元プロジェクトのM1 3件、公開API 2件が合格。tokenizer全15件、CPU / GPUCompute全15件、APIで再利用後の推論も確認 |
| 配布内容 | 重み・生成モデル・ネイティブプラグインを含めない。モデル不要のTestsとAPI文書を同梱 |

すべてのUnity成功結果はfailed / skipped / inconclusive = 0。
新規consumerの29件は小さいモデルのAPI契約であり、その環境で実モデルGPUまで実行した証拠ではない。
実モデルの証拠は別に実行した元プロジェクトの回帰結果。保存・量子化・性能測定全体は今回再実行していない。
[小さい数値レポートと19ファイルのソース対応](results/m2-package-windows-20261009.json)を保存した。

## TDD

| 対象 | Red | Green |
| --- | --- | --- |
| 必須判定 | ci module未実装のimport error | 成功・失敗・skip・cancel・欠落・不正JSONの14件合格 |
| 全PRの入口 | ci.yml欠落で3 failed | 全PR起動・reusable構成・always集約・Secrets境界を確認 |
| UPM監査 / consumer | package module未実装のimport error | 不正依存・モデル / native混入・UnityEditor参照・meta / GUID・既存プロジェクト保護を確認 |
| 実パッケージ | 未移行で1 failed / 17 passed | 移行後18件合格 |
| 埋め込み依存 | 元manifestにpackage依存がなくKeyError | versionを明示し、Package Manager再解決後にcompile / contracts合格 |
| 必須監査 | package job未接続で1 failed | CIの4番目の必須jobへ接続 |
| 監査の失敗伝播 | tee利用時の明示bash不足で1 failed | GitHubのbash -eo pipefailを使用 |
| mainポリシー | main-protection.json欠落で1 failed | 宣言を追加し、サーバー適用・読み戻し |

全PythonテストはActionsの4環境で実行する。ローカルでは追加範囲の小さいred / greenのみ実行。
CI導入時のrun [37809737222](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37809737222)は成功したが、UPM監査の追加前。
UPM実装commitのrunは [37811183097](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37811183097)。
Python 4環境各102件、lint、実モデル15件（最小cosine `0.9999998807907104`）、パッケージ監査、Required CIがすべて成功した。
確認済みhead `8a585bc` の [run 37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)も同じ8 jobがすべて成功した。
実際のcheckout `9ccff937ae9fc675c12aee8e1533d08cc0502e9a` とPR headのtree一致を確認済み。

文書のみPRの実行証拠は [PR #7](https://github.com/ayutaz/unity-embeddinggemma-2/pull/7) / [run 37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)。
baseは `feat/ci-upm-package`、headは `9633a1ca8d5f3a01eff427599187329c2447b68c`。
差分は `docs/ci-docs-only-probe.md` 1ファイル・6行追加のみ。全8 job成功、Python4環境各102件、実モデル15件・同じ最小cosine、Required CI成功をログで確認した。
結果をPR #6本文に記録し、検証用PR #7は未マージで閉じた。PR #6の全体diffを文書のみと扱った検証ではない。

## 別プロジェクトで再現

リポジトリのtools/から、存在しないか空のディレクトリを指定する。既存プロジェクトは上書きしない。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.package
uv run --locked python -m embeddinggemma_tools.package --consumer ../artifacts/upm-consumer --automation
uv run --locked python -m embeddinggemma_tools.unity --suite compile --launch --project ../artifacts/upm-consumer --output ../artifacts/unity-harness/upm-consumer
```

初回のlaunch readinessがタイムアウトした場合は失敗ログを保持し、起動済みEditorの状態を確認する。
`--launch`を外したcompileや次のrun-testsの結果を別に取得し、launchを成功扱いにしない。
起動済みconsumerで、リポジトリルートから:

```powershell
& artifacts/uloop/bin/uloop.exe --project-path artifacts/upm-consumer run-tests --filter-type assembly --filter-value EmbeddingGemma.Package.Editor.Tests --test-mode EditMode --unsaved-changes fail
```

consumerにはパッケージ・Test Frameworkだけを宣言し、`--automation`指定時だけ検証用uloopを追加する。
URP・元プロジェクトのAssets・モデル・測定fixtureをコピーしない。
`--git-revision <40桁commit>`でGit subfolder依存のmanifestも生成できる。今回Editorで実行した導入経路はローカルフォルダ依存。
Git URLでの消費側Editor実行・公開tag導入はまだ検証していない。

## 失敗とCLIの注意

- 最初の元プロジェクトcompileは新規embedded packageがまだ認識されず16 errors。依存を明示して `PackageManager.Client.Resolve()` を実行。domain reloadでCLI接続が切れたが、lockのembedded登録と後続compile / testsの成功を確認した。
- consumerの直接起動は子プロセスに `ALLUSERSPROFILE` がなく、UPMのpath undefinedで依存解決に失敗。自分で作成した検証用Editorだけを再起動し、既存Pythonハーネスの環境補完で解決した。
- 新規consumerのuloop launch readinessはタイムアウト。テストコマンドは別にcompile成功と29 passedを返した。post-compile warmup skipped警告はNUnitのSkippedCount=0と区別し、launch成功としては数えない。
- `status`のReady、DLL生成、CLI設定ファイルの存在だけではテスト成功と判定せず、実際のrun-tests結果を採用した。

元ログと実行JSONは `artifacts/ci-upm/`、`artifacts/unity-harness/upm-*/` に保持する。大きいログとモデルはGitへ入れない。
CLI起動待ちの問題は検証用ツールの制約として残し、パッケージRuntimeの成功と分離する。

## 残作業

PR #6は確認したheadでCI成功・競合なし。merge依頼後にmainへ統合し、統合後のCIとpackage反映を確認する。
PR #5の文書変更もPR #6に含まれ、#5は未マージで閉じた。文書のみの検証用PR #7もクローズ済み。
その後は [M2計画](m2-plan.md)のテキスト検索サンプル、モデル配布手順、macOS / iOS / Android、リリースへ進む。
`Samples~`には現状説明だけを置き、未実装の検索シーンをPackage Managerへ登録しない。
