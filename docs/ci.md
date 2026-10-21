# CIと検証の実行方針

更新: **2026-10-21**。Windows版[v0.1.0の公開](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0)と実GPU受け入れは完了しています。必要CIは全15 job成功・13 archiveの独立照合で確認しました。[現状](status.md) / [正式版の記録](results/windows-v010-release-20261021.json) / [今後の計画](roadmap.md)。

以下のworkflow契約と過去runはそれぞれのsourceに対応します。CPU参照、Unity build、実モデルGPU、端末の実行を分け、skip・未実行・cancelを合格へ数えません。

## 現行の必須CI

`.github/workflows/ci.yml`はPR・mainへのpush・手動起動で実行します。まず実Git差分とeventのbase / head / sourceを照合し、次の経路を選びます。

| 変更 | 必須の検査 |
| --- | --- |
| `README.md`と`docs/`内のMarkdownだけ（`docs/privacy.md`を除く） | 差分・リンク・元の送信祖先すべて・公開profile・実PR本文のprivacy検査、workflow lint |
| Runtime、Editor / sample、tests、参照 / モデル、依存、workflow、privacy規則、その他のpath | 従来のPython3.13 / 3.14 × Windows / Ubuntu、workflow lint、package監査、実モデル参照・exportと固定音声6条件のONNX比較 |
| 手動起動、比較元がない場合、mode変更、renameでcodeを隠す場合、判定できない差分 | 完全な既存回帰。分類job自体が失敗した場合はRequired CIも失敗 |

集約checkの名前は`Required CI`です。別のjobでGit差分を再計算し、分類結果・対象source・差分digestの一致を確認します。必要なcheckの欠落・failure・cancel・予期しないskipは失敗です。文書経路ではモデルjobは明示的に未実行と記録し、成功したモデル / GPU検証とは数えません。

この振り分けはPR #118で実装しています。[文書専用の実run](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39555715582)では、Required CIまで39秒で完了しました。
[sourceと実jobの照合記録](results/documentation-CI-route-readback-20261021.json)では、モデル3 jobの明示的な未実行と`model_execution=false`を確認しています。
文書経路はrunner待ちを除く10分以内を目標にしますが、1回の実測で将来の時間を保証するものではありません。コード変更を含む完全回帰と最終headの受け入れは、別途完了を確認します。

完全回帰のCPU参照生成・比較はUnity RuntimeのCPU推論設定ではありません。Unity実GPU・UI・Player・実機の受け入れは別に実施します。CI成功だけで正式公開条件を満たしたとしません。branch protectionと、各artifactの公開前guardは変更していません。

### 文書経路の結果を読む

`documentation`の`model_execution=false`は、リンク・送信履歴・公開payloadを検査し、モデルを実行していないことを表します。`Required CI`の成功を、新しい実モデルやGPUの検証成功へ読み替えないでください。コード変更などの完全回帰では、元の各jobの実結果を別に確認します。

## 固定参照の取得と保持

現行のモデル回帰は[明示的な配布入力とaudit](native-regression-distribution.md)を使用します。
元のartifact ID・期限と後続の配布bytesを同一扱いにせず、workflowが選択したproducer / source / size / SHA256を記録します。
古い保存artifactは期限だけで存在を推測せず、必要な原本の全bytes・hashを確認します。
新規実行の前に、実行するworkflowと入力契約を確認してください。参照原本・許容差の変更や、配布provenanceを旧raw loaderへ混ぜることはしません。


## 2026-10-14時点のCIとGPU受け入れの区別

2026-10-14、PR #81のコード基準 `a237dee2874cdb0fbcdf4fe333aebae7193ad390` のCI `38571522044` と前headの `38569646238` は停止指示によりcancelled。最新headのRequired CI合格を主張しない。画像role参照のActions `38562190876` はPython18件合格、CPU文脈matrix `38563722587` は診断実行完了で、Unity / GPU全chainの合格ではない。

通常CIの `model-reference` は実重み生成を含むため、文書だけのpushでも自動開始し得る。実モデル検証停止中は不要な重いrunを識別してcancelし、cancelを成功へ読み替えず記録する。workflowとRequired CI規則は今回変更しない。新しいGPU検証が必要な場合に限り、runner / ライセンスと[今後の計画](roadmap.md)の開始条件を再確認する。CPU Pythonの参照生成は独立期待値用で、Unityモデル推論をCPUへ切り替える許可ではない。

更新日: 2026-10-10。確認基準main: `4ee0cbb`（PR #12統合後）。
CI・UPM監査・検索基盤・サンプル・準備最適化はmain統合済み。[CI run 37975720534](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975720534)の全8 job成功を確認した。

PR #31の旧checkpoint（2026-10-10）:
更新日: 2026-10-10。確認基準main: `6f55a85`（PR #30統合後）。
CI・UPM監査・検索基盤・サンプル・準備 / cache・Editor font寿命修正はmain統合済み。[main CI run 38031441901](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38031441901)の全8 job成功。draft PR #31の実装checkpoint `cd6b8bf`も[CI run 38040187805](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38040187805)全8 job成功。通常表示sample Playerの終了クラッシュは未解決で、CI成功だけでmergeやM2完了としない。

このPR runの実checkoutはmerge ref `3a8f52e`（head `cd6b8bf` + base `6f55a85`）。ログでPython4環境各253 passedを確認した。実Unity・GPU・モバイル実機を実行した8 jobではない。追加の文書headは同じ必須CIをPRで確認する。
文書のみのPR #7も全8 job成功後に未マージで閉じた。[現在の状態](status.md)を参照。
GitHub側のmain保護は設定済み。M1のWindows実測は [完了検証](m1-completion-validation.md)、UPM移行の結果は [パッケージ検証](m2-package-validation.md)を参照。

## 必須CIの導入時の記録

以下はPR #6 / #7当時の導入記録です。変更分類を加えた現行の経路・skipの判定は上記を参照してください。
`.github/workflows/ci.yml`を入口とし、全PR・main push・手動実行を対象にしました。
導入時は文書のみのPRも同じ完全な検証を実行しました。以下のreusable workflowは単独のPR / pushトリガーを持たず、重複実行を避けます。

| workflow / job | 実行内容 | 判定 |
| --- | --- | --- |
| `python-tests.yml` / `python` | Ubuntu / Windows × Python 3.13 / 3.14、uvによるオフラインpytest | matrix全環境の成功が必要 |
| `workflow-lint.yml` / `workflow-lint` | actionlint 1.7.12 / ShellCheck | 成功が必要 |
| `model-reference.yml` / `model-reference` | 固定revisionの実モデル取得、M1の15入力と検索6文書 / 4queryのPython参照生成・保存済み`.pt2`照合・検索順位確認 | 成功が必要。Sentis実行の代わりにはしない |
| `package-validation.yml` / `package` | manifest・直接依存・Runtime参照・GUID・配布禁止ファイルの監査、別consumerのmanifest生成 | 成功が必要。Unityを実行した結果ではない |
| `ci.yml` / `Required CI` | 上記4 jobの結果を `always()` で集約 | success以外、欠落、skipped、cancelled、不明値は失敗 |

必須チェック名は `Required CI`。GitHub Actions App ID `15368`を提供元に固定する。
集約処理は `tools/embeddinggemma_tools/ci.py`、構成と失敗判定はTDDで検証する。
Secretsは渡さず、`pull_request_target`は使わない。外部forkの実行承認はGitHubの制御に従う。

導入前のpaths filter付きworkflowでは文書のみPRにcheckが発行されなかった。
この導入前の問題はPR #6の統合で解消した。古いmainを基準にした作業ブランチは最新mainへ更新し、対象headのRequired CIを確認する。
PR #6は文書PR #5の変更も含み、重複する#5は未マージで閉じた。
PR #7はPR #6のブランチをbaseにした実PRで、Markdown 1ファイル・6行追加だけでもRequired CIを含む全8 jobが実行・成功した。
検証後に閉じており、mainへ統合したPRではない。

## main保護とPR運用

mainへの直接pushは禁止。作業ブランチからPRを提出し、mergeは依頼があるまで行わない。
GitHubのbranch protectionを2026-10-09に設定し、2026-10-10にもAPIで次を読み戻して確認した。

- PR必須。単独開発でも運用できるよう人手の必須承認数は0。
- `Required CI`必須、GitHub Actions提供元に固定、baseに対して最新であることを要求。
- 管理者にも保護を適用。force pushとbranch削除を禁止。
- 未解決会話の解決とlinear historyを要求。

設定の単一ソースは [.github/main-protection.json](../.github/main-protection.json)。
将来再適用する際は管理権限で次を実行し、必ずAPIを読み戻す。

```powershell
gh api --method PUT repos/ayutaz/unity-embeddinggemma-2/branches/main/protection --input .github/main-protection.json
gh api repos/ayutaz/unity-embeddinggemma-2/branches/main/protection
```

CIの成功は自動merge・自動releaseを意味しない。モデル・大きいログをGitへ含めない。

## 参照生成とソース対応

モデルは `google/embeddinggemma-2`、revision `914f7f89142e33e77833254d9c9b90c3cef7303b`。
batch 1 / length 128 / fp32の`.pt2`を生成し、保存済みexportをPython eagerと全15件比較する。閾値は0.999999。
生成はtools/のuvで行い、失敗時の部分成果物を成功扱いにしない。開始時に以前の成功レポートを無効化する。

3日保持の `m1-reference-<checkout-SHA>` は `reference.json`、`search-reference.json`、`tokenizer.json`、`model.pt2`、`export-validation.json`を含む。軽量な `search-reference-<checkout-SHA>` は検索参照とexport報告のみを含む。互換モデルの再利用は[モデル準備手順](model-preparation.md)で生成元とhashを別々に監査する。
pytestとパッケージ監査artifactは7日保持。失効時はActionsで再生成し、小さい数値要約と再現手順をGitへ残す。

PRでは既定のmerge refをcheckoutする。`github.sha` / metadataのsource commitと、PR head SHAは区別する。
run URL、PR head、実際のcheckout SHA、model revision、ファイルhashを揃えて監査する。
[GitHubのpull_request仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)を参照。

## Unityの検証

クラウド `.github/workflows/unity-validation.yml` は信頼できるrefからの任意手動Linux CPU補助検証のみ。
ライセンスpreflight → 同一checkoutの参照生成 → `M1CPU`テスト、の順で実行する。
GameCI Actionの固定SHAはv4.4.0、CLIはv0.1.72に対応する。Docker経路はLinux対象。
Windows / GPUの合格には数えず、設定の存在を実行実績として扱わない。

2026-10-10確認時点でRepository Secrets / Variablesは未登録、クラウドEditor jobは未実行。
旧自動workflowはSecrets不足でpreflight失敗し、Editorを開始していない。
クラウド実行を利用する場合は [GameCI公式手順](https://game.ci/docs/github/test-runner/)に沿ってUnityライセンスを準備する。
Secret値はソース・チャット・ログへ書かない。

### Unity CIの開始条件

最新の保存preflightではmacOS / Android / iOSのUnity CIライセンスが未設定でfailureです。
小さいMetal probeや過去のAPK buildを、その後のUnity実行・実モデルGPUの合格に数えません。
利用可能なCIライセンスと対象module・hostが揃うまで、同じ失敗を再実行しません。

現行workflowは`UNITY_EMAIL` / `UNITY_PASSWORD`と、`UNITY_LICENSE`または`UNITY_SERIAL`を検査します。
これはworkflowの入力契約であり、ローカルPersonal entitlementをCIへ流用できるという説明ではありません。
ライセンス種別に対応した有効なCI利用・activation手段を確認した上で、承認された設定だけをSecretsへ登録します。
値をソース・チャット・artifactへ出さず、登録確認はSecret名だけで行います。

実行時はpreflight、Editor開始、build / NUnit、正常終了、元sourceに結び付くartifact回収を別々に判定します。
Windowsの新tag実GPU受け入れは[正式版の記録](results/windows-v010-release-20261021.json)、
他platformの開始条件は[ロードマップ](roadmap.md)を参照してください。

## 実行履歴

| 対象 | 成功したrun / 内容 |
| --- | --- |
| M1統合main `8146107` | [37801650704](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801650704)、Python 4環境各64件 |
| PR #4最終head `04a970f` | [37801226518](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801226518)、Python 4環境各64件 |
| PR #4実モデル | [37801226486](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37801226486)、Python全15件照合 |
| CI入口導入 `8122a13` | [37809737222](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37809737222)、Python・lint・実モデル・Required CI成功。UPM監査の追加前 |
| UPM実装 `b74180e` | [37811183097](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37811183097)、Python 4環境各102件、lint、実モデル15件、パッケージ監査、Required CIすべて成功 |
| PR #6確認済みhead `8a585bc` | [37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)、全8 job成功、Python 4環境各102件、実モデル15件・最小cosine `0.9999998807907104`。checkout `9ccff937ae9fc675c12aee8e1533d08cc0502e9a` とheadのtree一致 |
| PR #6 / #8統合後main `c7d1189` | [37949804912](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37949804912)、pushイベントの全8 job成功、Python4環境各102件、実モデル15件・最小cosine `0.9999998807907104`、Required CI成功。Unity実行は含まない |
| 文書のみPR #7 `9633a1c` | [37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)、全8 job成功。Markdown 1ファイル・6行追加、baseは `feat/ci-upm-package`、検証後クローズ |
| PR #11統合後main `758cb04` | [37975026704](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975026704)、全8 job成功。検索サンプル・モデル準備・最適化を統合 |
| PR #12統合後main `4ee0cbb` | [37975720534](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975720534)、全8 job成功。統合記録の文書更新。Unityの再実行ではない |

過去の失敗・cancel・基盤導入の記録は [基盤履歴](m1-validation.md)と [API実装履歴](m1-runtime-validation.md)を保持する。
上の結果は各commitの実行記録。新しい文書PRの結果はそのPRのChecksで確認し、過去runを新しいheadの成功として数えない。

## ローカルの必要最小限の確認

```powershell
cd tools
uv sync --locked
uv run --locked pytest -q tests/test_ci.py tests/test_workflows.py tests/test_package.py
uv run --locked python -m embeddinggemma_tools.package
```

全Pythonテスト、モデル取得・変換は原則Actionsを使う。変更に必要な小さいred / greenだけローカルで確認する。
新規consumerの初期UPM導入記録は [パッケージ検証](m2-package-validation.md)、現在の検索サンプル導入・再現は [モデル準備手順](model-preparation.md)を参照。
