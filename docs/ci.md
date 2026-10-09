# CI と開発手順

更新日: 2026-10-10。基準main: `c7d1189`。
PR #6の全PRのCIとUPM監査はmain統合済み。PR #8も統合後、main `c7d1189` の全8 job成功を確認した。
文書のみのPR #7も全8 job成功後に未マージで閉じた。[現在の状態](status.md)を参照。
GitHub側のmain保護は設定済み。M1のWindows実測は [完了検証](m1-completion-validation.md)、UPM移行の結果は [パッケージ検証](m2-package-validation.md)を参照。

## 全PRの必須CI

`.github/workflows/ci.yml` を入口とし、paths filterを付けず、全PR・main push・手動実行を対象にする。
文書のみのPRも同じ検証を実行する。以下のreusable workflowは単独のPR / pushトリガーを持たず、重複実行を避ける。

| workflow / job | 実行内容 | 判定 |
| --- | --- | --- |
| `python-tests.yml` / `python` | Ubuntu / Windows × Python 3.13 / 3.14、uvによるオフラインpytest | matrix全環境の成功が必要 |
| `workflow-lint.yml` / `workflow-lint` | actionlint 1.7.12 / ShellCheck | 成功が必要 |
| `model-reference.yml` / `model-reference` | 固定revisionの実モデル取得、Python参照生成、保存済み`.pt2`の全15件照合 | 成功が必要。Sentis実行の代わりにはしない |
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

3日保持のモデルartifactは `reference.json`、`tokenizer.json`、`model.pt2`、`export-validation.json`を含む。
pytestとパッケージ監査artifactは7日保持。失効時はActionsで再生成し、小さい数値要約と再現手順をGitへ残す。

PRでは既定のmerge refをcheckoutする。`github.sha` / metadataのsource commitと、PR head SHAは区別する。
run URL、PR head、実際のcheckout SHA、model revision、ファイルhashを揃えて監査する。
[GitHubのpull_request仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)を参照。

## Unityの検証

クラウド `.github/workflows/unity-validation.yml` は信頼できるrefからの任意手動Linux CPU補助検証のみ。
ライセンスpreflight → 同一checkoutの参照生成 → `M1CPU`テスト、の順で実行する。
GameCI Actionの固定SHAはv4.4.0、CLIはv0.1.72に対応する。Docker経路はLinux対象。
Windows / GPUの合格には数えず、設定の存在を実行実績として扱わない。

2026-10-09時点でRepository Secrets / Variablesは未登録、クラウドEditor jobは未実行。
旧自動workflowはSecrets不足でpreflight失敗し、Editorを開始していない。
クラウド実行を利用する場合は [GameCI公式手順](https://game.ci/docs/github/test-runner/)に沿ってUnityライセンスを準備する。
Secret値はソース・チャット・ログへ書かない。

必要なWindows GPUと新規プロジェクトへのUPM導入は、ローカル6000.3.16f1とuloopで検証する。
[自動操作とハーネス](automation.md)を使い、未保存Scene / Prefabは自動保存・破棄しない。
CPU / GPUCompute、実モデル / 小さい契約モデル、compile / 数値一致を別々に記録する。skipをGPU合格にしない。
macOS / iOS / Androidのbuild・実機・測定は [M2計画](m2-plan.md)の後続段階。

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
新規consumerの作成・Unityでの導入検証は [パッケージ検証](m2-package-validation.md)を参照。
