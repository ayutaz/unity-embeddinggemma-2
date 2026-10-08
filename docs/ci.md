# CI と開発手順

更新日: 2026-10-08

ユーザー指定により、Unity の検証はローカル 6000.3.16f1 + Unity CLI Loop のハーネスでも実行する。
依存解決・コンパイルは成功し、M1 Editor テスト3件は参照データ未配置で失敗した。
Python / モデル変換は引き続き Actions を利用する。[ローカル自動操作](automation.md)を参照。

Unity workflow は任意の **Linux CPU 手動検証** に変更した。PRでは起動しない。
使用する Action の固定 SHA は `v4.4.0`、CLI は `v0.1.72` に対応する。
Action が呼ぶ `game-ci test --docker` のコンテナ経路は Linux のみ対応する。
以前の `windows-latest` + CPU/GPUCompute matrix は削除した。
Windows CPU/GPUCompute の M1 合格はローカルハーネスで検証する。Linux CPUの合格では代替しない。
手動クラウド workflow の Editor 実行は未検証で、引き続きライセンスSecretsが必要。
[GameCI の対応条件](https://game.ci/docs/cli/build/#classic-docker-test-flow)を参照。

## 現在の CI 状態

4 workflow は [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1) で main `60f906d` に統合済み。
統合後の main CI で Python matrix の4環境各47件と workflow lint が成功。
最終 PR head `673cb9a` では実モデルの15件照合と全自動チェックが成功した。
以前の自動Unity workflowはSecrets不足でpreflight失敗し、Editorは開始していなかった。
以後も関連パスを変更する PR では Python / 実モデル参照 / workflow lint を自動実行し、Windows Unity 検証はローカルで行う。
基盤の統合は M1 完了・UPM リリースを示さない。Dependency Graph の成功も推論の検証には数えない。

| workflow | 起動条件 | 実行内容 | 現在の状態 |
| --- | --- | --- | --- |
| `workflow-lint.yml` | `.github/**` の push / PR、手動 | actionlint / ShellCheck | main `60f906d` 成功（run `37758476440`）、最終 PR も成功 |
| `python-tests.yml` | `tools/**` または自身の変更を含む push / PR、手動 | Ubuntu / Windows × Python 3.13 / 3.14 の uv / pytest | main `60f906d` の4環境各47件成功（run `37758476415`） |
| `model-reference.yml` | `tools/**` または自身の変更を含む PR、手動、reusable call | 固定 revision の実モデル取得、参照生成、`.pt2` export / 再読み込み照合 | 最終 PR head `673cb9a` 成功（run `37755304407`）。main push では起動しない |
| `unity-validation.yml` | 信頼できるrefからの手動実行のみ | ライセンス preflight → 同一 revision のモデル生成 → Linux CPU | 構成変更後は未実行。Windows / GPU / M1合格の証拠にはしない |

手動実行の設定が存在することは実行実績ではない。旧自動Unity workflowの失敗も履歴として保持する。
最新の外部状態と実行済みテストの証拠は [検証記録](m1-validation.md)を参照。

### 検証したソース SHA の扱い

現在の workflow は `actions/checkout` の `ref` を上書きしない。
`pull_request` では既定の merge ref を checkout し、`GITHUB_SHA` はその merge commit を指す。
PR の作業ブランチの最新 commit（`github.event.pull_request.head.sha`）とは区別する。
これは [GitHub の公式仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)に従う。
`prepare` が出力する `metadata.source_commit` と artifact 名には `GITHUB_SHA` が入る。
PR 番号・head SHA・実際に検証した merge SHA・run URL を合わせて記録し、参照生成と Unity のソースが一致することを確認する。
push 起動の結果を PR 起動の結果と比較するときも、この違いを確認する。

PRのpaths filterは、直前の追加commitだけでなくPRのthree-dot diff全体で判定する。
PR #1 のように tools / workflow の変更がある PR では、文書のみの追加commitでもPR CIが再起動し得る。
最新 main から作成した文書のみの PR は、現在の paths filter では上記3つの自動 workflow の対象外。
pushのpaths判定とは分けて扱う。[GitHub公式の差分仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#git-diff-comparisons)を参照。
検証記録は成功したrunと対象SHAのスナップショットとして残し、後続runの状態はActions画面で確認する。

## ブランチと PR

`main` への直接 push は禁止する。変更は作業ブランチで commit / push し、PR を作成する。
PR の CI 結果を確認する。GitHub API では main の branch protection が `Branch not protected`、
repository rulesets が空であり、サーバー側で直接 push を防ぐ設定は未完了。
この方針は `AGENTS.md` にも記録している。必須 CI の check 名を確認した後、PR 必須・必須 check・
force push / branch 削除の禁止などを GitHub 側に設定する作業を残す。
paths filter 付きの check を全 PR で必須にする場合は、文書のみの PR でも判定が完了する起動条件を先に整える。
自動 merge / release / 公開はこの CI に含めない。

PR #1 の基盤ブランチ `feat/m1-text-tdd` は統合済み。以後は最新 main を基点に作業ブランチを作成する。
merge は依頼があるまで行わない。今回の計画更新も文書専用ブランチと PR で進める。

## Python tests

`.github/workflows/workflow-lint.yml` で actionlint 1.7.12 による workflow の静的検査も実行する。
ローカルでは全 workflow の actionlint 合格を確認済み。GitHub Actions 上のactionlint / ShellCheckも修正後runで合格。

`.github/workflows/python-tests.yml` は関連ファイルの push / PR と手動実行を対象にする。
Ubuntu / Windows と Python 3.13 / 3.14 の組み合わせで `uv sync --locked` とオフライン pytest を実行する。
モデルのダウンロードを禁止した環境で小さいモデルを使い、結果を JUnit XML として保存する。
テスト対象は prompting、masked mean、正規化、text-only 重み抽出、Core ATen export、
保存後の別入力推論、公式 sentence-transformers pooling による参照生成、CLI の入力検証。

## Model reference and export

`.github/workflows/model-reference.yml` は関連する PR と手動実行を対象にする。
公開された Hugging Face モデルを SHA 固定で取得し、固定 15 ケースの参照と batch 1 / length 128 の
fp32 `.pt2` を作成する。保存済み export を再読み込みし、全ケースの Python eager 出力と比較する。
ここでの閾値は 0.999999。これは Python export のチェックであり、Sentis の M1 合格ではない。

成果物（3 日保持）:

- `reference.json`: 入力、プロンプト、ID、mask、参照ベクトル、モデル revision、ライブラリ build、ソース SHA
- `tokenizer.json`: 参照と同じ固定長 padding / truncation を設定した tokenizer
- `model.pt2`: fp32 text-only export
- `export-validation.json`: 全ケースの cosine、ファイル SHA-256、生成条件

モデルのファイル自体は Git 管理しない。参照生成失敗時は job を失敗させ、部分成果物があれば調査用に保存する。
この参照生成 workflow では PR の変更コードを実行しても Secrets を利用しない。
Unity workflow は後述の信頼できるrefからの手動実行に限ってライセンス Secrets を使う。

`prepare()` は開始時に前回の `export-validation.json` を削除し、今回の全ケース照合が
完了した場合だけ新しいレポートを保存する。失敗 job の部分成果物はデバッグ用。
成功判定では job の conclusion、ソース revision、全 15 ケースの結果を確認する。

## Unity CI の準備条件

`.github/workflows/unity-validation.yml` は信頼できるrefからの手動実行のみを対象にする。
ライセンスの存在を最初に検証し、同じソース revision の参照生成 workflow を呼び、Linux runner で `M1CPU` カテゴリだけを実行する。
Unity の acceptance tests はローカル Editor でコンパイル成功。実行すると参照データ未配置で3件失敗した。
クラウドの Linux Editor job は未実行。初回の手動runでimport / tokenizer / CPUの結果を確認する。
GitHub Actions 上の Unity にはライセンス設定が必要。2026-10-08 の再確認でもリポジトリの Secrets / Variables は未登録。
[GameCI の公式手順](https://game.ci/docs/github/test-runner/)に従い、Personal は
`UNITY_LICENSE` / `UNITY_EMAIL` / `UNITY_PASSWORD`、Pro は `UNITY_SERIAL` / `UNITY_EMAIL` /
`UNITY_PASSWORD` を GitHub の Repository Secrets に設定する。値をソース・チャット・ログへ書かない。
外部 fork の PR を Secrets 付きで実行する `pull_request_target` は使用しない。

Sentis CPU と GPUCompute の結果は区別する。一般の hosted runner に GPU があるとは仮定しない。
GPU 機能がない runner のテストを skip しても M1 の GPU 合格にはしない。
現在の acceptance test は compute 非対応を明示的な失敗にし、graphics device / API / OS / Editor / backend をログへ記録する。
実行可能な GPU runner、または必要最小限のローカル GPU 実行を用意して実測する。

## 次の実装と CI の確認順序

1. 最新 main に同期して新しい作業ブランチを作成する。基盤の Python / lint は合格済みで、次は参照成果物の配置から進める。
2. 成功 run `37755304407` の artifact を取得し、全15件の結果・固定 model revision・source SHA / tree・ファイル digest を監査する。期限と取得候補は [計画](m1-plan.md) を参照。失効、モデル生成コード・依存・入力条件の変更時は最新 main でモデル workflow を手動再実行する。
3. ローカル Unity 6000.3.16f1 に配置して tokenizer / import / CPU / GPUCompute の結果を切り分ける。GitHub Secrets は不要。失敗を再現してから互換性修正を TDD で実装する。
4. Python や export を変更した場合は、PR の関連 workflow で単体テストと実モデル照合を再確認する。新しい成果物の source SHA を記録し、同じモデルを Editor へ再配置する。C# 変更はローカル compile / M1 ハーネスで確認する。
5. C# ランタイム API、保存、量子化、測定を [M1 計画](m1-plan.md) の合格条件に従って進める。Editor 生成の packages-lock は更新時に確認する。クラウド Linux CPU 検証を使う場合だけ Secrets を準備する。
6. CI run URL、PR head と実際の checkout SHA、環境、結果を docs に記録する。モデル artifact は3日、pytest / Unity artifact は7日で失効するため、要約と再生成手順を残す。
7. PR の差分・CI・未解決指摘を確認し、merge は依頼があるまで行わない。文書のみの PR で workflow が起動しない場合は、未実行と記録して過去の成功を今回の実行として数えない。

fork PRからはUnity workflowを起動しない。手動クラウド検証の未実行も、M1の成功には数えない。
新しい `.cs` のコンパイル成功と tokenizer / import の成功も、768 次元出力の数値一致とは別に確認する。

## ローカルで必要になった場合のコマンド

原則は CI を利用する。小さい単体テストだけを確認するとき:

```powershell
cd tools
uv sync --locked
uv run --locked pytest -q
```

モデル取得を伴う以下のコマンドは通常 GitHub Actions が実行する:

```powershell
uv run --locked python -m embeddinggemma_tools prepare --output ../artifacts/m1
```

既存 snapshot を利用する場合は `--snapshot <directory>` を指定する。
ローカル snapshot の由来は実行者が確認し、`--revision` を実際に取得した SHA と合わせる。
