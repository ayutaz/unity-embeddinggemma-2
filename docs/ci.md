# CI と開発手順

更新日: 2026-10-08

ユーザー指定により、Unity の検証はローカル 6000.3.16f1 + Unity CLI Loop のハーネスでも実行する。
依存解決・コンパイルは成功し、M1 Editor テスト3件は参照データ未配置で失敗した。
Python / モデル変換は引き続き Actions を利用する。[ローカル自動操作](automation.md)を参照。

現在の Unity workflow はクラウドでの検証準備が未完了。
使用する Action の固定 SHA は `v4.4.0`、CLI は `v0.1.72` に対応するが、
Action が呼ぶ `game-ci test --docker` のコンテナ経路は Linux のみ対応する。
`windows-latest` との現構成は、Secrets の追加だけでは動作しない。
Windows のホスト Editor を使う経路への修正・実測を別途行う。
[GameCI の対応条件](https://game.ci/docs/cli/build/#classic-docker-test-flow)を参照。

## 現在の CI 状態

4 workflow は作業ツリーに追加済みで、actionlint 1.7.12 による静的検査は合格。
`feat/m1-text-tdd` の初回実装コミット `4fec060` を push し、[draft PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1)を提出済み。
4 workflow のPR runを起動済み。初回実モデルの参照生成 / export照合は成功。
workflow lint の ShellCheck SC2129 は修正後CIで合格。UnityはSecrets不足でpreflight失敗。
初回pushと修正後PRのPython matrixは4環境で各33件合格。実モデル再runも成功（最小cosine 0.9999997616）。詳細なrun履歴は検証記録を参照。
既存の Dependency Graph 2 件の成功を、今回の Python / モデル / Unity CI の成功と混同しない。

| workflow | 起動条件 | 実行内容 | 現在の状態 |
| --- | --- | --- | --- |
| `workflow-lint.yml` | `.github/**` の push / PR、手動 | actionlint / ShellCheck | 修正後CI成功（run `37733804135`） |
| `python-tests.yml` | `tools/**` または自身の変更を含む push / PR、手動 | Ubuntu / Windows × Python 3.13 / 3.14 の uv / pytest | 初回push / 修正後PRで4環境各33件成功（PR run `37733804075`） |
| `model-reference.yml` | `tools/**` または自身の変更を含む PR、手動、reusable call | 固定 revision の実モデル取得、参照生成、`.pt2` export / 再読み込み照合 | 初回 / summary修正後とも実モデル成功（再run `37733804078`） |
| `unity-validation.yml` | テスト・Packages・tools・関連 workflow の変更を含む同一 repo の PR、手動 | ライセンス preflight → 同一 revision のモデル生成 → Windows CPU / GPUCompute | Secrets不足でpreflight失敗（run `37733804252`）。model / unity jobsはskip、合格ではない |

手動実行の設定が存在することは実行実績ではない。初回は PR イベントでの実行を確認する。
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
このPRにはtools / workflowの変更があるため、文書のみの追加commitでもPR CIが再起動し得る。
pushのpaths判定とは分けて扱う。[GitHub公式の差分仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#git-diff-comparisons)を参照。
検証記録は成功したrunと対象SHAのスナップショットとして残し、後続runの状態はActions画面で確認する。

## ブランチと PR

`main` への直接 push は禁止する。変更は作業ブランチで commit / push し、PR を作成する。
PR の CI 結果を確認する。GitHub API では main の branch protection が `Branch not protected`、
repository rulesets が空であり、サーバー側で直接 push を防ぐ設定は未完了。
この方針は `AGENTS.md` にも記録している。必須 CI の check 名を確認した後、PR 必須・必須 check・
force push / branch 削除の禁止などを GitHub 側に設定する作業を残す。
自動 merge / release / 公開はこの CI に含めない。

現在の作業ブランチは `feat/m1-text-tdd`。承認モード変更後、実際のブランチ作成に成功し、
以前の実行ポリシーによる制約は解消した。main への直接 commit / push は行っていない。

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
Unity workflow は後述の同一 repository の PR に限定してライセンス Secrets を使う。

`prepare()` は開始時に前回の `export-validation.json` を削除し、今回の全ケース照合が
完了した場合だけ新しいレポートを保存する。失敗 job の部分成果物はデバッグ用。
成功判定では job の conclusion、ソース revision、全 15 ケースの結果を確認する。

## Unity CI の準備条件

`.github/workflows/unity-validation.yml` は同一リポジトリの PR と信頼できる作業ブランチからの手動実行を対象にする。
ライセンスの存在を最初に検証し、同じソース revision の参照生成 workflow を呼び、Windows runner で CPU / GPUCompute を別 job として検証する。
Unity の acceptance tests はローカル Editor でコンパイル成功。実行すると参照データ未配置で3件失敗した。
GameCI と Windows runner の組み合わせを含め、Editor job は未実行。初回runのpreflight以降の結果から互換性を確認する。
GitHub Actions 上の Unity にはライセンス設定が必要。2026-10-08 の再確認でもリポジトリの Secrets / Variables は未登録。
[GameCI の公式手順](https://game.ci/docs/github/test-runner/)に従い、Personal は
`UNITY_LICENSE` / `UNITY_EMAIL` / `UNITY_PASSWORD`、Pro は `UNITY_SERIAL` / `UNITY_EMAIL` /
`UNITY_PASSWORD` を GitHub の Repository Secrets に設定する。値をソース・チャット・ログへ書かない。
外部 fork の PR を Secrets 付きで実行する `pull_request_target` は使用しない。

Sentis CPU と GPUCompute の結果は区別する。一般の hosted runner に GPU があるとは仮定しない。
GPU 機能がない runner のテストを skip しても M1 の GPU 合格にはしない。
現在の acceptance test は compute 非対応を明示的な失敗にし、graphics device / API / OS / Editor / backend をログへ記録する。
実行可能な GPU runner、または必要最小限のローカル GPU 実行を用意して実測する。

## 初回 CI の確認順序

1. 実行ポリシーの問題を解消し、作業ブランチへ変更を commit / push して draft PR を作成する。main へ直接 push しない。
2. workflow lint と Python matrix を確認する。同じ PR に push と PR 起動の両方が出る場合も、PR の head SHA と実際の checkout / `GITHUB_SHA` を区別して記録する。
3. Secrets を使わない実モデル参照生成を確認する。全 15 ケースで Python eager 対保存済み export の cosine >= 0.999999、出力 / tokenizer の digest と metadata を保存する。
4. GitHub の Repository Secrets に Unity ライセンス設定を用意する。欠落していれば Unity preflight は失敗し、モデル生成と Editor job は開始しない。Python 単体 / standalone model workflow は独立して進められる。
5. Editor の依存解決、コンパイル、tokenizer、CPU 推論、GPUCompute 推論を切り分けて確認する。C# ランタイム API、保存、量子化、測定は今後の TDD 作業。
6. 生成された `Packages/packages-lock.json` を既存 lockfile と比較し、必要な変更を作業ブランチへ反映する。現時点の lockfile には Sentis が含まれていない。
7. CI run URL、ソース SHA、実行環境、結果の要約を docs に記録する。モデル artifact は 3 日、pytest / Unity artifact は 7 日で失効するため、再生成手順も保持する。

fork PR の Unity job は起動条件で除外する。この場合の skip は検証完了ではない。
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
