# M1 検証記録

日付: 2026-10-08

## 最新の確認結果

**M1 は未完了。ローカル Unity 6000.3.16f1 の起動・依存解決・コンパイルが成功。M1 の Editor テスト3件は参照未配置で失敗し、GPUの数値一致は未検証。**
この文書の後半にある 31 件の結果は修正過程の履歴。ハーネス追加前の Python 全体は33件で、ハーネス専用14件は別途検証した。

### ローカル自動操作ハーネス（2026-10-08 17:17 JST）

- ユーザー指定で Unity を 6000.3.16f1 (`a56f230f6470`) に変更。インストール済みの同版で Editor の起動を確認。
- Unity Personal ライセンスをローカルで認識。GitHub Secrets を使用していない。
- 初回起動は `ALLUSERSPROFILE` 欠落により UPM が `The "path" argument must be of type string. Received undefined` で停止。Unity 子プロセスだけに標準の CommonApplicationData を補うと解決し、依存取得・インポートが完了。システム設定は変更していない。
- Unity CLI Loop パッケージ 3.14.0、プロジェクト内 dispatcher 3.8.1 / project runner 3.8.0 を導入。旧グローバル npm CLI 2.1.9 は変更していない。
- `uloop compile` は `Success=true, ErrorCount=0, WarningCount=0`。Sentis / tokenizer API を使用する既存テストもコンパイルできた。Editor が生成した packages-lock に Sentis / uloop と依存が追加された。
- `uv run --locked python -m embeddinggemma_tools.unity --suite compile --launch` は成功。証拠: `artifacts/unity-harness/compile/`。
- 同ハーネスの `--suite m1` は `TestCount=3, PassedCount=0, FailedCount=3, SkippedCount=0`。3件とも `reference.json` 未配置のアサーションで失敗。tokenizer / CPU / GPUCompute の数値照合まで進んでいない。証拠: `artifacts/unity-harness/m1-missing-reference/` と `.uloop/outputs/TestResults/` の NUnit XML。
- ハーネスはTDDで実装。未実装モジュールによる red と、失敗時の結果・XML未回収による red を確認後に修正し、専用 Python 単体テスト14件がローカルで合格。これを実モデル推論の成功とは数えない。
- NVIDIA GeForce RTX 4070 Ti SUPER / Direct3D 12 を Editor ログで認識。GPUの認識はSentisの動作確認ではない。
- 実行方法と再導入手順: [自動操作](automation.md)。以下のCI表はハーネス追加前の実行履歴。

| 項目 | 証拠 | 判定 / 限界 |
| --- | --- | --- |
| Python 単体テスト | `artifacts/python-tests.xml`: tests=33、failures=0、errors=0、skipped=0。2026-10-08 14:18 JST の実行。CLI 表示 14.59s、XML time 14.572s | 小さいモデル、Windows / Python 3.14 のオフライン実行のみ合格 |
| 実装との対応 | XML の生成時刻は `prepare.py` / `test_prepare.py` の最終変更より後。export、loader、参照、CLI、失敗時レポートの testcase を含む | 今回の文書更新ではテストを再実行していない |
| workflow | 4 YAML のローカルactionlint 1.7.12 合格、修正後CI `37733804135` 成功 | 初回ShellCheck SC2129は修正済み。lint合格はSentis合格ではない |
| Unity 依存（CI導入時） | 変更前の CI は 6000.3.19f1 指定でSecrets preflight失敗 | その時点では Editor は未実行。後続のローカル結果は上記を参照 |
| 実モデル設定 | 固定 revision の config は hidden_size=512、embedding_dim=768、24 層、語彙 262144、sliding_window=512 | 実モデルのtext-only load / Python参照生成 / export照合は下記CIで成功。Sentis互換性は未検証 |

`artifacts/` は Git 管理対象外。上記 XML はローカルの検証証拠であり、他の checkout には含まれない。
リモートCIのrun URLと成果物は下記に記録する。ローカルXMLをリモートCIの証拠として流用しない。

### GitHub と作業ツリー（文書更新時に再確認）

- repository: `ayutaz/unity-embeddinggemma-2`、public、default branch は main。
- `git status --short --branch` / `git branch --list`: 承認モード変更後に `feat/m1-text-tdd` を作成。main への commit / push は未実施。
- `gh pr view 1`: [draft PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1)、base=`main`、head=`feat/m1-text-tdd`、初回実装 head SHA=`4fec06088b5a535430995936cd1d2e5284a8f559`。
- main のリモート SHA は `e19e116428bbf488dc6466f658156e3d7f6d87e8` のまま。mainへのpush / mergeは行っていない。
- 既存 Dependency Graph 2 件の成功（run ID `37728681777` / `37585043501`）は M1 の実行ではない。以下に新規runを分けて記録する。
- `gh secret list` / `gh variable list`: ともに登録なし。値の取得・ログ出力はしていない。
- main protection API: `Branch not protected` (HTTP 404)。repository rulesets API: 空配列。GitHub 側の直接 push 防止は未設定。
- 以前のブランチ作成は `approval required by policy, but AskForApproval is set to Never` で拒否された。ユーザーが権限モードを変更した後、`git switch -c feat/m1-text-tdd` が成功し、制約の解消を確認した。

これは 2026-10-08 のスナップショット。再開時にはブランチ・PR・run・Secrets 名・protection を再確認する。

### 初回 PR CI の run 記録

初回実装コミット `4fec060` を対象に起動。2026-10-08 の起動直後の確認では全PR runが `queued`。
待機中は成功・失敗を判定しない。pushイベントの Python tests / Workflow lint も別runとして起動している。
PRのpaths filterはthree-dot diff全体を見るため、文書のみの追加commitでもPR CIが再起動し得る。
以下はrunごとの履歴であり、後続runの成功を含めて主張しない。

| workflow | run | 起動直後の状態 |
| --- | --- | --- |
| Workflow lint | [37733486201](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733486201) | queued |
| Python tests | [37733486327](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733486327) | 4環境 queued |
| Model reference and export | [37733486276](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733486276) | queued |
| Unity M1 validation | [37733486514](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733486514) | license job queued。Secrets未登録のためpreflight未成立 |

再開時は上記run IDを `gh run view <id> --json status,conclusion,jobs,url` で確認する。
`metadata.source_commit` の実際のcheckout SHAは参照生成結果から照合し、PR head SHAと同一と仮定しない。

### 初回CIの失敗と修正

14:42 JST の再確認で Workflow lint run `37733486201`（PR）/ `37733471105`（push）が失敗。
`model-reference.yml` の summary 出力に ShellCheck `SC2129`（同じファイルへの連続追記をまとめる）を検出した。
これはCIで失敗を実行確認してから修正したもの。echo / cat の各行でのリダイレクトを
グループ全体の一回のリダイレクトにまとめ、成功 / 失敗のsummary本文は維持する。
修正後のCI合格は別runで確認する。Windowsでのローカルactionlint合格は、CIのShellCheck合格を含まない。

同時点で Model reference run `37733486276` は `uv sync --locked` が成功し、実モデル生成stepが `in_progress`。
生成stepの開始だけでは全15ケースの照合成功を示さない。

### 確認できたCI結果（2026-10-08）

| 検証 | 対象のhead SHA / run | 実行結果 |
| --- | --- | --- |
| Python matrix（push） | `4fec060` / [37733471063](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733471063) | success。Windows 3.13: 33 passed / 21.91s、Windows 3.14: 33 passed / 14.57s、Ubuntu 3.13: 33 passed / 14.96s、Ubuntu 3.14: 33 passed / 15.66s。ログで各件数を確認 |
| 実モデル参照 / export（PR） | `4fec060` / [37733486276](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733486276) | success。15ケース、最小cosine `0.9999998807907104`。実モデルのeager対保存済み`.pt2`照合。Sentisの結果ではない |
| lint修正後（PR） | `3df9ebe` / [37733804135](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733804135) | success。CIのactionlint / ShellCheckでSC2129解消を確認 |
| Unityライセンス（PR） | `3df9ebe` / [37733804252](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733804252) | failure。Secrets不足でlicense jobがexit 1。model / unity jobsはskipped。Editor / GPUは未実行 |
| Python matrix修正後（PR） | `3df9ebe` / [37733804075](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733804075) | success。Windows 3.13: 33 passed / 23.12s、Windows 3.14: 33 passed / 24.26s、Ubuntu 3.13: 33 passed / 15.43s、Ubuntu 3.14: 33 passed / 16.90s。各件数はrunログで確認 |
| 実モデル再run（PR） | `3df9ebe` / [37733804078](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37733804078) | success。15ケース、最小cosine `0.9999997615814209`。閾値 0.999999 以上を維持 |

初回実モデルartifactは `m1-reference-3472830afe4e7a7c6ce2840022cbc8435abee1f2`。
このsuffixはPRのmerge checkout SHAで、head SHA `4fec060`と異なる。
artifact ID `11530499604`、サイズ `1125922580` bytes、zipのSHA-256は
`95b34d2f2ad4d850946a283aec1d419a66e6dc78069a16edd444081b334ee0f6`。
ログとAPIで確認し、大きいモデルartifactはローカルへ取得していない。
このdigestはアップロードされたzip全体の値であり、`model.pt2`単体のdigestとは区別する。

実モデル再runのartifactは `m1-reference-ad60ba066d9cbe1c1e5e123d03bcc82d7840ac4f`。
merge checkout SHAは `ad60ba066d9cbe1c1e5e123d03bcc82d7840ac4f`、artifact ID `11530858602`、
サイズ `1125922493` bytes、zipのSHA-256は
`bf26dccb19b85194db29f943b3c66080b7f6703e43d2b27443b0bffaa7a62213`。
14:48 JSTにAPIで未失効を確認。モデルartifactはGitに含めず、ローカルにも取得していない。

以前のPR matrix / Unity runは新しいpushによりcancelledになったため、合格数に含めない。
Unity Secretsを設定後、同じソースrevisionの参照生成とEditor検証を再実行する。
この表は検証対象コミットの記録。後続の文書commitでPR CIが再起動した場合、そのrunの成功を先取りしない。

## 実行済み: 小さいモデルによる Python TDD

Python 3.14.0 / Windows、uv 0.12.20。torch 2.14.1+cpu、transformers 5.19.0、
sentence-transformers 6.1.0。モデルのダウンロードを伴わない単体テスト。

| 対象 | Red の確認 | Green の確認 |
| --- | --- | --- |
| prompting / pooling / normalization | `tests/test_text.py`: 対象モジュール未実装で collection error | 13 passed |
| text-only loader / Core ATen export | `tests/test_model.py tests/test_export.py`: 対象モジュール未実装で collection error | 4 passed |
| 公式 pooling による参照生成 | `tests/test_reference.py`: 対象モジュール未実装で collection error | 9 passed |
| CLI / 生成から再読み込みまで | `tests/test_prepare.py`: 対象モジュール未実装で collection error | 実装後、torch build suffix の記録漏れを検出。修正後、全体で合格 |

初回の全体実行（履歴）:

```text
cd tools
uv run --locked pytest -q --junitxml=../artifacts/python-tests.xml
31 passed in 14.41s
```

小さいモデルでも実際の EmbeddingGemma2TextModel（局所注意と全体注意、PLE、出力 projection）を使う。
モデル本体を stub にした export テストではない。異なるトークン / mask を保存後の `.pt2` に入力して検証する。
参照生成の pooling / normalization は sentence-transformers の公式モジュールを使い、export 側の自作後処理と独立に比較する。

## 実装初期の静的確認と外部状態（履歴）

以下は権限モード変更・PR提出前の記録。現在の状態は冒頭の確認結果を参照。

- リポジトリは PUBLIC。
- GitHub の Repository Secrets / Variables は開始時点で未登録。
- Sentis の公開レジストリで 2.6.1 を確認。manifest に追加済み。
- 実モデルの safetensors ヘッダのみを HTTP Range で確認。1,376 tensor のうち 413 が `language_model.` 配下で、`language_model.embedding_projection.weight` が存在する。重み本体のローカル取得・推論はしていない。
- 最初の 3 つの workflow YAML を解析。その後 workflow-lint を追加し、最新の全 4 workflow は actionlint 合格。GitHub Actions 上での実行成功を意味しない。
- `git diff --check` は成功。
- 作業ブランチ作成は実行環境の自動承認レビューに拒否された。ユーザーがブランチ / PR 運用を明示した後も拒否。
- commit / push / PR 作成 / Actions 起動は未実施。`main` への push は行っていない。

## 未検証・結果待ち

- Unity 6000.3.16f1 のローカル起動・依存解決・packages-lock 更新は完了。実モデルの配置と数値照合が残る。
- Unity acceptance tests のコンパイルと red / green。
- 実モデルの CPU / GPUCompute 推論と、全 15 ケースの cosine >= 0.999。
- `.sentis` 保存、量子化、精度・メモリ・時間の測定。

M1 は未完了。最新の 33 件の Python 単体テスト合格でも、実モデル・Sentis・GPU の一致を代替しない。

## 続行時の export 互換性修正（履歴）

保存済みの小さいモデルの `.pt2` を静的に確認し、次の二点を検出した。

1. `aten.slice.Tensor` の終端が `9223372036854775807`。Sentis 2.6 の JSON reader は `as_int` を
   `System.Int32` として読むため、範囲外になる。
2. torch 2.14 の `aten._assert_tensor_metadata.default` がグラフに残る。
   Sentis 2.6 の importer の対応演算子一覧には含まれない。

先に保存後のグラフを検証するテストを追加し、1 に対する失敗を確認。
再帰的に scalar args / kwargs を int32 範囲へ制限した後、2 に対する失敗を確認した。
数値出力を持たない metadata assertion のみを削除し、再読み込み後の異なる ID / mask の
結果が公式モデルと一致することを再確認した。入力長を変えた場合の shape guard も維持される。

参考: [Sentis の公式 export 手順](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/export-convert-torch.html)
にも int32 の範囲に制限する工程が明記されている。

export 互換性修正時の結果（履歴）:

```text
uv run --locked pytest -q --junitxml=../artifacts/python-tests.xml
31 passed in 13.69s
```

actionlint 1.7.12 による全 workflow の静的検査も合格。`workflow-lint.yml` を追加して CI でも検査する。
これらは importer の既知の問題への対処であり、実際の Unity import 成功を証明するものではない。
再確認時点でもブランチは main のまま、open PR は 0、登録済み Unity Secrets は 0。
リモート CI の実行に必要な条件はまだ変化していない。

## 権限モード変更前の再開確認と成功レポートの更新（履歴）

再開時にも `git switch -c feat/m1-text-tdd` が
`approval required by policy, but AskForApproval is set to Never` で拒否された。
ブランチ作成、commit、push、PR 提出、リモート CI 実行は未実施。
GitHub API で PUBLIC と Unity Secrets 未登録を再確認した。

参照生成を同じ出力先で再実行して失敗すると、前回の `export-validation.json` が残ることを検出。
モデル読み込み時と export 時の失敗を注入する回帰テストを先に追加し、2 件とも
古い成功レポートが残るという意図した理由で失敗した。
`prepare()` の開始時に前回のレポートを削除し、今回の全ケース照合が完了した場合だけ保存するよう修正。
部分成果物は調査用に残るため、成功判定には今回の job の成功とレポートの両方が必要。

修正後の全体結果（小さいモデル・オフライン）:

```text
uv run --locked pytest -q --junitxml=../artifacts/python-tests.xml
33 passed in 14.59s
```

実モデル / Unity / GPU の検証は引き続き未実行であり、M1 は未完了。
