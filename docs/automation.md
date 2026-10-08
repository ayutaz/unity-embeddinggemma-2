# Unity のローカル自動操作

更新日: 2026-10-09

Unity 6000.3.16f1 をローカルで起動し、Unity CLI Loop を介してコンパイル・EditMode テスト・ログ取得を行う。
Python のモデル取得・変換は GitHub Actions に残す。ローカルで認証済みの Editor を使うため、GitHub Secrets は不要。

## 導入済みの構成

| 対象 | 固定バージョン |
| --- | --- |
| Unity | 6000.3.16f1 (`a56f230f6470`) |
| Sentis | 2.6.1 |
| Unity CLI Loop UPM パッケージ | 3.14.0（manifest / packages-lock） |
| uloop dispatcher | 3.8.1（`artifacts/uloop/bin/uloop.exe`） |
| uloop project runner | 3.8.0（UPM パッケージ指定に従い dispatcher が取得） |

CLI はこのプロジェクト内に配置し、既存のグローバル npm CLI は変更していない。
単に `uloop` と入力すると旧版 2.1.9 が選ばれる環境のため、ハーネスまたは明示的なパスを使う。
バイナリ・ログ・モデルは Git 管理対象外。UPM パッケージと CLI は別々のバージョン番号を持つ。

## 新しい checkout での CLI 準備

リポジトリルートで PowerShell から実行する。公式リリースの zip と checksum を取得し、照合する。

```powershell
New-Item -ItemType Directory -Force artifacts/uloop | Out-Null
gh release download dispatcher-v3.8.1 --repo hatayama/unity-cli-loop --pattern uloop-dispatcher-windows-amd64.zip --pattern uloop-dispatcher-windows-amd64.zip.sha256 --dir artifacts/uloop
$uloopExpected = ((Get-Content artifacts/uloop/uloop-dispatcher-windows-amd64.zip.sha256 -Raw) -split '\s+')[0]
$uloopActual = (Get-FileHash artifacts/uloop/uloop-dispatcher-windows-amd64.zip -Algorithm SHA256).Hash
if ($uloopExpected -ne $uloopActual) { throw 'uloop checksum mismatch' }
Expand-Archive artifacts/uloop/uloop-dispatcher-windows-amd64.zip -DestinationPath artifacts/uloop/bin -Force
& artifacts/uloop/bin/uloop.exe --version
```

Unity 側パッケージは manifest に追加済みなので、Editor を開くと解決される。
別の既存プロジェクトへ同じ版を追加する場合は `uloop package install --version 3.14.0` を使う。
[公式の導入手順](https://github.com/hatayama/unity-cli-loop#quickstart)を参照。

## ハーネス

```powershell
cd tools
uv sync --locked
# Editor の起動・準備待ち → コンパイル → ログ保存
uv run --locked python -m embeddinggemma_tools.unity --suite compile --launch
# 起動済み Editor で M1 の3テスト → ログ・失敗時XML保存
uv run --locked python -m embeddinggemma_tools.unity --suite m1
# C# API 経由の実モデル15ケース×CPU/GPUComputeを検証
uv run --locked python -m embeddinggemma_tools.unity --suite runtime
# 保存・量子化・両backend実モデル照合と測定（NUnit / CLIとも20分上限）
uv run --locked python -m embeddinggemma_tools.unity --suite completion --timeout 1200
```

`--launch` を指定したときだけ起動コマンドを送る。起動済み Editor への通常の検証では省略し、毎回ウィンドウを前面へ移動しない。
`--project` / `--uloop` / `--output` で対象プロジェクト・CLI・証拠の保存先を明示できる。
`--timeout` は各コンパイル・テストの待ち時間で、既定900秒、範囲1～1200秒。
ハーネスはWindowsで欠落する `ALLUSERSPROFILE` を子プロセスにのみ補い、UPMの起動エラーを防ぐ。
ユーザーやシステムの環境変数は変更しない。

テスト対象は `EmbeddingGemma.Editor.Tests` assembly 内のclassを明示する。
`m1` は `M1ReferenceTests` の tokenizer・CPU・GPUCompute 計3件、`runtime` は
`TextEmbedderReferenceTests` のAPI経由CPU / GPUCompute 計2件（各15ケース）。単体契約テストはこの件数に混ぜない。
Scene / Prefab の未保存変更があれば `--unsaved-changes fail` で停止する。コンパイルにも外部Scene変更の停止オプションを指定する。
検証のために未保存編集を自動保存・破棄しない。

証拠は既定で `artifacts/unity-harness/<UTC timestamp>/` に保存する。

- `summary.json`: 対象・指定Unityバージョン・開始/終了時刻・結果・失敗理由
- `launch.json`（指定時）/ `compile.json` / `run-tests.json`: 引数・exit code・標準出力・標準エラー
- `get-logs.json`: Console の直近200件とstack trace（失敗後も取得を試みる）
- `test-results.xml`: uloop が生成した失敗時の NUnit XML（存在する場合にコピー）

exit code 0 は指定した scope の成功、1 は失敗。`--suite compile` は必ず `m1_reference_passed=false`。
M1 scope は `Success=true` だけでは合格にせず、3件すべて合格・失敗0・skip0・inconclusive0を必要とする。
runtime scope は両backendの2件すべて合格を必要とし、`runtime_reference_passed` に記録する。
runtime の合格だけで `m1_reference_passed` は true にしない。
未実行・不明なJSON・タイムアウトも成功にしない。開始時に以前の成功レポートを上書きする。
`m1_reference_passed=true` はこの3テストの合格を表す。`.sentis` 保存・量子化・性能測定を含むM1全体の完了ではない。
`completion`は `M1CompletionTests` の1件でfp32 / Float16重みの保存・再読み込み、CPU / GPUCompute各15ケース、warmup・45サンプル測定を検証する。
1件passed / failed=skipped=inconclusive=0の時だけ `m1_completion_passed=true`。
数値・段階別メモリは `artifacts/m1-completion/results.json`、モデルも同じGit管理外ディレクトリに保存する。
以前のresults.jsonのsuccessは実行開始時に無効化する。結果ファイルだけでなくNUnit / ハーネスの成否も確認する。
M1全体の完了監査は [検証記録](m1-completion-validation.md) に従い、ソースSHA・固定参照・実行結果・CIを照合する。

## M1 参照成果物の配置

コンパイルだけならモデルは不要。M1 数値検証では成功した Model reference and export run の artifact が必要。
run と artifact の対応、モデルrevision、`export-validation.json` のSHA-256を確認する。
モデル変換はローカルでやり直さず、CI成果物を約1.1GBダウンロードする。

リポジトリルートで（`<...>` は対象runの実値に置換）:

```powershell
gh run download <successful-run-id> --name <m1-reference-artifact-name> --dir artifacts/m1
cd tools
uv run --locked python -m embeddinggemma_tools.stage --source-commit <CI-checkout-SHA>
```

配置処理は固定 revision・生成元・全15件のPython結果・参照ベクトルの条件と3ファイルのSHA-256を監査し、
合格時だけ `Assets/M1Generated/model.pt2` を置換する。`artifacts/m1-stage.json` に小さいレポートを保存する。
監査失敗時には以前の成功レポートを無効化する。監査の成功はSentis合格ではない。
`reference.json` / `tokenizer.json` は `artifacts/m1/`、モデルは `Assets/M1Generated/model.pt2` をテストが参照する。
その後 `--suite m1` と `--suite runtime` を実行し、Sentis import・tokenizer・CPU・GPUCompute とAPIを確認する。
artifact の保持期間は3日。失効した場合は CI で再生成する。生成モデルをcommitしない。

## 確認済みの範囲

基盤 PR #1 の導入時は、Editorの起動・依存解決・コンパイルが成功し、M1テスト3件は参照未配置で失敗した。
後続 PR #3 のブランチで実モデル成果物の監査・配置が完了。空文字例外を修正後、
`--suite m1` は3件すべて、`--suite runtime` は両backendの2件すべて合格した。skip / failed / inconclusiveは0。
各backendの全15ケースでfp32一致を確認し、C#単体契約24件も合格。詳細は [ランタイム検証記録](m1-runtime-validation.md)。
基盤は PR #1 で main `60f906d` に統合済み。統合後の main CI（run `37758476415`）で既存33件とハーネス14件を合わせた47件が4環境で合格。
参照成果物の取得候補・期限・ソース対応と、次の保存・量子化・測定は [M1 計画](m1-plan.md) を参照。
クラウドUnity workflowはSecretsが必要なLinux CPUの手動補助検証のみ。M1のWindows CPU/GPUCompute合格はこのローカルハーネスで確認する。
詳細な履歴は [検証記録](m1-validation.md)を参照。
