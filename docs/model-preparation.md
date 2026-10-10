# 検索サンプル用モデルの取得・変換・配置・更新

対象: Unity 6000.3.16f1 / Sentis 2.6.1、固定`google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`。TextSearchの6文書 / 4queryとM1の15入力を使う。モデルをGit / LFSへコミットしない。

モデル参照と`.pt2`の生成はActions、Sentisの`.sentis`変換と実GPU検証は認証済みEditorで行う。ローカル操作にGitHubのUnity Secretsは不要。Linux CPUの任意手動Unity workflowは別経路で、未実行を合格にしない。

## 1. 成功したCI成果物を取得する

新規checkoutでも最初に`tools/`で`uv sync --locked`を実行する。Pythonはuvだけを使う。
最新の実装ブランチまたは統合後mainの`CI`で、Python・lint・model-reference・package・Required CIが成功したrunを選ぶ。PRのhead SHAと実際のcheckout SHAは異なる場合がある。

リポジトリルートからPowerShellで実行する。`$runId`を選んだ成功runの番号に置き換える。

```powershell
$runId = <successful-run-id>
gh run view $runId --repo ayutaz/unity-embeddinggemma-2
$referenceName = gh api "repos/ayutaz/unity-embeddinggemma-2/actions/runs/$runId/artifacts" --jq '.artifacts[] | select(.name | startswith("m1-reference-")) | .name'
$checkoutSha = $referenceName -replace '^m1-reference-', ''
gh run view $runId --repo ayutaz/unity-embeddinggemma-2 --log | Select-String -SimpleMatch $checkoutSha
gh run download $runId --repo ayutaz/unity-embeddinggemma-2 --name $referenceName --dir "artifacts/download/$runId"
```

checkoutログの40桁SHA、artifact名末尾のSHA、`export-validation.json`の`metadata.source_commit`が一致することを確認する。runが失敗・cancel・未完了、artifactがexpired、SHAが不一致なら使用しない。
約1.1GBのモデルと約32MBのtokenizerを含む。`reference.json`、`search-reference.json`、`tokenizer.json`、`model.pt2`、`export-validation.json`が必要。

## 2. 監査して配置する

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.stage --source "../artifacts/download/$runId" --project .. --source-commit $checkoutSha --search
```

`stage --search`は固定revision・実際の生成SHA・入力条件・M1の15件・検索10件のexport cosine >= 0.999999・全順位・SHA-256を監査する。
合格時に`Assets/M1Generated/model.pt2`と`artifacts/m1/`の参照・tokenizerを配置し、`artifacts/m1-stage.json`を保存する。不正な成果物では既存モデルを置き換えず、監査の成功を無効にする。
配置成功はSentis推論の合格ではない。

## 3. サンプルを導入し、Sentis形式へ変換する

別のUnityプロジェクトでパッケージを導入し、Package Managerのサンプル一覧から**Text Search**をImportする。
検証用に空のconsumerを作る場合は、リポジトリの`tools/`から次を使う。既存プロジェクトを上書きしない。

```powershell
uv run --locked python -m embeddinggemma_tools.package --consumer ../artifacts/c --sample --automation
uv run --locked python -m embeddinggemma_tools.stage --source "../artifacts/download/$runId" --project ../artifacts/c --source-commit $checkoutSha --search
```

consumerをUnity 6000.3.16f1で開き、依存解決・compileの完了を確認する。上の`--automation`は検証用uloopと、外部Acceleratorを無効にするプロジェクト設定を加える。
起動方法と失敗ログの扱いは [自動操作手順](automation.md)。起動待ちが失敗しても、後続の成功と混ぜない。

Windowsではconsumerの絶対パスを短くする。今回の長いconsumerパスでは、267文字のuloopテンプレートをPowerShellは認識したがUnity側で`DirectoryNotFoundException`となり、起動接続待ちもタイムアウトした。短い`artifacts/c`で同じ新規導入を行うと起動・compile・自動操作が成功した。既存プロジェクトの移動やLibraryのコピーで回避せず、空の短いパスへ導入し直して検証する。

Editorの**Tools → EmbeddingGemma → Prepare Text Search Models**を実行する。
このメニューは`stage --search`の成功記録と配置済みファイルのhashを読み戻し、`.pt2`をSentisでimportして保存する。モデルをダウンロードしない。

```text
Assets/StreamingAssets/EmbeddingGemmaTextSearch/
  model-fp32.sentis
  model-float16.sentis
  tokenizer.json
  preparation.json
```

`preparation.json`には成功 / 失敗、生成元SHA、Unity版、各ファイルのサイズとSHA-256を記録する。Float16は保存重みの形式であり、全演算fp16化や速度改善を意味しない。
ルート検証プロジェクトではこの生成先をGit管理外にしている。別プロジェクトでもモデル・tokenizerをコミットしない設定を用意する。

## 4. 検索を操作する

インポートした`TextSearch.unity`を開き、Playする。検証用consumerでは`Assets/EmbeddingGemmaTextSearch/TextSearch.unity`。

1. `.sentis`と`tokenizer.json`のパスを指定し、CPUまたはGPUComputeを選択する。
2. **モデルと文書を準備**で6文書を一度だけ埋め込む。
3. 検索文を入力し、**検索**で全6件の順位とcosineを表示する。
4. **解放 / モデルを変更**で推論器を解放し、形式・backendを変更して再準備できる。
5. Play停止・GameObject無効化時にも推論器を解放する。

Float16重みを試す場合は`model-float16.sentis`を指定する。GPU非対応・モデル未準備・不正な入力・推論失敗は画面に表示し、CPUへ自動切り替えしない。
同点は文書IDのordinal順。固定例の成功から任意の検索文の品質を保証しない。
現在のサンプルは通常のファイルパスを同期読み込みする。Androidのjar内StreamingAssetsなどを含むPlayer配置は後続の端末検証で対応を確認する。

## 5. 更新とartifact失効

Git URL consumerの依存解決は、Unityが作成したlockとmanifestを次のコマンドで照合できる。`<consumer-path>` と `<40-digit-commit>` は実際のprojectと導入対象commitへ置き換える。これはGit解決の監査であり、compile・実モデル・GPU実行の証拠ではない。

```powershell
uv run --locked python -m embeddinggemma_tools.package --verify-git-consumer <consumer-path> --git-revision <40-digit-commit>
```

モデルrevision・生成コード・依存・入力条件を変更したらActionsで参照を再生成する。現在のstageは上記の固定revisionだけを受け入れ、勝手に別モデルへ更新しない。
artifact保持は3日。失効した場合は`CI`を対象ブランチで再実行し、新しい成功runとcheckout SHAで取得・監査・準備する。
再準備の前にサンプルでリソースを解放する。途中の準備失敗や以前の結果を新しい成功として扱わず、`preparation.json`・Console・監査結果を確認する。

実モデルのWindows回帰には、`tools/`から`uv run --locked python -m embeddinggemma_tools.unity --suite search --timeout 1200`を使う。`--project`の既定はこのリポジトリの検証projectで、consumerを自動選択しない。
4件（fp32 / Float16重み × CPU / GPUCompute）がすべてpassed、failed / skipped / inconclusive = 0の時だけ検索参照合格。
結果は`artifacts/m2-search/results.json`とハーネス出力に分けて保存する。[検証記録](m2-search-validation.md)に実行済みの範囲を記載する。

### 別consumerで実モデル回帰を行う

`package --sample`で入るsampleテストは小さいモデルの契約であり、実モデル参照fixtureは配布UPMに含まない。consumerの4条件を検証するには、検証リポジトリのfixtureを明示的に配置する。今回のconsumerで実行したfixtureは元の `TextSearchReferenceTests.cs` とSHA-256一致を確認した。

consumerのEditorを停止し、リポジトリルートから次を実行する。`artifacts/c`は作成済みの短いconsumer。既存の検証フォルダがある場合は上書きせず、内容を照合してそのfixtureを使用する。

```powershell
$validationPath = 'artifacts/c/Assets/SearchValidation'
if (Test-Path -LiteralPath $validationPath) { throw 'Validation directory already exists; compare the existing fixture first.' }
New-Item -ItemType Directory -Path $validationPath | Out-Null
Copy-Item -LiteralPath Assets/Tests/Editor/TextSearchReferenceTests.cs -Destination $validationPath
Copy-Item -LiteralPath Assets/Tests/Editor/TextSearchReferenceTests.cs.meta -Destination $validationPath
$validationAssembly = @{
  name = 'EmbeddingGemma.Search.Consumer.Validation'
  references = @('EmbeddingGemma.Runtime', 'Unity.InferenceEngine')
  includePlatforms = @('Editor')
  optionalUnityReferences = @('TestAssemblies')
  autoReferenced = $false
}
$validationAssembly | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$validationPath/EmbeddingGemma.Search.Consumer.Validation.asmdef" -Encoding utf8
```

consumerを同じUnity版で開き、依存解決・compileと上記の`stage --search` / 準備メニュー完了を確認した後、`tools/`から実行する。

```powershell
uv run --locked python -m embeddinggemma_tools.unity --project ../artifacts/c --suite search --timeout 1200
```

consumerの数値結果は `artifacts/c/artifacts/m2-search/results.json`。ハーネスの保存先は既定では検証リポジトリの `artifacts/unity-harness/` で、`--output`でも明示できる。fixture欠落によるNoTestsFound、GPU skip、接続切断を4条件合格にしない。

## ローカル検証の待機を減らす

自動検証用consumerを`--automation`で新規作成すると、Unity Acceleratorをプロジェクト単位で無効にする。通常consumerやグローバルEditor設定は変更しない。検証で到達できない外部Acceleratorへのタイムアウトが繰り返し観測されたため、継承した接続先による待機を避ける。

既存consumerではProject Settings > EditorのCache ServerをDisabledにしてから、必要なら再起動する。[UnityのCacheServerMode](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/CacheServerMode.html)でDisabledはプロジェクトの外部キャッシュを無効にする設定。

重いモデル準備と実モデル照合はまとめて実行し、ドキュメントだけの編集で繰り返さない。準備メニューは入力モデルSHA-256、生成元commit、Unity版、tokenizer、出力3ファイルのサイズとSHA-256が一致した場合だけ再利用する。旧receiptや破損・欠落・不一致は再変換し、失敗時に過去のsuccessを残さない。再利用ではModelAsset読み込み・変換・書き込みを省き、全体Refreshも実行しない。

従来実装のWindows consumerの単回測定は初回202.76秒、再利用174.52秒。出力hash・receipt・全ファイル更新時刻は不変だった。[検証記録](m2-search-validation.md)に当時のTDD・CLI完了応答・画面操作と未解決ログを保存している。

その後、Windows Editorの1MiB以上の完全なSHA-256照合をWindows標準の`certutil.exe`へ委譲した。絶対パスでsystem utilityを起動し、shellを使わずウィンドウも表示しない。終了codeが0で、64桁hexのhash行が1つだけある場合に採用する。起動失敗・timeout・不正出力では従来の完全なSHA-256へ戻る。他のOSと小さいファイルは従来の処理を維持する。照合をサイズ・mtimeだけに置き換えない。

同じWindows consumer / 実モデルのキャッシュ再利用は変更後の単回測定でEditor内7.38秒だった。モデルとreceiptのサイズ・更新時刻は不変、入力と出力の完全hashも一致した。過去174.52秒はCLIを含む別時点の測定であり、正確な改善率やcold-cacheの速度を示さない。初回変換・domain reload・他OSの高速化は未測定。[高速化の実測記録](results/m2-editor-sha256-windows-20261010.json)に条件と失敗・未実行を記載する。

テストが数値を保存していても、CLIの完了応答を取得できなければハーネス合格にはしない。接続切断時の`SafeToRetry: false`を無視して同じ重い操作を自動再実行しない。

## 検証済みモデルを再利用する場合

すでに成功runから取得した完全なモデル成果物がある場合は、新しい検索参照だけを`search-reference-<checkout-SHA>` artifactで取得できる。
この小さいartifactは`search-reference.json`と`export-validation.json`を含み、重みとtokenizerを含まない。

```powershell
gh run download <successful-search-run-id> --repo ayutaz/unity-embeddinggemma-2 --name search-reference-<search-checkout-SHA> --dir artifacts/download/search-reference
cd tools
uv run --locked python -m embeddinggemma_tools.stage --source <cached-complete-artifact-directory> --source-commit <model-checkout-SHA> --search --search-source ../artifacts/download/search-reference --search-source-commit <search-checkout-SHA>
```

双方のSHAはそれぞれ成功runの実際のcheckoutログで確認する。モデル側は元の15件とファイルhashを再監査し、検索側は全10件・順位・hashを別に監査する。
固定モデル・revision・export条件・依存の全metadata（生成SHAを除く）とtokenizerのSHA-256が一致する場合だけ組み合わせる。異なるモデルや依存を互換と仮定せず、不一致は失敗にする。
`m1-stage.json`の`source_commit`はモデル生成元、`search_source_commit`は検索参照生成元。別の検索export報告は`artifacts/m1/search-export-validation.json`へ保存する。
Sentis準備報告も`sourceCommit`と`searchSourceCommit`を別に記録し、実モデル検索の新しい結果で4条件を照合する。キャッシュ監査だけでは検索合格にしない。
