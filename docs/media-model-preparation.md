# 画像・音声・テキストモデルの準備

変換済みの全6 moduleは、正式SDK Releaseとは別の実験用モデルpreviewとして公開しました。全assetの実payload guardとremote size/digest、本文、producer tagを[照合済み](results/model-preview-publication-567befb-20261018.json)です。匿名の空cache取得・中断復帰・最終consumer受け入れは進行中で、第三者導入の完了とはまだ扱いません。

| 形式 | 配布元 | bundle.jsonの独立SHA256 | object合計bytes |
| --- | --- | --- | --- |
| FP32 | [モデルpreview](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/models-preview-567befb-fp32) | `40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647` | 8,584,030,668 |
| Float16重み | [モデルpreview](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/models-preview-567befb-float16) | `8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e` | 5,874,675,948 |

各配布は820 object・553固有Sentisモデルです。LICENSE.txt、NOTICE.txt、MODIFICATIONS.txtも取得・保持してください。モデルのproducerは固定source `567befbd64287a706edaac59c978927e0277188a`で、SDKの実行sourceとは別に記録します。Float16重みの保存サイズは速度・全演算FP16化・モバイル対応の保証ではありません。

## Release assets形式の準備

上表のpreviewはGitHub Release assetsを使います。全objectの出所・hash・モデル記述内の個人情報は別に監査しています。公開した事実と、空cacheからの取得・GPU実行の成功は別々に判定します。

descriptorの `object_layout` は省略時または `nested` では従来の `objects/<sha>.<suffix>` URL、`flat` ではdescriptorと同じURL階層の `<sha>.<suffix>` を使います。ローカルの保存名はどちらも `objects/<sha>.<suffix>` のままです。レイアウト変更もdescriptorのSHA256に含まれ、未知の値や不一致pinはobject転送・出力作成の前に拒否します。別サーバーのobject URLをdescriptorから受け取る機能ではありません。

全module・全宣言objectを準備した場合は、hashを検証した元descriptorのbytesをatomicに保存します。JSONの再整形・改行・キー順・文字escapeで固定SHAを変えません。一部moduleだけを取得した場合や未参照objectを除く場合は、準備していない宣言を含めず新しいdescriptorを作ります。receiptの `source_descriptor_bytes_preserved` と `prepared_descriptor_sha256` で、入力pinと準備後のdescriptorを区別します。[TDD checkpoint](results/media-descriptor-byte-preservation-tdd-20261018.json)は10件のRedから関連87件Greenを確認しました。LF / CRLF等の入力bytesは検査しましたが、Mac実機やGPU実行の合格ではありません。

maintainer用の `embeddinggemma_tools.media_release_assets` は、hash固定の既存text / image bundleと新audio bundleを入力し、全6 moduleを持つ新しい配布候補を組み立てます。コピー対象の全source objectを照合してから新規出力を作成し、audio moduleだけを置換します。元のmodel・係数・module内容は変更せず、共有objectは重複排除します。producerと両入力descriptorのpin・producerを記録し、コピー先も全hash確認後にdescriptorと `release-assets.json` を出力します。既存出力は上書きしません。

```powershell
uv run --no-project --python 3.13 python -m embeddinggemma_tools.media_release_assets `
  --base $reviewedBaseBundle --base-sha256 $reviewedBaseDescriptorSha256 `
  --audio $reviewedAudioBundle --audio-sha256 $reviewedAudioDescriptorSha256 `
  --output $newDistributionDirectory --producer $fixedDistributionSource `
  --precision fp32
```

これはモデル変換・GPU実行・privacy承認・uploadを行うコマンドではありません。`privacy_review_completed=false` のinventoryをそのまま公開承認として使いません。GitHubの[Release制限](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)に合わせ、license / notice用の枠を残して1,000 asset以内、各objectは2 GiB未満を要求します。モデルをGit / Git LFSへ追加しません。

[TDD checkpoint](results/media-flat-assets-and-composition-tdd-20261018.json)では、flat取得の8件とcomposition未実装の12件の失敗を確認しました。実HTTPSのfixture拡張前の2 setup errorも区別して保持し、修正後はHTTPS取得・warm再利用・404・同サイズ破損修復を確認しています。小さい不透明fixtureの成功は、実モデルの公開取得・GPU数値合格を証明しません。

## 取得対象と検証の境界

対象は `text`、`image-square`、`image-landscape`、`image-portrait`、`audio-1119`、`audio-839` です。公開previewはFP32 / Float16重み、Sentis 2.6.1、GPUCompute、元モデルrevision `914f7f89142e33e77833254d9c9b90c3cef7303b` に対応します。公開先からの取得成功と最終環境の受け入れは引き続き別に確認します。

ツールは変換済みgraphを取得します。元checkpointからの変換を実行したという意味ではありません。bundleのproducer commit、元モデルrevision、objectごとのサイズ・SHA256を保持し、GPU実行・数値合格は別に検証します。SDKの実行commitとモデルを生成したcommitも別に記録してください。

## コマンド

リポジトリの `tools/` で実行します。取得処理はPython標準ライブラリだけを使い、モデル推論やTorchの導入は行いません。

```powershell
uv run --no-project --python 3.13 python -m embeddinggemma_tools.media_setup --help
```

上表の固定URLと、独立に確認したdescriptorのSHA256を指定します。SHA256を同じ未検証サーバーから自動的に信頼する手順にはしません。

```powershell
$bundleDescriptorUrl = 'https://github.com/ayutaz/unity-embeddinggemma-2/releases/download/models-preview-567befb-fp32/bundle.json'
$bundleDescriptorSha256 = '40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647'
$bundleDestination = Join-Path $env:LOCALAPPDATA 'EmbeddingGemmaModels/fp32'
uv run --no-project --python 3.13 python -m embeddinggemma_tools.media_setup `
  --descriptor-url $bundleDescriptorUrl `
  --descriptor-sha256 $bundleDescriptorSha256 `
  --output $bundleDestination
```

`--modules` を省略すると全6 moduleです。必要なmoduleだけを取得する場合は、例えば `--modules text image-square` を追加します。共通objectは一度だけ取得し、出力bundleには選択したmoduleとその参照objectだけを記載します。モデルをGitやGit LFSへ追加しないでください。

## 結果と再実行

- 正常完了は終了code0。`bundle.json` と `preparation.json` を出力します。`preparation.json` の `success`、`producer`、`model_revision`、`descriptor_sha256`、選択module、取得・再利用件数とbytesを確認します。
- 各objectをサイズ・SHA256まで検証してから確定し、最後にbundle descriptorを公開します。取得途中のbundleを利用側へ渡しません。
- tokenizer、text model、media backbone・tokenizer config、音声preprocessorの必須参照を取得前に検証します。model参照は`.sentis`、module backendはGPUCompute、model segment数はoperation数+1を要求します。欠落したrecipeを準備成功にしません。この確認はUnityのrecipe全体・graphの実行可能性の検証を代替しません。
- 同じコマンドの再実行でも、既存objectを全体ハッシュで検証します。正常objectは再利用し、サイズが同じ破損も検出して再取得します。
- 通常の中断・転送失敗では未完了objectの一時ファイルを片付け、完了済みobjectを残します。プロセス強制終了で一時ファイルが残った場合は、次の準備が排他ロックを取得してから所有形式の一時ファイルを回収します。同じ出力先と固定参照で再試行してください。転送処理に入った後の捕捉可能な失敗は `preparation.json` に失敗として記録します。以前の公開済みdescriptorは成功するまで置き換えません。強制終了した呼び出し自体の成功判定にはしません。
- Ctrl+Cはキャンセルのメッセージと終了code130を返します。object準備に入った後は `success=false` / `error_type=KeyboardInterrupt` を記録し、一時ファイルを削除して排他ロックを解放します。descriptor受信中など準備開始前には新しいreceiptがない場合があります。再試行は同じ固定URL・SHA・出力先を使い、完成済みobjectをhash照合して再利用します。ブロッキングI/O中の即時停止時間は保証しません。
- descriptorのURL・SHA256・revision自体の検証に失敗した場合は、転送準備に入る前に終了するため、出力receiptがない場合があります。終了codeと標準エラーも保存してください。

HTTP 404・期限切れ・認証不足は取得失敗です。ハッシュ検証を無効にしたり、似た名前のモデルへ置き換えたりせず、公開配布側の固定参照を確認してください。別revisionを使う場合は、producer・graphと数値参照を改めて監査します。

取得receiptの `inference_executed` と `gpu_validated` は常に `false` です。取得成功を、Unityの実モデル推論・CPU fallbackなし・検索順位・GPU互換性の合格として数えません。

## 残る受け入れ

公開配布後、既存model cacheを使わない取得で全6 moduleの出力を照合し、warm再利用・中断からの再試行・破損修復を実データでも確認します。そのモデルを固定Git consumerに導入し、対応する実GPU・数値・検索順位・正常終了を別記録で検証してから、この手順を利用者向け完成手順に更新します。取得契約テストは41件です。追加の必須recipe検証は16件の意図した失敗から修正後41件成功を確認し、実FP32 / Float16重みの各820 objectを宣言したdescriptorのmetadata読み取りも確認しました。[修正と確認範囲](results/media-setup-required-recipe-guard-20261018.json)。小さな不透明fixtureと既存descriptorの読み取りは、実objectの再取得・再hash・GPU実行や公開empty-cache受け入れを代替しません。

## 実objectによるローカルHTTPSの検証

[f09317e checkpointの統合source照合](results/real-local-https-integration-source-20261018.json)では、当時実行したCLI・HTTPS fixture・選択test sourceのGit blob一致を確認した。後述の排他・回収修正は別sourceで検証している。公開取得とGPUの未確認は維持する。

2026-10-18の最初の `f09317e` checkpointは、recipe guard修正時と同じ取得実装で実行した。[実データの監査](results/real-local-https-media-recovery-f09317e-20261018.json)は、既存FP32 bundleの全6 module・820 object・8,583,084,520 bytesを、127.0.0.1だけで待ち受けるHTTPSから空の出力cacheへ転送した記録である。取得元は既存ローカルbundleであり、公開元からの新規利用者の取得は未確認のままとする。

| 確認した条件 | 結果 |
| --- | --- |
| 未信頼の一時TLS証明書 | CLI exit1、object request0、出力cache未生成 |
| descriptorの404 / 410、pin違い、model revision違い | 各exit1、object request0 |
| 空cacheで3番目のHTTP responseを途中切断 | exit1、完成した2 objectを保持、一時ファイルなし、bundle未公開 |
| 同じcacheへの再試行 | exit0、2 object再利用・818 object受信、全6 module完成 |
| warm再利用 | exit0、820 objectをhash照合して再利用、object request0 |
| 1,572,864 bytesの実FP32重みを同サイズで破損 | exit0、対象1件だけ再受信・819件再利用 |
| 同じ重みの404から再試行 | 失敗時は旧descriptorを保持してexit1、再試行は1件修復・819件再利用でexit0 |

最後に元と出力の各820ファイル、計1,640ファイルをサイズ・SHA256で独立に再照合し、sourceとdestinationが同一file / hardlinkでないことも確認した。元descriptorとproducer / revision、元のuser変更4ファイルも保持した。入力descriptorのpinと、出力descriptorのJSON内容の一致は分けて記録している。

TLSの信頼は検証用child processの `SSL_CERT_FILE` だけに指定し、OSの信頼storeは変更していない。証明書やモデルを外部へ公開していない。検証した中断はHTTP responseの途中切断であり、Ctrl+C・強制終了・OS停止の実証にはしない。取得receiptのinference / GPU flagはfalseで、GPU数値合格や転送速度の保証へ広げない。

既存41件に、実CLI・HTTPSを通る7件を加えた48件が成功した。最初の7件のsetup errorは未実装のHTTPSテストfixtureによるもので、実装後は証明書拒否・転送・中断・修復を確認した。HTTPS fixtureは既存のOpenSSLから一時証明書を生成する。モデル・証明書・秘密鍵・生の転送logはGitに追加しない。

この観測で、実objectと通常のCLIを使うローカル転送・故障注入・修復は確認できた。公開した固定URLからのempty-cache取得、その出力での固定Git consumer・GPU・操作受け入れは引き続き必要である。

## 強制終了後の回収と同時実行

[強制終了復旧の統合source照合](results/media-crash-recovery-integration-source-20261018.json)で、実行したCLI・HTTPS fixture・選択test moduleの一致も確認した。

後続のsource `85a2d71` は、[実CLIを強制終了した復旧監査](results/real-media-crash-recovery-85a2d71-20261018.json)に対応する。修正前は、再試行が成功しても前のプロセスの一時ファイルが残り、同じ出力先への二重実行も拒否しなかった。所有形式の一時ファイル・二重writer・実HTTPS受信中の強制終了の3件で意図したRedを確認し、修正後の関連51件は成功した。

同じ出力先ではOSの排他ロックを取得した1つの準備だけが動く。2つ目はobjectを取得したりownerのreceiptを書き換えたりせず、終了code1となる。ロックは通常終了・プロセス終了でOSが解放する。空の `.preparation.lock` は出力先に保持する。実装はWindowsの[msvcrt.locking](https://docs.python.org/3.13/library/msvcrt.html#msvcrt.locking)とUnixの[fcntl.flock](https://docs.python.org/3.13/library/fcntl.html#fcntl.flock)を使い、このcheckpointの実機結果はWindowsである。

排他を取得した準備は、出力rootとobjects直下にある既知の形式・UUID名の一時ファイルだけを回収する。名前の異なる利用者ファイルやsymlinkは回収しない。receiptの `stale_pending_files_removed` に回収件数を記録する。descriptor・model revision・object hashの検証、完了後のdescriptor確定は維持する。

実FP32 bundleのコピーcacheでは、9MiB重みの受信中に、検証が起動したCLIだけを強制終了した。別CLIは拒否され、ownerの一時ファイル・旧receipt・descriptorを保持した。強制終了で残った受信済み一時ファイルは次の準備で1件回収し、重み1件を受信・819件再利用して終了code0。その後のwarmも全6 module・820件再利用・新規受信0で終了code0だった。出力820ファイルを独立に再hashし、元重み・元descriptorとuser変更4ファイルも照合した。

この確認はOSが終了させた所有CLIからの復旧で、Ctrl+C signalの実証ではない。公開取得・GPU推論の受け入れは未完了のままである。過去の48件成功やf09317eの実データ受信を、変更後sourceの全条件再実行として数えない。

## Ctrl+Cによる中断と再試行

source `2b6f45d` では、descriptor受信中と実HTTPS object受信中のCtrl+Cを2件のRedで確認し、CLIの境界でKeyboardInterruptを捕捉する修正後、関連53件が成功した。[実Ctrl+Cと実FP32重みの監査](results/real-media-ctrl-c-2b6f45d-20261018.json)。旧sourceでは例外のtraceを表示し、Windowsの実CLIはキャンセルcode130を返さなかった。objectの検証・排他・receipt・後始末の処理は変更していない。

Windowsの実イベントは、非表示の新規consoleに検証用relayと所有CLIだけを起動し、[GenerateConsoleCtrlEvent](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent)のCTRL_C_EVENTを使った。送信前にconsole参加PIDがこの2件だけであることを確認し、呼び出し元のconsoleへ接続しない。先行fixtureではvenvの起動用processも参加し、検査で送信せず停止した。同じPythonのbase interpreterで2件だけを起動するようfixtureを直してから、意図したRedとGreenを確認している。POSIX分岐は所有PIDにSIGINTを送るが、ここでの実行結果はWindowsであり、他OSは最新CIで別に判定する。

既存のコピーcacheを使った実FP32の9MiB重みでも、受信済み一時ファイルを確認してからイベントを送信した。CLIは強制killなしで終了code130、キャンセル表示、失敗receipt、旧descriptor保持、一時ファイルなしとなった。その後の再試行は1件受信・819件再利用、warmは全6 module・820件再利用・新規受信0で、どちらもexit0だった。出力820 objectのサイズ・SHA256をCLIとは別に照合し、元重み・固定source・既存user4ファイルを保持した。

SSL受信のpauseはイベント送信後に解除した。この試行は実際のCtrl+C・後始末・再試行の確認であり、通信が停止したままの即時キャンセル時間、公開元からの取得、GPU推論や優先1〜3全体の合格ではない。source descriptorの入力pinと、選択module・object順序を整えて書き出す出力JSONのbyte hashも区別する。JSON内容・全objectは照合し、入力pinを変更していない。
