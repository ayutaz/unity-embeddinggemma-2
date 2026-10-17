# 画像・音声・テキストモデルの準備

この取得ツールは開発中です。変換済みモデルの公開配布先はまだ確定しておらず、第三者が空の環境から全6 moduleを取得する実証も未完了です。以下のURL変数は、公開後に固定commitを含む実URLへ置き換えます。過去の期限付きActions artifactや開発者のローカルcacheを、利用者の取得元として扱いません。

## 取得対象と検証の境界

対象は `text`、`image-square`、`image-landscape`、`image-portrait`、`audio-1119`、`audio-839` です。現在の配布候補はFP32、Sentis 2.6.1、GPUCompute、元モデルrevision `914f7f89142e33e77833254d9c9b90c3cef7303b` に対応します。候補の820 object、8,583,084,520 bytesは開発環境でサイズ・SHA256を照合済みですが、公開先からの取得成功ではありません。

ツールは変換済みgraphを取得します。元checkpointからの変換を実行したという意味ではありません。bundleのproducer commit、元モデルrevision、objectごとのサイズ・SHA256を保持し、GPU実行・数値合格は別に検証します。SDKの実行commitとモデルを生成したcommitも別に記録してください。

## コマンド

リポジトリの `tools/` で実行します。取得処理はPython標準ライブラリだけを使い、モデル推論やTorchの導入は行いません。

```powershell
uv run --no-project --python 3.13 python -m embeddinggemma_tools.media_setup --help
```

公開後、配布監査で示された固定URLと、独立に確認したdescriptorのSHA256を指定します。SHA256を同じ未検証サーバーから自動的に信頼する手順にはしません。

```powershell
$bundleDescriptorUrl = '公開後の固定commitを含むHTTPS bundle.json URL'
$bundleDescriptorSha256 = '配布監査で確認した64桁のSHA256'
$bundleDestination = 'C:\EmbeddingGemmaModels\fp32'
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
- 中断・転送失敗では未完了objectの一時ファイルを片付け、完了済みobjectを残します。同じ出力先と固定参照で再試行してください。転送処理に入った後の失敗は `preparation.json` に失敗として記録します。以前の公開済みdescriptorは成功するまで置き換えません。
- descriptorのURL・SHA256・revision自体の検証に失敗した場合は、転送準備に入る前に終了するため、出力receiptがない場合があります。終了codeと標準エラーも保存してください。

HTTP 404・期限切れ・認証不足は取得失敗です。ハッシュ検証を無効にしたり、似た名前のモデルへ置き換えたりせず、公開配布側の固定参照を確認してください。別revisionを使う場合は、producer・graphと数値参照を改めて監査します。

取得receiptの `inference_executed` と `gpu_validated` は常に `false` です。取得成功を、Unityの実モデル推論・CPU fallbackなし・検索順位・GPU互換性の合格として数えません。

## 残る受け入れ

公開配布後、既存model cacheを使わない取得で全6 moduleの出力を照合し、warm再利用・中断からの再試行・破損修復を実データでも確認します。そのモデルを固定Git consumerに導入し、対応する実GPU・数値・検索順位・正常終了を別記録で検証してから、この手順を利用者向け完成手順に更新します。取得契約テストは41件です。追加の必須recipe検証は16件の意図した失敗から修正後41件成功を確認し、実FP32 / Float16重みの各820 objectを宣言したdescriptorのmetadata読み取りも確認しました。[修正と確認範囲](results/media-setup-required-recipe-guard-20261018.json)。小さな不透明fixtureと既存descriptorの読み取りは、実objectの再取得・再hash・GPU実行や公開empty-cache受け入れを代替しません。

## 実objectによるローカルHTTPSの検証

2026-10-18。CLIの取得実装はrecipe guard修正時と同じGit blobで、test source `f09317e` のfixtureを使った。[実データの監査](results/real-local-https-media-recovery-f09317e-20261018.json)は、既存FP32 bundleの全6 module・820 object・8,583,084,520 bytesを、127.0.0.1だけで待ち受けるHTTPSから空の出力cacheへ転送した記録である。取得元は既存ローカルbundleであり、公開元からの新規利用者の取得は未確認のままとする。

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
