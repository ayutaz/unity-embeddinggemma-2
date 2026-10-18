# 保存済みraw候補のローカル診断

更新: 2026-10-19。通常の数値回帰は[公開native配布](native-regression-distribution.md)を使います。この文書は、選択済み再生成世代`e9de158`の保存ZIPを使う歴史的なraw診断の取得方法を説明します。消失した別の旧世代graphを復旧したという意味ではありません。

## 明示的なローカル原本

`audio_generation_candidate.fetch_candidate`、`audio_candidate_distribution.fetch_inputs`、`audio_candidate_onnx_identity.fetch_native`は、`source_archive`の指定が必要です。対応するCLIと`selected_audio_generation fetch`には`--source-archive`を渡します。指定なしの旧raw取得は、network接続や出力作成の前に失敗します。

取得処理は、`tools/references/audio-candidate-e9de158.json`の固定`artifact_sha256`に対してZIP全体を照合します。必要なメンバーだけを取り出した後も、元のgraph・audit・input hashを照合します。配布用clean graphをraw graphとして認めず、raw loader / auditorの数値基準と世代選択を維持します。

モデルと元入力を取得する2経路では、`original` directoryに事前に準備した元参照JSON・audit JSONも照合します。保存ZIPからの読み込み中に、古いhosted artifactやgolden取得へfallbackしません。元参照は、例えば次の公開配布経路で別に準備します。`tools/`から実行するPowerShell例です。

```powershell
uv run --locked python -m embeddinggemma_tools.native_regression_distribution fetch-original --output ../artifacts/original-private
uv run --locked python -m embeddinggemma_tools.audio_generation_candidate fetch --source-archive $env:EMBEDDINGGEMMA_RAW_ARCHIVE --original ../artifacts/original-private --case silence --output ../artifacts/raw-candidate-private
```

`EMBEDDINGGEMMA_RAW_ARCHIVE`は実行者の私的な保存先を指します。raw ZIP・graph・診断ログは公開しません。取得関数のhash照合と、後続のモデル実行・数値比較・GPU受け入れを区別します。

## 来歴とhosted経路

ローカルreaderは`transport=verified_local_archive`、完全な`archive_sha256`とサイズ、`provider_metadata_revalidated=false`を返します。ZIPの固定hashに結び付く歴史的なproducer情報と、その場でGitHubから取得したmetadataを混同しません。新しいprovider IDやworkflow metadataを捏造せず、世代・GPUの合格も付与しません。

手動workflowの`diagnose_candidate_onnx_identity`と`preserve_candidate_distribution`は、historical raw原本の取得を伴うためhosted実行を廃止しました。既存のinput名を残し、選択時はcheckout・依存download・モデル取得・artifact uploadの前に明示的に失敗します。ローカルCLIの診断・保存処理は継続できます。

通常の公開配布による全6信号回帰と、元JSON / 6 raw inputだけを保存する`preserve_original_inputs`は維持します。後者は別の保存済み入力成果物を使い、旧rawモデルZIPを取得しません。

## 旧成果物の除去条件

[移行のTDD・実archive確認](results/raw-candidate-archive-retirement-tdd-20261019.json)では、13件Red→13件Greenの後、local golden必須の追加3件Red / 12件passを経て関連170件Green・skip0、actionlintが成功しました。保存ZIP 4,938,559,326 bytesの私的完全2コピー・全CRCと、実際の全6 case・raw graph 2 profile・native ONNX 2 profileのローカル取得を照合しました。新しいモデル推論・数値回帰・GPUの合格ではありません。最初のchild起動はimport前提の不一致でarchive I/O前に失敗し、その記録を保持した上で、明示した同一interpreterによる実確認はexit0でした。source全体のRequired CIとmainへの統合は別に残ります。

コード変更だけでhosted artifactは削除しません。現行mainと実行中のjobに必要な取得依存が残らないこと、完全ZIP・必要graphの私的バックアップ、代替経路の全数値回帰、公開payloadのprivacy確認を先に検証します。削除した場合は実DELETEと後続GETの結果を別に記録し、workflow runや残す数値成果物を一緒に消しません。GitHubの旧PR / cacheに残る個人情報は、別のprovider対応として追跡します。
