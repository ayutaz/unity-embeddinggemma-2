# 正式リリースまでの計画

確認日: 2026-10-19。対象はテキスト・画像・音声のGPU推論と横断検索です。[ゴール](goal.md)の必須環境・精度条件を維持します。

直近の到達点は **#99までの統合と#95・#96の整理、公開経路の実数値回帰、画像のGPU反復測定と画像・音声のCPU診断** です。main `c6dcde4` は#99 head `cc92d4b`と同treeで、全15 CI job・公開履歴・公開identity・provider署名を照合しました。元の特徴量に対するCPU診断は画像3・音声2条件とも中間値に不一致が残ります。両精度の公開取得、実マイクGPU登録、実画面、Windows混在Player全4起動とテキストの別Player 2起動の既存合格を保持し、残る操作・性能・必須実機と公開前監査を進めます。

## 現在地

正式SDK tag・Releaseは未作成です。Sentisモデル配布用のpreview 2件とnative回帰データ用のpreview 1件を公開済みで、正式SDK公開とは区別します。以下は同日の実データ・GitHub照合に基づくcheckpointであり、正式候補全体の合格ではありません。

- [#99の受け入れ](results/manual-native-distribution-integration-cc92d4b-20261019.json)：run `39163550227` は全15 job成功・skip0。7成果物の実digest・CRC・私的2コピーと、元参照824,320値の超過0・集計一致を確認しました。main `c6dcde4` の同tree・公開著者・provider署名・全履歴も照合。手動回帰の期限付きraw artifact依存を固定公開配布へ移し、元の数値gateを維持しました。SDK package / Runtimeは不変で、新しいGPU受け入れやmain push CIの別実行成功とは区別します。
- iOS設定は[初回19件Red→19件Green](results/ios-build-profile-tdd-20261019.json)と、[Xcode構成の追加8件Red / 15件pass→最終23件Green](results/ios-xcode-scheme-tdd-20261019.json)を保存。Windows / Androidを含む最終関連44件が成功・skip0です。Release / IL2CPP / Metal・Device SDKに加え、Xcode Release設定とDebug / Releaseへの復元を確認し、署名・provisioning値は変更・掲載しません。実export・署名・端末GPUは未実行です。[端末用入口・fixture配置を含む残件](device-validation.md)を維持します。
- [#98の受け入れ](results/candidate-performance-integration-869cadd-20261019.json)：run `39159816416` は全15 job成功・skip0。7成果物の実digest・CRC・私的2コピーと、元参照824,320値の超過0・集計一致を確認しました。main `dbe39f5` の同tree・公開著者・provider署名・全履歴も照合。SDK package / Runtimeは不変で、新しいGPU受け入れやmain push CIの別実行成功とは区別します。
- [旧providerコピーの確認](results/provider-cleanup-status-20261019.json)：既知の書換前commit APIに非公開メールmetadataが残存。Support案件はOpenで、削除完了は未確認です。元の値やpayloadは非公開に保持し、現行sourceの履歴・payload監査と分けて追跡します。正式公開のprivacy条件は未達です。
- [#97の受け入れ](results/native-distribution-integration-344f07a-20261019.json)：run `39155770986` は全15 job成功。実アップロード7 ZIPのdigest・CRC・私的2コピー、元参照824,320値の許容差超過0と集計一致を独立確認しました。公開main `4d66d92` のtree・公開著者・provider署名・履歴も確認。#95・#96のexact headは取り込み済みで、統合先を本文へ記録してcloseしました。新しいGPU検証とは区別します。
- [画像GPUの24反復](candidate-media-performance.md)は3形状×両精度・各4回で参照一致・解放・exit0です。初回CPU診断と、その後の[元特徴量による画像3・音声2条件の診断](reference-cpu-diagnostics.md)を区別して保存しました。後続の最終20 vectorは元参照へ独立に一致しましたが、中間encoder値は全5条件で許容差超過・exit1です。元Document特徴量6件は回収・全hash/CRCと完全な私的2コピーを確認し、元vision JSONは未回収。CPU差の原因は未確定で、CPU比較全体は未合格です。音声4条件のGPU測定、専有GPU/VRAM・最低要件も残ります。
- [手動回帰の公開配布経路](results/manual-native-distribution-cli-tdd-20261019.json)：全6信号・full/cutと元数値gateを維持し、9件Redから関連96件Green・actionlint成功。新CLIで既存の実CI capture 824,320値を再監査し、全6 case詳細と超過0が以前の独立監査に一致しました。新しいモデル実行やsource全CIの合格ではなく、旧artifactの削除も行っていません。

- [最新統合の受け入れ](results/release-integration-8649c94-acceptance-20261018.json)：head `8649c94` のRequired CIを含む15 jobと別のnative-dxgが成功。実行treeはheadと一致し、数値7 ZIPの全hash・CRC・2コピーと元参照824,320値の許容差超過0を独立確認しました。
- 同じheadの空Git consumerでpackage 600ファイル・sample 86ファイル・公式framework 1,168ファイルを照合し、policy 34件が合格。FP32 / Float16重みの画像3・音声2と固定4 queryの実GPU PlayMode実行、各34破損controlの独立監査・正常終了が成功しました。ホストピークは14,669,389,824 / 10,761,338,880 bytesで、物理RAM確認後の明示16 GiB allowance下の結果です。既定12 GiB、SDK専有VRAM、latency、手動UI・全role・全環境の合格へは読み替えません。
- [モデルpreview公開](results/model-preview-publication-567befb-20261018.json)：FP32 / Float16とも825 assetの実payload guardとremote size/digest、本文、producer tagを照合して公開しました。元モデルはGit / LFS外です。[匿名の空cache取得](results/public-model-consumer-8649c94-20261018.json)も両精度で実中断・再開・820 object照合・warm転送0・1 object破損修復まで成功し、公開取得モデルの5 media profile・4 query GPU PlayModeを独立監査しました。新しいsource全体の受け入れは別に追跡します。
- [許可された実マイクGPU登録](results/native-microphone-gpu-8649c94-20261018.json)：追加で許可された8.4秒・134,400 sampleを取得し、自動停止・audio-839 GPU登録・text検索・解放・破棄が成功。実audio 270 Workerとtext 1 Workerの計算CPU fallback・中間flushは0、有限unit 768次元vectorを確認しました。1 pass・skip0・exit0です。音声は保存・外部送信せず、音声・embeddingも公開しません。source `8649c94`の追加harnessの結果であり、手動録音UI・任意音声の意味精度・他端末への読み替えはしません。
- [descriptor bytesの保持](results/media-descriptor-byte-preservation-tdd-20261018.json)：10件Redから関連87件Green、#92最新headの必須CI・公開履歴・merge treeを確認して統合済みです。全宣言を準備した場合だけ元bytesを保持し、subsetは未取得objectを公開しません。
- [修正後の実画面とGPU](results/native-ui-7754ce4-20261019.json)：固定Git `7754ce4`の新規consumerでpolicy等84件が合格。実マウス・キーボードによる画像・合成音声登録と全4検索を操作し、2 media vector・8順位/score・16破損control拒否を独立監査しました。GPU Workerは181 / 270 / 1、計算CPU fallback・中間flushは0、正常終了です。observer警告は保持し、全手動項目・性能全体の合格へは広げません。
- [追加の途中停止](results/native-playback-stop-7754ce4-20261019.json)：同じSDK bytesで固定合成音声を実UIから再生し、残り3.349375秒の観測から0.251744秒後に明示停止しました。登録・preview・session保持後の全解放、PlayMode停止・Editor exit0、元証拠不変と2コピーを確認。追加録音はなく、性能測定や全操作合格とは区別します。
- [混在Playerの契約](results/final-mixed-player-contracts-20261019.json)と[cache接続](results/final-mixed-player-cache-integration-20261019.json)：21・43・22件の未実装失敗から86件Green、既存回帰込み112件、実tiny cacheの7件Redから関連119件Green。元の失敗・不変の参照/gateと、契約段階の実行範囲を保存しました。
- [最終混在Player](results/final-mixed-player-3a02fc0-20261019.json)：公開取得済みFP32 / Float16重みを新cacheと独立warm processで実行し、全4起動が正常終了。全roleの144 base・180 warm、18文書×4 query×4起動の288順位/score、88破損control拒否を独立確認しました。各warmはモデル転送0、各coldは820、descriptor読み込みは別に6件です。最大観測native working setはFP32 8,676,872,192 / Float16 6,024,720,384 bytes。source `3a02fc0`のGPU Worker・fallback/flush・解放とcompiled hash67件を確認しました。SDK専有VRAM・最小hardware・全環境の合格ではありません。
- [既存テキスト全ケース](results/final-text-player-3a02fc0-20261019.json)：同sourceのRelease Player 2起動で各25行のtoken照合、両精度×CPU/GPU計8条件の15 case・6文書・4 queryが合格。CPUとGPUを別集計し、192順位/score・34破損report controlを確認しました。raw vectorはruntime内の照合で、保存値からの独立vector再計算ではありません。固定queryのwarm各6回のCPU/GPU時間を記録し、一般入力のlatency・GPU専有時間・最小要件へは広げません。
- [数値回帰データの公開と取得実装](native-regression-distribution.md)：16 assetの実サイズ・hash、本文・tagを読み戻し、元参照/metadata 6ファイルを匿名HTTPで照合。明示input/audit・privacyの関連284件とactionlintが合格しました。元のgolden・許容差・採用根拠7記録は維持しています。新経路の全6信号・824,320値の実回帰は#97のCIで別に判定します。
- [CIの旧コマンド名検査を修正](results/native-distribution-workflow-assertion-repair-20261019.json)：run `39150835197` のPython全4環境は同じテスト1件で失敗しました。明示的な公開配布経路を検査するよう更新し、1件のRedから関連293件Green・skip0を確認。実数値jobの結果とCI全体の不合格を区別し、修正後headの必須CIを再確認してから統合します。参照・許容差・6信号の必須条件は変更しません。
- [公開経路の実数値回帰](results/native-distribution-real-numeric-73fa675-20261019.json)：同runの6信号と集計は成功。実アップロード7 ZIPのhash / CRC・私的2コピーを確認し、元参照824,320値の許容差超過0と集計JSONの一致を独立監査しました。通常CIの不合格を全体成功へ読み替えません。
- [モデル内部メタデータの検査を修正](model-artifact-metadata.md)：新たなCIモデル内にビルド環境の絶対パスが残る境界を確認し、後続run `39153769849` は停止しました。PT2 / ONNXの文字列検査と新規PT2保存前の任意trace除去を追加。最終関連117件と、実raw PT2 2件の拒否・配布PT2 2件と元ONNX 2件の合格を確認しています。原本の書換え・数値基準の変更はなく、修正sourceの全CI・旧provider成果物の整理は別に残ります。

### 音声候補の採用経緯と固定sourceの保存記録

- 新しい音声候補は、source `b6ede71` の[変換CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39120142716)で両profileの生成に成功。保存ZIP全体のCRC・hashと私的な2コピーを確認しました。
- 固定Git consumerのsource `a164cba` で、全6信号・824,320値を実GPU照合し、元の要素別許容差超過は0でした。
- source `c1e9aa8` で、実PCMから全6信号×Query / Document / Rawの18最終vectorを実GPU照合しました。全6processはexit0、skip0。元のcosine条件0.999に対する最小値は0.9999997097692651でした。破損control18件も拒否しました。
- 先行する15条件まとめ実行は既定180秒を超えて失敗しました。この失敗を保持した上で、上限を変えず1信号3条件ずつに分割して上記の合格を確認しています。
- source `39be89d` で保存bundleを生成。434ファイル・285個の固有Sentisモデルのhash、operation / coefficient対応、私的な2コピーを検査しました。モデル記述内の文字列とtokenizer JSONも監査しました。
- 保存bundleのEditor GPU照合はDocumentの全6信号で成功。元参照を独立再計算し、最小cosineは0.9999998052055729、破損control6件を拒否しました。このgateだけでは保存bundleの全roleやPlayer合格を意味しません。
- 新FP32 bundleのWindows IL2CPP / Release / High stripping Playerは、両音声profileで実行と独立再監査が成功。通常18件・in-process warm18件を元参照から再計算し、前後の全270 Worker監査、キャンセル復帰、解放・exit0と24件の破損control拒否を確認しました。実build sourceは `89b1ab9`、bundle生成sourceは `39be89d`。Runtime Git tree一致により既存binaryを再使用しており、新buildとは扱いません。
- source `c18b4e5` の新規固定Git consumerで、新候補285モデルのFloat16重み派生を生成しました。全434ファイルとFP32係数の不変、module対応、文字列privacy、私的な2コピーを独立確認。保存サイズは3,657,076,501から2,573,229,069 bytesです。この派生bundleも、両profileの実Release Playerと独立再監査で通常18件・warm18件、24破損control拒否、全GPU監査・exit0が成功しました。全演算FP16化・速度改善・独立warm起動の合格には読み替えません。
- 既存bundleではsource `89b1ab9` の実Playerで画像3・音声2×FP32 / Float16重みの10条件が合格しています。新候補や最終統合版へそのまま付け替えません。
- #89旧head `39be89d` の[通常CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39123968041)には、固定音声6信号と集計の失敗があります。最新headの合格と区別して保持します。旧native PT2の同一bytes復旧も未達であり、新候補の成功を原本の復旧とは扱いません。

7種類の証拠をcanonical JSON SHA256とSDK Runtime treeへ固定し、採用候補を使う必須数値回帰の取得経路を `58f48ac` で実装しました。27件の未実装失敗とworkflowの1件の失敗から、関連95件・actionlint成功を確認。元の全6信号・5区間・824,320値の実実行条件は維持しています。[初回選択CI 39126214004](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39126214004)は音声6信号・集計が成功し、全7 archiveの独立読み戻しでも824,320値の許容差超過0でした。ただし旧取得経路を前提とするテスト1件が各Python環境で失敗し、CI全体は不合格です。1件のローカルRedから関連111件Greenへ修正後、source `392ced7816ff5c724ccd902b0dcbf488b79cee18` の[最新CI 39127876689](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39127876689)は全15 job成功しました。Pythonの4環境、package、actionlint、通常reference、音声6信号と集計、Required CIを含みます。#89はそのheadでmain ca2ff1aへマージ済みです。これは最新CIの成功であり、新しいGPU実行や工程1全体の完了ではありません。

全6 moduleの[実配布候補の組み立て](results/media-distribution-composition-567befb-20261018.json)はFP32 / Float16とも成功しました。各820 object・553モデルの元bytes・module対応・個人情報metadataを独立照合し、合計1,640ファイルの[実公開payload checker](results/model-publication-payload-guard-392ced7-20261018.json)も成功しました。flat HTTPS取得・回復とcompositionの関連77件が合格しています。その後のpreview公開・匿名の空cache取得結果は現在地の記録を参照してください。

新配布候補の[サンプル検証入口](candidate-media-sample-validation.md)も準備しました。旧bundle用のpinと数値gateを維持し、新しいFP32 / Float16 descriptorと画像3・音声2 profileを別に検査します。未実装10件・cohort不足7件のRedから、旧policyを含む34件Green・skip0・Unity exit0と新fixtureのcompileを確認しました。この契約検査自体はlocal-file / no-graphicsの小規模確認です。後続の実モデル・PlayMode・Playerの結果は、現在地の各固定sourceの記録で区別しています。

## 優先順位と完了条件

| 順序 | 実施すること | 完了条件・成果物 |
| --- | --- | --- |
| 1 | 基盤復旧と既存PRの整理 | #99まで統合、旧PRの取り込みとcloseが完了。各headの必須CI・数値成果物の独立照合、mergeの同treeと公開identityを確認 |
| 2 | 空checkout・空cacheから利用者向けの準備を再現 | 公開手順だけでtext・画像3・音声2の全6 moduleが準備でき、source / model hash一致。欠落・破損・中断・期限切れから回復 |
| 3 | 最終候補のsample操作と性能を確認 | 日英検索・横断検索、画像、音声再生 / 停止、実マイク、キャンセル・再試行・解放が成功。権限拒否・deviceなしも確認。初回 / warm、CPU / GPU時間、RAM / VRAMを実測し最低要件を決定 |
| 4 | 最終統合版のWindows Playerを受け入れ | 機能側はsource 3a02fc0で全4混在起動・別text 2起動が合格。IL2CPP / Release / High、全media role・18文書の検索・text 15 case、GPU監査・独立cold/warm・解放・exit0を記録。性能・最低要件の未達は工程3と合わせて追跡 |
| 5 | macOS Editor・iOS / Android実機を受け入れ | 端末・OS・graphics APIを記録し、3入力の実GPU・権限・suspend / 復帰・メモリを確認。skip・未実行・CPU fallbackをGPU合格に数えない |
| 6 | 最終配布候補を固定し、公開前監査 | 必須回帰全成功。README / API / sample / CHANGELOG / 利用条件 / 最低要件を整備。履歴・commit identity・実公開payload・画像の目視 / OCR / hashを確認。旧PR / cacheのprivacy対応も追跡 |
| 7 | 正式公開と公開後の導入確認 | version・tag・GitHub Releaseを一致させ、公開物を読み戻す。公開tagを新規consumerへ導入し直し、モデル準備とsample smokeが成功 |

工程1は#99までの統合と既存PR整理が完了しました。GitHubがsquash著者へ使用する公開名と指定noreplyは[限定的な履歴分類](results/provider-public-author-tdd-20261018.json)で扱い、実際のmerge・CI実行commitでも公開identityとprovider署名を確認しています。ローカルcommit identityや他の個人情報検査は維持します。工程2の公開取得と工程4のWindows機能側は上記の実績があり、残る工程3の操作/性能・工程5〜7の条件は未完了です。

工程2のモデル配布は正式SDK Releaseと分け、ライセンス・変更点・全ファイルhashを添えたpreviewの公開と公開URLからの取得確認まで完了しました。モデルはGit / LFSへ含めません。数値回帰データも別previewへ保存し、元参照の同一bytesと、任意stack traceだけを除いた配布graphの計算内容・全非modelメンバーの一致を確認しました。元graphと実行graphのhashを区別する取得処理と全6信号CI・独立監査は成功しました。旧成果物の退避・依存棚卸しは維持し、原本が失われたものを再生成品で原本扱いしません。

## PRの整理

- #89: 最新CI成功・差分・privacy監査を経てmain `ca2ff1a` へマージ済みです。
- #86: 取得・回復の変更を#91へ取り込み済み。元PRはcloseしました。
- #87: Float16・sample gate・測定の変更を#91へ取り込み済み。元PRはcloseしました。
- #90: Player監査とrelease payloadの変更を#91へ取り込み済み。元PRはcloseしました。
- #91・#92: 統合候補とdescriptor bytes保持をマージ済みです。
- #93: 実画面のパス表示修正をマージ済みです。
- #94: 最終混在Playerの検証経路を、全必須CIと実Player監査後にマージ済みです。
- #95・#96: exact headを#97へ取り込み、統合先を記録してclose済み。個別mergeはしていません。
- #97: head `344f07a` の全15 CI job・7成果物の独立数値監査・privacy確認後、main `4d66d92`へ統合済み。
- #98: head `869cadd` の全15 CI job・7成果物の独立数値監査・privacy確認後、main `dbe39f5`へ統合済み。
- #99: head `cc92d4b` の全15 CI job・7成果物の独立数値監査・privacy確認後、main `c6dcde4`へ統合済み。

#89の後、#86・#87・#90の変更を#91へまとめ、全必須CI・新規consumerと両精度の実sampleを確認して統合しました。3 PRのexact headが統合候補の祖先に含まれることを検査し、本文へ統合先を記録してcloseしています。個別PRのmergeとは区別します。今後の取得・操作・環境別修正もtask branch / PRを使い、最新headのCI・差分・privacyを確認して統合します。正式SDK tag / Releaseは工程6の完了後です。

## 実行環境・日程・報告

- macOS / iOS / Androidの端末・署名・build環境は、工程1と並行して先に確保します。直近の環境確認は接続Android端末0、self-hosted runner0、Mac / iOS未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先します。ローカルのUnity / GPU実行は必要な確認に絞り、1processずつ動かします。
- #94までの統合、公開モデル取得・復帰、実マイク、表示修正後の画面、最終混在Playerの独立cold / warmとtext全ケースは成功しました。次は「残る実操作・各stage性能とCPU比較」「数値参照の恒久取得・privacy対応」「必須実機」です。正式SDK公開は全条件の完了後に行います。合格・失敗・未実行とsource / modelを分けて記録します。
- リリース日は未確定です。統合、公開モデルからの新規consumer導入、Windows機能側の受け入れは成功しました。工程5の端末・署名・実行担当を確定した時点で、残る測定・修正・環境別検証の日程を見積もります。端末が未確保のまま完了日を約束しません。
- 次の報告は「残る操作・性能」「恒久参照とprivacy対応」「必須端末の実行可否」の単位で行います。新しい不具合がない同一モデルの重い変換は繰り返さず、CI終了待ちと実作業時間を分けて示します。
- 公開表示名 `yousan` は維持します。公開文書には相対 / 環境変数のパスだけを使い、元画像・元数値証拠は非公開で保持します。証拠画像を加工・再生成して漏えいや失敗を隠しません。
