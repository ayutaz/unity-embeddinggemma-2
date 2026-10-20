# 正式リリースまでの計画

確認日: 2026-10-20（JST）。対象はテキスト・画像・音声のGPU推論と横断検索です。[ゴール](goal.md)の必須環境・精度条件を維持します。

## 最新の計画（CIを先行、PR #111統合後）

基準mainは `de56db8c84984dc84fc2597f60cc1c6fbd3bf456`、PR #111まで統合済みです。採用head `1280948` の[必須CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39347537063)は全15 job成功。実13 archiveを独立照合し、元824,320値・CPU encoder 120 capture / 30,822,400値の許容差超過0を確認しました。SDK実行コードはWindows GPU検証source `f0fc2a7`から不変ですが、package内READMEとCHANGELOGは変更されています。package全bytesの一致や、新mainでのGPU再実行は主張しません。

**実機待ちと分離してmacOSテスト・Android APK・iOS Xcode出力のCIを先行します。** PR #112の初回head `6c1ee2f`の[実preflight](results/platform-ci-preflight-6c1ee2f-20261020.json)は全3 jobがライセンス未登録でfailureでした。実3 archiveを独立照合し、Editor導入・build・モデル推論未開始を確認しました。本branchにはさらにMetalの16整数compute probeと、iOS export後の署名なしXcode compile・実arm64 app照合を接続しました。追加したnative処理の実runner結果は未確認です。[追加TDD](results/platform-native-contracts-tdd-20261020.json)は新規20件Redと保護・終了監督計4件Redを経て、platform関連68件・関連回帰計105件Greenです。原モデル / fixture配置、署名・端末操作、実モデルGPU・全実機は残ります。以下は全工程の計画であり、入口の単体テスト合格を実Editor・build・実機合格には数えません。

| 順序 | 作業 | 開始条件と完了条件 |
| --- | --- | --- |
| 1 CIの入口と環境確認 | 固定Unity 6000.3.21f1、macOS / Android / iOSのmodule・license・空き容量・実行時間を検査。source / model / fixture hashを固定 | 設定不足も明示的に記録するpreflightをTDDで追加。認証値をログや成果物に含めず、未設定を成功にしない。環境不足があるplatformだけを保留し、他platformの準備を継続 |
| 2 hosted CIでbuildとテスト | macOSのcompile / Editorテスト、Android ARM64 / IL2CPP / High / Release APK、iOS Device / Metal / IL2CPP / Release Xcode出力を並列化 | 有効なUnity CIライセンスと対応module、十分な容量が必要。実Editor実行・build結果・正常終了・成果物hashを回収して完了。Androidエミュレータは補助検証。iOSのXcode compileはexportと別結果とし、実機用署名・配置とも分ける |
| 3 実GPUと実機の受け入れ | macOS Metal、iOS / Android実機で全入力・全role・混在検索、数値、fallback / readback、権限、キャンセル、suspend / 復帰、解放を検証 | hosted MacはまずMetal / compute能力と容量を小さいprobeで確認。不足時は利用可能なMacへ切替。端末は接続hostまたは検証可能な実機サービスが必要。実機CIはtrusted source限定の隔離された実行環境で動かし、エミュレータを実機合格に数えない |
| 4 性能と公開前整理 | 同一hardware・source・モデル・入力・仕事量でCPU/GPU、cold / warm、stage時間・RAM / VRAM・解放を測定。公開source・全祖先・実payloadと旧providerコピーを確認 | 画像CPUの未合格条件と過去の失敗を保持。元数値gateを満たす実測と最低要件を記録。現行履歴の合格とprovider側削除確認を分け、画像は実bytesの目視 / OCRを実施 |
| 5 正式SDK公開 | 版・CHANGELOG・license・固定tag・Release、新規consumerへのそのtagの導入 | 2〜4の必須条件が完了後に実施。モデル取得から全入力・検索・解放・終了まで再現。previewを正式SDK Releaseに数えない |

工程1〜2を先行し、工程3の端末・署名準備、工程4のGPU資源確保とprovider対応を並行させます。ローカルAndroid moduleのUAC待ちはhosted CIのAPK buildを止める理由にしません。実装は関連workflowと契約テストをまとめたtask branch / PRで行い、既存Required CIは維持します。計画変更だけでモデル変換・実機検証を開始したとは扱いません。

直近のsource別結果: `08d7e33`のhosted MacではSwift compile・Metalの16整数照合が成功し、不一致0です。Unity licensingは未設定で、全3 preflightのfailureとbuild未開始を保持します。同sourceのRequired CIでLinuxのCLIテスト各2件が失敗したため、[子プロセスのsource選択](results/platform-cli-source-selection-20261020.json)を修正しました。親の `PYTHONPATH` に依存しない関連105件が合格しましたが、修正sourceのRequired CI・成果物照合を完了してから統合を判断します。次の実行条件はCIライセンスの準備で、small Metalの成功を実Sentisモデル合格にしません。

### 実行上の判断点と所要時間

- 最初の成果物は3platformのpreflightと、実行できるplatformのbuild / test結果一覧です。Unityライセンスの登録方法は選んだrunner・ライセンス種別で確認し、Secretsの値をチャット・ソース・公開ログへ出しません。利用者側に残るのは必要なライセンス / iOS署名設定と実機の提供・接続です。通常のworkflow実装は開発側で進めます。
- hosted runnerのOS提供だけでは実GPU対応や大きいモデルの実行容量を保証しません。GitHubの[runner仕様](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)と実probeを併用します。モデルは必要moduleだけをhash検証して取得し、機密設定・生ログを含むartifactを無条件に公開しません。
- GPU性能測定の既存開始条件は音声で空きVRAM 14 GiBです。これはSDK最低要件ではなく、実行前に再確認します。他アプリの終了や新しい録音は自動では行いません。異なるCI host間の時間を同一hardwareのCPU/GPU比較に合算しません。
- Unityの[iOS build手順](https://docs.unity.com/en-us/engine/6000.3/manual/platform-specific/iphone/ios-building-and-delivering/build-process)に従い、Xcode export、compile、署名、実機起動をそれぞれ判定します。build成功だけで端末の実GPU・権限・復帰を合格にしません。
- 標準hosted runnerを優先します。有料runner・実機サービス・追加購入をこの計画だけで開始しません。利用可能な無料枠・既存環境で不足する場合は、具体的な不足と代替案を提示します。
- preflightが失敗したら原因を特定して修正し、環境変化や修正なしの同じ失敗を再実行しません。重いモデル検証は変更をまとめたsourceで行い、数値原本・許容差・失敗記録を維持します。公開前にはhook、履歴、実payload、公開identityを検査します。
- 期間の見積りは最初のpreflightでlicense・module・容量を確認した時点と、最初のbuild完了時に更新します。現状では実機確保・署名・provider対応の所要日数が不明なため正式公開日は未確定です。各報告で実行source、実行した範囲、成功 / 失敗 / 未実行、次の開始条件を示します。

## PR #110統合後のcheckpoint（履歴）

以下はCI先行計画へ変更する前の記録です。「外部条件待ち」やローカルUAC待ちを全工程へ適用せず、現在の実行順は上の計画を使います。

計画の基準はmain `110969884d1da0c24aea7677cb6d5d694bd7a607`です。[PR #110](https://github.com/ayutaz/unity-embeddinggemma-2/pull/110)まで統合済みです。head `810d49a`の[必須CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39336685624)は全15 job成功、実13 archiveと元6入力824,320値・CPU encoder 120 captureの独立照合も合格しました。[Windows音声CPU比較](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39336685770)は2条件×4回・全32 captureで元数値条件に合格しました。実行sourceは同treeのtest-merge `82506a6`です。実mainの署名・公開identity・全祖先も確認済みです。重複main CIのcancelledを成功には数えません。

**次の優先作業は、同じhardwareでのCPU/GPU性能測定と必須実機の検証です。** Windowsの音声CPU比較を再び未着手扱いにしません。一方、画像CPUの過去の数値差、全体性能、最低要件、必須他環境、provider側コピー、正式SDK公開は未完了です。SDK packageは既存Windows GPU検証source `f0fc2a7`と全bytesが一致しますが、新mainでのGPU再実行を示す結果ではありません。

| 順序 | 作業と担当 | 開始条件 | 完了条件 |
| --- | --- | --- | --- |
| 1 統合・基準固定 | 開発側：#110の採用source・元参照・許容差・モデルhashを測定の基準として固定 | 基盤の統合とWindows音声CPU照合は完了 | 後続変更はtask branch / PRで検証し、実行sourceを混同しない |
| 2 数値差と性能 | 開発側：未合格CPU条件を追跡し、同じhardware・入力・モデル・仕事量でtext / image / audioのCPU/GPU、初回 / warm、各stage、RAM / VRAM、解放を測定 | 測定用GPU容量の確保と、元数値gateに合格するCPU比較条件 | 元の数値条件と正常終了を満たす比較表、実測に基づく最低要件。失敗・未実行は別記 |
| 3 必須実機 | 利用者側：利用可能なMac / iOS / Android環境と署名環境を確保。開発側：build・実行・独立監査 | Mac / iOS / Android環境、Android module導入時のUAC対応 | 全入力・全role・混在検索、実GPU、権限拒否、キャンセル、suspend / 復帰、解放を各必須環境で確認 |
| 4 公開前整理 | 開発側：最終source・全祖先・実payload・導入文書を監査。provider：古いコピーの削除対応 | 現行監査は継続可能。provider側削除は外部対応待ち | 現行公開物と古いコピーを別々に確認。画像は実bytesの目視 / OCR、hookを維持。公開名 `yousan`は維持 |
| 5 正式公開 | 開発側：SDK版・CHANGELOG・license・固定tagを準備し、空consumerでそのtagを導入して正式Releaseを公開 | 2〜4の必須条件が完了 | 新規導入からモデル取得・全入力・検索・解放・終了まで再現し、公開source / model hashと一致 |

工程2と3は環境確保後に並行し、工程4のprovider対応も並行して追跡します。重い処理はCIを優先し、同じsourceで成功済みのモデル変換・検証を無条件に繰り返しません。

### 次に必要な環境と判断点

- GPU空き容量は計画作成時に3,866 MiBでした。音声GPU性能測定の既存開始条件14 GiBを下回るため、開始前に再確認します。14 GiBはSDKの最低VRAM要件ではありません。他アプリの終了は自動で行いません。
- 直近の監査ではAndroid接続端末0、Mac / iOS環境は未確認です。Androidの取得済みtoolsとprivate consumerを使い、UAC対応の明示的な回答後にmodule導入、実build、端末配置へ進みます。設定テストやtool展開だけを実機合格にしません。
- provider側の古いcommitコピーは直近の匿名API再確認でも取得可能でした。削除完了は未確認です。個人情報の値や元応答は公開しません。
- 現在は上記の外部条件待ちです。GPU容量の確保、端末・UAC対応、provider対応の変化があれば該当工程を再開します。SDKは`0.1.0-pre.1`で、公開済みpreviewを正式SDK Releaseには数えません。

GPU・必須端末・provider対応の日程が未確定のため、正式公開日はまだ確定しません。次の見積り更新は、環境確保時、性能測定終了時、必須実機検証終了時に行います。

## PR #109時点のcheckpoint（履歴）

以下の未着手・次の作業・環境値は#109時点の記録です。現在の優先順は上の計画を使います。元の失敗・未実行・数値基準を維持します。

[#105](https://github.com/ayutaz/unity-embeddinggemma-2/pull/105)〜[#108](https://github.com/ayutaz/unity-embeddinggemma-2/pull/108)に続き、[#109](https://github.com/ayutaz/unity-embeddinggemma-2/pull/109)を統合しました。このcheckpointのmainは`811c0d2da93328a429ee54466f8b0e1eba7909d9`です。[#109の統合記録](results/audio-cpu-pipeline-integration-4367677-20261020.json)ではhead `4367677`のCI全15 job成功・skip0、実13 archiveのdigest / size / 全CRC・完全な私的2コピーと元数値を独立照合しました。実mainと採用headの同tree、公開identity・署名・全祖先も確認しました。重複main CIはcompleted / cancelledで、成功として数えません。後続sourceは別に検査して統合します。

音声encoderのCPU比較は全6入力・120 capture・30,822,400値で元の許容差超過0です。これはencoder区間の結果であり、PCM前処理から最終vectorまでのCPU/GPU性能比較や、全環境の受け入れ完了ではありません。SDKの推論要件はGPUのままです。

音声のPCM前処理→監査済み5つのONNX CPU session→共通backboneを4回測定する[比較用入口](audio-cpu-encoder.md)では、source `9bd7a26`の[初回実モデル結果](results/audio-cpu-export-preprocessing-9bd7a26-actual-20261020.json)を保持しています。無音は合格、padded_noiseはencoder各回33値の超過で不合格でした。公式CPU前処理へ修正したsource `4367677`の[Linux実測](results/audio-cpu-official-preprocessing-4367677-actual-20261020.json)は、2条件・各4回・全32 captureを独立照合して元数値条件へ合格しました。旧失敗・元参照・許容差・SDKのGPU処理は変更していません。次はWindowsで同じ元PCM・数値gate・実接続を実行します。Windows用workflowとメモリ・CPU型番の取得は関連144テスト・skip0で確認済みですが、Windows実モデルは未合格です。

画像の先行する[実CPU測定](results/image-cpu-reference-8c60111-actual-20261020.json)は、landscape / portrait合格・square各回600値の元許容差超過で不合格でした。全12最終vectorとpixels / positionsは一致し、全48 capture・3 archiveを独立照合しました。CPU型番等は未記録で原因は未確定です。追加の[同一host比較](results/image-cpu-paired-hosts-1789f0c-actual-20261020.json)ではEPYC 9V45の2 host・3形状・各4回のnative実測は全24回合格し、encoderは元参照とbitwise一致しました。COMPATIBLE設定は全24回不合格（square / landscape / portraitで各回361 / 95 / 128値の超過）です。全192 capture・2 archiveと完全な私的2コピーを独立照合し、run全体のfailureと旧600値の失敗を保持します。記録したnative条件の結果であり、任意CPUやWindowsの比較・性能全体の完了ではありません。

GPU検証で固定したencoder参照と、Document最終参照のencoder生成世代は区別して保持します。先行CIの固定GPU参照への6件bitwise一致と、旧Document生成世代への26値の超過、ローカルAVX2の失敗も保存します。これらを新しい実測の600値と混同せず、元参照・許容差を変えません。CPU比較・性能全体は未合格です。

| 工程 | ここからの作業 | 完了条件 |
| --- | --- | --- |
| 1 統合 | #109までの採用sourceとmain、既存Windows機能検証の適用範囲を固定 | 基盤は統合済み。後続変更はtask branch・PR・必須CIと成果物で個別に検証する |
| 2 数値比較と性能 | 合格したnative条件を固定し、未合格環境の数値差を追跡する。同じhardware・入力・モデル・仕事量のCPU/GPU、音声の未測定条件、初回 / warm、stage時間、RAM / VRAMを埋める | 元の数値条件を満たす完全な比較記録、正常終了と解放、実測に基づく最低要件。未合格・未実行を別記 |
| 3 必須環境 | macOS Editor、iOS / Android実機で全入力・role・混在検索を確認し、権限拒否、キャンセル、suspend / 復帰も検証 | 各環境で実GPU・fallback / readback・数値・操作・解放の証拠。設定テストやWindows成功では代用しない |
| 4 公開前整理 | 古いprovider側コピーの削除確認、最終source・全祖先・公開payload・導入文書を監査 | 現行履歴とprovider側残存を別々に解決。画像は実bytesの目視 / OCR確認。公開名 `yousan` を維持 |
| 5 正式公開 | 1〜4の完了後にSDK版・tag・Releaseを作り、そのtagを空consumerへ導入する | モデル取得から全入力・検索・終了まで再現し、公開版のsource / model hashと結果が一致 |

今すぐ進められるのはPRの統合準備、工程2の残る比較・測定準備と、工程4の文書・公開payload監査です。端末準備とprovider対応は並行して進めます。重いモデル変換・Unity実行はCIを優先し、同じsourceの成功済み処理を無条件に再実行しません。

### 直近の実行順（2026-10-20確認）

1. **照合済みの比較結果を適用範囲ごとに記録する。** [#108](https://github.com/ayutaz/unity-embeddinggemma-2/pull/108)の実行source `1789f0c` に対する同一host比較は回収・独立照合済みです。nativeの24回合格とCOMPATIBLEの24回不合格を両方残し、旧600値の原因を推定で断定しません。MKL設定値を実operator dispatchの証拠にも使いません。
2. **Windowsの実CPU接続を通常CIと並列に検証する。** #109はhead `4367677`の必須CI全15 job・実13 archiveと、別のLinux音声全体8回・32 captureを確認してマージ済みです。Windows専用workflowは元PCMの2条件を使い、同じ5区間の元数値gateを通してから公式CPU前処理→encoder→backboneを測定・監査します。対象コードのPRと手動実行へ限定し、生ログは非公開、数値成果物はprivacy guardの成功後だけ公開します。修正と文書をまとめてからPRを作り、重いCIの不要な再実行を抑えます。
3. **同じ実行環境の比較と性能表を完成する。** 合格したLinux native画像とLinux音声2条件、旧square600値、Windows CPUの先行不合格を分けて追跡します。Windows CIでの数値合格も、利用者の同一hardwareのCPU/GPU性能の代わりにはしません。同じ入力・モデル・仕事量・hardwareの初回 / warm、各stage、RAM / VRAMと解放を測定し、最低要件を実測から決めます。元参照・許容差と製品のGPU推論要件は維持します。
4. **必須端末とprivacyの残件を並行して完了する。** macOS Editor・iOS / Android実機で全入力、検索、権限・復帰・解放を確認します。provider側の古いコピー除去を別途確認し、最終source・履歴・公開payloadを監査します。
5. **全条件の達成後に正式公開する。** SDK版・tag・Releaseを作り、空consumerへのそのtagの導入からモデル取得、全入力・検索・終了まで確認します。preview公開を正式SDK公開の完了として数えません。

次の判断点はWindowsの実モデルCIと、保存した前処理・mask・encoder・最終vectorの独立照合です。Linux source `4367677`の成功をWindowsや別sourceの実行成功へ読み替えず、失敗があれば元の数値条件のまま保存します。端末・GPU資源・provider対応は並行して進めます。

### 実行の開始条件と時間の見積り

- 音声GPU性能測定の既存開始条件は空きVRAM 14 GiB以上です。直近の再確認は約4 GiBで未開始です。開始前に空きを再確認します。他アプリの終了や追加録音を自動では行いません。この条件はSDKの最低VRAM要件を示す値ではありません。
- Mac / iOS / Androidの利用可能な実機と実行環境は確認待ちです。接続Android端末0、self-hosted runner0を再確認しました。Androidツールの取得・展開と固定fixture 28ファイルのprivate consumer準備は済みましたが、対応EditorのAndroid module導入はUAC承認待ちで、APK・実機GPUは未実行です。未接続・未実行を成功として扱いません。
- provider側の古いコピーは今回の匿名API再確認でもHTTP 200でした。SupportはOpen・コメント1件・provider返信0件で、削除完了は未確認です。元応答は完全な私的2コピーを保持し、個人情報の値を公開しません。最終確認記録とSupportの状態を更新してから工程4を判定します。
- GPU資源、必須端末、provider対応が未確定のため、現時点で正式リリース日を確約できません。工程2の比較契約を整えた時点、性能測定終了時、必須実機検証終了時に残件と所要時間を更新します。

正式SDK tag・Releaseは未作成です。公開済み4件はいずれもモデル / 回帰データ / 元画像参照のpreviewです。次の報告では「実行した範囲・結果・残る開始条件」を示します。

## 過去のcheckpoint（2026-10-19以前）

以下は各source時点の保存記録です。「未回収」「未統合」「未合格」等は当時の状態を表し、最新の順序・残件は上の表を優先します。

直近の到達点は **#102までの統合、端末用入口の準備、最新sourceでのWindows混在Player全4起動の成功** です。[統合と実検証](results/device-runtime-integration-f0fc2a7-20261019.json)ではmain `f9b22ce`とhead `f0fc2a7`の同tree、全15 CI job、元参照824,320値の超過0、両精度cold / warmの全role・混在検索・解放を確認しました。元の特徴量に対するCPU比較は未合格で、元workflowのLinux既定条件を復元した追加診断でも143,360値中288値が元許容差を超過しました。元参照・許容差は変更していません。必須実機、音声性能・専有GPU / VRAM・最低要件、旧providerコピー除去、正式SDK公開と新tag consumer確認を残しています。

### 当時の現在地

正式SDK tag・Releaseは未作成です。Sentisモデル配布用のpreview 2件とnative回帰データ用のpreview 1件を公開済みで、正式SDK公開とは区別します。以下は同日の実データ・GitHub照合に基づくcheckpointであり、正式候補全体の合格ではありません。

- [#100の受け入れ](results/ios-profile-integration-7171a20-20261019.json)：run `39166352666` は全15 job成功・skip0。実7成果物のdigest / CRC・私的2コピーと、元参照824,320値の超過0・集計一致を確認し、main `2a17421` の同tree・公開identity・署名・履歴も照合しました。iOS設定の23 / 関連44テストを統合しましたが、実export・署名・実機GPUは未実行です。SDK / Runtimeは不変で、main pushの重複CI成功には読み替えません。
- [#99の受け入れ](results/manual-native-distribution-integration-cc92d4b-20261019.json)：run `39163550227` は全15 job成功・skip0。7成果物の実digest・CRC・私的2コピーと、元参照824,320値の超過0・集計一致を確認しました。main `c6dcde4` の同tree・公開著者・provider署名・全履歴も照合。手動回帰の期限付きraw artifact依存を固定公開配布へ移し、元の数値gateを維持しました。SDK package / Runtimeは不変で、新しいGPU受け入れやmain push CIの別実行成功とは区別します。
- iOS設定は[初回19件Red→19件Green](results/ios-build-profile-tdd-20261019.json)と、[Xcode構成の追加8件Red / 15件pass→最終23件Green](results/ios-xcode-scheme-tdd-20261019.json)を保存。Windows / Androidを含む最終関連44件が成功・skip0です。Release / IL2CPP / Metal・Device SDKに加え、Xcode Release設定とDebug / Releaseへの復元を確認し、署名・provisioning値は変更・掲載しません。実export・署名・端末GPUは未実行です。[端末用入口・fixture配置を含む残件](device-validation.md)を維持します。
- [固定入力のstaging](results/device-fixture-materialization-tdd-20261019.json)：29件Red→29件Green。元28ファイル・34,474,958 bytesを実file URIで取り出し、全hashの独立再計算・既存loaderの36 media roleケースとtext 6文書 / 4 query・私的完全2コピーを確認しました。Windows Editorでの入力統合1件・skip0であり、Android jar・実機起動・モデルやGPUの実行は未確認です。元参照・許容差は変更せず、端末用runnerへの接続を残します。
- [#98の受け入れ](results/candidate-performance-integration-869cadd-20261019.json)：run `39159816416` は全15 job成功・skip0。7成果物の実digest・CRC・私的2コピーと、元参照824,320値の超過0・集計一致を確認しました。main `dbe39f5` の同tree・公開著者・provider署名・全履歴も照合。SDK package / Runtimeは不変で、新しいGPU受け入れやmain push CIの別実行成功とは区別します。
- [旧providerコピーの確認](results/provider-cleanup-status-20261019.json)：既知の書換前commit APIに非公開メールmetadataが残存。Support案件はOpenで、削除完了は未確認です。元の値やpayloadは非公開に保持し、現行sourceの履歴・payload監査と分けて追跡します。正式公開のprivacy条件は未達です。
- [歴史的raw ZIP依存の整理](raw-candidate-archive.md)：13件Red→13件Greenとlocal golden必須の追加Redを経て、関連170件Green・skip0とactionlintが成功しました。元ZIP 4,938,559,326 bytesの私的完全2コピー・全CRC、6 caseとraw graph / native ONNX各2 profileのローカル取得を確認。元の数値基準・世代選択を維持し、旧hosted rawモデル2モードを明示的な失敗へ変更します。source全CI・main統合・旧artifact削除は未実施です。元入力だけの別保存経路は継続します。
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
| 1 | 基盤復旧と既存PRの整理 | #101まで統合、旧PRの取り込みとcloseが完了。各headの必須CI・数値成果物の独立照合、mergeの同treeと公開identityを確認 |
| 2 | 空checkout・空cacheから利用者向けの準備を再現 | 公開手順だけでtext・画像3・音声2の全6 moduleが準備でき、source / model hash一致。欠落・破損・中断・期限切れから回復 |
| 3 | 最終候補のsample操作と性能を確認 | 日英検索・横断検索、画像、音声再生 / 停止、実マイク、キャンセル・再試行・解放が成功。権限拒否・deviceなしも確認。初回 / warm、CPU / GPU時間、RAM / VRAMを実測し最低要件を決定 |
| 4 | 最終統合版のWindows Playerを受け入れ | 機能側はsource 3a02fc0で全4混在起動・別text 2起動が合格。IL2CPP / Release / High、全media role・18文書の検索・text 15 case、GPU監査・独立cold/warm・解放・exit0を記録。性能・最低要件の未達は工程3と合わせて追跡 |
| 5 | macOS Editor・iOS / Android実機を受け入れ | 端末・OS・graphics APIを記録し、3入力の実GPU・権限・suspend / 復帰・メモリを確認。skip・未実行・CPU fallbackをGPU合格に数えない |
| 6 | 最終配布候補を固定し、公開前監査 | 必須回帰全成功。README / API / sample / CHANGELOG / 利用条件 / 最低要件を整備。履歴・commit identity・実公開payload・画像の目視 / OCR / hashを確認。旧PR / cacheのprivacy対応も追跡 |
| 7 | 正式公開と公開後の導入確認 | version・tag・GitHub Releaseを一致させ、公開物を読み戻す。公開tagを新規consumerへ導入し直し、モデル準備とsample smokeが成功 |

工程1は#101までの統合と既存PR整理が完了しました。GitHubがsquash著者へ使用する公開名と指定noreplyは[限定的な履歴分類](results/provider-public-author-tdd-20261018.json)で扱い、実際のmerge・CI実行commitでも公開identityとprovider署名を確認しています。ローカルcommit identityや他の個人情報検査は維持します。工程2の公開取得と工程4のWindows機能側は上記の実績があり、残る工程3の操作/性能・工程5〜7の条件は未完了です。

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
- #100: head `7171a20` の全15 CI job・7成果物の独立数値監査・privacy確認後、main `2a17421`へ統合済み。
- #101: head `0f0153e` の全15 CI job・7成果物の独立数値監査・privacy確認後、main `d529626`へ統合済み。元参照824,320値の超過0、同tree・公開identity・署名を照合。旧raw artifact 12128476472だけを完全保存・現行依存確認後に削除し、producer runと他成果物を保持しました。

#89の後、#86・#87・#90の変更を#91へまとめ、全必須CI・新規consumerと両精度の実sampleを確認して統合しました。3 PRのexact headが統合候補の祖先に含まれることを検査し、本文へ統合先を記録してcloseしています。個別PRのmergeとは区別します。今後の取得・操作・環境別修正もtask branch / PRを使い、最新headのCI・差分・privacyを確認して統合します。正式SDK tag / Releaseは工程6の完了後です。

## 実行環境・日程・報告

- macOS / iOS / Androidの端末・署名・build環境は、工程1と並行して先に確保します。直近の環境確認は接続Android端末0、self-hosted runner0、Mac / iOS未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先します。ローカルのUnity / GPU実行は必要な確認に絞り、1processずつ動かします。
- #94までの統合、公開モデル取得・復帰、実マイク、表示修正後の画面、最終混在Playerの独立cold / warmとtext全ケースは成功しました。次は「残る実操作・各stage性能とCPU比較」「数値参照の恒久取得・privacy対応」「必須実機」です。正式SDK公開は全条件の完了後に行います。合格・失敗・未実行とsource / modelを分けて記録します。
- リリース日は未確定です。統合、公開モデルからの新規consumer導入、Windows機能側の受け入れは成功しました。工程5の端末・署名・実行担当を確定した時点で、残る測定・修正・環境別検証の日程を見積もります。端末が未確保のまま完了日を約束しません。
- 次の報告は「残る操作・性能」「恒久参照とprivacy対応」「必須端末の実行可否」の単位で行います。新しい不具合がない同一モデルの重い変換は繰り返さず、CI終了待ちと実作業時間を分けて示します。
- 公開表示名 `yousan` は維持します。公開文書には相対 / 環境変数のパスだけを使い、元画像・元数値証拠は非公開で保持します。証拠画像を加工・再生成して漏えいや失敗を隠しません。

## 端末用入口の追加checkpoint

[Android / iOS混在入口](device-mixed-runtime.md)は、CLIを使わない端末内設定、固定fixtureのStreamingAssets転送、既存の全role・全module・混在GPU protocolを接続します。専用build markerと可逆profileを使い、Editor・Development・target違い・Android非Vulkan / iOS非Metalを拒否します。48件Red→48件Green、関連190件Green・skip0はEditor契約検査です。実APK / Xcode export・端末へのbundle配置・GPU実行、permissions・suspend / 復帰・性能は未実行で、工程5の完了条件は維持します。
