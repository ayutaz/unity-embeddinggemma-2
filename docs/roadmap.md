# ロードマップ

更新: **2026-10-18**。履歴整理後のmain `4b5bf72`（PR #85、Runtime改善#88とprivacy guardを含む）を基準とする現在の作業計画です。本文の過去の実行SHAは原本の識別子で、履歴整理後の導入pinとして使わず、sourceとmodel hashを個別に照合します。確認済みの事実は[検証状況](status.md)、変更しない目標は[プロジェクトのゴール](goal.md)を参照してください。

テキスト・画像・音声のGPU API、Windows text Player、画像・音声検索サンプルとcacheは統合済みです。現在はprivacy対応と固定参照の復旧が先です。その後、他の利用者が準備・操作を再現できる状態を作り、そのcandidateで性能と対象環境を検証します。正式リリースは全必須条件の後に行います。

## リリースまでの実行計画

正式リリースの対象は、Sentisによるtext・image・audioのGPU推論と横断検索です。必須環境はWindows Editor / Player、macOS Editor、iOS / Android Player実機です。[ゴール](goal.md)にあるこの範囲を維持し、各工程の未実行・失敗を残件として管理します。現行版は `0.1.0-pre.1`、正式Releaseは未作成です。

| 工程 | 実施内容 | 次へ進む完了条件 |
| --- | --- | --- |
| 1. 基盤復旧・PR整理 | #89の最新CI確認、元の数値参照のCI保存、音声モデルの復旧または別世代としての完全な再検証。#89 → #86 → #87を順に統合 | 必須の参照・モデル取得が再現可能で、各PRの最新Required CI成功と差分監査が完了。新しいモデルは元の数値・GPU条件をすべて満たす |
| 2. 新規利用者の導入再現 | 空checkout / 空cacheから、公開手順でtext・画像3・音声2の全6 moduleを用意。固定Git版を新規consumerへ導入 | 開発者のローカルモデルやLibraryを渡さずに準備・推論できる。切断、破損、期限切れ、revision不一致から復帰し、source / model hashが一致する |
| 3. 実操作・性能・安定性 | 画像プレビュー、音声再生・途中停止、実マイクの明示開始/停止、日英検索、キャンセル・復帰、切替・解放を最終candidateで確認。遅延と資源残留を計測して必要な修正を行う | 正常系に加え権限拒否、deviceなし、欠落、対応外入力、中断・復帰を確認。初回/warm、load・各stage・release、CPU時間・GPU時間・RAM/VRAMを範囲を明示して記録し、必須の欠測・帰属不明を解消する |
| 4. WindowsマルチモーダルPlayer | IL2CPP / Release / High strippingでbuildし、text・image・audio、全必要role、横断検索と操作を実GPUで実行 | 元参照のvector・順位・scoreと一致し、fallback/readback監査、独立cold/warm起動、キャンセル・回復・解放、メモリ測定と正常終了がそろう。契約テストだけでは完了にしない |
| 5. 必須他環境の受け入れ | macOS Editor、iOS / Android実機で3モダリティ・検索・操作・メモリを確認。OS固有の配置、cache、権限、suspend/復帰を修正 | 各端末・OS・graphics API・source / model / 精度を固定した実GPU結果がある。CPU比較を分け、未実行・skip・CPU fallbackをGPU合格に数えない |
| 6. リリース候補と配布監査 | candidateのsource・モデル・依存を固定。packageの表示名・説明・keywordsを3入力対応の実態に整え、README、API/sample手順、CHANGELOG、利用条件、最低要件、既知の制限を更新。全体回帰と公開内容のprivacy監査 | 空consumerの固定版導入が再現でき、全必須環境の証拠がcandidateに対応する。旧PR/cacheの個人情報対応も結果を再監査。モデルとコードを分離し、画像は元bytesの目視/OCR・hash確認を終える |
| 7. 正式Releaseと公開後確認 | レビュー可能なRelease本文・版・tag・配布物一覧を準備し、公開依頼に基づきGitHub Releaseを作成。公開された固定tagから導入する | package version、CHANGELOG、tag、Releaseが一致し、公開物の読み戻しと新規consumerでの導入・sampleのsmoke確認が成功する |

**並行して先に行うこと**：Mac・iPhone/iPad・Android端末、Unityのbuild module、Xcode/署名、実GPU・マイクと実操作の検証手段を確認し、担当と実行条件を確定します。端末準備をWindows完了後まで先送りしません。旧PR refs・cacheの削除と元モデル回収に関するSupport対応も追跡します。 現在のWindows環境ではADBは利用可能ですが、接続端末は0台です。repositoryの登録済みself-hosted runnerも0台で、Mac・iOSの利用環境は未確定です。端末の接続・署名・実GPU検証手段の確保を工程5の開始条件として明示します。

**維持する精度と実行条件**：有限・単位長の768次元vector、FP32 cosine >= 0.999、採用Float16形式 >= 0.99、元のencoder要素別許容差、必要なQuery / Document / Raw、固定検索の全順位とscoreを確認します。モデル再検証ではnative full/cutのbit一致と全6信号も維持します。GPU前処理からencoder・backboneをつなぎ、中間CPU計算・readbackで成功扱いにはしません。tokenize・decode・音声取得等の入力境界と最終vector取得は別に記録します。

**時間と実行管理**：固定リリース日は、音声モデル方針と必須端末・署名環境の見通しが立ち、基盤復旧の実測時間を確認した時点で設定します。現時点で日数を確約しません。重い取得・変換・回帰・buildはGitHub Actionsを優先し、実GPU・実機・UIの確認は確保した環境で行います。ローカルUnityを重複起動せず、試行ごとに時間・メモリ上限を定め、同じ巨大モデルの探索・生成を根拠なく繰り返しません。

**実行範囲と条件**：利用者から工程7までの実行依頼を受けています。公開表示名は `yousan` の維持で確定しています。音声モデルは元の数値参照・許容差を保って再検証を進めます。別世代の採用には識別可能な世代名と全受け入れ条件の合格を必要とし、必須条件の省略や原本と異なるモデルの原本扱いは行いません。正式Releaseは工程1〜6の完了後に実施します。

各工程の詳細と既存checkpointは以下に保持します。工程ごとにsource・モデル・精度・環境、合格・失敗・未実行、残件を記録し、最終candidateでの証拠をそろえてから次の完了判定を行います。

## 直近の作業順

### 元の数値参照の復旧と次の数値比較（2026-10-18）

#89の公開head `317bae3` の[CIは終了](results/privacy-guard-ci-317bae3-20261018.json)し、Python全4環境・package・workflow lint・model referenceが成功しました。元音声モデル取得の6信号と集計は失敗し、Required CI全体は不合格です。#89・#86・#87は未マージです。

[元の数値参照だけを保存したCI](results/original-audio-golden-ci-readback-20261018.json)は成功しました。実際にアップロードされたZIPのSHA256・CRCと元JSON2件のSHA256を読み戻し、私的な2コピーも照合しました。元モデル2件とGPUの復旧・合格ではありません。

診断workflowの消失した参照取得先を復旧済みの数値原本へ接続し直しました。元JSONのhashと元の許容差を維持し、新たな収集で保存するnative・ONNX出力を元の数値と比較します。前回のbyte復旧runは比較用出力を保持していなかったため、1回の限定実行で必要な証拠を取得します。Required CIの原本モデル条件は変更しません。新たな出力のsource・実行環境・model hashは原本と分け、比較失敗・未実行を合格扱いにしません。


### 0. CI・個人情報対応・モデルの再受け入れ

表示名とJUnit出力の修正は公開済みで、`317bae3` のPython全4環境は合格しました。元数値参照のCI保存も完了しています。残る順序は、元の数値参照を使うモデル比較、モデル取得経路の復旧、各PRの最新Required CI、#89 → #86 → #87の統合です。

元モデル2件のbyte復旧は、私的・公開成果物の探索と元のPython tree・依存lockでの再生成とも不成功でした。原本回収はSupportへ照会中です。別世代を検証する場合も元の数値参照・許容差、native full/cutのbit一致、全6信号・全carryのONNX検証と実GPU・consumer・Playerの条件を維持し、合格前に必須条件を置き換えません。

旧PR refs・cache・残るActionsメタデータの個人情報対応は、モデルの数値比較とは別にSupportの結果と照合します。固定参照・モデルは完全なbytesとhashを私的に退避し、元の検証画像を加工・再生成して証拠に置き換えません。大量削除は再開しません。

以降の最終受け入れでは、新規consumerの3入力・検索・操作・速度とメモリ、Windows Release Player、必須他環境を確認します。旧sourceの合格・skip・CPU fallback・契約テストだけで最終実GPU合格にはしません。[過去の統合監査](results/runtime-ready-integration-2a29c68-20261018.json)は当時のcheckpointとして保持します。

Windowsの画像・音声Player検証基盤は私的な統合候補に保持しています。[契約テストと元入力の読み込み65件](results/media-player-contracts-20261018.json)は合格しましたが、最終candidateでの実モデルGPU・Release Player実行は未実施です。[選択したPython 133件とworkflow lint](results/private-final-integration-20261018.json)も当時の固定sourceの確認であり、最新headのRequired CIを代替しません。

### 1. 新規利用者のモデル準備

最新componentをまとめた[ツール関連132件のpreview](results/combined-tools-preview-86afcbc-20261018.json)は成功した。これは以前の69/72件とは別の固定sourceで、公開配布元のempty-cache取得・実モデル/GPUの追加合格ではない。[自動化pluginなし新規consumerの実操作](results/fresh-git-native-ui-b519-no-uloop-partial-timeout-20261018.json)は画像登録・参照一致までで900秒上限に達したため、最終の一連操作は未完了のまま維持する。停止済みの同じ試行を成功に読み替えず、公開取得を伴う最終candidateの確認へ対応付ける。

固定参照・現行bundle形式を使い、空checkoutからtext / image / audioの参照・qualified graph取得、変換、bundle生成を再現可能にします。期限切れartifact、欠落・破損・revision違いからの回復も文書化します。

**完了条件**：既存cacheを所有しない第三者が全6 moduleを準備できること。producer / graph / model SHA、全ファイルhash、成功と失敗を記録します。モデルはGit / LFSへ入れません。

モデル取得の追加checkpointとして、PR #86で実FP32 bundle全6 module・820 object・8,583,084,520 bytesの[ローカルHTTPS転送と修復](https://github.com/ayutaz/unity-embeddinggemma-2/blob/e02663d50f539b3fb49674442747a46c85f35450/docs/results/real-local-https-media-recovery-f09317e-20261018.json)を確認した。切断後の818件受信・2件再利用、warmの820件再利用、同サイズ重み破損1件だけの修復、404時の旧descriptor保持と再試行を含む。元と出力の計1,640ファイルを独立に再hashした。sourceは既存ローカルbundleで、公開配布元のempty-cache取得と最終consumerのGPU・操作受け入れは未完了のまま維持する。

取得処理の後続では、[PR #86の排他・強制終了後回収](https://github.com/ayutaz/unity-embeddinggemma-2/blob/ce7e13689f42950937671e0e8436deca61b7d363/docs/results/real-media-crash-recovery-85a2d71-20261018.json)を確認した。実9MiB重みの受信中に所有CLIを終了し、競合拒否、残った一時ファイル回収、1件受信・819件再利用とwarm全820件再利用を確認している。関連51件はWindowsで成功。これを含むローカル統合previewの関連72件とsource対応は[後続checkpoint](results/priority-1-to-3-followup-0658004-20261018.json)に分ける。さらに[別sourceの実Ctrl+C追試](https://github.com/ayutaz/unity-embeddinggemma-2/blob/678eb504ea0a7817ea34ae67b4cc8bb485fdf06a/docs/results/real-media-ctrl-c-2b6f45d-20261018.json)は、2 Red→関連53 Green、実9MiB重みの受信中のexit130・後始末・再試行・warm・820件のhash照合まで確認した。実行結果はWindowsであり、通信停止中の即時キャンセル時間・他OS・公開固定元からの取得は別に確認する。

### 2. 利用者操作とサンプルUX

更新済み#85の固定Git `1d8d868`による[新規consumerの一連操作](results/fresh-native-ui-1d8d868-full-local-bundle-20261018.json)は、既存ローカルFloat16モデルで確認済み。Library・モデルコピー・SDK overlayなしで画像・音声の実GPU登録、日英2検索と元参照、プレビュー、エラー復帰、cacheキャンセル・再試行・切替、再生と明示途中停止、解放・PlayMode停止・Editor通常終了code0を確認した。最初の再生停止は終端付近で未達、別の再生/Stopのみ残り2.688秒・8.557482秒で元gate合格という両記録を残す。以前のfresh timeoutは保持し、公開手順で取得したモデル・最終統合sourceとの対応付け、実マイク・権限/device・中断復帰を次に確認する。

1の手順で用意したモデルと固定Git版を空consumerへ導入し、手動で登録→検索→キャンセル→再登録→切替→解放を確認します。既存bundleでの操作確認は先行できます。結果画像プレビュー・音声再生、マイクの明示開始 / 停止は機能ごとに別PRで実装します。

2026-10-18の固定Git `e28a5fd` 新規consumerでは、既存ローカルFloat16 bundleによる画像・音声の実画面GPU登録、日英検索の参照順位・score、画像プレビュー、音声再生位置の進行、エラー復帰・キャンセル再試行・切替・解放・正常終了code0を確認しました。[操作監査](results/fresh-git-native-ui-e28a5fd-20261018.json)。別のwarm起動では11.2秒音声を7.232秒の位置で停止し、解放と正常終了code0も確認しました。[途中停止監査](results/native-audio-early-stop-e28a5fd-20261018.json)。実マイクと公開入力からの空cache取得は別の未完了項目として維持します。非録音時のpauseで録音破棄のstatusが出る問題、Editor終了時のJobTempAlloc警告は別に扱います。

**完了条件**：日英入力・対応外入力・モデル欠落・権限拒否・deviceなし・中断・復帰を確認すること。固定PCMの数値検証と実マイクの動作検証を分け、無断の保存・送信・収集はしません。

最新sampleを含む固定Git `680af07` の[途中停止のwarm追試](results/native-audio-early-stop-680af07-followup-20261018.json)も、11.2秒の音声を9.130625秒の位置で観測した後に実Stop操作で停止し、解放・正常終了code0を確認した。音声vectorと日本語検索scoreは元参照・許容差内、315 package・86 sampleファイルは終了後に照合済みである。先行試行の自然終了に近い停止・監査gate失敗は保持し、今回の成功を公開取得・実マイクやfresh全操作の合格へ広げない。

### 3. 性能・安定性の測定と改善

candidate・入力・model hashを固定し、最初に測定します。遅い区間・資源残留の原因を絞って修正し、Float16 / GPU単独と連続実行も区別します。測定設計は2の完了を待たず進められます。

**完了条件**：load・初回・warm・前処理・encoder・backbone・解放、CPU / GPU時間、ホストメモリ / VRAMを別記すること。Editor import・検索DB・allocation・domain reloadも分離し、前後で精度・順位・正常終了を維持します。

2026-10-18時点のSDK `d7b2233` では、[GPU出力の不要なCPU storageを省く変更](gpu-write-only-output-allocation.md)をTDD・GPU演算等120件で検証しました。同条件の計測付き画像反復で旧出力とのbyte一致と改善を観測し、全6 module×FP32 / Float16重みの単独参照32ケース、両形式の混在検索の全順位・再試行・解放・正常終了も確認しています。保存重みのFloat16化を全演算FP16化や形式間の高速化にはしません。

同SDKの[画像3・音声2 profile×両形式・40反復](media-gpu-submission-observations.md)で元参照・解放・exit0、encoder 17,176区間と前処理/backbone 80区間、CPU/wall・9プロセスのGPU memoryを確認しました。[text両形式の固定query各4反復](text-current-sdk-observations.md)はmain-thread marker内のmanaged bytesを校正済みの停止CPU履歴で記録しています。GPU Recorderと履歴の値の差、短いstageのGPU memory欠測は保持し、CPU wallとの加減からkernel時間を作りません。

Editorでは[検索割り当て・空/再利用Library](editor-search-allocation-observations.md)、[両sampleと直接index再構築](editor-index-rebuild-observations.md)、[EditMode GC校正・実再構築中のframe全体allocationとnative memory](editor-frame-allocation-observations.md)を分けて確認しました。最後の条件は7 pass / skip0 / exit0で、40文書とfixture・両sceneの保持を2回確認しています。背景index専有bytesは未測定であり、frame全体の値や重なるstartup/import/reload時間を加算しません。

[ロード・解放の範囲監査](load-release-measurement-scope.md)と[Windows engine別のOS観測](windows-gpu-engine-observations.md)では、既存12実行のCPU/wallと、別の実text2実行のcounter時刻・範囲・欠測を照合しました。[frame GC controlと代表実音声](async-frame-allocation-observations.md)も、入力準備・load・検証I/O・release・Editorを含む累積量として保持します。load/release全体のGPU時間、個別非同期stage・背景indexの専有allocationを、この観測から確定はしません。

次は、これらの測定範囲・失敗・欠測を最終candidateへ対応付け、最新Required CIと要件別の監査を閉じます。性能全体は未合格です。過去の12GiB guard中断・16GiB EditMode900秒timeout・Profiler不整合・テスト未実行も[検証状況](status.md)と各記録に保持します。公開取得・実マイク・最終candidateの実操作を含む1〜3全体の条件を維持し、異なる実行条件から速度改善率や最低RAM / VRAMを推定しません。

粗いOS counterの後続として[native PDH追試](results/native-pdh-text-stage-observation-f604baf-20261018.json)を追加した。同じFP32 text4 vectorの元参照・解放・exit0を維持し、8,783点、取得間隔中央値4.6489msとclockを検査した。load内の正数はprocess活動量として保持し、queryの3Dで経過時間を超える差分と、releaseのpair不足はnullとした。26件の新しいRed→関連43 Greenはツールの検証であり、性能全体・SDK専有load/release時間の合格ではない。同方式の単純な再実行で未取得を埋めず、帰属と更新の制約を次の方式の選択へ反映する。

### 4. WindowsマルチモーダルPlayer

1〜3で安定したcandidateを使い、画像・音声sampleの配置・cache・切替・失敗回復をPlayerへ接続します。

先行作業として[画像・音声の固定入力Player検証基盤](windows-media-player-validation.md)を私的なbranchで準備しています。元の参照・入力hash、5 profile、全最終vector、GPU Worker割当、キャンセル後の復帰と解放を記録します。契約テスト・元入力の読み込みと、実際のRelease build・GPU実行は別に判定し、この基盤だけで全体合格にはしません。

**完了条件**：IL2CPP / Release / High stripping、実モデルと全モダリティの固定順位・score、実GPU、独立cold / warmの終了code0を記録すること。旧text Playerの証拠を流用しません。

### 5. 必須他環境の受け入れ

macOS Editor、iOS / Android実機を環境ごとのPRで進めます。環境確保は今から並行し、最終受け入れは安定candidateで行います。

**完了条件**：次表の端末・backend・モデル・source・精度・操作・メモリ・終了を記録すること。未実行やCPU fallbackをGPU合格にしません。

### 6. 最終candidateの導入・配布監査

1〜5の必須条件が合格したら、版更新、CHANGELOG、利用手順・ライセンスを整備し、空consumerで再現します。

**完了条件**：全結果とcandidate source / model hashの対応、固定Git導入、packageとモデル配布の分離、文書コマンド・sample再現、required CI成功を確認すること。

### 7. 正式公開

6が完了し、公開の明示依頼を受けてから、承認されたcandidateをtag固定で導入確認し、GitHub Releaseを作成します。

**完了条件**：tag / version / Releaseの一致と読み戻し。画像・音声を含む必須検証を省いてtext-onlyの正式版を先行公開しません。

各行を巨大な1本のPRにはしません。準備、画像表示、音声再生、収録、測定、原因別修正、環境別受け入れをレビュー可能な単位に分割します。所要時間は実機・runnerと測定で決め、未実測の性能目標を達成済みにしません。

## 正式公開まで維持する条件

| 対象 | 必須の証拠 | 現時点で不足するもの |
| --- | --- | --- |
| Windows Editor / Player | 固定Git導入、3モダリティの実モデル・固定順位、sample実操作、CPU / 実GPUを別判定、PlayerのIL2CPP / stripping・独立2起動・正常終了 | 最終candidate全体、画像・音声Player、反復測定・一部UX |
| macOS Editor | 実Mac、CPU / Metal GPUの3モダリティ、sample操作・解放・測定 | 環境確保と実測 |
| iOS Player | Mac / Xcode / build module / 署名・実機、IL2CPP / stripping、CPU / Metal GPU、配置・cache・権限・suspend / 復帰 | 環境確保と実測 |
| Android Player | 対象UnityのSDK / NDK / JDK、実ABI・graphics API・実機、CPU / GPU、jar配置・cache・権限・終了 | 最新toolchain・実機受け入れ。過去APKのbuild監査だけでは不足 |

FP32最終vectorは有限768次元・単位長・参照cosine >= 0.999、採用するFloat16重み形式は >= 0.99、固定横断検索は全順位一致を要求します。encoder中間値・scoreには元の固定許容差を使い、失敗に合わせて緩めません。各profile・role・混在索引とmodule切替を対応する参照で確認します。

CPUでの明示比較はGPU受け入れと分けます。利用者向け画像・音声推論の主経路はGPUであり、非対応演算・OOMをCPUへ切り替えて成功扱いにしません。CPUの数値文脈差、失敗・skip・未実行はそれぞれ記録します。

Web / WebGPU・動画・学習・本格的なベクトルDBは必須公開範囲に追加しません。モバイルのメモリ不足が判明した場合も記録と計画を更新し、未達を黙って除外しません。

## 開発・実行のルール

- 振る舞いの変更は、独立した期待値で意図したRed→最小実装→Green→整理。文書変更に架空のRed / Greenは付けません。
- Pythonは`tools/`とuv、重い取得・変換・回帰・buildはActionsを優先。実GPU・UI・端末の代替できない確認だけをローカルで行います。
- 監査済みmodel・Libraryを再利用し、変更のない巨大graphを再生成しません。Unityは重複起動せず、timeoutでは同じPID・operation・保存結果を確認します。
- 各PRにsource・model / fixture hash、Red / Green、実モデル・実backend、失敗 / skip / 未実行、CIと残件を記録します。main直pushは禁止、マージは依頼された範囲で行います。
- 実GPUの実行条件と数値結果、Python CIの成功、UI操作、公開操作を別々に判定します。最新headで未実行の条件を過去の成功で埋めません。

実装時のC1〜C7・I1〜I6・A1〜A7の詳細は[全体TDD設計](multimodal-tdd-plan.md)・[画像設計](m3-plan.md)・[音声設計](m4-plan.md)に保持します。そこにある過去checkpointの「次の作業」より、本ページの現在の順序を優先してください。
