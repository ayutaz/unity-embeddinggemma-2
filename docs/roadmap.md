# ロードマップ

更新: **2026-10-18**。main `d9ae07b`（PR #88）を統合済み基準とする現在の作業計画です。過去の基準SDK `e28a5fd` と統合済みRuntimeに対応する `d7b2233` を区別し、各実行のsourceを記録します。確認済みの事実は[検証状況](status.md)、変更しない目標は[プロジェクトのゴール](goal.md)を参照してください。

テキスト・画像・音声のGPU API、Windows text Player、画像・音声検索サンプルとcacheは統合済みです。次は、他の利用者が準備・操作を再現できる状態を作り、そのcandidateで性能と対象環境を検証します。正式リリースは全必須条件の後に行います。

## 直近の作業順

Runtime負荷軽減の[PR #88](https://github.com/ayutaz/unity-embeddinggemma-2/pull/88)はRequired CI成功後にmain `d9ae07b`へ統合済み。[単独候補](results/runtime-ready-integration-2a29c68-20261018.json)の517 pass / skip0 / exit0、既存実モデル参照sourceとのRuntime一致、統合treeとCI候補の一致を確認し、#85 / #87へ実mainの祖先を取り込んで重複Runtime差分を整理した。優先1〜3全体の公開取得・実マイク・最終操作・性能欠測は、この先行統合によって完了扱いにしない。

依頼された優先1〜3の要件ごとの現状・残作業・完了条件は[残タスク監査](priority-1-to-3-acceptance.md)を参照してください。固定sourceの数値成功、最新sampleの操作、公開元からの取得と性能全体の受け入れをそれぞれ確認します。

### 1. 新規利用者のモデル準備

最新componentをまとめた[ツール関連132件のpreview](results/combined-tools-preview-86afcbc-20261018.json)は成功した。これは以前の69/72件とは別の固定sourceで、公開配布元のempty-cache取得・実モデル/GPUの追加合格ではない。[自動化pluginなし新規consumerの実操作](results/fresh-git-native-ui-b519-no-uloop-partial-timeout-20261018.json)は画像登録・参照一致までで900秒上限に達したため、最終の一連操作は未完了のまま維持する。停止済みの同じ試行を成功に読み替えず、公開取得を伴う最終candidateの確認へ対応付ける。

固定参照・現行bundle形式を使い、空checkoutからtext / image / audioの参照・qualified graph取得、変換、bundle生成を再現可能にします。期限切れartifact、欠落・破損・revision違いからの回復も文書化します。

**完了条件**：既存cacheを所有しない第三者が全6 moduleを準備できること。producer / graph / model SHA、全ファイルhash、成功と失敗を記録します。モデルはGit / LFSへ入れません。

モデル取得の追加checkpointとして、PR #86で実FP32 bundle全6 module・820 object・8,583,084,520 bytesの[ローカルHTTPS転送と修復](https://github.com/ayutaz/unity-embeddinggemma-2/blob/e02663d50f539b3fb49674442747a46c85f35450/docs/results/real-local-https-media-recovery-f09317e-20261018.json)を確認した。切断後の818件受信・2件再利用、warmの820件再利用、同サイズ重み破損1件だけの修復、404時の旧descriptor保持と再試行を含む。元と出力の計1,640ファイルを独立に再hashした。sourceは既存ローカルbundleで、公開配布元のempty-cache取得と最終consumerのGPU・操作受け入れは未完了のまま維持する。

取得処理の後続では、[PR #86の排他・強制終了後回収](https://github.com/ayutaz/unity-embeddinggemma-2/blob/ce7e13689f42950937671e0e8436deca61b7d363/docs/results/real-media-crash-recovery-85a2d71-20261018.json)を確認した。実9MiB重みの受信中に所有CLIを終了し、競合拒否、残った一時ファイル回収、1件受信・819件再利用とwarm全820件再利用を確認している。関連51件はWindowsで成功。これを含むローカル統合previewの関連72件とsource対応は[後続checkpoint](results/priority-1-to-3-followup-0658004-20261018.json)に分ける。さらに[別sourceの実Ctrl+C追試](https://github.com/ayutaz/unity-embeddinggemma-2/blob/678eb504ea0a7817ea34ae67b4cc8bb485fdf06a/docs/results/real-media-ctrl-c-2b6f45d-20261018.json)は、2 Red→関連53 Green、実9MiB重みの受信中のexit130・後始末・再試行・warm・820件のhash照合まで確認した。実行結果はWindowsであり、通信停止中の即時キャンセル時間・他OS・公開固定元からの取得は別に確認する。

### 2. 利用者操作とサンプルUX

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
