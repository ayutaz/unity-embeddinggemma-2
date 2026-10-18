# 優先1〜3の残タスクと完了条件

更新: 2026-10-18。依頼「3まで進めてください」の範囲は、[ロードマップ](roadmap.md)のモデル準備・利用者操作・性能改善です。測定項目だけを終えて全体を完了にはしません。以下は固定された検証記録の監査であり、進行中のプロセスの成功を先取りしません。

[元の9要件と実行記録の照合](results/priority-1-to-3-requirement-audit-adf7c24-20261018.json)を追加しました。PR #85 / #86 / #87の固定headを組み合わせたローカルpreviewで、取得41契約・実HTTPS7件・Windows counter21件の計69件が成功しました。SDK packageとvalidationは実行済みcomponentから変更していません。これはツールの組み合わせ検証であり、実モデル・実GPU・公開取得・実UIの追加合格ではありません。CI欄は取得時点のsnapshotで、最新headのchecksは各PRで確認します。

後続の[取得修正・操作追試のcheckpoint](results/priority-1-to-3-followup-0658004-20261018.json)では、取得PR `ce7e136` と性能PR `19d1edc` のローカルpreview `0658004` のツール関連72件、同一package treeの固定Git `680af07` での実音声途中停止を対応付けた。前の69件監査・途中停止未達の記録は変更していない。この追加監査も、公開取得・実マイク・性能全体・最終統合の完了にはしない。

さらに[PR #86のCtrl+C追試](https://github.com/ayutaz/unity-embeddinggemma-2/blob/678eb504ea0a7817ea34ae67b4cc8bb485fdf06a/docs/results/real-media-ctrl-c-2b6f45d-20261018.json)で、CLIのキャンセル表示・exit130を2 Red→関連53 Greenで確認した。Windowsの実Ctrl+Cイベントと9MiB実重みで、強制killなしの後始末・失敗receipt・旧descriptor保持、再試行1件受信/819件再利用、warm820件再利用まで成功した。これは前の72件を更新した統合実行ではなく、別sourceの取得処理の追加証拠である。SSL受信のpauseはイベント送信後に解除しており、停止した通信の即時中断時間は保証しない。公開取得と他OSの最新CIは別に残す。

## 残タスク

最新の[統合ツールpreview `86afcbc`](results/combined-tools-preview-86afcbc-20261018.json)では、取得PR #86 `678eb50` と性能PR #87 `34afe64`、先行Runtime #88を組み合わせ、取得・HTTPS・native PDH・GPU engine collector・packageの関連132件が成功した。SDK package treeとvalidation、取得CLIの5ファイルを固定componentへ照合し、実Ctrl+C fixtureのexit130・強制後始末なしも確認した。これは小さなfixtureのツール検証で、実モデル・実GPU・公開取得の追加合格ではない。以前の69件・72件のcheckpointは変更しない。

同じpackage tree `b5195f3` の[新規consumer・自動化pluginなしの実操作追試](results/fresh-git-native-ui-b519-no-uloop-partial-timeout-20261018.json)も900秒上限で終了code1となった。モデル欠落からの復帰、cache準備のキャンセル・再試行、画像のGPU登録と768次元の元参照一致（cosine 0.9999999999997693）までを部分証拠として保持する。音声登録・日英検索・再生・解放・正常終了はこの試行では未実行で、一連の操作は不合格。controllerの900秒はimport・ネイティブ操作・ツール・待機を含み、推論時間やplugin除去の速度比較にはしない。終了後315 package・86 sample・observer・root user4ファイルを照合した。過去の正常終了したwarm追試も別source/起動条件の記録として保持する。

| 要件 | 現在の証拠 | 残る作業・完了条件 |
| --- | --- | --- |
| 1a: 全6 moduleの公開入力からの新規準備 | PR #86に固定descriptor・object検証を伴う取得ツール。配布候補は既存ローカルbundle | 公開先と固定commitを確定し、空の入力・出力cacheから全6 moduleを取得。producer、revision、descriptorと全objectのhashを照合し、実行receiptを保存する |
| 1b: 中断・破損・期限切れからの回復 | PR #86の[実FP32 bundle・820 objectのローカルHTTPS故障注入](https://github.com/ayutaz/unity-embeddinggemma-2/blob/e02663d50f539b3fb49674442747a46c85f35450/docs/results/real-local-https-media-recovery-f09317e-20261018.json)に加え、[OS排他と強制終了後の回収](https://github.com/ayutaz/unity-embeddinggemma-2/blob/ce7e13689f42950937671e0e8436deca61b7d363/docs/results/real-media-crash-recovery-85a2d71-20261018.json)を3 Red→関連51 Greenで確認。実9MiB重みの受信中に所有CLIを終了し、競合拒否・残った一時ファイル1件回収・1件受信/819件再利用・warm820件再利用・出力820件hash照合まで成功 | 公開した固定元でも同手順と文書コマンドを確認する。既存ローカルsourceでの受信・再hashを公開empty-cache取得・GPU合格へ広げない。Windowsの実強制終了とCtrl+C、他OSの検証を区別する |
| 2a: 固定Git consumerの操作 | 更新済み#85の固定Git `1d8d868`を新規consumerへ導入し、[画像・音声GPU登録、日英検索、cacheキャンセル・再試行・切替・解放と正常終了](results/fresh-native-ui-1d8d868-full-local-bundle-20261018.json)を実マウス・キーボードで確認。モデル欠落・非対応入力・空queryの拒否と復帰、元vector・全順位・scoreを照合 | 既存ローカルFloat16 bundleの一連操作は確認済み。公開手順で取得したモデルと最終candidateへ対応付け、公開取得・実マイクの証拠を別に揃える |
| 2b: プレビュー・再生・停止 | 上記の新規consumerで画像プレビュー、音声再生・明示途中停止・解放・exit0を確認。再生2回のうち最初は終端付近で旧gate未達、別の再生/Stopは残り2.688秒・観測開始から停止8.557482秒で旧gate合格。両試行を監査へ保存 | 公開手順で取得したモデルと最終candidateへ対応付ける。最初の再生の途中停止成功・聴感品質・実マイク・公開取得の合格にはしない |
| 2c: 実マイク | 明示操作・失敗処理の実装と契約テスト。実録音は未実行 | 録音の明示許可後、開始・停止・中断復帰を実機で確認。権限拒否・deviceなしの扱いも記録。固定PCMやfake deviceを実マイク成功にしない |
| 3a: 初回・warm・反復とstage別CPU/GPU時間 | 新SDK `d7b2233` の画像3・音声2 profile×両形式・40回で同一入力CPU/wall・独立参照・解放・exit0、encoder 17,176と前処理/backbone 80のGPU区間を確認。[media集約](media-gpu-submission-observations.md)と[同SDKのtext両形式4反復・停止CPU履歴](text-current-sdk-observations.md)を照合 | [load/release](load-release-measurement-scope.md)・[OS counter](windows-gpu-engine-observations.md)・非同期区間の範囲と欠測は監査済み。load/release全区間のGPU時間、個別stageの専有allocationは未取得であり、性能全体の合格にはしない。GPU Recorderと履歴の差を保持し、CPU wallとGPU queue区間を加減してkernel時間にしない |
| 3a: ホスト・VRAM・Editor負荷 | media 9プロセスとtext両形式でtimestamp付きGPU memoryを確認し、欠測を明記。[Editor検索割り当て・空/再利用Library](editor-search-allocation-observations.md)は各10 pass / skip0 / exit0。[frame GC controlと代表実音声](async-frame-allocation-observations.md)も校正・記録欠落のRed・接続後Greenを確認。[両sample導入後の明示再構築](editor-index-rebuild-observations.md)に加え、[EditMode GC校正・実再構築中のframe全体allocationとnative memory](editor-frame-allocation-observations.md)は7 pass / skip0 / exit0、2反復で40文書と両scene保持 | 背景indexの専有allocation・非同期allocationの測定範囲と欠測の根拠を最終監査する。frame全体の値・プロセス全体・stage・寿命ピークと、重なるEditor時間を加算しない |
| 3b: FP32/Float16と原因別改善 | [出力確保の新SDK候補](gpu-write-only-output-allocation.md)はTDD・GPU演算等120件、両形式の固定画像各4回byte一致と計測付きwarm反復の改善、[全6 module×両形式の単独参照32ケース](results/gpu-output-d7b2233-all-module-isolated-reference-20261018.json)、両形式の混在全順位・再試行・解放・exit0を確認 | 固定Windows/GPU・入力での原因別改善と回帰は監査済み。元の精度・全順位・GPU要件・正常終了を維持し、通常API全体・任意入力・形式間の速度保証へ拡張しない。3aの欠測と最終candidateの受け入れは別に残す |
| delivery: PRと最終監査 | Runtime先行統合のPR #88はRequired CI成功後にmain `d9ae07b`へ統合済み。統合treeはCI候補 `57d2cdb`と一致し、#85 / #87へmain祖先を取り込んで重複Runtime差分を整理。PR #85 UX、#86取得、#87性能・Float16は未統合で、#87は#85に依存 | 更新した各headのRequired CI、差分・依存を確認し、依頼範囲内で統合。最終sourceとmodel hashへ全記録を対応させ、上記全行の証拠を揃えてから優先1〜3を完了とする |

## そのまま維持する失敗と境界

- 固定Git `1d8d868`の[新規consumerの後続成功](results/fresh-native-ui-1d8d868-full-local-bundle-20261018.json)は、過去の900秒timeoutを変更しない別試行。初回importと全ネイティブ操作を含め803.890085秒、1800秒の事前設定上限内で通常終了した。Library・モデル・SDK overlayをコピーせず、同じpackage tree `b5195f3`を使用。終了後315 package・86 sample・observer・root user4ファイルを照合した。controller時間は推論latencyやplugin比較ではない。最初の再生は自然終了まで0.021375秒しか残さず旧gate未達、別の明示再生/Stopだけが元の停止位置・時間条件を満たした。両方の生記録と判定を保持し、gateを緩めていない。observerのclip未割当時のtimeSamples警告も残るため無警告の合格とはしない。公開empty-cache取得・実マイク・性能全体は未完了。

- 最新sampleを含む固定Git `680af07` の新規consumerでは、実マウス・キーボードで画像と音声を登録し、日英2 queryの順位・scoreと768次元vectorを独立参照へ照合した。ただし所有Editorは900秒の上限でexit1となったため、[部分実行の監査](results/fresh-git-native-ui-680af07-partial-timeout-20261018.json)は操作全体を不合格とする。cache準備中のキャンセルと再試行、モデル欠落からの復帰、画像表示は観測済み。別の[同consumerのwarm追試](results/native-audio-warm-completion-680af07-20261018.json)では、音声のGPU登録・日本語検索を再照合し、空query拒否後の再実行、再生位置の進行、停止ボタン、登録・preview・session解放、終了code0を確認した。2回の停止は11.2秒音声の終端に近く、早い途中停止の証拠にはしない。終了後のpackage 315ファイル・sample 86ファイル・受動observerとroot user 4ファイルを照合した。Unity CLI Loopの欠落assembly診断・observerのtimeSamples警告・JobTempAlloc等は監査に保持し、無警告の成功としない。さらに[非対応入力の追試](results/native-input-recovery-680af07-early-stop-unproven-20261018.json)で、画像modeへの音声入力を拒否し、audio-1119への切替・再登録後の768次元参照一致と解放・exit0を確認した。同追試の十分早い停止gateは監査scriptのAssertionErrorとして保持する。この先行追試時点では、公開取得・十分早い音声途中停止・実マイクは未完了だった。後続の途中停止の合格は次段落に分ける。
- 後続の[同consumerの途中停止追試](results/native-audio-early-stop-680af07-followup-20261018.json)は、元のgate（最後の再生位置が159,200 sample未満、観測した再生開始から停止まで10秒未満）を変更せず合格した。11.2秒の固定音声を9.130625秒で観測した後、実Stop操作で停止し、解放・PlayMode停止・正常終了code0まで確認した。音声768次元の参照cosineは0.9999999905661618、固定日本語queryのscore誤差は0.0000026369377859047916で元の許容差内。終了後の315 package・86 sampleファイル、observer、root user 4ファイルを照合し、今回もtimeSamples等の警告を保持した。先行試行の失敗は残し、今回のwarm一連の成功をfresh全体・公開取得・実マイクへ広げない。
- FP32の12GiB guard中断、16GiB EditModeでの900秒fixture timeoutを保持する。別のPlayMode成功で過去の失敗を消さない。
- GPU RecorderとProfiler履歴の値の差、既知配列を検出できなかった割り当てAPI、短い呼び出しのVRAM欠測を保持する。未知を0や合格へ置き換えない。
- SDK `e28a5fd`、UX修正 `99cc826`、この監査前の性能branch `10a3ed5` の `Packages/com.ayutaz.embeddinggemma/Runtime` Git treeは全て `4e59c35842ef153f8ab65e46fa848c380d3c5844`。これはruntime sourceの同一性だけを示し、sample・計測コード・consumer・実行条件の同一性や最新版の操作合格は示さない。
- モデル公開と実マイクの許可待ちの間も、独立して進められる測定・文書・回帰を進める。公開先へのuploadや録音を許可済みと推定しない。

## 作業順と後続範囲

PDHに代わる[ETW経路の対照確認](results/dxg-method-controls-72bd2b1-20261018.json)では、現在のtokenでDxgKrnlの有効化・停止がreturn0、未変更のMicrosoft decoder `72bd2b1` の単体130件と実D3D12対照2件がfail0 / skip0 / exit0だった。コピーpacketのPID帰属・時刻順序とresource破棄イベントの時刻窓は検査済みで、終了後の所有ETW session残留はない。権限変更・Unity・モデル実行は行っていない。実SDK stageのGPU時間・driver退役・専有allocationは未取得であり、この小規模な対照を全体合格にしない。次は上限付きの対象PID consumerとraw clock/lossの校正を用意し、固定SDK/model/inputの元参照・正常終了へ対応付ける。

3aの後続では、[native PDHと同SDK text4 vector](results/native-pdh-text-stage-observation-f604baf-20261018.json)を実行し、参照・解放・exit0、終了後554 source一致と、所有process handleの生存を検査した8,783点を照合した。取得間隔中央値4.6489msでload内2,687点に正数を観測したが、queryの3Dは隣接差分が経過時間を超え、releaseは1点だけでpair不足だった。26件の新しいRedと既存を含む43件のGreen、clockとengine別の棄却を保持する。SDK専有load/releaseの合格にはせず、このPDH方式を繰り返すだけでは解消しない帰属・provider更新の制約として次の測定方式を判断する。

実GPUの残る計測を1本ずつ進め、Editor負荷と分離して原因を特定する。並行して公開手順と最終candidateの受け入れ準備を整え、許可された公開取得・マイク確認を実施する。最後に最新CI・source対応と全行の監査を閉じる。

Windows画像・音声Player、macOS/iOS/Androidの最終受け入れ、版更新・正式Releaseはロードマップの4〜7であり、今回の1〜3の完了条件に置き換えたり、黙って完了扱いしたりしない。
