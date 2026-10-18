# 現在の対応範囲と検証状況

更新: **2026-10-18**。統合済み基準はmain `d9ae07b`（PR #88）。追加検証は記録ごとの固定sourceに対応し、過去の基準SDK `e28a5fd` と実GPU参照済みRuntime `d7b2233` を区別します。開発版 `0.1.0-pre.1`、公開tag・正式Releaseはありません。

候補SDKの[GPU出力確保の変更](gpu-write-only-output-allocation.md)では、CPU storageの不要なゼロ初期化・転送を省きました。GPU演算等の回帰120件、全6 module×FP32 / Float16重みの単独実GPU参照32ケース、両形式の混在検索の全順位・再試行・解放・正常終了を新SDKで確認しました。同じ計測付き画像呼び出しのwarm反復はFP32が19.3–29.4秒から4.8–5.5秒、Float16重みが30.2–34.0秒から4.6–5.1秒でした。少数の固定入力観測を通常API全体や形式間の高速化へ拡張しません。性能全体・公開モデル取得・実マイク・最終candidate操作の受け入れは残っています。

[導入](../README.md) / [ドキュメント一覧](README.md) / [今後の計画](roadmap.md)

Runtime負荷軽減の[PR #88](https://github.com/ayutaz/unity-embeddinggemma-2/pull/88)はRequired CI成功後、2026-10-18 00:57:10 UTCにmain `d9ae07b`へsquash統合された。main既存サンプルを保った[単独候補の回帰517 pass / skip0 / exit0](results/runtime-ready-integration-2a29c68-20261018.json)と、統合commitのtreeがCI候補 `57d2cdb`に一致することを確認した。同候補と本branchのRuntime treeは `7d6e4314552b221d06db881405e7ccff3f449a77` で一致する。#85 / #87には実mainの祖先を取り込み、重複Runtime差分を整理した。公開取得・実マイク・最終操作と性能全体の受け入れは別に残す。

## 統合済みの機能

後続の[統合ツールpreview132件](results/combined-tools-preview-86afcbc-20261018.json)は、最新取得CLIとnative PDH・packageを同時に確認した。一方、同じpackage treeの[自動化pluginなし新規consumer](results/fresh-git-native-ui-b519-no-uloop-partial-timeout-20261018.json)は画像GPU登録・参照一致までの部分実行で、900秒上限・exit1のため操作全体は不合格。時間にはimport・操作・待機も含まれ、推論latencyやplugin有無の性能差を示さない。公開取得・実マイク・最終candidate全操作・性能欠測は残す。

別の[更新済み#85 `1d8d868`の新規consumer](results/fresh-native-ui-1d8d868-full-local-bundle-20261018.json)では、既存ローカルFloat16 bundleを使い、実マウス・キーボードによる画像・音声GPU登録、日英2検索の元参照vector・全順位・score、プレビュー、エラー復帰、cacheキャンセル・再試行・切替、再生と明示途中停止、解放、PlayMode停止・Editor正常終了code0まで確認した。Library・モデル・SDK overlayのコピーなしで、初回import・操作・ツール・待機を含むwallは803.890085秒。最初の再生/Stopは終端付近で旧gate未達、別の再生は残り2.688秒・8.557482秒で旧gate合格という両記録を保持する。315 package・86 sample・observer・root user4ファイルも終了後に照合済み。timeSamples警告は残っており、無警告・通常API速度・公開取得・実マイクの合格にはしない。

| 項目 | 現状 | 主な根拠 |
| --- | --- | --- |
| テキスト推論・検索 | UPM、保存・Float16重み、Text Searchを実装。固定Git consumerとWindows Playerの実モデル検索を確認 | [PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31)、[Player監査](results/m2-current-main-player-acceptance-20261017.json) |
| 画像・音声API | 実Texture / PCMからGPU前処理→encoder→共通backbone→正規化768次元。Query / Document / Rawを固定入力で検証 | [PR #79](https://github.com/ayutaz/unity-embeddinggemma-2/pull/79)、[元API完了監査](results/goal4-current-completion-audit-20261016.json) |
| 画像・音声検索サンプル | Package ManagerからImport可能。実ファイル登録、混在検索、module切替、キャンセル・復帰・解放 | [PR #83](https://github.com/ayutaz/unity-embeddinggemma-2/pull/83)、[サンプル検証](multimodal-search-sample.md) |
| モデルbundle・cache | 6 moduleの準備処理、サイズ・SHA256照合、選択moduleだけのcache・破損修復・排他lease | [サンプル検証](multimodal-search-sample.md)。空checkoutからの利用者向け自動準備は未整備 |

PR #31と#83はマージ済みです。#83の検証head `613ada824bd13a297660d8c4de4c2c14b4e317e8` は[CI 38939320058](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38939320058)の15 jobすべて成功。統合後mainと同じtreeです。これはコード統合の確認であり、正式リリースの完了ではありません。

## 確認済みの範囲

実測はそれぞれの固定sourceに対応します。異なるsource・Unity版の結果を、最新headの全条件再実行として合算しません。

| 検証 | 結果・範囲 | 固定記録 |
| --- | --- | --- |
| 元画像encoder | 6画像・3 profile、778,240値が参照とbit一致 | [画像結果](results/m3-adjacent-reciprocal-all-image-gpu-passed-20261015.json) |
| 元音声encoder | 6信号・2 profile、824,320値で元許容差超過0 | [音声結果](results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json) |
| 元APIの最終vector | 画像・音声それぞれ全6入力×3 role、各18条件合格。画像RenderTexture・復帰も確認 | [画像API](results/m3-final-image-api-all-cases-all-roles-gpu-20261016.json)、[音声API](results/m4-real-pcm-all-roles-staged-gpu-20261016.json)、[RenderTexture](results/m3-rendertexture-recovery-and-text-gpu-regression-20261016.json) |
| 保存bundleから実画像・PCM | 5 profile・12条件、最終Document vectorの最小cosine 0.9999998052055729。5プロセスすべて正常終了 | [全profile](results/multimodal-prepared-media-all-profiles-gpu-20261017.json) |
| 混在サンプル | 実音声・画像、固定4 query、GPUキャンセル→再実行→切替→解放・終了。修正後warmのホストピーク約9.49 GiB | [混在実行](results/multimodal-search-mixed-sample-cache-retirement-20261017.json) |
| 実PlayMode画面 | 固定Git `1d8d868`新規consumerで画像・音声GPU登録、日英検索の元参照、画像表示、再生・明示途中停止、エラー復帰・解放・exit0を実マウス・キーボードで確認。既存ローカルFloat16モデルの範囲 | [新規consumer監査](results/fresh-native-ui-1d8d868-full-local-bundle-20261018.json) |
| Unity全体回帰 | package + Text Search + Image and Audio Search、505 pass / fail0 / skip0、正常終了 | [全体回帰](results/multimodal-sample-canonical-regression-20261017.json) |
| Windows text Player | IL2CPP / Release / High stripping、独立cold / warm、4 query×6順位・score、FP32 GPU、正常終了。CPU計算fallback0 | [Player監査](results/m2-current-main-player-acceptance-20261017.json) |

元APIは主にUnity6000.3.16f1、最新サンプル・consumer・Playerは6000.3.21f1です。いずれもSentis2.6.1 / Windows / RTX4070 Ti SUPER / D3D12の固定実測です。Playerの合格範囲はtext検索で、画像・音声Playerの合格を含みません。

## GPUとメモリ

画像・音声の受け入れ経路はGPUComputeで実行します。中間tensorのCPU readback・計算fallbackを許可せず、入力の取得・upload、CPU tokenizer、最終768次元のCPU取得を境界として扱います。

音声キャンセル後の再実行でホストメモリが16,757,751,808 bytesに達した失敗を保持しています。GPU fence完了を待つ前処理とcache解放へ修正後、同じ混在warm手順は10,190,360,576 bytesで正常終了しました。修正前cold合格と最終vectorはbit一致しています。

全profile検証の最大ホストピークは12,090,900,480 bytes（約11.26 GiB）。これらは実測であり、VRAM必要量・全端末の最低メモリ・長期安定性・速度の保証ではありません。Float16重み保存も高速化の保証にはしません。

既定Assets indexの[明示的な再構築](editor-index-rebuild-observations.md)では、SDK `d7b2233` / validation `9b6f28e` の両sample 86ファイルを導入した既存consumerで、readyでも0文書だったcacheを40文書へ構築し、固定fixture・両sceneの直接検索を2回確認した。6 pass / skip0 / exit0。最初の再構築9.146613秒、同Editorの反復0.0232275秒で、平均性能・推論時間・cold importの値にはしない。失敗した前のprobe、managed Process memoryのraw 0を利用不能とした記録、背景専有allocationの欠測も保持する。

## 未完了・未検証

ロード・解放は[同じSDKの12実行と固定sourceを照合](load-release-measurement-scope.md)しました。ロード中に大きな定数・独自operationのGPU転送があるためCPUだけの仕事とは扱わず、未取得のGPU時間はnullを維持します。4回の推論完了後のcloseは、処理中キャンセルやdriverのVRAM返却完了の時間を示しません。この範囲監査でGPU時間の取得を完了扱いにはしません。

同文書の小規模な転送・解放対照実験では、SetDataの3区間はGPU値・データ一致を確認しましたが、DisposeのGPU値は711描画frameでも欠測で、全体は0 pass / 1 fail / skip0 / exit2でした。先行の記録保存失敗と独立記録欠落のRedも保持し、実装後の試行名 `green` や演算controlの成功を、解放計時・実モデルの受け入れ成功とは扱いません。

[Windowsのraw Timer100Nsによる別観測](windows-gpu-engine-observations.md)も追加しました。収集protocolを含む21件が成功し、実textを2プロセス・計8 vectorの参照一致と正常終了へ対応付けています。修正後のload内3観測点・14 engineで観測点間の差分0を確認しましたが、load全体の境界は覆えておらず、約10msのreleaseには観測点がありません。SDK専有のGPU時間とは扱わず、欠測は維持します。

[Editor再構築のframe全体GC](editor-frame-allocation-observations.md)は、EditModeの既知8MiB校正と実indexの2反復を7 pass / skip0 / exit0まで確認しました。40文書と両sample sceneを保持し、累積24,371,666 / 274,417 bytesとnative processメモリを記録しています。index専有bytesやGPU/実モデルの合格ではなく、Profiler不整合・compile未実行等の先行失敗も保持します。

### 進行中PRの確認済み範囲

- [#85](https://github.com/ayutaz/unity-embeddinggemma-2/pull/85): 画像プレビュー・音声再生・明示的なマイク操作の実装。固定Git consumerで画像・音声登録、日英検索、エラー復帰・キャンセル再試行・切替・解放と正常終了を実際のマウス・キーボードで確認しました。実マイクは未実行です。

途中停止の追加操作では、既存consumerのwarm起動から固定11.2秒の合成音声をGPU登録して再生し、7.232秒の位置から停止状態へ移行しました。登録内容の解放とEditor正常終了code0も確認しました。[途中停止監査](results/native-audio-early-stop-e28a5fd-20261018.json)。新しい空consumerの再検証や聴感品質の評価には数えません。

非録音時のpauseによるstatus上書きは、PR #85の追加差分 `99cc826` で修正しました。録音中のキャンセルとpause時の再生停止は維持し、意図したstatus不一致1件のRedから関連9件Green・skip0・exit0を確認しました。これはモデルを使わないlocal-file consumerの回帰であり、新sourceの実GPU・固定Git操作再検証とは別です。

固定Git `e28a5fd` を新規consumerへ解決し、SDK311ファイルとsample86ファイルを照合した実画面操作では、Float16重みの画像1件・合成音声1件をGPU登録し、日英2 queryの順位とscoreを独立Python基準へ照合しました。プレビューを目視し、AudioSourceの再生位置の進行、登録・プレビュー・モデルの解放、PlayMode終了とEditor正常終了code0を確認しました。[固定操作監査](results/fresh-git-native-ui-e28a5fd-20261018.json)。既存のローカル準備済みbundleを使用しており、公開入力からの空cache取得や実マイクの証拠ではありません。再生停止操作は自然終了付近だったため早期停止の証明を残しません。また、非録音時のpauseによる誤解を招くstatus上書きとEditor終了時のJobTempAlloc警告を保持し、正常終了だけで警告の解消とは扱いません。
- [#86](https://github.com/ayutaz/unity-embeddinggemma-2/pull/86): 固定bundleの取得・監査・再試行・修復。公開取得元から全6 moduleを空cacheへ取得する実行証拠は未完了です。
- [#87](https://github.com/ayutaz/unity-embeddinggemma-2/pull/87): Float16重みの全6 module・固定16ケースは、単独Editorの実GPUで参照比較・正常終了を確認しました。[固定結果](results/all-six-module-float16-e28a5fd-acceptance-20261018.json)。同一プロセスの連続実行、検索順位、性能、実UIの合格を含みません。

SDK `e28a5fd`のFP32連続試験は、12GiB guardでの中断を保持しています。16GiBの明示割り当てによる別試験も、画像3件と音声1件の登録後、2種類目の音声処理中に既存900秒のfixture上限を超えました。後続テキストfixtureは合格しましたが、全体は1 pass / 1 fail / skip0 / exit2です。[失敗記録](results/combined-fp32-e28a5fd-16gib-timeout-20261018.json)。上限変更や過去sourceの成功をもって、安定性の改善とは扱いません。

同じSDK・16GiB割り当てのPlayMode試験では、FP32とFloat16重みの両方で、画像・音声5件の登録、日英4 queryの全順位・score、GPU-stageでのキャンセルと再試行、Worker・kernel・tensor・cache leaseの解放、続く直接テキスト4件を確認しました。各2 pass / fail0 / skip0 / exit0で、ホスト生涯ピーク最大はFP32が14,559,334,400、Float16が11,157,610,496 bytesです。[対応する入力・順位・sourceの照合](results/fp32-float16-e28a5fd-playmode-correspondence-20261018.json)。これはlocal-file consumerを自動操作した結果で、固定Gitの手動操作や速度改善、最低RAMの証拠ではありません。[4画面の目視確認](results/fp32-e28a5fd-rendered-results-visual-inspection-20261018.json)でも、実際の音声再生とマイク確認は未実行です。

同一テキストの初期観測では、両形式で各4回の実GPU実行・参照一致・解放・exit0を確認しました。[初期観測と限界](results/text-fp32-float16-e28a5fd-measurements-20261018.json)。RecorderとProfiler履歴のGPU値の差、および割り当てAPIの全0を保持します。短い各呼び出し内のGPUメモリ観測も欠測であり、性能全体の合格ではありません。後述する校正結果を、この古い実行の割り当て量へ遡って代入しません。

初回command-buffer区間の実測は、4回目のGPU履歴走査で観測上限200,000 nodeに達し、NUnit 0 pass / 1 fail / skip0 / exit2でした。[失敗監査](results/text-command-buffer-failure-review-20261018.json)。その後、停止した履歴を読む方式では4回の参照一致・解放・exit0まで完走しました。GPU時間の解釈は別途校正中です。

管理メモリの割り当ては、既知の64KiB / 128KiB配列を検出できなかった `GC.GetAllocatedBytesForCurrentThread` を利用不能とし、停止したCPU履歴の `columnGcMemory` で別途校正しました。noop / 64KiB / 128KiBは0 / 65,568 / 131,104 bytes、差分は65,536 bytesで、1 pass / fail0 / skip0 / exit0です。[校正と失敗の区別](results/managed-allocation-scoped-byte-calibration-20261018.json)。FP32の実GPU SDK APIを同一入力で4回測定し、初回547,694 bytes、warm27,224 bytes、続く2回各26,036 bytesを記録しました。4 vectorの独立参照一致・解放・exit0も確認しています。[FP32 APIの割り当て監査](results/text-performance-fp32-e28a5fd-calibrated-cpu-allocation-allocation-audit-20261018.json)。この値は正確なCPU marker内のメインスレッド管理メモリであり、他スレッド・native / GPUメモリ・非同期stage全体の値ではありません。

Float16重みの同じSDK APIでも、初回610,830 bytes、warm26,100 bytes、続く2回各26,036 bytesを観測し、4 vectorの独立参照一致・解放・exit0を確認しました。[Float16 APIの割り当て監査](results/text-performance-float16-weights-e28a5fd-calibrated-cpu-allocation-allocation-audit-20261018.json)。これは個々の固定呼び出しの観測であり、初回条件の違いから精度形式による速度・割り当て改善を推定しません。その後、画像3・音声2 profile×両形式の全40回で、stage別CPU/wall・参照一致・解放・exit0を確認しました。[反復測定とSearchの観測](media-repeat-measurements.md)。stage別GPU時間の校正、非同期割り当て、必要なEditor index再構築・cold import / allocationの受け入れは残っています。

FP32のGPU区間の対照実験では、送信前の250ms CPU待機を加えた1回のdispatchでもRecorderは14.225 / 23.614msで、2回のdispatchでは41.850msでした。4回とも参照一致・解放・exit0を確認しました。[対照実験の固定記録](results/controlled-text-gpu-scope-fp32-20261018.json)。送信前のCPU待機とGPU仕事量への応答を区別できた観測ですが、事前転送済みWorkerの4観測に限ります。SDK API全体の遅延、純粋なkernel稼働時間、平均性能や他stageの校正へ拡張せず、異なる値を示すProfiler履歴の解釈も未確定のままです。

SDK `e28a5fd` と計測 `7ecec1f` をGitから固定書き出しした別consumerでも、実行後に534 sourceファイルを再照合しました。[FP32対照実験](results/controlled-text-gpu-scope-fp32-frozen-calibration-framework-repaired-20261018.json) / [Float16対照実験](results/controlled-text-gpu-scope-float16-weights-frozen-calibration-framework-repaired-20261018.json)は各4 vector・解放・1 pass / skip0 / exit0です。両形式で送信前250msの待機よりRecorderの区間が短い観測を得ましたが、Float16の初回単発値は2 dispatchの値より大きく、「2回がすべての単発より長い」という対照条件は不成立のまま保持します。少数の初回・warmを混ぜて平均性能や速度改善を主張しません。

この固定consumerの最初の起動はTest Frameworkの型解決エラーでsafe modeに入り、テスト未実行でした。正常終了code0だけを合格にせず、公式1.6.0の1168ファイルをbyte照合してembedded packageにした後、両形式が実行できました。検証用projectだけAcceleratorを無効化し、既存のuser設定は保持しています。[環境復旧と重なりのあるimport / reload観測](results/frozen-measurement-environment-recovery-20261018.json)。この復旧時点の実行ではsearch DBの所要時間は未測定でした。別のモデルなしconsumerでは既定indexの初期化・準備待ちとpackage検索4回・exit0を確認しましたが、ファイルパス検索も含むためindex単独・再構築・allocationの受け入れではありません。

固定FP32の旧計測では、Profiler履歴のms→ns変換から負のlong値も検出しました。元のfloat値は未保存で原因未確定です。非数・非正値・long範囲外・丸め後0を履歴の有効時間にしないguardを2件Red→関連10件Greenで追加しました。[変換のTDD](results/gpu-history-value-bounds-tdd-20261018.json)。これはモデルなしのguard検証であり、GPU履歴の解釈やstage別性能の合格には数えません。旧snapshotの不正な履歴値は未利用可能の観測として保持します。

固定sourceの追加試行では、両形式で250msの送信前CPU待機をGPU Recorder区間に含まず、2 dispatchが単発値より長い対照結果を得ました。[追加の制御実験](results/frozen-source-gpu-dispatch-scope-calibration-20261018.json)。上記の別試行で対照条件が不成立だった事実は保持します。待機なし・各1 dispatchの初回 / warm / 2反復も、両形式で4 vectorの参照一致・解放・1 pass / fail0 / skip0 / exit0を確認し、実行後の534 sourceファイルを再照合しました。[通常dispatchの観測](results/frozen-source-text-gpu-dispatch-observations-20261018.json)。入力転送・tokenizerは測定外で、CPU完了時間にはcommand構築・GPU fenceのframe待ち・最終readbackを含みます。SDK API全体の遅延、純粋なkernel稼働時間、平均速度や形式間の高速化は主張しません。Profiler履歴の別値は未確定のままです。

### 新SDKのGPU区間とEditor割り当て

SDK `d7b2233` / 計測 `10962fa` の[全5 profile・両形式の40反復](media-gpu-submission-observations.md)では、元参照・解放・正常終了と、encoder 17,176区間、前処理/backbone 80区間すべてを確認しました。CPU/wall・ロード/解放・ホストメモリを別記し、9プロセスで実際のcounter採取時刻によるGPU memoryも照合しました。FP32 squareのcounter traceと短い区間の欠測は保持します。計測付きAPI・共有cacheを残した固定入力の観測で、通常APIの平均速度・形式間の高速化・最低VRAMではありません。

別のモデルなしconsumerでは、[Editor検索の校正済み割り当てと空/再利用Library](editor-search-allocation-observations.md)を確認しました。noop / 64KiB / 128KiBは両実行で0 / 65,568 / 131,104 bytes、既知SDKファイルの検索4回、各10 pass / fail0 / skip0 / exit0、終了後558 sourceファイル一致です。main-thread marker内のmanaged bytesであり、背景index・native/GPUメモリ・非同期完了の割り当てではありません。検証専用consumerの空Library起動は実import 2658件、再利用起動は0件で、起動/import/reloadの時間は重なるため加算しません。

### 同SDKのtextとフレーム単位GCのcontrol

同SDKの[text両形式の固定query各4反復](text-current-sdk-observations.md)も、独立参照・停止CPU履歴のmanaged bytes・解放・各1 pass / skip0 / exit0まで確認しました。GPU Recorderと履歴の値の差は残っており、cross-method interpretationを合格にせず、純粋なkernel時間とは扱いません。

モデルなしの[frame GC control](async-frame-allocation-observations.md)は、既知の0 / 4MiB / 8MiBを実際のcounter更新に分けて取得し、10 pass / skip0 / exit0でした。描画frameだけを待った旧controlの9 pass / 1 fail / exit2を保持し、元の上限を変えていません。フレーム全体の値は無関係のEditor処理・observerも含み、実モデルの非同期allocationやSDKだけのbytesの合格ではありません。

続く[FP32音声の代表実行](results/async-real-audio-frame-allocation-9bed619-20261018.json)では、元のGPU検証4 vector・解放の後に割り当て記録の欠落で失敗するRedを確認し、接続後は1 pass / skip0 / exit0でした。counter sample 2052個の合計は6,984,175,075 bytesです。入力準備・ロード・検証用I/O・解放・含まれるframeのEditor処理を含む累積割り当てで、SDKの個々のstageのexclusive bytesやホストピークではありません。元のasync呼び出し本体・参照・GPU要件は維持しました。

### 残る受け入れ

- 空のcheckoutから画像・音声bundleを準備する公開手順と配布経路。
- 公開手順で取得したモデルと最終統合candidateを新しいconsumerへ導入する最終受け入れ。既存ローカルFloat16モデルと固定Git `1d8d868`の画像・音声・日英検索・解放・正常終了の一連操作は上記で確認済み。
- 結果画像プレビュー・音声再生・明示的なマイク操作のPR統合と実操作の受け入れ。
- load/release GPU区間の扱い、代表的な非同期割り当て区間と個別stageの測定範囲・欠測の最終監査、明示的index再構築時の専有allocation・背景処理。
- 画像・音声検索のWindows Player、macOS Editor、iOS / Android実機での精度・操作・cache・正常終了。Androidの過去のAPK build成功は実機GPU合格ではありません。
- 全モダリティの採用Float16形式・明示CPU経路の環境別受け入れ、最終candidate / tag固定導入、正式Release。

過去のCPU対ONNX診断失敗、旧同期音声のメモリ上限停止、未実行・skipは成功に置き換えません。[サンプル検証](multimodal-search-sample.md)と[過去の設計・検証資料](README.md)に保持しています。次の順序・依存・公開条件は[ロードマップ](roadmap.md)を参照してください。
