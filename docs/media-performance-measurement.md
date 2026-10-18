# モデル性能・GPU障害の測定手順（作業中）

この工程は未完了です。SDK d1548acの全6 moduleで、画像6件・音声6件・テキスト4件の実GPU最終vectorを固定参照と比較し、数値基準・解放・終了code0を確認しました。インポート済みサンプルの登録5件・日英検索4件の固定順位とscore許容差、GPU段階でのキャンセル・再試行・切替・解放も通過しました。新規Git環境の実操作、公開取得、Float16、性能合格・速度改善・最低VRAMは未確定です。

[全6 moduleの検証一覧](results/all-six-module-d1548ac-acceptance-20261018.json)と[実サンプルの統合検証](results/multimodal-sample-d1548ac-20261018.json)にscope・固定参照・失敗記録を分けています。横長画像の初回はsupervisor/Unityが途中終了しXMLなしとなったため合格に含めません。その後、既存WindowsTerminal PID7788の終了と別PIDでの起動、GPU空き約14GiBを観測して再実行しました。他アプリをagentが終了した事実はなく、終了原因は断定しません。正方形画像の約2GiB空き条件と後続の約14GiB空き条件を、同一条件の性能baselineへ合算しません。

## 現在取得できた観測

- Unity 6000.3.21f1 / Sentis 2.6.1 / RTX 4070 Ti SUPER / D3D12で、小さな計算のGPU marker時間を取得しました。通常EditorのPlayModeでProfiler GPU収集を有効にした場合に成功し、同環境のbatchmodeでは有効な値を取得できませんでした。GPU対応フラグだけで計時成功とは判定しません。
- 既存GPU呼び出しをGPU収集付きCustomSamplerで囲む経路も小さな計算で確認しました。これはmarkerのGPU timestamp区間であり、queueの待ちや区間内の他のGPU処理を含み得ます。command buffer内のkernel markerと同じ範囲として比較しません。
- WindowsのGPU Process Memory counterからプロセス別dedicated/shared割り当てを取得しました。これはサンプリングした割り当て量で、GPUの物理容量・連続した最大値・GPU実行時間ではありません。
- Unity終了中にも他アプリに大きなGPU割り当てが観測されました。過去のGPU障害への因果関係は未確定です。大きなモデルを再実行する前に、他プロセスの負荷と空きメモリを再確認します。他アプリは自動的に終了しません。

小さな計算の合格を実モデルの合格へ流用しません。修正前・修正後の実モデル失敗は[失敗監査](results/multimodal-usability-gpu-failure-and-upload-fix-20261017.json)で確認できます（この文書の依存先PR #85に記録）。

## 固定する条件

SDK / validationのcommit、Unity・Sentis・driver・graphics API・device、model producerとdescriptor/object SHA256、入力と参照SHA256、module/profile/role、実行順、cache状態を記録します。異なるsourceの結果を一つの最終candidateの実測として合算しません。

最初の起動は「新しいプロセス・GPU session・利用者cache」を条件とし、OSのdisk cacheを消した状態とは区別します。warmは同じsessionでの反復です。OS cacheや他アプリの設定を無断で変更しません。

## 測定表

| 対象 | 記録するもの | 混同しないもの |
| --- | --- | --- |
| 取得・照合・load | ファイル取得、全hash照合、モデル読込、Worker作成、GPU転送のwall / process CPU時間 | 推論時間やcold disk速度とは別 |
| 前処理・encoder・backbone | CPU submission、GPU marker時間とmarker名・範囲・sample数、最終readbackと待ち | CPU stopwatchをGPU時間として代用しない |
| 初回・warm反復 | 各試行の生値、実行順・入力・精度、失敗・中断 | 初回をwarmの中央値へ混ぜない |
| ホストメモリ | Windows native current/peak working set、測定時刻、Unity allocator | process全体のpeakを特定stageのpeakにしない |
| GPUメモリ | PID・adapter別dedicated/shared、サンプル間隔・欠測、GPU全体の負荷 | 容量を使用量にしない。欠測はnull |
| 解放・終了 | GPU fence後のClose、資源所有権、数frame後のcurrent値、正常終了code | editor共有cacheやglobal peakがゼロになるとは要求しない |
| Editor | import、domain reload、検索DB処理、allocationと対象marker | reloadの入れ子時間を足さない。例外0を処理時間0にしない |

## GPU計時の確認

`GpuTimerProbeTests` は明示的な診断用です。モデルを読み込まず、小さなGPU計算 `[1,2]+1=[2,3]` を実行します。`EG_GPU_TIMER_REPORT` で未使用の出力ファイルprefixを指定し、通常Editor・D3D12で実行します。各テストは `-command-buffer.json` / `-opaque.json` に別々に保存します。GPU profiler設定は終了時に戻し、生成camera / RenderTexture / Worker / Tensorを解放します。

`Recorder.gpuElapsedNanoseconds` はframe単位で遅れて到着するため、GPU sample数とともに排出期間を記録します。正のGPU時間が来ない場合は計時不能です。fence通過やCPU submission時間をGPU計時の成功に置き換えません。

GPUの内訳を読む場合は `-history.json` も出力します。Recorderの有効化だけではEditorの履歴は作られないため、このEditor自身のlocal targetを確認してProfilerDriverの収集も有効にします。EditMode test coroutineはPlayModeへ入った後もEditorLoopから送信するため、履歴収集ではEditorLoopも含めます。終了時には接続先と設定を戻します。

`GpuProfilerHistory` は最近25 frameから指定markerだけを探し、全threadを調べてGPU列を読みます。CPU列へ代替せず、正のGPU時間を持つ直下の子markerも区別して記録します。値はProfilerの浮動小数点msからnsへの単位変換であり、ns単位の精度を保証しません。親と子は時間が重なるため加算できず、この限定windowを長いモデル実行全体の測定として使うこともできません。

`GpuOperationTrace` は明示した診断observerへ、演算番号・種類・tensor shape/backend・CPU submission区間・例外型を送るための補助です。値・画像・波形・embeddingは読み出しません。元の演算とtensorの所有権を変えず、`submitted` はGPU完了を意味しません。実モデルへ接続した診断は、最後にGPU fenceを通過した区間と、最後に送信した演算を区別して記録します。

## 前処理・encoder・backboneのCPU観測

`GpuMediaBundleExecutionTests` の画像・音声検証に `EG_MEDIA_STAGE_TIMING=1` を指定すると、既存の最終vector・数値比較を維持して、reportの `stage_measurements` に各段階のbeginとcompleted/failed/cancelledを追記します。固定fixtureとmoduleを選ぶ既存の環境変数を併用します。対象は既にloadしたsessionの前処理・encoder・backboneで、モデルload・text・解放の内訳はまだ対象外です。

`MediaStageTiming` はvalidation側だけでstage interfaceを装飾します。非同期encoderは返されたTaskの完了まで待ち、同じ入力・token・結果を引き渡します。元の例外とキャンセルを維持し、stageやtensorを所有・破棄せず、tensor値の追加readbackもしません。測定終了時に元のstage参照へ戻し、既に閉じたfacadeの参照は復活させません。

`wall_seconds` はGPU待ち・frame待ちを含む区間です。`process_cpu_seconds` はその区間のEditor全体のCPU時間で、当該stageだけのCPU時間ではありません。ホストのworking setは終了時の値、peakはprocess生涯の値、Unity allocatedは現在量です。allocation rateやstage別peakへ読み替えません。`gpu_elapsed_ns` はnullのまま保持し、GPU時間は別のProfiler計測を使います。計測・report保存の負荷を含む実行なので、計測なしの性能値と混在させません。

生成fixtureによる8件の契約テストでは、同期待機・非同期待機・例外・キャンセル・呼び出しの分離と、実ImageEmbedder/AudioEmbedder facadeへの接続・復元を確認しました。実モデルでこの経路を実行した結果はまだなく、数値・GPU・性能の受け入れには数えません。

## 実モデルの停止区間と行単位dispatch

旧SDK ca7fd23ではbackboneを省いた実画像の区間診断でも、stage4のGPU fence通過後、stage5のモデル・出力コピー・softmaxを含む区間でD3D12 `887a0006` が再現しました。入力は `(1,12,2520,2520)`、約7,620万要素です。元のworker/backend cache解放を適用しても再現し、旧full実行と同時刻のWindows Display event4101も保存しました。特定のshaderだけが原因、またはメモリ不足が唯一の原因とは断定しません。

SDK d1548acではsoftmaxを最大1024行ずつに分割します。既存の8/16レーンkernelと指数関数は維持し、新kernelの各行の算術は整数の行offset以外同一です。小さな実GPUテストは5件のAPI/bound未実装Redから、従来契約を含む9件Green。旧kernelとの出力bit一致、最終の端数tile、反復・入力所有権を確認しました。

同じ実画像の先頭8段は修正後にGPUComputeで完了し、終了code0を確認しました。初回の修正後試行はstage5 GPU完了後に診断用4GiB host guardでキャンセルされ、合格に数えません。空き物理RAM約28GiBを確認して診断上限8GiBとした試行のhost lifetime peakは4,549,300,224 bytesでした。これは全モデルの最低RAMやGPU VRAMではありません。元のパイプライン全体や全6 moduleの合格へ流用しません。

監査は [区間診断](results/isolated-original-gpu-prefix-20261017.json) と、依存PR #85の [dispatch修正](results/bounded-gpu-softmax-dispatch-20261017.json) に記録します。再利用projectの旧同名fixtureの二重実行と3分timeoutは失敗として保持しました。その後、公式Test Frameworkの `-assemblyNames EmbeddingGemma.Validation.Editor.Tests` と8分のfixture timeoutを明示し、正方形画像2件の最終768次元・cosine >=0.999・正規化・session解放を確認しました。NUnit1 pass / fail0 / skip0、process exit0、全体349.26秒、host lifetime peak11,757,166,592 bytesです。保存したfloat32 vectorは小さなPython検証でも独立に再比較しています。[正方形画像の全体検証](results/image-square-bounded-softmax-acceptance-20261017.json)を参照してください。これは同じ入力の反復や全6 moduleの性能合格ではありません。

## 残る受け入れ

正方形画像の全体実行・終了確認から他のprofileと音声・テキストへ広げ、同一candidateで全6 module、採用するFloat16形式、GPU単独と連続実行を測定し、上表のload・初回・反復・各stage・解放・Editor負荷を埋めます。未採用・未検証のFloat16形式は未実行として残し、FP32の結果で代替しません。

FP32の有限768次元・単位長・参照cosine >= 0.999、採用するFloat16の >= 0.99、固定横断検索の全順位一致と元score許容差を維持します。GPU実行・操作・メモリ・終了の結果と、CPU比較・CI・小さなfixtureの結果を別々に判定します。
