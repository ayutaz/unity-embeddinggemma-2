# モデル性能・GPU障害の測定手順（作業中）

この工程は未完了です。実モデルの画像登録でD3D12 device error `887a0006` が再現しており、性能合格・改善効果・最低VRAMを確定していません。CPU fallbackで成功扱いにはしません。

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

`GpuOperationTrace` は明示した診断observerへ、演算番号・種類・tensor shape/backend・CPU submission区間・例外型を送るための補助です。値・画像・波形・embeddingは読み出しません。元の演算とtensorの所有権を変えず、`submitted` はGPU完了を意味しません。実モデルへ接続した診断は、最後にGPU fenceを通過した区間と、最後に送信した演算を区別して記録します。

## 残る受け入れ

まず既存の実モデルGPU障害を区間単位で特定・修正します。その後、同一candidateで全6 module、採用するFloat16形式、GPU単独と連続実行を測定し、上表のload・初回・反復・各stage・解放・Editor負荷を埋めます。未採用・未検証のFloat16形式は未実行として残し、FP32の結果で代替しません。

FP32の有限768次元・単位長・参照cosine >= 0.999、採用するFloat16の >= 0.99、固定横断検索の全順位一致と元score許容差を維持します。GPU実行・操作・メモリ・終了の結果と、CPU比較・CI・小さなfixtureの結果を別々に判定します。
