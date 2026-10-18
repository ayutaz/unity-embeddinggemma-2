# ロード・解放時間の測定範囲

2026-10-18。SDK `d7b2233` / validation `10962fa` の既存実GPU記録を再監査した。[画像・音声10プロセス](media-gpu-submission-observations.md)と[text両形式2プロセス](text-current-sdk-observations.md)は、ロード・解放のCPU process時間とwallを保存している。[対応監査](results/load-release-measurement-scope-d7b2233-20261018.json)では12件の元receipt・raw evidenceのSHA256、終了code0・NUnit各1 pass / fail0 / skip0、固定554 sourceファイルを照合した。新しいモデル実行やGPU時間の取得ではない。

## ロードに含まれる仕事

`GpuMediaBundle.LoadAsync` はファイル検証・CPU上のモデルと係数読み込み・tokenizer構築・Worker構築を行い、その間の `Task.Yield` でEditorの処理を挟む。wallにはこの待機が入り、process CPUには他のEditor threadの仕事も入る。

ロードを「CPUだけの処理」とは分類できない。`GpuConstantWorker` は大きなGPU用定数を8MiB以下のraw-word単位で `ComputeTensorData` のbufferへ `SetData` する。画像・音声encoderの各segmentとtext/backboneにも使われる。独自operationの係数も `GpuBlockedMatmul` / `GpuSequentialScale` の `SetData`、`GpuFixedLeftMatmul` の `ComputeTensorData.Pin` によってGPUへ渡す。監査JSONには固定Gitのファイルhashと該当行を記録した。

一方、chunk以下の定数はこの事前転送を通らず、CPUで必要な定数も残す。ロード完了は全定数のGPU転送完了や最初の推論と同じ条件を意味しない。FP32 / Float16重みとも、ロードと初回推論を分けて記録する。CPU上の準備処理があることを、モデル推論のCPU fallbackとは扱わない。

## 解放の測定範囲

`GpuMediaSession.CloseAsync` は終了要求を出し、実行中の呼び出しがあればその終了を待つ。resourceは逆順にDisposeする。今回の12件は固定入力を4回完了した後のcloseであり、処理中キャンセルの遅延を測ったものではない。元テストでWorker・kernel・tensorの解放を確認していても、closeのreturn時刻はdriverによるVRAM返却時刻や全GPU queueの終了時刻の測定ではない。

## GPU時間は欠測のまま保持する

mediaのGPU observerはロード後にattachし、推論の送信区間を測る。textはロード中と各queryの履歴保存後にprofilerを無効化している。どちらもロード・解放を囲むGPU markerを取得していないため、12件すべてのload/release GPU時間をnullにした。GPU仕事が0だったという意味ではなく、CPU/wall・VRAM採取点から時間を推定しない。

次のGPU測定には、実際のuploadとresource退役の区間を識別する校正が必要である。`Task.Yield` をまたぐ全ロードを単一の同期markerで囲んだり、測定の都合でSDKのawait・fence・呼び出し順を変えたりして同じ動作の計測とはしない。この監査で範囲と欠測理由は確定したが、load/release GPU時間および性能全体の受け入れは未完了である。

## 小規模な転送・解放markerの対照実験

SDKを変更せず、validation `2d221e6` の `GpuTransferTimingProbeTests` で独立したbufferを用意し、同期 `SetData` / `Dispose` をGPU有効なCustomSamplerで囲んだ。[3回の実行と固定sourceの監査](results/gpu-transfer-retirement-timing-probe-2d221e6-20261018.json)。実モデルは読み込んでいない。

| 観測区間 | GPU Recorder | CPU呼び出しwall |
| --- | ---: | ---: |
| 8MiBを1回SetData | 3.785728ms | 6.4593ms |
| 別々のbufferへ8MiBを2回SetData | 5.668608ms | 12.6443ms |
| markerの前に250msのCPU待機、その後8MiBを1回SetData | 2.546944ms | 3.0990ms |
| 計16MiBのbufferを2個Dispose | null | 0.0490ms |

GPU演算の独立controlはGPUCompute・期待値一致・18,432nsだった。各bufferの全要素を測定markerの外で照合した。Disposeでは711個の異なる描画frameを観測してもGPU値を取得できなかった。CPU上でDisposeが戻った時間を、driverの解放完了やGPU仕事0へ置き換えない。

テスト全体は0 pass / 1 fail / skip0 / exit2で、元のGPU正数条件を維持した。実行ラベルの `green` は実装後の試行名であり、成功ではない。転送3件の正数・データ一致だけで、解放計時・copy engineへの時間帰属・実モデルのload/releaseを合格にしない。表は各1観測で、平均帯域やCPU待機を引いたkernel時間でもない。

先行 `88b6536` はPlayMode domain reload前に初期化した保存先が失われ、ArgumentNullExceptionで失敗した。記録の初期化を切替後へ移した `c95e620` ではGPU controlを確認した後、独立した転送・解放の記録欠落で意図したRedを確認した。この2件も各exit2で保持する。新しいmarkerの欠測を解消するために既存の成功したdispatch計時を代入しない。

Windowsには `GPU Engine / Running Time` counterも存在したが、ここでは実モデルの値は取得していない。別方式を用いる場合は、PID・adapter/engine・counter時刻・単位を校正し、複数engineの重なりとEditor自身のGPU仕事を区別する必要がある。

その後の[Windows raw Timer100Nsの別観測](windows-gpu-engine-observations.md)では、同じSDKの実text4反復を2回、元参照・解放・正常終了と対応付けた。修正後のload内3観測点・14 engineの差分は、load全体16.308541秒のうち観測点間10.620599秒で0だった。境界のGPU転送や、観測点のない10.1547msのreleaseへ拡張しない。OSのprocess/engine活動量であってSDK専有のload/release GPU時間ではなく、欠測と元のproviderエラーを保持する。

後続の[native PDH追試](results/native-pdh-text-stage-observation-f604baf-20261018.json)は、同じSDK/validationのFP32 text4 vector、参照・解放・1 pass / skip0 / exit0と、8,783回の所有process counterを対応付けた。取得間隔の中央値4.6489msで、load内に2,687点、各query内に326 / 18 / 18 / 18点、release内に1点を採用した。loadの3Dに1.489ms、firstのCopyに2.5074msという内側区間の正数があったが、queryの3Dは隣接差分が経過時間を超え、releaseはpair不足でnullだった。1msのclock整合性幅を含むcall全体が区間内に収まる点だけを用い、itemの実時刻をquery時刻へ置換していない。

これはSDK専有load/release時間の取得にはならない。細かいsampleを増やすだけでは解決しないprovider側の値と帰属の制約を保持し、同じPDH手法の反復で全体合格に置き換えない。clock診断・TDD・CPU観測・source対応と、未確定の原因は[Windows観測](windows-gpu-engine-observations.md)へ分けて記録する。
