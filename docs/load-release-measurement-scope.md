# ロード・解放時間の測定範囲

## Native QPCによるstage対応の追加確認

[FP32 textの固定入力4回](results/raw-qpc-text-fp32-raw-qpc-warm-20261018.json)では、検証コードだけにWindows QPCの開始・終了値を追加し、同じSDK/model/inputの元参照・解放・exit0を維持した。Unity側20契約とPython集計33契約はRed→Greenを確認した。Mono Stopwatchのepochを生QPCと同じとする初期案は実機で失敗したため採用せず、Native QPCを直接記録する。最初の描画あり起動は180秒でPlay Mode再読み込み中にtimeoutし、モデルreportも欠落した失敗として保持する。別の同じ上限・再利用Libraryの起動では全4 vectorとcollectorが正常終了し、loss0でload・first/warm/repeat・releaseの6区間へ対応付けた。

loadの対象processのGPU workは101件、同一queue内の重複を除いた観測区間は9,715,676nsだった。release窓内に該当GPU workは観測されなかった。これはSDK専有load/release時間や物理VRAM返却を表すものではない。各queryのGPU区間もqueue別のunionとして保持し、engineを加算しない。CPU/wall・Unity Recorder/履歴・host peakは元の別項目を維持する。公開版のpath欄は伏せ、raw evidenceのhashは非公開の原本を指す。固定Git導入pinは履歴整理後のsourceで再確認する。

続く[Float16重みのtext4回](results/raw-qpc-text-float16-weights-raw-qpc-20261018.json)も同じSDK・計測コード、元参照・同じ許容差、GPUCompute、解放・exit0、collector loss0で6区間の照合に成功した。load内91件の同queue unionは5,565,136ns、release内GPU workは0件だった。両形式とも初回と3回の後続queryは正数のGPU workを観測している。別processでの観測値であり、precision間の速度比較や全演算FP16の証明にはしない。

後続の[実text・raw DirectX観測](results/raw-dxg-real-text-observer-fixes-20261018.json)では、SDK `d7b2233` / validation `10962fa`を変えず、初回・warm・反復の4 vectorと正常終了を独立確認した。最終collectorは39,362件の`OnGpuWork`を含む45,085行、loss0・診断0・不正/破棄記録0で終了した。初回の依存decoderのPSO統計assert、次の12件lossは失敗のまま保持し、不要な統計keywordとbuffer設定をTDDで修正した。48 native unit・9 CLIが成功し、upstreamは未変更。外部QPC/precise UTCの観測はあるが、既存Mono `DateTime.UtcNow`のstage境界の誤差は校正していないため、load/releaseのGPU時間はまだ確定しない。次はvalidation側だけにraw QPCのstage境界を追加し、元参照・SDK・model/inputを維持して対応付ける。resource破棄がない3件も保持し、driver退役・VRAM返却を証明しない。

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

## DirectXイベントによる次の測定経路

PDHと別の経路として、[ETWの開始権限とMicrosoft製decoderの対照実験](results/dxg-method-controls-72bd2b1-20261018.json)を確認した。現在のtokenで、providerなしの小さな空session、およびDxgKrnlのBase/GPUSchedulerを有効化・無効化するsessionは、全APIがreturn0で開始・停止できた。空sessionは最大256KiBを要求し、各owned handleで直ちに閉じた。イベントconsumer・ファイル・Unity・実モデルは起動しておらず、この結果からGPU時間は得られない。最初の記録のFormatError(0)がthreadの古いエラー文字列を返した点は、後続記録でnumeric return0が成功であると訂正し、元記録も保持した。

[MicrosoftのDxTimingCaptureLibrary固定commit `72bd2b1`](https://github.com/microsoft/DxTimingCaptureLibrary/blob/72bd2b18553493aa9850e434caf71bfa18952aac/README.md)を変更せずにビルドし、upstream単体130件・実D3D12対照2件はfail0 / skip0 / exit0だった。後者はdefault adapterの256-byte copyに対象PIDのpacketと順序付きsubmit/complete時刻が存在すること、および別のresource破棄callbackがCPU Releaseを囲む時刻内に入ることを検査する。終了後、所有test PIDのETW session残留は0だった。upstreamの対照は最低64 / 最大1290個の1MiB buffer設定で、空sessionの256KiB設定とは区別する。C4819警告と標準buildのPIX decoder stubも記録している。

これはSDK load/release、driver退役・VRAM返却、専有allocationの証拠ではない。破棄callbackをGPU解放完了と同一視しない。次は対象PIDのイベントだけを出力する上限付きconsumerを用意し、raw clock・単位・loss・終了を校正したうえで、同じSDK/model/inputのstageと元参照・正常終了へ対応付ける。非管理者であることだけを理由にETWを不可能と扱わず、既存PDHの欠測と別に進める。

別の[upstream Perfetto policy単体9件](results/dxg-perfetto-policy-72bd2b1-20261018.json)もfail0 / skip0 / exit0だった。ただしsample writerの `ToPft` は最初のイベントを取得時点のPerfetto SDK clockへ置き直すため、出力時刻を既存UTC/QPC stageへそのまま照合しない。libraryの `GpuEngineActivity` はdriverへのsubmitからcompleteまでの上限区間であり、GPU実行だけの時間でもない。次のconsumerでは元のepochを保持した `OnGpuWork` の区間、queue待ちを含むpacket区間、resource破棄イベントを分け、clock anchor・帰属・loss・終了を確認する。9件の成功はwriter policyの確認で、追加の実GPU captureやSDK計測ではない。

## 元のQPC時刻を保持するobserver

[実装と独立した小規模GPU対照](results/native-dxg-raw-observer-control-20261018.json)では、`tools/native/dxg-observer` に対象PIDだけを出力する上限付きCLIを追加した。記録・callback・所有sessionの終了は、21件のscaffold Red、callback8 Red、ownership6 Red、packet情報1 Redを経て最終40 pass / fail0 / skip0。CLIも9 Red→9 Greenで、既存outputの保持、欠落PIDの拒否、raw QPC/UTC anchor・target生成FILETIME・loss・closureを確認した。初期のリンク設定・entrypoint・構文のbuild失敗は、挙動のRedと別に保持している。

Microsoftの未変更256-byte D3D12 workloadを所有handleでsuspend起動し、collectorのready後にresumeした。所有PIDの `OnGpuWork` 1区間は736nsの正数で、実行前に決めた外部QPC窓±5msに収まった。256-byte resource2個の作成・解放とpacket envelope9件を別記し、target/collectorはexit0、loss・diagnostics・decode error・残留sessionは0だった。この窓は精度保証ではなく、小規模な時計epoch・帰属・終了の対照である。default adapterの対照をRTX/Sentis互換性や実モデル受け入れにはしない。

手順・上限・JSONの範囲は[observer README](../tools/native/dxg-observer/README.md)を参照。新しいWindows Actionsは40 native unit＋GPU収集なしの8 CLI preflightを実行する構成で、追加時点では未確認。localの実GPU対照と混同しない。SDK・validation・モデルは変更していない。次は同じSDK/model/inputの実測をこのobserverへ接続し、stageごとのclock対応、元参照vector・解放・正常終了を照合する。load/release GPU時間・driver退役・SDK専有allocationは引き続き未完了。
