# 非同期区間のフレーム単位GC割り当てを校正する

2026-10-18。Unity6000.3.21f1 / validation `9e79167` のモデルなしPlayMode control。Unityの[GC Allocated In Frame](https://docs.unity3d.com/6000.3/Documentation/Manual/ProfilerMemory.html)と[ProfilerRecorder](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Unity.Profiling.ProfilerRecorder.LastValue.html)を使い、実際のcounter更新をまたいで既知サイズの配列を2回確保する。SDKやSynchronizationContextの置き換えは行わない。

[独立監査](results/async-frame-counter-aligned-calibration-9e79167-20261018.json)は10 pass / fail0 / skip0 / exit0、終了後570 sourceファイル一致。集計の契約テスト9件を含み、standaloneでは9件Red→9件Greenだった。

| 既知のpayload合計 | counterで記録したmanaged bytes | 含めたcounter samples |
| --- | ---: | ---: |
| 0 | 83,845 | 2 |
| 4MiB | 4,264,309 | 2 |
| 8MiB | 8,459,089 | 2 |

既知配列を囲む2個のmain-thread CPU markerも停止した履歴で独立に照合した。4MiBと8MiBでは両方の大きなpayloadが別々のcounter sampleに現れ、8MiB−4MiBの差は4,194,780 bytesだった。事前に決めた各windowのpayload+1MiBの背景・observer上限、差分3〜5MiBの条件を維持した。noopにも83,845 bytesがあるため、このcounterはSDKだけの割り当てではない。

windowはoperation完了時点のsample数を記録し、その後最初の新しいcounter sampleまでを含める。後から余分に採取したsampleは集計から除外する。欠測・負数・overflow・満杯/wrapしたbufferを拒否し、同じ値が続く別frameを重複として消さない。含まれるframeには無関係のEditor処理やobserverの割り当てもあり、重なるwindow同士を加算しない。

最初のcontrol `664ab20` は描画frameだけの進行を待ったため、4MiBの2回の確保が1個のcounter sampleへ集まり、9 pass / 1 fail / skip0 / exit2だった。[元の失敗](results/async-frame-render-counter-boundary-failure-20261018.json)を保持する。修正は、校正用の2回目の確保を実際のcounter更新後まで待つもの。2個の大きなsample・元の上限・同じSynchronizationContextの条件は緩めていない。

このcontrol自体は実モデルやGPU互換性の代用ではない。

## 代表実音声の検証区間へ接続

既存の実音声検証に `EG_ASYNC_FRAME_ALLOCATION=1` のopt-inを追加した。最初に必要な記録の存在を要求し、SDK `d7b2233` / validation `d1df229` のFP32 / audio-1119で4 vectorの参照一致と解放の後、記録欠落による0 pass / 1 fail / skip0 / exit2を観測した。実装後の `9bed619` は同じ元の入力・参照・GPU・メモリ上限・終了条件で1 pass / fail0 / skip0 / exit0だった。[独立したRed/Green監査](results/async-real-audio-frame-allocation-9bed619-20261018.json)。両試行の計8 vectorを独立に再計算し、終了後570 sourceファイルも照合した。

Greenのwindowはcounter sample 2052個、managed bytes合計6,984,175,075、ホスト生涯ピーク9,465,491,456 bytesだった。累積割り当てとピークを混同しない。SDKへの元のasync呼び出しを含む `Run` 本体はRed/Greenで同一で、追加したcaptureは検証driverの外側に置いた。SDKの演算・await・GPU fence・cache解放は変更していない。

windowには入力準備・ロード・4回のGPU呼び出し・既存のstage observer・参照照合・JSON/binary I/O・解放・含まれるframeのEditor処理が入る。個々のSDK stageのexclusive bytesや全profile・全precisionの割り当て量ではない。モデルなしcontrolと、代表実モデルの広い区間の結果を分け、測定範囲を最終監査で保持する。
