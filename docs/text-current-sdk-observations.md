# 新SDKでの同一テキストの反復測定

2026-10-18。画像・音声の40反復と同じSDK `d7b2233` / validation `10962fa` / Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12で、固定の日本語Queryをfirst / warm / 2反復した。新しいプロセス・sessionごとにFP32とFloat16重みを分け、共有OS/shader cacheとproject Libraryは保持した。

[FP32](results/text-current-fp32-d7b2233-first-20261018.json) / [Float16重み](results/text-current-float16-weights-d7b2233-first-20261018.json)は各4 vectorが独立Python参照・有限768次元・単位長に合格し、解放・各1 pass / fail0 / skip0 / exit0と終了後554 sourceファイル一致を確認した。各形式の4回は同一queryの反復であり、異なる4 queryや全roleの受け入れではない。

| 観測 | FP32 | Float16重み |
| --- | ---: | ---: |
| loadのwall | 12.850433秒 | 11.152465秒 |
| 最初のqueryのwall | 1.210824秒 | 0.996098秒 |
| 続く3 queryのwall | 0.052038–0.056351秒 | 0.056599–0.060708秒 |
| 続く3 queryのGPU Recorder値 | 19.227–25.428ms | 22.525–23.195ms |
| queryのmain-thread marker内managed bytes 初回 / warm / 反復1 / 反復2 | 547,694 / 27,224 / 26,036 / 26,036 | 610,830 / 26,100 / 26,036 / 26,036 |

停止したCPU履歴の割り当て値は、[同じUnity版の既知サイズ校正](editor-search-allocation-observations.md)で利用できたcolumnGcMemoryを使う。Monoの `GC.GetAllocatedBytesForCurrentThread` は既知64KiBを検出せず、この試行でも利用不能だった。監査中のlegacyな `managed_allocation_counter_calibrated` はprobeを実行した意味で、利用可能・校正合格の意味ではない。`managed_allocation_counter_available=false` とraw delta 0を保持し、CPU履歴の値と区別する。load/releaseの非同期managed allocationへは拡張しない。

GPU RecorderとProfiler履歴の値は一致していない。例えばFP32 warmはRecorder 19.227ms、履歴271.884msで、両者を同じkernel時間とは扱わない。別々の観測として保存し、cross-method interpretationの合格flagはfalseを維持する。CPU process時間とwall/GPU区間も重なり得るため、加減しない。少数の固定入力を平均性能・形式間の高速化へ一般化しない。

[FP32のGPUメモリ](results/text-current-fp32-d7b2233-first-memory-20261018.json) / [Float16重みのGPUメモリ](results/text-current-float16-weights-d7b2233-first-memory-20261018.json)は正確なPIDと実counter採取時刻でload/query/releaseへ対応付けた。短いqueryに採取点がなければnullを保持し、観測最大値を保証されたpeak・必要VRAMにしない。GPU load/release時間も未測定のままである。

起動補助スクリプトの最初の試行は、置換件数の事前checkで停止し、Unity/modelは未実行だった。元のcontroller記録を保持し、修正後の別controllerで上記2形式を実行した。実モデルの失敗やGPU passに数えない。画像・音声の対応する記録は[同じsourceのstage測定](media-gpu-submission-observations.md)、残作業は[優先1〜3の監査](status.md)を参照する。
