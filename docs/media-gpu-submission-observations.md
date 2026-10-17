# 前処理・backboneのGPU送信区間を測る

2026-10-18。SDK `d7b2233`、計測 `10962fa`、Windows / Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定consumer。計測の追加でSDKの演算・await・GPU fence・cache解放は変更しない。

## 測る範囲

前処理とbackboneの元のメソッド呼び出しを、その呼び出しから戻るまでGPU markerで囲む。非同期メソッドではTaskを返すまでであり、完了待ち全体ではない。固定版の `AudioPreprocessor.ProcessAsync` は最初のawaitより前にGPU処理を送信し、その後のfence待ち・cache解放は元のstage CPU/wallで別に測る。任意のasync実装へ一般化しない。encoderは元のmodel segmentとGPU operationの各markerを使う。

CPU送信wall、stage全体のwall、Editorプロセス全体のCPU、GPU queue区間は異なる観測で、重なり得る。足したり差し引いたりしてkernel稼働時間を作らない。GPU queue区間にはcopy・queueの待ちやbubbleが含まれ得る。Profilerの影響も含むため、別の計測sourceや通常APIとの速度比較には使わない。

## 完了した最初の2条件

| 条件 | 同一入力反復・元参照 | encoderのGPU区間 | 前処理・backboneのGPU区間 | 終了 |
| --- | --- | --- | --- | --- |
| image-square / FP32 | first / warm / 2反復、4 vector合格 | 1,436 / 1,436 | 8 / 8 | 1 pass / fail0 / skip0 / exit0 |
| audio-1119 / FP32 | first / warm / 2反復、4 vector合格 | 2,140 / 2,140 | 8 / 8 | 1 pass / fail0 / skip0 / exit0 |

[画像のGPU監査](results/submission-error-gpu-image-square-fp32-10962fa-first-audit-20261018.json) / [前処理・backbone](results/submission-error-gpu-image-square-fp32-10962fa-first-stages-20261018.json) / [CPU・wall](results/submission-error-gpu-image-square-fp32-10962fa-first-cpu-wall-audit-20261018.json)。[音声のGPU監査](results/submission-error-gpu-audio-1119-fp32-10962fa-first-audit-20261018.json) / [非同期前処理・backbone](results/submission-error-gpu-audio-1119-fp32-10962fa-first-stages-20261018.json) / [CPU・wall](results/submission-error-gpu-audio-1119-fp32-10962fa-first-cpu-wall-audit-20261018.json)。終了後は554 sourceファイルを再照合し、Worker・kernel・tensorの解放を確認した。

[音声のGPUメモリ](results/submission-error-gpu-audio-1119-fp32-10962fa-first-memory-audit-20261018.json)は正確なPIDとcounter採取時刻で対応付け、プロセス区間28観測中26観測が有効だった。観測最大Dedicatedは9,383,141,376 bytes、Sharedは245,952,512 bytesで、保証されたpeak・必要VRAMではない。短いstageに採取点がなければnullを保持する。この画像試行はcounter採取開始前に終了したため、対応するGPUメモリの記録はない。

## 失敗からの修正

最初の計測 `5a907a2` は推論4回と解放まで完了したが、GPU送信区間8件がすべてaborted・値nullとなり、NUnitは0 pass / 1 fail / skip0 / exit2だった。内部reportのsuccess=trueだけで合格にしない。[元の失敗と修正の監査](results/submission-typed-null-failure-and-tdd-20261018.json)に保持する。

Newtonsoftのnull stringは、JSONではnullでもメモリ上のtoken型がStringのままの場合がある。token型だけで例外の有無を判定していたため、成功もabortedとしていた。値で判定するように修正し、型付きnull・明示null・実例外・JSON往復・欠落・不正な型・空名を検証した。standaloneの関連12件中4件Redから12件Green、Unityの関連27件もfail0 / skip0 / exit0だった。欠落・不正な値を成功に置き換えず、GPU値の正数・Enabled / Gathered・元の参照と終了条件も維持する。

## 残る受け入れ

残る画像profile・音声839・Float16重みを同じsourceで順番に測定中であり、上の2条件だけで全profileを合格にしない。全条件のGPU区間・CPU/wall・メモリを集約し、非同期allocation・Editor測定を別途受け入れる。公開empty-cache取得、実マイク、最終candidateの手動操作と最新head CIを含む優先1〜3全体は未完了である。
