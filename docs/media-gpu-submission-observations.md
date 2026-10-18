# 前処理・backboneのGPU送信区間を測る

2026-10-18。SDK `d7b2233`、計測 `10962fa`、Windows / Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定consumer。計測の追加でSDKの演算・await・GPU fence・cache解放は変更しない。

## 測る範囲

前処理とbackboneの元のメソッド呼び出しを、その呼び出しから戻るまでGPU markerで囲む。非同期メソッドではTaskを返すまでであり、完了待ち全体ではない。固定版の `AudioPreprocessor.ProcessAsync` は最初のawaitより前にGPU処理を送信し、その後のfence待ち・cache解放は元のstage CPU/wallで別に測る。任意のasync実装へ一般化しない。encoderは元のmodel segmentとGPU operationの各markerを使う。

CPU送信wall、stage全体のwall、Editorプロセス全体のCPU、GPU queue区間は異なる観測で、重なり得る。足したり差し引いたりしてkernel稼働時間を作らない。GPU queue区間にはcopy・queueの待ちやbubbleが含まれ得る。Profilerの影響も含むため、別の計測sourceや通常APIとの速度比較には使わない。

## 全5 profile・両形式の同一入力反復

全10プロセス・40 vectorの元Python参照・単位長、Worker・kernel・tensor解放、各1 pass / fail0 / skip0 / exit0を確認した。encoderは17,176区間、前処理・backboneは80区間すべて正数で、元のEnabled / Gathered条件を維持した。終了後554 sourceファイルを再照合し、[集約監査](results/all-media-d7b2233-stage-gpu-cpu-wall-memory-10962fa-20261018.json)から個別のGPU・CPU/wall・メモリ監査へたどれる。[統合sourceとの対応](results/submission-measurement-integration-source-20261018.json)はruntimeと計測validationのGit tree一致を確認する。sampleのUX差分は別であり、手動操作の合格を含まない。

下表は初回後3回の計測付きAPI経過時間の範囲（秒）。各profileで同じ入力を両形式に使った。project Library・共有OS/shader cacheは残しており、形式間の速度比較、通常APIの平均性能や以前の別計測sourceとの改善率にはしない。

| profile | FP32 warm・反復 | Float16重み warm・反復 |
| --- | ---: | ---: |
| image-square | 4.557–4.732 | 4.549–4.614 |
| image-landscape | 4.850–5.092 | 4.651–4.762 |
| image-portrait | 4.979–5.034 | 4.660–4.771 |
| audio-1119 / silence | 5.235–5.307 | 5.215–5.354 |
| audio-839 / padded_noise | 5.234–5.540 | 5.180–5.449 |

[画像のGPU監査](results/submission-error-gpu-image-square-fp32-10962fa-first-audit-20261018.json) / [前処理・backbone](results/submission-error-gpu-image-square-fp32-10962fa-first-stages-20261018.json) / [CPU・wall](results/submission-error-gpu-image-square-fp32-10962fa-first-cpu-wall-audit-20261018.json)。[音声のGPU監査](results/submission-error-gpu-audio-1119-fp32-10962fa-first-audit-20261018.json) / [非同期前処理・backbone](results/submission-error-gpu-audio-1119-fp32-10962fa-first-stages-20261018.json) / [CPU・wall](results/submission-error-gpu-audio-1119-fp32-10962fa-first-cpu-wall-audit-20261018.json)。終了後は554 sourceファイルを再照合し、Worker・kernel・tensorの解放を確認した。

9プロセスでGPUメモリを正確なPIDとcounter採取時刻へ対応付けた。例として[FP32音声](results/submission-error-gpu-audio-1119-fp32-10962fa-first-memory-audit-20261018.json)はプロセス区間28観測中26観測が有効で、観測最大Dedicated 9,383,141,376 bytes、Shared 245,952,512 bytesだった。保証されたpeak・必要VRAMではなく、短いstageに採取点がなければnullを保持する。FP32のsquare画像試行には対応するcounter traceがなく、欠測のまま集約した。全10条件のホスト生涯ピーク最大は9,744,793,600 bytesで、混在検索のピークとは別の観測である。

## 失敗からの修正

最初の計測 `5a907a2` は推論4回と解放まで完了したが、GPU送信区間8件がすべてaborted・値nullとなり、NUnitは0 pass / 1 fail / skip0 / exit2だった。内部reportのsuccess=trueだけで合格にしない。[元の失敗と修正の監査](results/submission-typed-null-failure-and-tdd-20261018.json)に保持する。

Newtonsoftのnull stringは、JSONではnullでもメモリ上のtoken型がStringのままの場合がある。token型だけで例外の有無を判定していたため、成功もabortedとしていた。値で判定するように修正し、型付きnull・明示null・実例外・JSON往復・欠落・不正な型・空名を検証した。standaloneの関連12件中4件Redから12件Green、Unityの関連27件もfail0 / skip0 / exit0だった。欠落・不正な値を成功に置き換えず、GPU値の正数・Enabled / Gathered・元の参照と終了条件も維持する。

## 残る受け入れ

GPU load/releaseの時間、非同期allocation・明示的なEditor index再構築の受け入れを、上のstage測定から推定しない。Editor測定、公開empty-cache取得、実マイク、最終candidateの手動操作と最新head CIを含む優先1〜3全体は未完了である。
