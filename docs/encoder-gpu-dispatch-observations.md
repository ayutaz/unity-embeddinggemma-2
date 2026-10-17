# EncoderのGPU区間の観測

2026-10-18。SDK `e28a5fd`、計測source `c2f766e`、FP32の固定square-0画像を同じsessionで初回・warm・2反復しました。SDKやschedulerは変更せず、検証側から既存cache callbackとGPU演算を借用してmarkerを配置し、終了時に復元します。

## 確認結果

- [callbackのTDD](results/encoder-gpu-stage-callback-tdd-20261018.json)：空のscaffoldで6件Red、同じテストが.NETで6件Green。Unityでも6 pass / fail0 / skip0 / exit0。fake callbackの順序・例外・中断・所有権の検証であり、GPU時間の合格ではありません。
- [実画像の監査](results/encoder-gpu-dispatch-image-square-fp32-c2f766e-first-audit-20261018.json)：各回モデル180区間・演算179区間、4回合計1436区間のGPU Recorder値を取得。全区間でGPU statisticsのEnabled/Gatheredを確認し、この実行の欠測は0でした。
- 4 vectorの独立参照cosine最小は0.9999999999997138。有限768次元・単位長、Worker/kernel/tensorの解放、1 pass / fail0 / skip0 / exit0まで確認しました。546 sourceファイルを実行前後と監査時に照合しています。

warmの区間別合計は次のとおりです。同一実行の観測であり、別のCPU/wall baselineとの速度比較には使いません。

| 区間 | 数 | CPU側の送信区間の経過時間 | GPU queue区間 |
| --- | ---: | ---: | ---: |
| model segmentと出力copy | 180 | 1.247秒 | 0.493秒 |
| GPU演算 | 179 | 13.350秒 | 2.385秒 |

GPU演算のCPU側経過時間ではBatchedMatmulの32回が6.603秒、Softmaxの16回が5.446秒でした。これらはmain threadの経過時間であり、CPUの実稼働時間ではありません。割り当て、転送、driver待ち等の内訳は未分離です。

## 次に検証する改善

現行の `GpuBatchedMatmul` / `GpuSoftmax` などは、書き込み先を `new Tensor<float>(shape)` で作った後に `ComputeTensorData.Pin(output)` へ渡しています。固定Sentis 2.6.1のshape-only constructorはCPU tensor dataを既定でゼロ初期化し、Pinは既存storageをGPU側へ変換します。全要素をGPU演算で書く出力では、CPU storageを作らずにGPU storageを割り当てる経路を候補として検証します。

まず書き込み専用出力の初期CPU storageを不要にする契約をRed→Greenで確認し、同じ入力・GPU marker条件で前後比較します。実GPUの演算テスト、全moduleの元参照と検索順位・解放・正常終了を新しいSDK sourceで確認するまで、高速化や数値互換を達成したとは扱いません。重みやgraphは再生成しません。

## 測定の限界

- GPU値はqueue上のmarker区間であり、command/copy・queue bubbleの影響を含み得ます。純粋なkernel稼働時間とはしません。
- profiler・marker作成・samplingにより呼び出しの経過時間は変わります。この実行を通常APIの遅延・平均性能・精度形式間の速度差へ拡張しません。
- この追加観測はFP32・square-0のencoderです。他profile、Float16、前処理・backboneのGPU区間はこの記録では未検証です。
- 元の12GiB停止やEditMode timeout、過去のGPU時間・割り当て欠測は保持します。優先1〜3全体も未完了です。
