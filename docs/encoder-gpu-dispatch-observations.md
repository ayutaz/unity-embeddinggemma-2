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

## 固定音声による別の観測

同じSDK・計測source・FP32 bundleで、audio-1119の固定silence入力を4回実行した。[音声の監査](results/encoder-gpu-dispatch-audio-1119-fp32-c2f766e-first-audit-20261018.json)。各回モデル268区間・演算267区間、合計2140区間がすべて観測できた。4 vectorの独立参照cosine最小は0.9999999999928919で、解放・1 pass / fail0 / skip0 / exit0と実行後546 sourceファイル一致を確認した。

[画像・音声の演算別集計](results/encoder-image-audio-operation-observations-20261018.json)では、bundle記載の全stageが各回1回ずつ対応し、両入力合計3576区間で中断なし・GPU statistics Enabled/Gathered・正の値があることを検査した。これはこの2入力のencoder送信区間の観測であり、前処理・backbone・他profile・Float16のGPU時間の受け入れではない。

音声warmでの送信区間の経過時間は、BatchedMatmul 36回が0.011150秒、Softmax 12回が0.002844秒だった。異なる仕事量のため、画像との数値差をモダリティ間の速度比較や改善率と解釈しない。画像で観測した長い送信区間を、音声全体の遅さの原因へ一般化する証拠は得られていない。画像の大きな出力確保を改善候補としつつ、音声はmodel送信・frame待機・cache処理なども別に追う。

## 次に検証する改善

Float16重みのsquare-0でも同じ計測を4回実行し、全1436区間と4 vectorの参照一致・解放・1 pass / fail0 / skip0 / exit0を確認した。[Float16画像の監査](results/encoder-gpu-dispatch-image-square-float16-weights-c2f766e-first-audit-20261018.json)。同じ入力・計測方法での修正前baselineとして保持し、FP32との速度差や改善は主張しない。上記のFP32画像・音声と合わせた[3 sessionの監査](results/encoder-three-session-reference-queue-coverage-20261018.json)では、12 vector・全5012区間を確認した。

現行の `GpuBatchedMatmul` / `GpuSoftmax` などは、書き込み先を `new Tensor<float>(shape)` で作った後に `ComputeTensorData.Pin(output)` へ渡しています。固定Sentis 2.6.1のshape-only constructorはCPU tensor dataを既定でゼロ初期化し、Pinは既存storageをGPU側へ変換します。全要素をGPU演算で書く出力では、CPU storageを作らずにGPU storageを割り当てる経路を候補として検証します。

まず書き込み専用出力の初期CPU storageを不要にする契約をRed→Greenで確認し、同じ入力・GPU marker条件で前後比較します。実GPUの演算テスト、全moduleの元参照と検索順位・解放・正常終了を新しいSDK sourceで確認するまで、高速化や数値互換を達成したとは扱いません。重みやgraphは再生成しません。

## 測定の限界

- GPU値はqueue上のmarker区間であり、command/copy・queue bubbleの影響を含み得ます。純粋なkernel稼働時間とはしません。
- profiler・marker作成・samplingにより呼び出しの経過時間は変わります。この実行を通常APIの遅延・平均性能・精度形式間の速度差へ拡張しません。
- この追加観測はFP32のsquare-0 / audio-1119 silenceとFloat16重みのsquare-0のencoderです。他profile、Float16音声、前処理・backboneのGPU区間はこの記録では未検証です。
- 元の12GiB停止やEditMode timeout、過去のGPU時間・割り当て欠測は保持します。優先1〜3全体も未完了です。
