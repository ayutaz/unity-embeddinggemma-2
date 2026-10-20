# CPU参照比較の計測契約

更新: 2026-10-20。[Windows機能検証の工程1〜3](results/windows-first-stages-1-to-3-20261020.json)は完了しました。同条件CPU比較・性能・最低要件は[実行計画](next-release-plan.md)の工程4で、機能検証とは別の未完了項目です。他環境の実機受け入れはWindows公開後に分けます。以下の契約・失敗診断は元sourceの範囲で保持します。

画像・音声のGPU実行中に観測したprocess CPU時間とは別に、同じ固定入力と
元のFP32参照を使うPyTorch CPU基準を測ります。SentisのCPU API対応を追加する
変更ではなく、SDK Runtimeは維持します。frameworkの違いを記録し、CPU時間と
GPU marker区間の差をkernel専有時間として扱いません。

`reference_cpu_performance.measure_reference_pipeline` は、前処理の実出力を
encoderへ、その実出力をbackboneへ渡します。正解特徴量の代入やstage省略を
行いません。同じ入力のfirst / warm / repeat-1 / repeat-2を測り、各stageの
wall / process CPUと最終768値を保存します。元参照cosine 0.999・単位長条件を
維持し、非CPU tensor、非FP32浮動小数、入力変化、異なるsource / descriptor /
referenceを拒否します。既存report・captureは上書きしません。

[契約段階のTDD](results/reference-cpu-performance-protocol-20261019.json)は、
未実装26件から26件合格・失敗0・skip0です。実モデルの数値・性能比較では
ありません。実行時は元モデルrevision・重み・設定のhashを照合し、画像 / PCM、
mask・prompt・profile・仕事量をGPU側と揃えて固定sourceへ結び付けます。

この入口はFP32だけを対象にします。Torch model全体をhalf化してfloatへ戻す
処理は、SentisのFloat16重み派生と同じ対象だけを丸める保証にならないため、
比較対象として受け入れません。GPUの両精度測定と、CPUの表現・frameworkの
範囲は別に示します。実モデルの照合・正常終了・独立監査、音声のGPU追加測定、
専有GPU / VRAM・最低要件・将来の他環境受け入れを、この小規模テストで完了扱いに
しません。

## 実画像のCPU診断

[source `3cc930a` の独立監査](results/image-cpu-diagnostics-3cc930a-20261019.json)では、
元revisionの重み・設定6ファイルを照合し、画像3形状を各4回、実前処理・公式
Gemma4VisionModel・projector・backboneの順にCPUで実行しました。4 thread・
FP32・MKL AUTOです。保存した最終768値を独立再計算し、12回すべてで元の
最終参照・単位長条件に一致しました。CUDA contextは作成していません。

| profile | CPU first 秒 | CPU warm 3回 秒 | GPU FP32 warm 3回 秒 |
| --- | ---: | ---: | ---: |
| square | 13.965 | 13.827–15.252 | 4.719–5.227 |
| landscape | 14.305 | 13.982–15.475 | 4.673–4.941 |
| portrait | 15.038 | 14.513–15.980 | 4.595–4.861 |

同じPNG・descriptor・最終参照を使用しています。PyTorch CPUとSentis GPUの
別framework・別processの観測値で、loadは表の推論時間に含めません。一般入力や
純kernelの速度比・Sentis CPU対応・最低hardware要件を示すものではありません。

**squareは中間encoder値の検査で各回3要素が許容差を超え、process exit1です。**
landscape・portraitは現在保有する中間参照に一致しexit0でした。中間参照の
metadata hashは元の最終参照が指定した世代と異なり、元世代の中間参照を回収
できていません。このため全条件の合格・元encoder回帰の受け入れ・性能全体の
完了は主張しません。許容差は変更せず、失敗数・最終値の一致・測定値を分けて
保存しています。原因は未確定です。音声のCPU比較も未実行です。

上記は初回sourceのcheckpointです。[後続のCPU診断と元データ回収](reference-cpu-diagnostics.md)
でDocument特徴量6件を回収し、元の最終参照auditに結び付けました。画像3・音声2
条件の全20最終vectorは一致しましたが、元の中間値に対して全5条件に許容差
超過が残り、いずれもexit1です。元vision JSONの回収、CPU全条件の合格や既存の
失敗の取消しとは扱いません。
