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

## 音声比較の途中capture保持

旧source `8f1bb97`では、停止した変換が残した全5 ONNXの実hashを、独立監査済み
CIの元reportへ照合して再利用しました。変換を繰り返さず実PCMからのCPU比較へ
進みましたが、1,200秒上限でexit124となり、最初の1回も完了していません。
数値比較・CPU速度・最低要件の合格ではありません。元の変換停止receiptと
非terminalの親記録を上書きせず、各processの実終了を別に保存しています。

音声CPUの測定入口は、完了した前処理・encoderの実出力をstage observerから
直ちに保存し、`partial-audio-CPU-captures.json` を一時fileの置換で更新します。
推論timerから除外したobserver区間で行い、次のstageには保存fileや参照値ではなく
直前の実tensorを渡します。全4回を待つ前に中断しても、完了済みstageの
元byte数・hash・shapeを後から確認できます。

この途中journalは合格・process生存・途中からの再実行許可を示しません。
全4回の元中間値・最終vector・正常終了と独立監査が、引き続き受け入れ条件です。
前処理後 / encoder後の中断について2件Redから関連54件Greenを確認しました。
実行中の旧sourceや元の失敗記録は書き換えず、新実装の実モデル実行は別途記録します。

## 既存exportのsourceを保った再利用

既定では、chain reportとCPU実行sourceの一致を引き続き必須にします。
測定の `--retained-chain-source` を明示した場合だけ、元exportのproducerと
現在の実行sourceを分けて記録します。実際のclean HEADを検査し、generatorの
全module・参照・依存入力のGit blobとmodeを比較します。CPU消費側の
`audio_cpu_reference.py` / `audio_cpu_encoder.py` と独立した
`native_gpu_memory.py` 以外に変更があれば拒否します。

元reportのproducerを現在のsourceへ書き換えません。元の数値監査、5実graphの
読み込み時の完全hash、CPU provider、4本の実carry、前処理・中間・最終値の
元gateは維持します。captureの別process監査でも保存したproducer資格を再検査し、
audit commandからのproducer上書きは拒否します。source再利用だけでは新しい
export実行・CPU比較・性能全体・GPU・正式公開を合格にしません。

[source再利用のTDD](results/retained-audio-graph-source-tdd-20261020.json)は、
generator・参照・依存の変更、dirty executor、誤ったproducer、保存した資格の
改変を拒否する小規模確認です。実モデルと性能の結果は別に記録します。

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
