# CPU参照比較の計測契約

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
専有GPU / VRAM・最低要件・必須実機の残条件を、この小規模テストで完了扱いに
しません。
