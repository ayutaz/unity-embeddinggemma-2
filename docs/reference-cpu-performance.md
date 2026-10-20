# CPU参照比較の計測契約

テキストには、[既存Windows Playerの同一query CPU / GPU比較](text-paired-performance.md)が
あります。現在のSDK入力との対応と元timingを再監査し、media用PyTorch比較と別に記録しています。
初回caseとwarm queryの違い、実行source、runtimeの数値検査と独立raw vector採取の違いを保持します。

更新: 2026-10-20。[Windows機能検証の工程1〜3](results/windows-first-stages-1-to-3-20261020.json)は完了しました。同条件CPU比較・性能・最低要件は[実行計画](next-release-plan.md)の工程4で、機能検証とは別の未完了項目です。他環境の実機受け入れはWindows公開後に分けます。以下の契約・失敗診断は元sourceの範囲で保持します。

後続の[Windows CPU原本再照合](results/windows-image-CPU-and-bounded-diagnostics-20261020.json)では、
source `7d22168` の画像3形状・各4回について全12最終vectorを独立再計算しました。
横長・縦長は中間captureも合格、正方形は各回3要素が超過してexit1です。
source `4e235b6` の正方形1回ずつの診断では、hardware FMA linear・固定profile pooling・
明示的RMS参照式の実captureが全てbit単位で同じで、超過を解消しませんでした。
入力・元参照・許容差は維持し、これらの同じ変更での重い再試行は行いません。
小規模の[明示的CPU参照](cpu-fp32-reference.md)は音声の次の診断にも使用できますが、
画像・音声の数値合格や性能改善を自動的には示しません。

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

後続の[明示的native CPU経路](native-audio-CPU-results.md)ではsource `c458937` で
音声2profileを各4回実PCMから計算し、全8回の元中間値・最終vectorと正常終了を確認しました。
encoderは全8回bit単位で元参照に一致し、GPU側との同じ入力・profile・descriptor・最終参照も照合しています。
これは下記の停止したONNX経路と別のexecutorです。CPUの独立load / release計時と公開CLIの統合は残っています。

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
後続の明示的CPU経路では、producerからimportされない `cpu_fp32_reference.py`、
`native_audio_cpu_encoder.py`、`vision_cpu_simd_reference.py`、`image_cpu_reference.py` も消費側として列挙しています。
generator・参照・依存の変更は引き続き拒否します。

元reportのproducerを現在のsourceへ書き換えません。元の数値監査、5実graphの
読み込み時の完全hash、CPU provider、4本の実carry、前処理・中間・最終値の
元gateは維持します。captureの別process監査でも保存したproducer資格を再検査し、
audit commandからのproducer上書きは拒否します。source再利用だけでは新しい
export実行・CPU比較・性能全体・GPU・正式公開を合格にしません。

[source再利用のTDD](results/retained-audio-graph-source-tdd-20261020.json)は、
generator・参照・依存の変更、dirty executor、誤ったproducer、保存した資格の
改変を拒否する小規模確認です。実モデルと性能の結果は別に記録します。

## 明示的native音声CPU測定入口

[音声の8実行](native-audio-CPU-results.md)で使用した数値経路を、task branchの
`audio_cpu_reference` の明示的optionへ統合しました。[TDD](results/native-audio-CPU-lifecycle-tdd-20261020.json)は
実入力のcarry、例外時の復帰、元係数の検査、load / releaseとcapture監査の契約を確認しています。
統合後source `fd7f375` の[実モデル8回とlifecycle](results/native-audio-CLI-lifecycle-fd7f375-20261020.json)は、
別processの元capture再照合まで合格しました。以前の非公開orchestrationとは分けて記録します。

`--native-candidate` と `--native-build` を同時に指定し、取得済みの `--snapshot` を明示します。
この経路では `--chain-report` / `--retained-chain-source` を混在させません。
native helperのbuildはmodel-load timerの外です。loadはprocessor / coefficient graph / backbone構築、
元係数の変更検査は別区間、releaseはencoderを閉じて所有model参照を捨てた後のGCまでです。
release完了をallocatorの0 byte保証や最低RAM要件には読み替えません。

```sh
uv run --group cpu-comparison python -m embeddinggemma_tools.audio_cpu_reference measure \
  --preprocessing "$EG_CPU_PREPROCESSING" --encoder-reference "$EG_CPU_ORIGINAL_ENCODER" \
  --document-reference "$EG_CPU_DOCUMENT_REFERENCE" --snapshot "$EG_CPU_SNAPSHOT" \
  --native-candidate "$EG_CPU_CANDIDATE" --native-build "$EG_CPU_NATIVE_BUILD" \
  --case silence --source "$EG_CPU_SOURCE" --output "$EG_CPU_OUTPUT"
```

実行後は同じ固定source・元参照で `audit` を別processから実行します。
auditへのnative選択optionの指定は拒否し、保存されたexecutorと元の配布inventoryを照合します。
最終vectorだけでなく全4回の元中間値を照合し、失敗・欠測と成功を分離します。

## 画像の明示的な集計順

`image_cpu_reference measure --statistics-reference sixteen-lane` は、vision towerの
元のGemma4 RMSだけを16 laneのFP32集計順へ置き換えます。scaleとepsilonは借用し、
projector / backbone・元参照・許容差は変えません。hardwareを見た自動選択は行わず、
既定の `native` 経路と過去の失敗を保持します。新たなモデルの重みは生成しません。

元参照に一致した2つのCI記録はAVX512、ローカルはAVX2でした。モデル読込・encoderのASTと
依存versionも照合しました。命令セットだけで原因を断定せず、
[PyTorchの集計実装](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp)を
独立scalar oracleと照合してから、source `fb097a5` の正方形1回で元の中間値・最終vectorを確認しました。
この1回を全形状・4反復・性能全体の完了には数えません。

共有用の入口は[小規模TDD](results/vision-sixteen-lane-reference-tdd-20261020.json)で、
全置換の事前検査・元scaleのidentity・projectorの保持・mode記録とaudit時の上書き拒否を確認しました。
loadとreleaseも別計時し、releaseはmodel所有参照の破棄とGCまでの範囲です。
統合sourceでの3形状・各4回と、別processでのraw capture照合を次に受け入れます。

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
