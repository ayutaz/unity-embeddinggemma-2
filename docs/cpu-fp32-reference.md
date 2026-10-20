# 明示的なCPU FP32参照の小規模検証

`tools/embeddinggemma_tools/cpu_fp32_reference.py` は、比較用Python toolsから明示的に選ぶ
x64 AVX2 / FMAの数値診断です。Unity packageへのnative pluginやCPU fallbackは追加しません。
実モデルの数値受け入れ・性能比較を、小規模テストの成功から推測しません。

`NativeFp32Reference.build` はインストール済みC++ compilerを使い、指定した非公開directoryへ
helperを生成します。import時のbuild・download・自動fallbackはありません。
source / compiler binary / flags / libraryのhashを保持し、既存buildの不一致は上書きせず拒否します。
MSVCは `/fp:strict` と明示的FMA intrinsicを使用し、一般の加算・乗算の自動再結合を許可しません。
生成library・compiler log・一時objectはGitへ含めません。

FP32 FMAを逐次実行した192要素のpartialを、block順に別々のFP32加算で接続します。
biasがある場合は最初のpartialへ加算してから後続blockを接続します。
CPUのFMA / AVX2・OS vector state・round-to-nearestを確認し、各workerへMXCSRを引き継ぎます。
native threadは最大4です。入力重みは変更せず、非連続arrayは読み取り用にcopyします。

Torch adapterはno-grad・CPU FP32に限定します。`linear_scope` は隔離した診断processでだけ使用し、
functional.linearを一時的に差し替え、正常終了・例外の両方で元へ戻します。
同時・入れ子のscopeは拒否します。元の関数へ黙ってfallbackしません。
matmul / FMAも明示的な呼出しで、実呼出し件数を記録します。

音声stemの明示的な診断は3×3、stride2、padding1、入力1 channelまたは16 channel単位に限定します。
Unfoldで元画素を選び、kernel位置→channelの順にpartialを作り、channel blockのpartialを別々に加算します。
元ONNX exporterのeager conv2d passthroughを実行証拠には使いません。scalar C99 FMAで独立照合しています。

[TDD記録](results/cpu-fp32-reference-tdd-20261020.json)はscalar C99 `fmaf` を独立oracleにした
bit単位の比較、192境界・bias順・batch・tail・broadcast・非連続入力・不正入力、
Torchのrank保持・実呼出し・例外時復帰・cache改変拒否を確認しています。
このhelperのhardware FMAと、既存ONNXのDekker / TwoSum式が全入力で同じとは主張しません。
既存ONNXのeager passthroughをordered graphの実行として数えることもありません。
実モデルへ適用する場合は実executor・演算方式・sourceを分け、元の全数値gateで判定します。

[固定画像の実診断](results/windows-image-CPU-and-bounded-diagnostics-20261020.json)では、
hardware FMA linear、固定profile pooling、既存の明示的RMS参照式を各1回確認しました。
全て最終vectorは一致しましたが、中間encoderの許容差超過は同じ3要素で、正常合格には数えません。
3診断の実encoder / final captureはbit単位で同じでした。FMA linearはencoder約44.36秒・backbone約17.85秒で、
通常CPUの固定profile診断の約13.45秒・2.92秒より遅く、この置換を画像の高速化として採用しません。
一条件一回の別process診断であり、一般的な性能比や全画像の合格は示しません。

この段階では元golden・許容差・モデル係数・Unity GPU推論を変更していません。
生成libraryを含む公開配布、全モデル合格、CPU速度改善、最低要件、正式Releaseは未確認です。
