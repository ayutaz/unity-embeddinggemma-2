# 現在の保存encoderの層別数値差

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-14 JST。Runtime requested / resolved Git `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、固定Python producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`。元のFP32全12条件の不合格、以前のproducerの参照drift、Pow候補の不合格は維持する。正式Release / tagは作成しない。

## 重みを読み込まない演算metadata

tools/とuvで`onnx_metadata.scan`を追加した。seekable protobufを境界付きで読み、ModelProtoのgraphと演算名・Tensor名・shapeを取得する。大きいinitializer raw_dataはseekで飛ばし、4 KiB以下の係数だけSHAを求める。metadata fieldは1 MiB以下、切れたvarint / 親境界超過・graph欠落を拒否する。CLIは入力modelの上書きを拒否し、`model_executed=false` / `binary_sha256_audited=false`を記録する。

scannerはAPI不足5 assertion red → 5 green、CLIは2 red → 関連7 green、音声norm係数anchorは4 red → 関連11 green。CPUでの小さいfixtureだけであり、モデル推論の成功ではない。Pythonの依存追加・新しい重み取得はない。

既存実ONNX1,224,214,325 bytesから1,852,533 bytesだけを読み、5,639 nodes / 586 initializersを取得した。12個の`/norm_out[_N]/Mul_1`を、標準Mulのinputである小さいFP32係数の名前・bytes・dims・SHAへ結び付けた。missing / duplicate / unhashed係数を拒否する。JSONのmetadataだけから全modelのSHA監査を済ませたとは扱わず、Unityの実試験で元ONNXの`7770a8af…`を別途CNG helperへ照合した。

## 保存モデルへの対応と不変観測

test-only `AudioCoefficientLayerMap`は、保存モデルの同bytesのFP32 constantだけをSHA照合し、その係数を使用する一意なMulのoutput indexを求める。missing / ambiguous consumer・stage名重複を拒否する。小さい4 assertion red → 4 green、元のModel保存byte列不変を確認した。

`WorkerObservation.ObserveIndexes`を2 assertion redから追加し、既存4条件と合わせ6 green。対応した12個のTensorだけをlive storageからコピーする。Model outputs / layersや最適化結果は変更せず、通常Scheduleの最終値とScheduleIterableで観測した最終値をbit比較する。診断readbackは明示し、本番chainのreadbackなしの証拠には使わない。

同じproducerのsilence各層native出力12個を13,779,130 bytes / 40 bounded range requestsで取得した。archive全体とstage ONNXは取得していない。native reference / validation JSONの固定SHA、encoder参照へのbinding、各binaryのSHA、元ONNX・保存FP32モデル・inputのSHAを実試験で照合した。

## 実層別結果

固定harness `9625613`、Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / Direct3D12。CPU / GPUの2 NUnit passed / failed0 / skip0 / CLI exit0、2026-10-13 19:38:10 UTC完了。[全数値・対応index・source](results/m4-current-coefficient-bound-layers-windows-20261014.json)。これは診断の完了で、数値gateの合格ではない。

両backendで12個を観測し、通常実行と最終featureのbitが一致した。係数の一意なidentityにより、shapeが同じだけのTensorをstageへ採用していない。norm_outのMul outputはlayer index151 / 287 / … / 1647、tensor index407 / 765 / … / 4345へ対応した。

silenceの最初の固定elementwise条件超過は両backendで`layer_0`。CPU9要素 / 最大許容差比約1.40、GPU8要素 / 約2.15。後続layerも不合格で、layer11はCPU2324要素 / 約27.42、GPU1886要素 / 約37.34。固定`rtol=1e-4` / `atol=1e-5`を変えていない。

以前の保存subsample投影候補はCPU / GPUとも条件内だったため、次は最初のaudio layer内のfeed_forward1 / attention / convolution / normalizationを分ける。以前のnative Torch分解で最初の超過がlayer10だった結果とは、source・変換経路・実backendが異なる。現在のlayer0超過を過去の実行へ付け替えない。

未完了: layer0内部の同一文脈native参照・最初の原因演算の特定と修正、他5信号の層別診断、元encoder全12条件の合格、Float16単独・全chain / 各role・境界、画像 / 音声sample、反復 / memory・実モデルPlayer / 必須実機、最終空consumer、配布準備。工程1〜6の完了とはしない。
