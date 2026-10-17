# M4: native SiLU保持と標準ONNX Swishの小さい比較

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

## 2026-10-14の追加状況

このSwish単独probeの結果は元sourceの履歴。現在のnative音声exportは `Neg→Exp→Add→Div` に変更したが、実モデル未検証。[修正receipt](results/native-silu-division-order-green-20261014.json)。

## 当時の実測・実装記録

2026-10-13。音声encoderのexp分解で生じる丸め差を避ける候補として、native SiLUを保持した小さい`.pt2`参照と、標準ONNX Swishだけのimportを比較した。全音声encoderの不合格解消、元6信号・実CPU / GPUの受け入れは未完了。正式Release / tagは作成しない。

[ONNX Swish仕様](https://onnx.ai/onnx/operators/onnx__Swish.html)はopset 24、alphaの既定値1である。導入済みSentis 2.6.1のONNX converterはSwishを専用layerへ変換し、CPU Burst / GPUCompute kernelは `v / (1 + exp(-alpha * v))` を実装している。native Torch `.pt2`のSiLUはSentis Torch converterに対応mappingがないため、native保持のPython成功をそのままTorch importの成功にしない。

## TDDと実測

PR #58の[CI 38361093854](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38361093854)ではUbuntu / Python 3.14の17要素fixtureでnative対div-expのexact mismatchが0だった。以前のテストの「必ず差が出る」というassertionが失敗したが、保存native対公式nativeの完全一致は合格。診断件数は各環境の入力から独立に再計算した件数・最大絶対誤差へ照合し、0も有効な実測として扱う。Windows実測の1 / 381件は当時の環境の記録として保持し、精度許容差は変更しない。

probe生成・native op保持・変更入力・immutable source拒否・モデルを取得しないCLIは5 assertion red → 関連9 passed。shape 17と[1,2,4096]、seed 412 / scale 2を使う。raw `aten.silu`を保存し、再読込と変更入力が公式SiLUに全値一致することを確認した。ONNXは標準domainのSwish 1 node / opset 24 / IR 10でcheckerが成功した。

固定source `ef7a45e237ced12c7b7b0466e23bb7e653bb8a58`で小さいfixtureをローカル生成した。モデル重み・tokenizer取得はない。同じUnity 6000.3.21f1 Editorを再利用し、各ONNXを1回import・保存し、同じ保存modelをCPU / GPUComputeで再読込した。import後の専用Swish layer 1個を確認した。vendor cacheの変更・native plugin追加はない。

2026-10-13 06:36:43 UTCに2 tests passed / failed 0 / skipped 0 / compile成功。内部のCPU / GPU × 2 shapeの全4条件が固定 `rtol=1e-4` / `atol=1e-5`を満たした。CPUは両shapeともnative参照との全値一致、GPUの最大差は17要素で2.384185791015625e-7、8192要素で7.152557373046875e-7。GPUにはexact value不一致が11 / 3263要素あるが、固定許容差超過は0。計算 / 入力CPU fallback・中間readback flushは0。最後の出力downloadは診断境界である。

元のexp分解はこのWindows fixtureで1 / 381要素のexact value差がある。単独SwishのCPU一致は、全model・他OS・他CPUの一致を証明しない。[scopeと証拠receipt](results/m4-native-silu-swish-probe-windows-20261013.json)。

## 次の変換gate

インストール済みPyTorch 2.14.1の実値はONNX max opset 23、TorchScript exporter max 20。opset 24をそのまま指定するfull encoder exportが使えるとは扱わない。native SiLUを保持するfull Python graphと、標準Swishへ変換するONNX経路を小さい変換テストから検証し、非標準domain / runtime依存を最終成果物へ残さない。元6信号の全feature許容差と、Sentis CPU / 実GPU・保存 / 再読込が揃うまで音声encoder不合格を解消済みにしない。
