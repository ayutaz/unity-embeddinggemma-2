# native inverse power候補の不合格

2026-10-14 JST。Runtimeは固定Git `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、元参照producerは`4edbe17a8105233c5aaa1514faa2862b08a4f7e0`。正式Release / tagは作成しない。

公式Gemma4RMSNormは`pow(mean_squared, -0.5)`を使い、実装commentもTorch / JAXのcompiler差を避けるためpowを選ぶと記している。一方、Sentis2.6.1の`ContractToSimplerLayerPass`は定数指数-0.5のPowをRsqrtへ置き換える。この演算差が現在の不合格に影響するかを調べた。原因を確定したものではない。

test-only `NativeInversePowerCandidate.Create`はModelのinputs / outputs / constants / layersのlistを別に作り、各Rsqrtを同じinput / output indexのPowと、新しい-0.5のscalar constantへ置き換える。他の演算・元の重みを変更せず、Runtime packageや保存モデルのfileを変更しない。

API不足2 assertion redから候補を実装した。小さいCPUは合格、GPUは0.25の逆平方根でexpected2.0 / actual1.99999988となりexact値のassertionに失敗した。元Modelの保存byte列不変とPowへの変換は確認できたが、小さい結果は1 pass / 1 fail / skip0 / CLI exit1。2 greenやbit一致とはしない。

固定harness / candidate source `188a6da`で、既存FP32保存encoderのfull / partial両profileを使用した。各profileの109個のRsqrtを置き換え、独立native featureと元6信号のCPU / GPUを元と同じ `rtol=1e-4` / `atol=1e-5` / cosine >=0.999999で比較した。再download / ONNX import / model再保存はない。

全12条件でcosineは合格したが、elementwise gateは全件不合格。NUnit0 pass / 2 fail / skip0 / CLI exit1、2026-10-13 19:20:26 UTC完了。[全結果と元の対照](results/m4-native-inverse-power-candidate-windows-20261014.json)。

| backend / signal | 元の不一致数 | 候補の不一致数 |
| --- | --- | --- |
| CPU silence | 728 | 1134 |
| CPU impulse | 1228 | 1470 |
| CPU sine | 1162 | 794 |
| CPU nyquist | 2279 | 2027 |
| CPU noise | 843 | 747 |
| CPU padded_noise | 743 | 610 |
| GPU silence | 665 | 559 |
| GPU impulse | 945 | 328 |
| GPU sine | 764 | 592 |
| GPU nyquist | 881 | 1029 |
| GPU noise | 94 | 126 |
| GPU padded_noise | 84 | 89 |

条件ごとの改善・悪化が混在し、数値修正として不採用。元の許容差を緩めず、元の不合格・GPUの1 ULP差を保持する。native Powを戻すだけでは全encoderを通せないことが確認できた。数式の一般的な同値性と、特定backendの浮動小数計算は区別する。

次は元ONNXのoutput名と保存layerの構造 / coefficientを監査し、同じproducerのlayer0以降を、最終経路を変えない観測へ対応付ける。原modelの数値gate、画像 / 音声全chain・各role / sample、反復・Player / 必須実機、最終空consumer、配布準備は引き続き未完了。正式リリースは行わない。
