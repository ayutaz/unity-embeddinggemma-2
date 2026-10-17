# 音声layer 0内部正規化の固定参照照合

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

2026-10-14 JST。layer 0の最終出力からさらに内部の差を調べるため、同じ保存FP32モデルの3つの正規化出力を観測した。Runtime / model / 元数値許容差は変更しない。正式Release / tagは作成しない。

## 参照と対応

元Python producerは`4edbe17a8105233c5aaa1514faa2862b08a4f7e0`。内部参照は[PR #75のActions run 38459688048](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38459688048)で、固定subsample入力から公式layer 0 finalをbit単位で再現した実行source `9800c52957478fafa1377e2caa215f9f2e1477b0`を使う。reportの固定SHA `39d563e2…`、各内部binaryのshape / size / SHAを照合した。

ONNX metadataの`/norm_pre_attn/Mul_1`、`/norm_post_attn/Mul_1`、`/norm_out/Mul_1`をFP32係数のbyte hashへ結び付けるproducerをTDD追加。4 assertion red → 4 green、既存scanner / CLI / 層別anchorを含む15 green。重みの再取得・依存追加なし。missing / duplicate stageとunhashed coefficientを拒否する。共通の係数検証を再利用する。

既存のTDD済み`AudioCoefficientLayerMap`で、保存モデル内の同じ係数を使う一意なMul outputへ対応付ける。入力とshapeが同じだけのTensorを選ばない。元ONNX `7770a8af…`、保存モデル `3214cc2b…`、silence入力 `366de71c…`をCNG hash helperで照合した。保存モデルのreimport / 再変換はしない。

## 実Windows結果

harness source `1f1414c7f4810fed440eb1b9a8aa0bad40f2a207`、Runtime requested / resolved Git `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、Runtime overlayなし。Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor本体121512の1つ、uloopで対象2件だけを実行。compile error / warning0、テスト登録確認後に実行した。[完了receipt](results/m4-layer-zero-three-norms-windows-20261014.json)。

| 段階 | layer / tensor index | CPU 不一致数 / 最大許容差比 | GPU 不一致数 / 最大許容差比 |
| --- | --- | --- | --- |
| norm_pre_attn | 44 / 122 | 2 / 2.12040 | 280 / 1.65915 |
| norm_post_attn | 93 / 259 | 1 / 1.05754 | 0 / 0.67595 |
| norm_out | 151 / 407 | 9 / 1.39837 | 8 / 2.14971 |

CPU / GPUとも3段階の観測は完了し、通常Scheduleと観測実行の最終featureはbit一致。診断のCPU readbackを本番chainのGPU保持の証拠には使わない。

固定`rtol=1e-4` / `atol=1e-5`の条件で両backendともnorm_pre_attnから不合格。これは今回観測した3段階の最初であり、最初の原因演算を確定したものではない。norm_pre_attnの前のfeed_forward1、subsampleの微差、正規化内部を切り分ける必要がある。GPU norm_post_attnの合格で他の不合格を消さない。

NUnit0 passed / 2 failed / skip0 / CLI exit1、2026-10-13 20:13:25 UTCに完了。各reportは`observation_completed=true` / `success=false`を保持する。現在の数値条件に実際に失敗する受け入れテストで、診断完了を数値passに置き換えない。初回のuloop権限不足は実テスト未実行であり、上記のassertion失敗と別に扱う。

次はnorm_pre_attn入力と独立した正規化処理の比較、feed_forward1内の差の分解から最小修正へ進む。元6信号・CPU / 実GPUのencoder gate、画像FP32差、全chain / 各role・入力境界、画像 / 音声sample、反復 / memory・実モデルPlayer / 必須実機、最終空consumerと配布準備は未完了。

## 追加: 正規化入力の照合

`ResolveNormInputs`は係数で確定した最終Mulから、normalized Mul → Rsqrt → epsilon加算 → ReduceMean → Squareをたどる。Squareの入力とnormalized Mulの入力が同じindexでなければ拒否する。missing APIの3 assertion redを確認し、最初の実装では合成fixtureがScalarMadへ統合されるため6 green / 1 fail。multiplier1 / 有限正epsilonのScalarMadを認識して7 green / skip0 / CLI exit0となった。単なる係数Multiplyや平方平均のないinverseをRMSNorm入力と誤認しない。元Modelの保存byte列不変を確認した。

追加harness source `8003bf54a92ae13e9701314d454da8cd0a785dc0`で、別の出力先へCPU / 実GPUの各4点を観測した。norm入力indexは107 / 244 / 392。norm_pre_attn入力のindex107はlayer38のClip出力であり、min / maxは±1e10。固定native feed_forward1出力は全要素がその範囲内なので、nativeのclampで値が変化しないことを確認して比較した。[追加receipt](results/m4-layer-zero-norm-input-windows-20261014.json)。

| 追加観測 | CPU 不一致数 / 最大許容差比 | GPU 不一致数 / 最大許容差比 |
| --- | --- | --- |
| norm_pre_attn_input（clamp後のfeed_forward1出力） | 281 / 2.05972 | 280 / 1.60989 |

両backendで正規化入力時点から固定数値条件を超えている。正規化だけが原因とは扱わず、次はfeed_forward1内部のpre norm / linear / activation / post normと、前段subsampleの差の影響を切り分ける。既存3出力の不一致数・最大比は前回と同じ。通常最終featureと観測最終featureのbit一致も両方維持した。

追加runはNUnit0 pass / 2 fail / skip0 / CLI exit1、2026-10-13 20:21:43 UTCに完了。元の3点run、mapperの中間失敗、追加4点runを別々に保持する。数値許容差、元参照、Runtime、モデルは変更していない。
