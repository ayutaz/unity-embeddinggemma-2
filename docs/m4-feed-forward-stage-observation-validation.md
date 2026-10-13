# layer 0 feed_forward1の保存モデル数値照合

2026-10-14 JST。norm_pre_attn入力から既に固定条件を超えるため、同一native文脈のfeed_forward1内部5段階を保存Sentis encoderへ対応付けて比較した。Runtime / model / 元参照 / 許容差を変更せず、正式Release / tagは作成しない。

## 固定文脈とTDD

元producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`、[詳細native run 38463246855](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38463246855)、実行source `6ac2c92191a62ada4830adeee5235fb3c4b90f15`。固定入力からnative layer finalをbit単位で再現し、元7段階のSHA不変を確認した参照を使う。report SHA `99529dc0…`と各binaryのshape / size / SHAを再照合した。

Python metadataのfeed-forward pre / post norm係数anchorは4 assertion red → 関連19 green。C#の5段階対応は4 assertion red → 既存を含む11 green。係数で確定したRMSNorm、平方平均 / epsilon / inverseの経路、2つのconstant-weight MatMulとSwishの接続を確認する。Relu・activation bypass・plain pre scaleを拒否し、元Modelの保存byte列を変えない。形や似た値だけからstageを選ばない。

## 統合係数と初回の未実行

最初のharness source `d7db2b6…`ではpost normの元係数が見つからず、CPU / GPUとも観測前に失敗した。NUnit0 pass / 2 fail / skip0だが、数値照合は未実行。再度CPUで小さい係数hashと45層の構造だけを調べた実行も、同じmapping failureで数値未実行。失敗ledgerを保持し、数値不合格へ付け替えない。

元ONNXからpost normの係数4096 bytesだけを読み、元SHA `57b626c2…`を監査した。Float32で0.5倍したSHA `1e6392c0…`は保存モデルのconstant102と一意に一致した。固定nativeデータでも`post norm * 0.5 + subsample input`はfeed_forward1出力へbit一致。保存モデルはこの残差倍率を係数へ統合していた。[小さいanchorとscale監査](results/m4-feed-forward-folded-anchors-20261014.json)。元重みや生成モデルをGitへ含めない。

診断readbackだけを2倍して元post normの尺度へ戻すhelperを5 assertion red → 既存と合わせ16 green / skip0 / CLI exit0で追加。借用配列を変更せず、nonfinite / subnormal / overflowを拒否する。任意倍率への復元ではない。統合係数を含むanchor manifest SHA `95f3e325…`をharnessで固定し、元post normの`rtol=1e-4` / `atol=1e-5`で比較する。半分の尺度へ許容差を当てて成功にしない。

## 完了した実測

修正後harness source `0cf171205ab1a2f09d12a7a5dbc61d5ea75014c2`、Runtime requested / resolved `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、Runtime overlayなし。Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor1個、uloopでCPU / 実GPUの対象2件だけを実行した。[全結果と失敗ledger](results/m4-feed-forward-stages-windows-20261014.json)。元ONNX `7770a8af…`、保存FP32 `3214cc2b…`、silence入力 `366de71c…`をCNG helperへ照合し、モデルを再取得・再変換していない。

| 段階 | layer / tensor | CPU 不一致数 / 最大許容差比 | GPU 不一致数 / 最大許容差比 |
| --- | --- | --- | --- |
| pre_layer_norm | 22 / 67 | 0 / 0.66380 | 0 / 0.61717 |
| ffw_layer_1 | 25 / 75 | 1 / 1.30831 | 0 / 0.94279 |
| act_fn | 26 / 77 | 0 / 0.69992 | 0 / 0.48900 |
| ffw_layer_2 | 30 / 88 | 0 / 0.46719 | 0 / 0.35123 |
| post_layer_norm（尺度復元後） | 36 / 103 | 284 / 6.15862 | 5 / 4.17788 |

linear2は追加のnorm-input clamp後のalias。native出力がその±1e10範囲内で変化しないことを確認した。post normは前記の固定係数統合に基づく2倍復元を明示する。両backendで5点の観測を完了し、通常Scheduleと観測Scheduleの最終featureはbit一致。診断readbackを本番GPU保持の証拠にしない。

CPUの今回最初の超過はlinear1、GPUはpost norm。GPUの前段4点が許容差内でも、微差が後段で増幅される可能性を残す。post normだけが原因とは確定しない。次は同じnative linear2入力によるpost norm単独を比較し、演算差と入力差の伝播を分けて最小修正へ進む。

NUnit0 pass / 2 fail / skip0 / CLI exit1、2026-10-13 20:42:35 UTC完了。`observation_completed=true` / `success=false`を維持。元全encoderの不合格、画像FP32差、全chain / role・境界、画像 / 音声sample、反復 / memory・実モデルPlayer / 必須実機・最終空consumer・配布準備は未完了。
