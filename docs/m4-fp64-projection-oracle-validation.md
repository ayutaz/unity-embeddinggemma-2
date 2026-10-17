# 固定native入力によるFP64 projection比較

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

2026-10-14 JST。保存Sentisのpost norm単独は同じnative入力で固定条件に合格したため、前段の微差と後段での増幅を調べる。固定native pre norm / activation入力と同じFP32重みから、線形積和だけをFP64で計算しFP32へ丸めた候補を比較する。Runtime・モデル・golden・許容差を変更せず、正式Release / tagは作成しない。

## TDDと比較の範囲

FP64 helper不足6 assertion red → 6 green。独立した桁落ちfixture、公式input / output clip、借用入力 / weight不変、dtype / shape / nonfiniteを確認した。比較API不足3 assertion red / CLI flag不足1 argparse red、producer flag不足1 TypeError redから実装し、既存を含む30 green。専用workflow条件1 assertion red → 1 green。ローカルは小さいfixtureのみ、tools/とuvを使用し、モデルの取得・実モデル実行は行っていない。

FP64計算はunbiased CPU FP32入力 / weightに限り、公式clippingを維持する。Gemma4RMSNormは内部でFP32へcastするため、単にlayer.double()を使ったものを全演算の高精度参照とは呼ばない。今回の候補はlinear1 / linear2のFP64積和と、FP64 linear2を入力にした元FP32 post normである。

比較前にlinear1 / linear2 / post normのFP32再実行が保存native値へ完全一致することを要求する。元の全12ファイルはshape / size / SHAを保持し、候補3ファイルを別名で保存する。変更されたbaselineを新しいgoldenとして採用しない。

固定`rtol=1e-4` / `atol=1e-5`で3候補を比較する。1候補でも外れれば`fp64_projection_gate_passed=false` / CLI success=false / exit1とし、失敗を通常成功に変えない。native文脈の一致と候補の数値合格は別fieldへ残す。候補が合格してもfull encoder / Unity / 全chainを合格にしない。

## 実モデル実行

専用Actionsで固定artifact `11814567883`のmanifest / native layer input / finalだけをrange取得し、元SHAを監査後に同revisionのsnapshotを取得する。候補のモデルweightや重いexportを成果物・Gitへ含めない。

実モデルの候補結果はまだ未取得。結果を確認してから精度修正候補を選ぶ。元音声ONNX・Sentis encoderの不合格、画像FP32差、全chain / role・境界、sample・Player / 必須実機、最終空consumerと配布準備は未完了。

## 固定実モデルの結果

[Draft PR #80](https://github.com/ayutaz/unity-embeddinggemma-2/pull/80)、head `fd228343882a49cc1e292bcacdcc21ec19b6e3eb`の[run 38468940928](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38468940928)は成功。実行source `3a409fee32cda13ded99c96ca9c6c701df8929ba`、report SHA `1e17becd…`。native layer final・3つのFP32 projection baselineはbit一致。元12ファイルのSHAは前回と全一致し、別のFP64候補3ファイルを含む全15 binaryのsize / SHAをダウンロード後に照合した。[小さいreceipt](results/m4-fp64-projection-actions-20261014.json)。

| 候補（FP32へ丸めて比較） | 不一致数 | 最大絶対差 | 最大許容差比 |
| --- | --- | --- | --- |
| FP64 linear1 | 0 | 2.86102294921875e-6 | 0.07906450 |
| FP64 linear2 | 0 | 1.9073486328125e-6 | 0.02833834 |
| FP64 linear2 → 元FP32 post norm | 0 | 2.288818359375e-5 | 0.22493946 |

このfixed silence / native exact-input条件では、高精度の線形積和を元FP32参照に照合しても、post normまで固定条件内だった。Sentisの実入力誤差を含む全経路に適用した結果ではなく、全encoder修正や全信号合格にはしない。次はGPUでも使えるFP32の分割積和・補償和を同条件へ比較し、実Runtime候補に進める。元の全経路不合格と全残作業は維持する。
