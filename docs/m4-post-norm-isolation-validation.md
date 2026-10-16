# 固定native入力によるpost norm単独検証

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-14 JST。実feed_forward1全経路ではCPUのpost normで284要素、GPUで5要素が元の固定許容差を超えた。演算自体と入力差の伝播を分けるため、同じ保存モデルからpost normだけを取り出し、固定native linear2出力を入力にした。Runtime・golden・モデル・許容差を変更せず、正式Release / tagは作成しない。

## TDDと固定Sentisの入力扱い

抽出API不足の5 assertion redを確認。初回実装では構造・元Model byte列不変・不正入力拒否の3件が合格したが、CPU / GPUの2件はmissing input index3で失敗し数値未実行だった。

固定Sentis Worker.SetInputは入力listの位置をModelStorage.SetInputへ渡すため、内部indexのまま入力を指定できない。入力をindex0に置き換え、その入力を参照するSquare / Mulの2つだけを新しいinstanceへコピーする。他4演算と参照定数は借用し、元Modelのinputs / layers / arrays / 保存byte列を変更しない。PackageCacheは編集していない。修正後は同じ5件がCPU / 実GPUを含め5 green / skip0 / CLI exit0。

係数SHAと既存の平方平均 / epsilon / inverseの経路でinput / outputを確定し、前段を含まない6演算だけを依存から抽出する。不明な依存・不正anchor・空shapeを拒否する。元encoderの実行やreimport / 再変換を行わない。

## 実モデル結果

harness source `cb55875526de1e9a24d5a83ec0b8ac38cf373fb4`、Runtime requested / resolved `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、Runtime overlayなし。Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor1個でuloopから対象2件のみ実行した。[全数値とTDD ledger](results/m4-post-norm-isolated-windows-20261014.json)。

native参照は元producer `4edbe17…`を完全一致で再実行した[run 38463246855](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38463246855)、実行source `6ac2c921…`。固定report SHA `99529dc0…`、統合係数anchor SHA `95f3e325…`、native input / outputのshape / size / SHA、保存FP32モデル `3214cc2b…`を照合した。native linear2出力が元のclip範囲内で変化しないことも確認した。

sliceの入力は元88 → 新0、出力103。Square / ReduceMean / ScalarMad / Rsqrt / Mul / Mulの6演算、axisと統合係数の2定数、GPUで必要な定数payloadは計4100 bytes。0.5統合係数の診断出力だけを既存TDD済みhelperで2倍し、元post normの尺度で固定`atol=1e-5` / `rtol=1e-4`に照合する。

| backend | 不一致数 | 最大絶対差 | 最大許容差比 |
| --- | --- | --- | --- |
| CPU | 0 | 7.62939453125e-6 | 0.002807873134 |
| GPUCompute | 0 | 7.62939453125e-6 | 0.002807873134 |

両backendで実出力backend一致、GPUの計算fallback0。NUnit2 passed / failed0 / skip0 / CLI exit0、2026-10-13 20:57:49 UTC完了。これは固定native入力でのpost norm単独の合格で、全encoder・GPU全chainの合格ではない。元の経路不合格は保持する。

このsilence条件では、同じpost normへ正しいnative入力を渡すと条件内になるため、前段の入力差の伝播を次の主要な調査対象とする。GPUの前段出力がそれぞれ許容差内でも、その微差が後段で拡大し得る。全信号・全層の原因が確定したとは扱わない。次は固定native activation / pre norm入力によるprojection単独や高精度演算参照との比較から、原因を絞って最小修正へ進む。

元音声ONNX・Sentis encoderの不合格、画像FP32差、全chain / role・境界、画像 / 音声sample、反復 / memory・実モデルPlayer / 必須実機、最終空consumerと配布準備は未完了。
