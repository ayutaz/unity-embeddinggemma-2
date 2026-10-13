# M4: native音声参照と標準ONNX Swish候補

2026-10-13。A4の保存graph数値差に対して、native SiLUを保持するPython参照と標準ONNX SwishのSentis向け候補をTDDで追加した。実重みの元6信号、Sentis全encoderのCPU / 実GPU、最終768次元API / sampleは未受け入れ。正式Release / tagは作成しない。

## 経路と固定した条件

`export_audio_pt2`はSiLUだけをCore ATen分解から除外し、native kernelを保持する。これはPython数値gate用であり、Sentisへの直接Torch importを成功扱いしない。既存の全6信号・profile・公式SDPA参照、`rtol=1e-4` / `atol=1e-5`、cosine >= 0.999999を維持する。

ONNXはTorchScript exporterのopset 20で一時的なSiLU markerを作り、version converterで24へ変換し、標準Swishへ置き換える。最終graphは標準domainのみとし、checkerを通す。custom markerはexport中だけで、native runtime pluginは追加しない。export用temporary directoryは出力内に限定し、temporary symbolic registrationはfinallyで解除する。

`prepare-audio-encoder --native-silu`はnative保存出力を公式参照へ照合した後、保存ONNXを独立`ReferenceEvaluator`で元の全6信号へ照合する。export時と同じ入力だけで検証しない。不正shape / dtype / 非有限値、固定elementwise条件またはcosine不合格は失敗JSONにsource / profile / case / graph hashを残し、成功receiptを作らない。数値不合格の公式 / ONNX出力もartifactへ残す。

native graphとONNX graphをprofileで区別し、全caseのONNX cosine / 最大絶対誤差 / 許容差比を記録する。`onnx_structurally_validated`、`onnx_numerically_validated`、`onnx_case_count`、`unity_validated`を分ける。重複native graph weightsを解放してからONNX evaluatorを読み込み、profileごとに解放する。時間 / peak memoryの改善はまだ測定していない。

## TDDと範囲

- native保持と標準ONNX APIの4 assertion red → 4 green。小さい全encoderを公式出力と独立ONNX評価で照合。
- candidate生成 / CLI選択2 red、default export保持1 red → 関連58 green。以前のnative vector比較2 redも、独立native期待値と完全一致する。許容差は変更しない。
- 数値成功flagの欠落と、構造上正常でも出力を反転したONNXを許可する穴の2 redを確認。red checkpoint `46d7e3a1048df3c269a8e82ef8899bfa3ccbd45b` → 数値gate実装commit `ab8e00367f4e95ac815a5ab0ee5c2d2328c5d851`。小さいモデルの関連43 passed / failed 0、feed-forward回帰17 passed。初回回帰コマンドのtestファイル名不備は未実行として扱い、assertion redに数えない。
- mainとのCLI競合を解消し、audio / prompt / probe各入口の3件を確認した。単独Swish probe PR #57のWindows全4条件の過去の実測は別scope。

[小さいTDD receipt](results/m4-audio-native-onnx-tdd-20261013.json)。ローカルでは重みを取得・実行せず、tools/とuvで必要な小さいテストだけを実行した。初回のnative58件と追加後43 + 17件は別checkpointで、件数を合算して一度の実行と扱わない。

## 次の受け入れ

Actionsで元の実重み6信号を実行し、native保存参照と全encoder標準ONNXの数値結果を分けて判定する。PR #40の過去のsilence不合格・残り5信号未実行はそのsourceの記録として保持する。候補が失敗した場合も固定許容差を緩めず、最初の演算差を小さいredへ落とす。

Python全gate合格後も、Sentis import / 保存 / 再読込、全6信号のCPU / 実GPUと採用Float16形式、GPU前処理 → 実encoder → 共通本体の連結、最終API / 検索・再生・明示収録 / Player / 必須実機は別gate。小さいONNX greenや構造checkerを実モデル・GPU成功へ流用しない。
