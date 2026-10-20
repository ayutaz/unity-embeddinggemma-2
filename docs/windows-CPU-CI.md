# Windows CPU参照のCI経路

音声・画像のCPU参照をWindows hosted runnerで実行するworkflowを用意しました。
製品のSentis GPU実装とは別の比較経路です。今回の確認はローカルの関連134テストと
actionlint 1.7.12による構文検査で、GitHub上の実モデル実行はまだ行っていません。
[検査記録](results/windows-CPU-CI-TDD-20261020.json)と
[実行計画](next-release-plan.md)を参照してください。

## 音声

`.github/workflows/audio-cpu-windows.yml`はsilence / padded_noiseを別jobで扱います。
PR時と手動実行の既定は`native-fp32`で、元PCM→前処理→5区間encoder→共通backboneを
実行し、元の全数値gateと保存captureの独立監査を通します。
入力snapshotは元cohortが指定するファイルと固定model revisionを使い、CLIがhashを確認します。
相対パスでnative helperを作る際の出力先を絶対化し、compilerの作業ディレクトリに左右されない
ことを実ビルド・load・再利用の回帰テストで確認しました。演算順序・許容差の変更ではありません。

旧`onnx-cpu`経路は手動で選ぶ診断として残します。既定では実行しない旧経路を合格とは
数えません。`Required CI`の`model-reference`→`pinned-audio-acceptance.yml`による
元の数値検証も変更していません。native経路の記録では、元ONNX bytesを実行したとは主張しません。

## 画像

`.github/workflows/image-cpu-windows.yml`は、実行前に次の条件を固定します。

| profile | statistics reference |
| --- | --- |
| square | sixteen-lane |
| landscape | native |
| portrait | native |

固定元参照を取得し、画像→前処理→encoder→backboneの各4回と、別commandのcapture監査を
実行します。失敗後の自動設定変更はありません。全profile共通設定や他hostでの合格を推測しません。
全形状をsixteen-laneにした際のlandscape不合格は、
[元の条件別記録](image-CPU-fixed-conditions.md)に保持しています。

## 失敗と公開範囲

command失敗時は終了コードを保持し、数値未達を成功へ変えません。生のstdout / stderrは
非公開用logへ送ります。完了・途中経過のJSON / 数値captureだけを、同じpathのprivacy guardが
成功した場合にartifact候補とします。snapshot・モデル・native library・compiler log・生logは
upload対象へ含めません。画像の入力bytesは既存のhash固定・目視 / OCR済みfixtureを使います。

ローカルの実測値は[音声8回](native-audio-CPU-results.md)・
[画像12回](image-CPU-fixed-conditions.md)に実行sourceを付けて記録済みです。
hosted runnerは別hardwareであり、その時間を同一実機のCPU / GPU比較へ混ぜません。
CIの実行、必要checkの判定、公開・mergeは別の未完了条件です。
