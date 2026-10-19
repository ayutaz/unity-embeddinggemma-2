# 実CPU比較と元画像特徴量の回収

source `869cadd` の比較用CPU処理を記録します。Unityの製品側GPU経路と分け、
画像は公式processor・vision model・projector・backbone、音声は実export前処理・
選択済みnative encoder graph・backboneを順につないでいます。正解特徴量は
照合専用で、測定した推論の入力へ代入していません。

## 回収した画像の元データ

[元の画像roleアーカイブ](results/original-image-document-feature-recovery-20261019.json)
はartifact `11893990321` / run `38562190876` の実配信サイズ・SHA256と全CRCを
検証し、完全な非公開2コピーを確保しました。Document最終参照は元の
`90692ad3…` と一致し、その元auditに紐づく特徴量6ファイルも全hashが一致します。
元vision参照JSONの回収ではありません。現在のvision JSONの特徴量とは完全一致
しておらず、その参照世代差と実行結果を区別します。

## 画像3・音声2条件の結果

Ryzen 9 5900X、物理RAM 68,628,013,056 bytes、FP32・4 thread・MKL AUTOで
各4回を測定しました。同じ入力のfirstと続く3回です。loadは下表の推論時間に
含めず、各stageのwall / process CPU、loadとhost peakはJSONへ記録しています。

| 条件 | first 秒 | 続く3回 秒 | 中間encoderの許容差超過（各回） |
| --- | ---: | ---: | ---: |
| image-square | 14.342 | 13.798–14.277 | 2 |
| image-landscape | 14.198 | 14.510–15.012 | 3 |
| image-portrait | 14.734 | 14.760–15.161 | 1 |
| audio-1119 / silence | 3.930 | 3.240–3.485 | 240 |
| audio-839 / padded_noise | 3.184 | 3.044–3.253 | 136 |

[画像の元Document特徴量による再照合](results/original-document-image-cpu-869cadd-20261019.json)
と[音声の元参照による診断](results/audio-cpu-diagnostics-869cadd-20261019.json)で、
保存された最終768値を独立再計算し、全20回が元のcosine・単位長条件に一致
しました。ただし上表の中間値が元の `atol=1e-5 / rtol=1e-4` を超え、**5条件
すべてprocess exit1・未合格**です。元の期待値・許容差と先行runの失敗を保持し、
最終値の一致や測定の終了を条件全体の成功には数えません。

中間値の超過数は固定sourceのharnessによる記録です。この反復測定では生の
中間captureを保持しておらず、保存済み最終vectorの独立再計算と区別しています。
CPUの完全な数値受け入れ・Sentis CPU対応・性能全体・最低要件は未達です。

## 音声の境界診断

無音入力では、実export前処理と公式feature extractorの出力が、元の入力特徴量
とbit単位で一致しました。その元入力をencoderへ渡す1回の別診断でも、MKL AUTO
で240要素、保存時の指定設定COMPATIBLEで294要素の差が残りました。保存した
中間captureからこの超過数を独立再計算しています。診断processのexit0は測定の
完了を示し、モデルの数値合格ではありません。無音の差は前処理では説明できず、
CPU実行環境による差の原因は未確定です。

音声GPUの追加反復計測、残る実操作・権限・必須端末検証、旧provider成果物の
整理、正式SDK公開は別に残ります。追加録音は行っていません。公開するのは
照合済みの数値要約・hashだけで、録音・画像・重み・生vectorは含めません。

追加の[採用済みONNXのCPU比較経路](onnx-cpu-comparison.md)では、元監査を通った実5分割をCPUだけで接続し、元の許容差を維持します。PyTorchの失敗記録を置き換えるものではなく、まずsilenceの実数値検証をCIで行います。小規模テストだけでは実モデル・CPU比較全体を合格にしません。
