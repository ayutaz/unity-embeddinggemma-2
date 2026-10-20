# 画像3条件のCPU比較とlifecycle

[固定条件の原本再照合](results/image-CLI-fixed-conditions-fe1cb9e-20261020.json)は、source `fe1cb9e` の
Windows CPU実行で3形状・各4回、全12回の元encoder・最終vectorのgateと正常終了を確認した記録です。
別processのcapture監査を再実行し、元report・生capture・process記録の保管2コピーを再hashしました。
snapshot・PNG・位置・descriptor・元参照は固定し、重み・期待値・許容差は変更していません。

## 条件ごとの明示的な設定

正方形には `sixteen-lane`、横長・縦長には `native` を指定しました。
設定は各実行の開始前にCLIで明示し、report / source journalにも結び付けています。
自動選択・失敗後の自動fallbackはありません。全形状が同じ集計設定で合格したとは主張しません。
変更は比較用Python toolsで、Unity SDKのSentis GPU実装には加えていません。

16 laneを横長へ適用した別試行は、最終vectorが通っても中間4要素が各回で超過しexit1でした。
その4回の元captureから超過を再計算し、失敗として残しています。その設定で縦長は未実行です。
既存native設定で合格していた横長・縦長を維持し、新しい入口でload / releaseまで測りました。
以前の正方形native、linear / pooling / RMS8 lane / attention matmulの診断失敗も取り消しません。

## 同じ入力・仕事量のFP32比較

CPUは同じWindows hostのRyzen 9 5900X / AVX2 / 4 thread / PyTorchです。
GPUは[既存の実測](windows-media-memory-results.md)のRTX 4070 Ti SUPER / D3D12 / Sentisで、
SDK source `7754ce4`、検証source `2cc7d94` を保持しています。
input・module・descriptor・Document参照のhashを照合し、初回とその後3回を同じ入力で測りました。

| module | CPU統計設定 | CPU load 秒 | CPU first 秒 | CPU warm 3回 秒 | GPU FP32 warm 3回 秒 | CPU release 秒 | CPU peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image-square | sixteen-lane | 17.322 | 20.443 | 18.753–19.593 | 4.623–4.949 | 0.240 | 3.121 |
| image-landscape | native | 14.330 | 15.717 | 15.381–16.383 | 5.253–5.375 | 0.197 | 3.121 |
| image-portrait | native | 13.296 | 17.915 | 15.915–17.894 | 5.098–5.305 | 0.226 | 3.121 |

loadはmodel構築、releaseはmodel所有参照を捨てた後のGCまでです。
whole-call wallはprotocolの確認等を含み、capture observer時間は除外しています。
各stage・process CPU時間はJSONに保持します。CPU / GPUは別framework・別processで、
一般入力・最適なCPU実装・純kernel時間の速度比には読み替えません。
peakはprocess lifetimeのWindows peak working setで、SDK専有RAMや最低RAMではありません。
release完了と正常終了を、allocatorの0 byte保証やGPU専有allocationの証明には数えません。

この記録は明示した3条件の比較です。CI経路、動作要件の最終整理、旧providerコピーの削除確認、
新tag consumerと正式SDK Releaseは別に残っています。
