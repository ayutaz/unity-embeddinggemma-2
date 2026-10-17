# ロード・解放時間の測定範囲

2026-10-18。SDK `d7b2233` / validation `10962fa` の既存実GPU記録を再監査した。[画像・音声10プロセス](media-gpu-submission-observations.md)と[text両形式2プロセス](text-current-sdk-observations.md)は、ロード・解放のCPU process時間とwallを保存している。[対応監査](results/load-release-measurement-scope-d7b2233-20261018.json)では12件の元receipt・raw evidenceのSHA256、終了code0・NUnit各1 pass / fail0 / skip0、固定554 sourceファイルを照合した。新しいモデル実行やGPU時間の取得ではない。

## ロードに含まれる仕事

`GpuMediaBundle.LoadAsync` はファイル検証・CPU上のモデルと係数読み込み・tokenizer構築・Worker構築を行い、その間の `Task.Yield` でEditorの処理を挟む。wallにはこの待機が入り、process CPUには他のEditor threadの仕事も入る。

ロードを「CPUだけの処理」とは分類できない。`GpuConstantWorker` は大きなGPU用定数を8MiB以下のraw-word単位で `ComputeTensorData` のbufferへ `SetData` する。画像・音声encoderの各segmentとtext/backboneにも使われる。独自operationの係数も `GpuBlockedMatmul` / `GpuSequentialScale` の `SetData`、`GpuFixedLeftMatmul` の `ComputeTensorData.Pin` によってGPUへ渡す。監査JSONには固定Gitのファイルhashと該当行を記録した。

一方、chunk以下の定数はこの事前転送を通らず、CPUで必要な定数も残す。ロード完了は全定数のGPU転送完了や最初の推論と同じ条件を意味しない。FP32 / Float16重みとも、ロードと初回推論を分けて記録する。CPU上の準備処理があることを、モデル推論のCPU fallbackとは扱わない。

## 解放の測定範囲

`GpuMediaSession.CloseAsync` は終了要求を出し、実行中の呼び出しがあればその終了を待つ。resourceは逆順にDisposeする。今回の12件は固定入力を4回完了した後のcloseであり、処理中キャンセルの遅延を測ったものではない。元テストでWorker・kernel・tensorの解放を確認していても、closeのreturn時刻はdriverによるVRAM返却時刻や全GPU queueの終了時刻の測定ではない。

## GPU時間は欠測のまま保持する

mediaのGPU observerはロード後にattachし、推論の送信区間を測る。textはロード中と各queryの履歴保存後にprofilerを無効化している。どちらもロード・解放を囲むGPU markerを取得していないため、12件すべてのload/release GPU時間をnullにした。GPU仕事が0だったという意味ではなく、CPU/wall・VRAM採取点から時間を推定しない。

次のGPU測定には、実際のuploadとresource退役の区間を識別する校正が必要である。`Task.Yield` をまたぐ全ロードを単一の同期markerで囲んだり、測定の都合でSDKのawait・fence・呼び出し順を変えたりして同じ動作の計測とはしない。この監査で範囲と欠測理由は確定したが、load/release GPU時間および性能全体の受け入れは未完了である。
