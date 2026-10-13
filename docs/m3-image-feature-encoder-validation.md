# 固定画像profileのencoder Runtime API

2026-10-14 JST。`ImageEncoderProfile` / `ImageFeatureEncoder` / `ImageEncodingResult`を追加した。公式exportは位置情報をgraphへ固定し、float `[1,2520,768]`の`pixels`だけを受け取る。APIは前処理の寸法・soft token数・全5040個の座標（padding `[-1,-1]`を含む）・tensor shape / backendを検査し、512次元のprojected soft tokensを返す。768次元の最終embeddingではない。callerは同じ位置情報を焼き込んだ監査済みmodelを供給する必要がある。constructorだけでgraph内部の座標の一致は証明できない。

WorkerはAPI所有、Model / preprocessing resultは借用。`CopyOutput`で独立した結果を作り、Worker破棄後も結果を利用できる。GPU pixel / soft tokenの途中CPU readbackは実装に含めない。現在のTexture前処理はGPU専用であり、このencoderのCPU合格をTextureからのCPU全chain合格にしない。

| 検証 | 固定sourceと結果 |
| --- | --- |
| 意図したred | test `833e7e0`をtest-only consumer assemblyへ配置、Runtimeは既存Git `9c6f7943ca6319849dd3db54611acc477e7d0981`。compile成功後、API不足のassertionで16 failed / 0 passed / skip 0。2026-10-13 15:31:05 UTC完了 |
| green | Runtime / package testsはGit `3c9bf7d616e0fccb4ec808280e1a59185da8e2b9`へ固定。重複するtest-only copyを削除し、requested / resolved SHA一致、Runtime overlayなし。16 passed / failed 0 / skip 0 / CLI exit 0。15:35:10 UTC完了 |
| package回帰 | 同じsourceで158 passed / failed 0 / skip 0 / CLI exit 0。15:36:56 UTC完了 |
| compile | 同じGit packageでerror 0 / warning 0 / CLI exit 0 |

小さい独立graphでsquare / landscape / portraitのfeature数、CPU / 実GPUのshape / backend / 値、Workerと結果の独立した寿命、借用input保持、寸法 / count / position違いの拒否、null / disposed inputとWorker、誤output shape、extra model inputを検証した。GPUはRTX 4070 Ti SUPER / Direct3D12、Editorは6000.3.21f1、Sentisは2.6.1、既存consumer `artifacts/k`。Editor本体はPID 149180の1つだけを起動した。公式Test Framework 1.6.0のembedded条件を保持する。

最初のpackage回帰はassembly名を誤指定して0件 / `NoTestsFound` / CLI exit 1だった。正しい`EmbeddingGemma.Package.Editor.Tests`で上記158件を実行した。0件のrunをgreenに数えない。

[機械receipt](results/m3-image-feature-encoder-tdd-windows-20261014.json)。実画像encoderのFP32不一致は未解消、実encoderでの新API検証・最終ImageEmbedder・sample / Player / 必須環境 / 最終空consumerは別gate。元Float16の実モデル成功を今回の合成graph成功へ付け替えない。
