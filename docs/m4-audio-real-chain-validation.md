# 実PCMから最終音声APIへの候補chain検証

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

## 2026-10-14の追加状況

過去の最終cosine全24条件合格をencoder elementwise合格へ読み替えない。最新encoder候補は元12数値条件すべて不合格、全role / 入力境界 / 所有権・資源の全chain受け入れは未達。

## 当時の実測・実装記録

2026-10-13。固定Git Runtime `9c6f794…`、参照producer `4edbe17…`、harness `da5e8f0…`で、実PCM → AudioPreprocessor → AudioFeatureEncoder → 音声専用backbone → AudioEmbedderを実行した。Document promptの全ID / maskも公式参照へ照合する。**FP32のCPU / GPUとFloat16重みのCPUは各6信号が合格。Float16重みのGPUはsineの最終正規化で失敗し、全体未達**。[全caseと原失敗](results/m4-audio-real-chain-windows-failed-20261013.json)、[TDDと実行receipt](results/m4-audio-real-chain-tdd-and-execution-20261013.json)。

| 条件 | 最終vector | minimum cosine | 状態 |
| --- | --- | --- | --- |
| FP32 CPU | 6 / 6 | 0.9999998013927136 | 既存最終cosine条件合格 |
| FP32 GPUCompute | 6 / 6 | 0.9999998039708355 | 既存最終cosine条件合格 |
| Float16重み CPU | 6 / 6 | 0.9999998007713093 | 既存最終cosine条件合格 |
| Float16重み GPUCompute | 2件が返却・合格 | 0.9999999884216414 | sineで`Exported embedding is not L2 normalized`。後続3信号未実行 |

CLI / NUnitは3 passed / 1 failed / skip 0 / exit 1、compile error / warning 0。source / PCM prefix / 不完全な全case / actual backend / normと、未達のencoder gateを合格へ上書きしない監査は17 assertion red → 17 green。失敗時の非finite・有限値の大きさ・ゼロnorm診断は2 assertion red / 既存17 pass → 19 green。診断は原因確定やRuntime修正ではない。

PCM前処理は別の段階診断としてCPU / 実GPU各6件が固定 `rtol=1e-4` / `atol=0.004`で合格し、mask完全一致・elementwise不一致0。最大絶対差0.0038890838623046875、最大許容差比0.8301271504209226。診断readbackはpublic API呼び出しの前に完了し、GPU featureの本番受け渡しとは分ける。3 WorkerのGPU計算 / 非empty入力fallbackとflushは0。device契約・固定Runtime source・Worker setsの証拠であり、GPU profilerの新規合格とはしない。

元の診断artifact `11814567883`には前処理JSONだけがあり、model / PCMはなかった。2 encoderは監査・取得済みのまま保持し、同producerのRequired artifact `11813259359`から前処理を取得した。metadataと全PCM / feature / maskのhashは元JSONと一致するが、`.pt2` graph hashは異なる。そのartifact自身の固定manifest hash `4e325f70…`で監査し、上記の実CPU / GPU段階比較を行った。[staging receipt](results/m4-audio-real-chain-artifact-staging-20261013.json)。別graphを同一binaryとは扱わない。

大きいfileの監査はmain `92eda69…`にある未変更のPlayerFileHashをtest-only helperとして再利用した。既存7テストが実Windows CNGを含め合格。Runtime source overlayはない。モデル・failure tensor binaryはignored artifactsへ保存し、Gitへ含めない。

最終vectorの成功は既存cosine条件に基づく。診断elementwiseにはFP32 CPUで439、FP32 GPUで441、Float16 CPUで506、途中のFloat16 GPUで28不一致があり、backbone単独24件のelementwise不一致0と区別する。元ONNX encoderの全6不合格、新旧参照drift、現在producerのSentis encoder単独feature gate、Raw / Query role、sample / Player / 必須実機 / 最終空consumerは未達・未実行のまま。M4完了・正式Releaseとはしない。

次は許容差を変えず、同じ4条件でGPU失敗時の前処理・encoder・最終Tensorの有限性と大きさを捕捉する。単独backboneの合格だけで連続chainの失敗を解消済みとせず、再現・原因修正・元6信号の再照合へ進む。

## 2026-10-14 JST: 再実行の失敗とGPU環境回復

診断付きの同じ4条件は、既存`.sentis`の置換をWindowsが拒否した2件と、GPU前処理の不合格2件で0 passed / 4 failed / skip 0 / CLI exit 1となった。[保存した再実行結果](results/m4-audio-real-chain-diagnostic-rerun-failed-20261013.json)。元のFloat16 GPU sineを局所化した結果ではなく、最初の失敗記録は保持する。

4要素だけの独立probeではCPUの`log(0.001)`は正しく、GPUは全ゼロを返した。全試験がterminal・Play停止・scene cleanを確認し、PID 16260を正常終了してEditor本体0を確認後、同じ6000.3.21f1 / `artifacts/k`を1つだけ起動した。portable Editorはuloopの既定探索外にあり、実際の確認済みexeを起動した。新PID 193788でReadyとなり、同じ小さいprobeはCPU / GPUとも合格。これは環境の回復であり、GPU異常の根本原因やFloat16 chainの修正とは扱わない。[回復とcache TDDのreceipt](results/m4-audio-gpu-health-recovery-and-cache-20261014.json)。

再実行時に同じ重いmodelを再import / 保存する経路を改善した。source / graph / Unity / Sentis / quantizationと保存fileのsize / SHAを束縛したcache監査を5 assertion red / 既存19 passから関連24 greenへ進めた。完全一致したfileだけを再利用し、新規生成は一意の別名へ保存して以前のfileを保持する。暗号学的監査は省略しない。性能改善の倍率や重いimport再利用の実測はまだない。数値許容差・M4全体の完了条件は変更していない。

### 回復後の実chainはEditor終了で未完了

回復後の実行は2026-10-13 15:08:42 UTCに開始したが、CLIが`UNITY_DISCONNECTED_AFTER_ACCEPT` / `Cause: EOF` / exit 1を返した。`uloop status`の`NotRunning`とUnity本体process 0を確認したため、単なる観測timeoutとは区別する。保存された[途中結果](results/m4-audio-real-chain-recovered-incomplete-20261014.json)には`completed_utc`がなく、FP32 CPUの6件は完了・合格、FP32 GPUは5件を保存して`evaluate_padded_noise`で途切れている。Float16 CPU / GPUはこの実行では未実行。NUnit完了集計はなく、GPU条件や全4条件を合格に数えない。

途中結果のSHA-256は`3629757438fdc97c413a7f902848f11843fc00e34ac5e9db2c124797f8abd939`、ignored CLI receipt `artifacts/audio-real-chain-unity/run-tests-recovered.json`は`d6af11528a86762818d1bc7d69ddadd591a34d31e13a7e236152b3a0f894454a`。Editor logには`d3d12: upload buffer was too small for the requested resource! Size: 16777216. Requested: 536870912.`がある。16 MiBのbufferに対して512 MiBを要求した証拠であり、数値不一致や終了の根本原因を確定したものではない。手動終了の有無も未確認。元の失敗を保持し、ComputeBufferへの転送と大きいconstantの扱いを調べてから重い再実行を判断する。texture / mesh向けのasync upload設定をこの転送の修正として扱わない。

### 最新Required CIでもencoder数値gateは未達

head `d1c0d8c`の[CI 38422572898](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38422572898)は`Validate real audio encoder and projector against official PCM-derived features`で`ValueError: audio ONNX export mismatch: silence`、exit 1となり、Required CIも不合格。Python全4環境、package、actionlintは合格した。元の数値gateの再現であり、network / runnerの一時障害として再実行しない。native-reference jobのskipを実モデル合格に数えない。このPRはDraftで保持する。
