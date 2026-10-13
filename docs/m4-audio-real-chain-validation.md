# 実PCMから最終音声APIへの候補chain検証

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
