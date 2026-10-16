# 現在の音声encoder数値gateと保存モデルの途中観測

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-14 JST。Runtime requested / resolved Git `adccab24dbcc3f5dd8c8ac79911a7604a6014021`、Python producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`。現在の実PCM chainと同じ参照で、encoder単体の未実行gateを確認した。正式Release / tagは作成しない。

## 全6信号の現在のgate

固定harness `2b8870e`の`CurrentAudioEncoderReferenceTests`。既存FP32 full / partial保存モデル、native feature・mask、元のencoder reference / validationをSHA / size / source / profileへ照合した。重いgraphの再download / import / exportはない。独立native featureからテスト用preprocessing resultを作り、実AudioFeatureEncoder APIへ与える。PCM前処理・最終embeddingの試験ではない。

CPU / GPUComputeの両条件で元6信号が最後まで実行され、全12件ともcosine条件は合格、固定 `rtol=1e-4` / `atol=1e-5`のelementwise条件は不合格。NUnit0 pass / 2 fail / skip0 / CLI exit1、2026-10-13 18:30:39 UTC完了。GPUの計算 / input fallbackとreadback flushは0。[全結果](results/m4-current-audio-encoder-windows-20261014.json)。

| signal | CPU mismatches | GPU mismatches |
| --- | --- | --- |
| silence | 728 | 665 |
| impulse | 1228 | 945 |
| sine | 1162 | 764 |
| nyquist | 2279 | 881 |
| noise | 843 | 94 |
| padded_noise | 743 | 84 |

前のproducer `271f…`の実FP32不合格、別producer `bd9…`の参照drift、NumPy ONNXの不合格も保持する。現在の最終768次元cosineの全24件合格で、このencoder単体の不合格を解消したことにはしない。

## graphを変更しない観測

同じproducerのstage JSON2つを89,053 bytes / 10 range requests、silenceのnative subsample出力1個を1,162,361 bytes / 7 requestsで取得した。7,486,823,730 bytesのarchive全体やstage ONNXは取得しない。stage metadataのencoder reference SHAは現在の`ac23f399…`と一致し、original ONNX SHAも現在の`7770a8af…`と一致した。以前の不一致producerのstage参照は採用しない。

test-only `WorkerObservation`は、同じModelとinputを通常Scheduleで実行し、次にScheduleIterableで実行する。live Tensorのコピーだけを取り、Model outputs / layers・最適化済み演算を変更しない。途中readbackはこの診断の境界であり、本番chainのGPU受け渡しの合格には数えない。

API不足の2 assertion red → CPU / GPUの2 green。CPU shape layerはSentis iteratorがyieldしないため、実layer indexとずれる問題を、正しいShape / Reshape layerで4条件中1 assertion red / 3 passとして再現した。`m_LayerCPUFallback`を用いてyield対象へ対応付け、4 green / skip0。元のbaseline / final値と、コピーしたlive出力、Model layer / output不変を独立fixtureで確認した。

途中のFunctional.Cast / dynamic ReshapeのfixtureにはCS0117 / CS1503があり、assertion redへ数えない。旧署名の2 testだけが見える状態やNoTestsFoundも4条件の成功にしない。force compileのCOMPILE_RESULT_UNKNOWNも成功にしない。検証コードを直してEditorを回復後、実際の4条件とGPU shape失敗を確認した。vendor cacheやRuntime sourceは変更していない。

## 実保存モデルのsilence候補

固定harness `8ceef94`、Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / Direct3D12。CPU / GPUの2 tests passed / failed0 / skip0 / CLI exit0、19:10:23 UTC完了。[全観測](results/m4-current-saved-model-observation-windows-20261014.json)。現在のEditorはPID121512の1個。

両backendで、通常実行の最終featureと途中観測した実行の最終featureがbit一致し、Model layers不変だった。silenceの通常最終値のelementwise不一致はCPU728 / GPU665で、先の全6信号の不合格と整合した。

shape [1,280,1024]の先頭16個を観測した。最初の同shapeはlayer14 / tensor47のReshapeで、subsample投影前の候補として大きく異なる。次のlayer15 / tensor49のMatMulはnative subsampleと全値が固定許容差内、CPU最大許容差比約0.5718 / GPU約0.6476、不一致0。次のClipも同じ結果だった。形が同じTensorを一律に同じstageと判断せず、元ONNX名から保存layer indexへの完全な対応付けは別途必要とする。

これはsubsample候補と不変な最終経路の実測であり、encoder全体や全layerの数値成功ではない。次は元のoutput名・演算列とlayer indexを監査し、同じ保存モデルでlayer0以降を比較して最初の差を局所化する。元の固定許容差・6信号・別producerの失敗を維持する。

未完了: 数値差の原因修正、Float16 encoder単体、Raw / Query・入力境界、画像 / 音声sample、反復 / memory、実モデルPlayer・必須実機、最終空consumer、配布準備。工程1〜6の完了とはしない。
