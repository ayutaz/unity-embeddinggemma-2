# 分割backboneを含む実PCM全chainの検証

2026-10-14 JST。Runtime Git `66c72fb5277dc5c3b2897cf71da583413411bc77`、harness `9c5198cf8c14d5f0099f4a6a80354e862231bf08`、Python producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`を固定し、実PCM → AudioPreprocessor → AudioFeatureEncoder → 分割constant転送のMultimodalBackbone → AudioEmbedderを実行した。

## 今回の完了結果

FP32 CPU / GPUCompute、Float16保存weight CPU / GPUComputeの全4条件が各6信号で既存最終cosine条件に合格。返却は全24ベクトル、最小cosine0.999999800599847、最大絶対差7.79302790760994e-5。Document roleの完全なprompt ID / mask、要求・実backend、768次元・有限値・L2 norm、GPUでの前処理 / encoder / backbone間のdevice保持を確認した。元のFP32 >= 0.999 / Float16 >= 0.99を維持する。[全実測](results/m4-bounded-audio-real-chain-windows-20261014.json)。

前処理単独の段階診断はCPU / 実GPU各6信号の全12件に合格、mask一致。診断readbackはpublic API chainの前に完了し、API内部の途中readbackにしていない。GPUのCPU計算 / 非empty input fallback、readback flushは0。GPU profilerの計測とは別gate。

CLI / NUnitは4 passed / failed0 / skip0 / exit0、compile error / warning0。2026-10-13 17:34:24 UTC開始、17:37:07 UTC完了。result SHA `66633153552934ee3e414f44d395b7931309d8f452e3a2aae2019c545620b293`、CLI SHA `220cb969dfb9cd102c0e917d345035aaef98213e64e88e81708749bcc248d438`。前回のFloat16 GPU sine正規化失敗と後続未実行は、今回の同じ公式6信号では再現しなかった。1 runだけで根本原因の確定、長期安定性や旧失敗の消去とはしない。

## source・cache・環境

Runtime requested / resolved Git一致、Runtime source overlayなし。保存済み前処理 / full / partial encoder cacheを、producer / graph SHA / Unity / Sentis / quantization、file名・size / SHAの元TDD済み監査で再照合して全3群を再利用した。cache不足時は例外とし、重いmodelのreimport / `.sentis`保存を行う経路をこのharnessから除いた。最終backboneは元のsaved-model receiptと各file hashを使用する。tokenizer / processor / PCM / feature / maskも固定manifestへ照合し、モデルや大きい入力・出力をGitへ含めない。

Editor6000.3.21f1 / Sentis2.6.1、RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor本体PID149180の1つ、公式TestFramework1.6.0 embedded。既知の新規asset検出問題を避け、固定harness sourceを既存の自作test-only aggregateへ組み込んで実行した。これはRuntime overlayではない。fresh consumer・Playerの証明にはしない。

## 残る未達条件

reportのoverall `success=false` / `end_to_end_validated=false`は保持する。encoder Python ONNX gateは不合格、current producerのSentis encoder単独feature gateは未実行であり、API最終cosineの24件合格では置き換えない。

最終vectorの診断elementwise不一致は計1890あり、native特徴量 → backbone単独の不一致0とは異なる。固定 `atol=1e-5` / `rtol=1e-4`を変更せず残す。Raw / Query role・長さ境界、元encoder数値不一致、独立2起動・反復memory・正常終了、IL2CPP High stripping、sample / Player / macOS / iOS / Android実機、最終空consumerは未達 / 未実行。M4・工程6の完了とはしない。

GitHub Unity secretsのname一覧は今回0件。ActionsでのUnity実行・IL2CPP buildは未実行として保持し、必要な最小ローカル検証を準備する。正式Release / tagは作成しない。
