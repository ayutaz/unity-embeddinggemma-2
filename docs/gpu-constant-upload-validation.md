# 大きいGPU constantの分割転送

2026-10-14 JST。Sentis 2.6.1の`ComputeTensorData.Upload`は全wordを1回の`ComputeBuffer.SetData`へ渡す。保存済みbackboneの最大constantはFP32 512 MiB / Float16 256 MiBで、先のD3D12 logは16 MiB bufferに512 MiBを要求していた。`GpuConstantWorker`はWorker所有の既存constant Tensorを、8 MiB以下のraw int32 wordコピーでGPUへ転送してからscheduleする。Modelのinputs / layers / constants / weightsは変更しない。

定数を追加inputへ移す方式は採用しない。Sentis Workerのinput indexは連続したordinalを前提とし、ModelStorageへそのindexで設定されるため、元constant indexをそのまま追加inputへ使うと契約が崩れる。Workerのprivate `m_Storage` / `m_LayerCPUFallback`とModelStorageの`GetTensor`は読み取りで参照し、GPU buffer操作とTensorへの所有権移譲は公開APIを使用する。CPUが必要とするshape constantはGPUへ移さない。Sentis依存は2.6.1に固定し、保持用`link.xml`を追加した。IL2CPP High strippingでの実行はまだ未検証。

Float / Int / packed Short / Byteは数値castせず32-bit word単位でコピーする。chunk budgetは4-byte aligned、4 bytes〜16 MiB、既定8 MiB。CPUは通常Workerを使用する。holderがWorkerと転送先constantを所有し、例外時もWorkerを解放する。worker getterは破棄後拒否し、借用Modelは再利用できる。今回のRuntime接続は`MultimodalBackbone`のみ。TextEmbedder / ImageFeatureEncoder / AudioFeatureEncoderのWorkerを変更したことにはしない。

## TDDと固定Git回帰

| source / 検証 | 結果 |
| --- | --- |
| test `6c681bba517ab681dde1393bd5fd499bf222fd7c`、既存Runtime `10fa49c…` | API不足の13 assertion red / skip0 / CLI exit1 |
| impl `d7435fc…` | 12 pass / 1 fail。量子化直後のin-memory Half fixtureがschedule時`KeyNotFoundException: 2`。元の失敗receiptを保持 |
| test `16a8c251bbfece6f203e669165b6dda5eafb5c03` | Half fixtureを実運用と同じ保存 / 再読込経路へ通すと合格。新しい不正input index拒否は1 assertion red、既存13 pass |
| impl `494e58f3cee3b213d14b7c628e12bb4e12217a3a` | 14 pass / failed0 / skip0 / CLI exit0。input index不正をallocation前に拒否 |
| integration test `f98e89ae1a1126fa09c9a735070aa0b21e629a64` | 既定8 MiBを超える独立graphでCPU / GPUのbackbone接続2 assertion red |
| Runtime `66c72fb5277dc5c3b2897cf71da583413411bc77` | 関連16 pass / failed0 / skip0 / CLI exit0、17:21:49 UTC完了。package全253 pass / failed0 / skip0 / CLI exit0、17:24:34 UTC完了。compile error / warning0 |

Float / Intの値、負のinteger bit、odd length15の保存済みHalf model、chunk境界、3回連続schedule、GPU backendとfallback / readback flush0、借用Model不変、Worker所有権・Disposeを検証した。Byte転送の専用caseは未実行。小さい結果を512 MiB modelやPlayerの合格へ流用しない。

redは既存自作test-only aggregateを使用した。greenはrequested / resolved Git一致のpackage testsへ移し、aggregateを復元した。Runtime source overlayはない。元のin-memory Half fixture失敗を[receipt](results/gpu-constant-upload-initial-half-layout-failed-20261014.json)へ残し、半精度値やprojection・数値許容差は変更しない。

## 実保存backboneの全24ベクトル

Runtime `66c72fb…`、harness `e3cc8401417e9654f4947afdbb7d54eae3460a50`、Python producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`。元のTDD済みcontext / vector guardを変更せず再利用し、最終reference / validationとsaved-model receiptのSHA、および全feature / modelのsize / SHAを再照合した。再download、`.pt2` import、`.sentis`保存は行っていない。

CPU / GPUCompute × FP32 / Float16保存weightの4条件、元6信号 / 2 profileの全24ベクトルが合格。元cosine条件FP32 >= 0.999 / Float16 >= 0.99を保持し、診断elementwise `atol=1e-5` / `rtol=1e-4`も不一致0。最小cosine0.9999999999855401、最大絶対差7.171183824539185e-7、最大許容差比0.06616953785056714。[全実測](results/bounded-backbone-real-windows-20261014.json)。

各profileのGPU transferは、FP32で2 constants / 67 SetData calls、Float16で2 constants / 34 calls、最大要求8388608 bytes。CPUではupload0。GPU計算 / 非empty feature input fallback・readback flushは0。最大値はAPIへ要求したサイズの実測であり、native allocator peakのprofiler計測ではない。元の512 / 256 MiB constantを実際に転送・推論した結果だが、先のEditor終了原因を確定したことにはしない。

実行は2026-10-13 17:25:14 UTC開始、17:26:11 UTC完了。NUnit4 passed / failed0 / skip0 / CLI exit0、CLI17:26:12 UTC完了。result SHA `bf066e4138e2cc7bf2ebe0189ea17bee346ac2d79330c6e995b5de2a86e5d20d`、CLI SHA `fec0021c1f9d5f58a7705adae921e6b688c1314b0a9ddc0144e3b95a0d29a072`。Editor6000.3.21f1 / Sentis2.6.1、RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor本体PID149180の1つ、公式TestFramework1.6.0 embedded。

この実測はnative特徴量から最終backboneまで。PCM前処理・実encoder・AudioEmbedder全chain、元ONNX / Sentis encoder数値gate、Float16 GPU sineの正常化失敗、Raw / Query / 境界、反復memory・正常終了・Player / 必須実機・最終空consumerは残る。旧失敗を合格で上書きしない。正式Releaseは作成しない。
