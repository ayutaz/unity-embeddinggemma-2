# M4: PCM前処理RuntimeとGPU出力所有権

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

更新日: 2026-10-13。A3の製品Runtime接続として `AudioPreprocessor`と`AudioPreprocessingResult`を追加する。実音声encoder・最終768次元API・音声sampleは別工程であり、正式Release / 公開tagは作成しない。

## 所有権と処理境界

固定profileの `AudioPcmInput`から波形とsample maskをuploadし、監査済みの前処理Modelを実行する。Modelは呼び出し側が所有、RuntimeはWorkerと入力Tensorを所有する。CPU / GPUComputeを明示し、名前 / dtype / static shapeと2出力の順序を検査する。GPUではPCM uploadを入力境界とし、以後のfloat特徴量[1,1119,128]とint mask[1,1119]をCPUへdownloadしない。

結果はdevice copyを含むbackend上の独立したTensorを所有する。Workerの再利用・破棄から独立して保持し、結果Disposeで両Tensorを解放する。部分copy失敗時は作成済みTensorを解放する。借用先の推論が完了するまで結果を保持する。GPU上のfinite検査やmask全値検査をCPUへ持ち出さず、採用する監査済みgraphの独立参照検証で保証する。

## TDD

初回fixtureでGatherElementsにrank 1のindexをrank 2入力へ渡したため、graph構築で失敗した。これはRuntimeのredへ数えず、fixtureをIndexSelectへ修正してstubを維持した。後続はcompile成功、21 failed / passed 0 / skip 0 / inconclusive 0、全件が期待したassertion failureと保存XMLで確認した。最小実装後は21 passed / fail 0 / skip 0 / inconclusive 0、CLI exit 0。

小さいSentis graphで入力 / graph不一致の拒否、null / Dispose、出力type / shape違いの反復失敗、CPU / 実GPU出力の独立所有とWorker再利用・破棄後の生存、GPU特徴量と整数maskを次段Workerへ直接渡す契約を検証した。requested / actual GPUCompute一致、両WorkerのCPU計算 / 入力fallback・readback flushなし、最終scalarだけのCPU取得を確認した。結果保持の確認では別途、処理後の診断readbackを行っている。

これは新Runtime / test sourceを既存consumerのEditor test assemblyへ配置したTDDであり、配布Runtimeや実FFT / mel精度の成功へ流用しない。[source・fixtureエラー・red / green結果hash](results/m4-audio-preprocessing-runtime-tdd-20261013.json)。

## 固定Git Runtimeと公式6信号の受け入れ

固定Git `9f99414bfa36c03431d12a23198c378a7235a6fe`を既存consumerへ更新し、Runtime / unit source overlayを退避した。requested / resolved SHA一致、106配布ファイルの105 canonical Git blob一致 + package.jsonのUPM生成`_fingerprint`だけを除いたsemantic一致、不一致0を確認した。両公開型は実際の `EmbeddingGemma.Runtime`所属。compile成功、package assembly全108 passed / fail 0 / skip 0 / CLI exit 0。

実受け入れharnessは `AudioPreprocessorRuntimeReferenceTests`。固定source / model revision・processor hash・参照と既存の保存graphを監査した。CPU / 実GPU各6信号の全featureは元の固定 `atol=0.004` / `rtol=1e-4`内、mask完全一致、有効frame数一致。両backendの最大差 `0.0038890838623046875`、最大許容差比 `0.830127150420923`。次段Workerが全特徴量とmaskを要素ごとに加算した後、独立した参照と全要素照合する。中間downloadなし、実GPUの両WorkerでCPU計算 / 入力fallback・readback flushは0。次段の出力取得後にのみ、元の特徴量 / maskを診断readbackする。最終2条件ともpassed / fail 0 / skip 0 / CLI exit 0、2026-10-13 02:15:00 UTC完了。[source・実測と失敗履歴](results/m4-audio-preprocessing-runtime-windows-20261013.json)。

初回の実受け入れでは測定用の大きいReduceMeanのCPU scalarが `-5.92099857`、独立平均は `-5.9077553749084473`で不一致（CPU failed、GPU 6信号passed）。CPU側の元feature全要素比較へ到達しなかったため、前処理結果の不合格原因とは確定しない。Runtimeを変更せず、測定を全要素の次段加算・元featureの全要素照合へ変更した。元の前処理許容差は維持する。大きいReduceMeanの根本原因は未確定で、最終合格によってこの失敗を消さない。

Git更新時に一度短いSHAを指定して自動解決が失敗したため、完全な40桁SHAへ修正した。Burst削除警告はLaterで閉じ、Editor再起動なし。完全SHAへのResolveは受理後EOF / SafeToRetry=falseを記録して再要求せず、同processのReadyと解決hashを確認した。失敗応答はterminal toolで確認したがTee fileには保存されず、その欠落もreceiptに記載する。初回source監査scriptのcounter衝突も合格に数えず、修正後の完全監査だけを採用した。

既存consumer・embedded公式Test Framework 1.6.0・監査済みmodel / Library再利用の条件。harnessはconsumer Editor test assemblyに配置、製品Runtimeは固定Git packageから実行した。モデル再取得・再exportは行っていない。音声encoder・共通本体・最終vector、最終空consumer・Player / 他OSの成功には数えない。次はencoder数値差の修正と実encoderへのGPU接続を進める。
