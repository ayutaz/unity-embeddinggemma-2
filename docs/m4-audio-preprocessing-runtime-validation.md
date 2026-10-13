# M4: PCM前処理RuntimeとGPU出力所有権

更新日: 2026-10-13。A3の製品Runtime接続として `AudioPreprocessor`と`AudioPreprocessingResult`を追加する。実音声encoder・最終768次元API・音声sampleは別工程であり、正式Release / 公開tagは作成しない。

## 所有権と処理境界

固定profileの `AudioPcmInput`から波形とsample maskをuploadし、監査済みの前処理Modelを実行する。Modelは呼び出し側が所有、RuntimeはWorkerと入力Tensorを所有する。CPU / GPUComputeを明示し、名前 / dtype / static shapeと2出力の順序を検査する。GPUではPCM uploadを入力境界とし、以後のfloat特徴量[1,1119,128]とint mask[1,1119]をCPUへdownloadしない。

結果はdevice copyを含むbackend上の独立したTensorを所有する。Workerの再利用・破棄から独立して保持し、結果Disposeで両Tensorを解放する。部分copy失敗時は作成済みTensorを解放する。借用先の推論が完了するまで結果を保持する。GPU上のfinite検査やmask全値検査をCPUへ持ち出さず、採用する監査済みgraphの独立参照検証で保証する。

## TDD

初回fixtureでGatherElementsにrank 1のindexをrank 2入力へ渡したため、graph構築で失敗した。これはRuntimeのredへ数えず、fixtureをIndexSelectへ修正してstubを維持した。後続はcompile成功、21 failed / passed 0 / skip 0 / inconclusive 0、全件が期待したassertion failureと保存XMLで確認した。最小実装後は21 passed / fail 0 / skip 0 / inconclusive 0、CLI exit 0。

小さいSentis graphで入力 / graph不一致の拒否、null / Dispose、出力type / shape違いの反復失敗、CPU / 実GPU出力の独立所有とWorker再利用・破棄後の生存、GPU特徴量と整数maskを次段Workerへ直接渡す契約を検証した。requested / actual GPUCompute一致、両WorkerのCPU計算 / 入力fallback・readback flushなし、最終scalarだけのCPU取得を確認した。結果保持の確認では別途、処理後の診断readbackを行っている。

これは新Runtime / test sourceを既存consumerのEditor test assemblyへ配置したTDDであり、配布Runtimeや実FFT / mel精度の成功へ流用しない。[source・fixtureエラー・red / green結果hash](results/m4-audio-preprocessing-runtime-tdd-20261013.json)。

## 次の受け入れ

固定Git版のRuntime導入、source / 型assembly監査、package assembly回帰を確認する。その後、既存の監査済み `.sentis`と公式6信号を使い、CPU / 実GPU各6件の全feature許容差とmask完全一致、GPUの次段Workerへの受け渡しを検証する。期待値はRuntime出力から作らず、既存の固定Python参照を再利用する。モデル再取得・再exportは不要。

実受け入れharnessは `AudioPreprocessorRuntimeReferenceTests`。固定source / model revision・processor hash・参照と保存graphを監査し、最終scalar取得後にのみ診断用feature / mask readbackを行う。合格時も音声encoder・共通本体・最終vector、最終空consumer・Player / 他OSの成功とは記載しない。現時点ではこれらの追加受け入れは未実行。
