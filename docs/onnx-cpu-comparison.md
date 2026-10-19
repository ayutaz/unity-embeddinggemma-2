# 採用済みONNXのCPU比較経路

既存の[PyTorch CPU診断](reference-cpu-diagnostics.md)では、最終vectorが一致した後も元のencoder中間値に不一致が残りました。その失敗・元参照・許容差を保持し、採用済みの実ONNX演算をCPUで実行する追加の比較経路を用意します。SDKの依存やSentisのGPU推論は変更しません。

## 実行条件

`onnx_cpu_comparison`は既存のdistribution監査を先に実行し、元参照に対する合格と同じ40桁のexecution sourceを必須にします。元入力から接続した5分割・各graphの全SHA256・captureを確認し、ONNX RuntimeのCPUExecutionProviderだけを明示して実行します。実行providerのfallbackは無効です。graph最適化は既定で無効です。CIの追加診断は、後述の区間単位の保持と最適化なしを明示して使用します。別ファイルの重みはgraph hashだけでは保証できないため拒否します。

各partへ渡すのは直前の実出力です。参照値は比較専用で、途中入力へ代入しません。最終partは元のencoder参照へ直接照合します。FP32の`atol=1e-5 / rtol=1e-4`を維持し、first / warm / repeat-1 / repeat-2のいずれかで超過したら失敗します。元reportとcaptureは上書きせず、新しい実captureと失敗reportを保存します。

依存はPython toolsの任意group `cpu-comparison`へ固定しています。CPU backendを使う小規模テストはこのgroupがなければ明示的なskipになり、CIのPython全4環境はgroupを指定して実行します。

```sh
uv --directory tools sync --locked --group cpu-comparison
uv --directory tools run --locked --group cpu-comparison python -m pytest tests/test_onnx_cpu_comparison.py tests/test_onnx_cpu_comparison_workflow.py -q
```

実モデルの最初の対象はCIのsilence 1ケースです。既存の全6信号・824,320値の元監査は維持し、そのsilence jobで元監査を通った実5 graphを追加実行します。追加stepは12分上限で、失敗を成功へ読み替えません。他5信号で追加CPU stepがskipされることも、CPU数値合格には数えません。新しいreportとcaptureも既存artifact privacy guardの対象です。graphや重みはuploadしません。

## 結果の範囲

準備段階の小規模テストは初回12件Red→12件Green、元監査への接続・外部重み拒否の追加8件Red→20件Greenです。CI接続も先に2件のRedを確認しました。追加の2件Redから既存reportの保護・実行後graph hash検査を修正し、最終の関連82件（実CPUの小規模23件・実hook関連6件を含む）が失敗・skipなしで成功しました。[TDD記録](results/onnx-cpu-comparison-tdd-20261019.json)を保持しています。実silenceでの成功は、実行後のreport・captureと元参照の独立照合が必要です。この文書の追加時点では実モデルのONNX Runtime実行を合格としていません。

計時はencoderの実CPU session呼び出しと入力・出力整合性チェックの区間です。参照比較・captureとreport書き込みを含めず、loadは別に記録します。GPU kernel時間やSDK専有allocationとの比較値にはしません。

reportの`encoder_replay_qualified`はこの1ケースのencoderに限ります。PCM前処理・backbone・最終768次元の連鎖、画像3形状、音声の他条件、全環境のCPU比較・性能全体・最低要件は別に残り、`CPU_reference_comparison_qualified`と`complete_performance_accepted`はfalseを維持します。

## Runner停止後の証拠保持

最初のCIと単一ジョブの再試行では、元参照回帰の実行後、追加CPU step中にrunnerの停止信号が記録されました。数値不一致やメモリ不足を確定できるログはなく、CPU captureも未回収です。同じsourceでの再試行は1回で止め、どちらの原本ログ・注釈も私的2コピーとhashで保持しています。元参照5ケースの680,960値は別に独立照合済み・超過0ですが、silenceを含む全6ケースやCPU全体の合格には数えません。

元の回帰成果物をCPU stepの前にprivacy guard経由で保存するよう順序を変更します。追加CPUのreport・captureは別artifactとして、専用のprivacy guard後に保存します。CPU失敗のhard gate、元6ケース・数値基準・入力・model係数は維持します。

CPU処理にはsource・case・part・roundと、全processのresident / peak memoryを記録するprogress journalを追加します。ONNXのparseとCPU session作成の前後、各実推論の前後を原子的に保存し、同じ許可済みfieldだけを直ちにstdoutへflushします。未知のmemory値はnullで、0や測定成功にはしません。path・例外本文はjournalへ記録しません。host全体が失われた場合、CPU artifactは依然として未回収になる可能性があるため、先に保存した元成果物とflush済みlogを使います。

journalのmemoryはSDK専有allocationや最低要件の証明ではありません。推論partの計時区間からjournal I/Oを除きますが、load区間は記録処理を含みます。数値合格・性能全体の未確認flagsは維持し、allocatorや演算順序をこの修正で変更しません。

この証拠保持の修正は[9件Red→関連92件Green](results/onnx-cpu-progress-preservation-tdd-20261019.json)で確認しました。実session作成失敗の直前journal、例外本文の非掲載、未知memoryのnull、原本の保護、推論計時からのjournal I/O除外を含みます。新sourceの実モデル結果は別途確認します。

## 実進捗に基づくメモリ保持設定の変更

source `4ae201b` のCI `39245552651`では、元の全6ケース・824,320値を独立に再照合し、許容差超過0を確認しました。先に保存した7 ZIPのdigest / CRC・私的2コピーと集計JSONの一致を確認しています。ただし追加CPU stepはrunner shutdownで中断し、Required CIを含むCI全体は不合格です。CPU artifactは未回収で、CPU数値合格とは扱いません。

flush済みの34件の進捗では、全5 sessionのロード後、第1区間の推論終了と第2区間の推論開始までを確認できました。その直前のresident memoryは10,075,226,112 bytes、観測OS peakは10,141,454,336 bytesです。以後のcaptureはなく、停止原因・OOMは未確定です。原本ログ・注釈はhashと私的2コピーを保持します。

次の実行では、CPU memory arena・memory patternを無効にし、`session.disable_prepacking=1`を指定します。[公式API](https://onnxruntime.ai/docs/api/python/api_summary.html)の先行確保・memory pattern設定と、[公式設定定義](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)の定数prepacking無効化を使用します。実sessionから読み戻した設定をreportに記録します。推論の全5 sessionを保持し、同じ元入力からfirst / warm / repeat-1 / repeat-2を接続する条件は維持します。途中参照の注入・partごとのcold reload・数値gateの緩和はしません。

この変更のメモリ削減量・実モデル互換性は未検証です。packing設定の変更による結果も、元の要素別許容差と全20実captureで判定します。小規模な実CPU sessionの4件Redから、密行列2段の実carry・4反復・既存privacy hook等を含む関連75件Green / skip0を確認しました。[TDDと失敗の記録](results/onnx-cpu-memory-policy-tdd-20261019.json)を保持しています。SDK / Assets、係数、元参照・許容差、backend / thread / graph最適化の指定は変更しません。実モデルの新source CIを確認してから統合します。

## 明示Basic最適化の追加診断

source `74f5d57` のrun `39251388877`も、最適化なしの第2区間推論開始後にrunner shutdownとなりました。直前residentは8,494,678,016 bytes、観測OS peakは9,982,025,728 bytesです。前sourceよりresidentは低下しましたが、host条件は異なり、原因・OOM・数値合格は未確定です。34件の進捗と原本ログ・注釈の私的2コピーを保持し、同じ設定の再試行は行いません。

元reportのoperator countsでは、第2〜第5区間がそれぞれ数十万演算を含みます。`--graph-optimizations basic`を追加し、定数畳み込みや冗長nodeの除去を許可する限定的な最適化を検証しました。[公式の最適化区分](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)に従い、Extended / Layoutは選択できません。既定の`disabled`は保持し、source `493f335`のCIは`basic`を明示しました。実sessionの設定読み戻しが選択と異なる場合も失敗します。

全5 sessionを保持し、元入力から全4反復を接続して全20出力を保存する条件、元の要素別許容差、元graph全hashの前後検査は維持します。Basic最適化の数値的な互換性も、実captureから判定します。元参照の変更、参照中間値への差し替え、partごとのcold reloadや最終値だけの合格判定は行いません。SDK / AssetsとUnityのGPU実装は変更しません。

初回9件Red（既存31件pass）と、実設定の不一致拒否の追加2件Redから、最終関連84件Green・fail0・skip0を確認しました。最適化なし / Basicの密行列2段・4反復とFP32の桁落ち対照、CLI・全5 sessionへの選択伝達、元証拠の保護を含みます。[TDDと停止記録](results/onnx-cpu-basic-mode-tdd-20261019.json)を保存しています。実モデルでのメモリ削減・全20出力の合格は未検証で、CPU比較全体・性能全体・最低要件・正式リリースも未完了です。

## 数値診断で保持するsessionを1区間へ制限

source `493f335` のrun `39255714739`は、Basicでも全5 sessionロード後、第2区間の推論開始を最後にrunner shutdownとなりました。34件の進捗の直前residentは14,337,609,728 bytes、観測OS peakは14,999,564,288 bytesで、数値capture artifactは未回収です。原因・OOMは未確定ですが、Basicによるメモリ改善は確認できませんでした。元の全6ケース・824,320値の超過0、7 ZIPのdigest / CRC・私的2コピーと集計JSONの一致は別に確認済みです。CI全体は不合格で保持しています。

`--execution-schedule partition-major`は、1区間をロードして4回実行し、native sessionの所有を解放してから次の区間へ進みます。4本それぞれの直前の実出力を別々に保持し、次区間へ渡します。元入力・graph・参照の全hashと元の要素別gate、実20 capture、最終区間の元参照への直接比較は維持します。遅れて読み込む元入力と最終参照も、使用直前にhashを再検査します。途中参照を入力へ代入せず、各区間は1回だけロードします。

既定の`chain-major`は、全5 sessionを保持して4本の連鎖を順に実行する従来方式です。CIの数値pilotは`--graph-optimizations disabled --execution-schedule partition-major`を明示します。どちらも全5区間・4本の実計算を維持します。新方式のfirst / warm / repeatは各ロード済み区間内の呼出順で、連鎖全体の連続warm時間ではありません。load時間も5回の区間setupの合計です。`end_to_end_warm_timing_qualified`・`CPU_reference_comparison_qualified`・`complete_performance_accepted`はfalseを維持します。連鎖全体の性能比較は別に残ります。

journalは区間ごとにロード、4回の推論、解放の順を記録します。成功時は82イベントです。native sessionの所有は最大1つですが、OS allocatorのメモリ返却は保証せず、解放前後のresident / peakを観測します。例外・数値超過時も所有sessionを解放し、原本・部分captureを保持して失敗します。

初回7件Red、CLI / CI接続の2件Red（先行8件pass）、遅延読込hash再検査の追加2件Redから、関連96件Green・fail0・skip0を確認しました。4本の異なる実carryが最終区間へ届くこと、微小な参照差を次区間で増幅する対照、失敗時の解放・原本保持、元graphの事後hash、82イベントの順序、推論計時からのjournal I/O除外を含みます。[TDDと停止記録](results/onnx-cpu-partition-major-tdd-20261019.json)を保持しています。これは小規模な実CPU / lifecycle検査で、実モデルでの新方式の数値合格・メモリ改善・正式公開は未確認です。
