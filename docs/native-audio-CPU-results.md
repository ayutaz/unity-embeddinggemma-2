# 音声2profileのCPU参照とSentis GPU比較

2026-10-20の[原本再照合](results/native-audio-two-profile-CPU-readback-20261020.json)で、
無音のaudio-1119とpadded_noiseのaudio-839を、実PCMから各4回接続しました。
公式CPU前処理→固定係数のencoder→元backboneの順に実値を渡し、正解特徴量を入力へ代入していません。
元の前処理・mask・encoder・最終768次元vectorのgateは全8回で合格しました。
encoderの最大絶対誤差は全8回で0です。最終vectorも保存した生bytesから独立再計算しました。

## 実行経路と適用範囲

CPU実行sourceは `c458937` です。既存の[係数配布](native-regression-distribution.md)のPT2を再利用し、
実hash・metadata・元modelとの対応を確認してから、元の5分割をFXで実行します。
行列積・stem・FMAは[明示的CPU参照](cpu-fp32-reference.md)を呼び、元の係数を借用しています。
各profileで実matmul 684回・elementwise FMA 4,800回・stem 8回を記録しました。
同じ出力を再利用せず、4回とも実前処理から計算しています。

巨大なONNXの読み込みを省く経路であり、元ONNX bytesを実行したとは数えません。
hardware FMAと既存Dekker式が全入力で等しいとの保証も加えません。
今回は固定2profileで元のencoder参照とbit単位で一致した結果です。
Unity SDKにnative plugin・CPU fallbackは追加していません。

測定にはコミット済みの `reference_cpu_performance.measure_reference_pipeline` を使用しました。
実callbackをtimerで囲み、stage後のcapture保存を除外しています。
各回前後の入力signature、全4回の生出力、元の中間gate、正常終了、変化していないCPU contextと係数を確認しました。
元report / pipeline / 16 captureを各profileで2コピー保管し、元bytesと全hashを照合しています。
実行した非公開orchestrationのhashも記録し、公開済みCLIによる再現性の完成とは区別します。

## 同じ固定入力のwall time

CPUはWindows上のRyzen 9 5900X、4 thread、FP32、PyTorch参照です。
GPUは同じWindows hostのRTX 4070 Ti SUPER / D3D12 / Sentisによる[既存のFP32実計測](windows-media-memory-results.md)です。
input・descriptor・Document参照のhash、profileと仕事量を照合しました。
CPU / GPUは別process・別frameworkで、数値検査や各実装の制御処理も異なります。
一般入力の速度比、純kernel時間、最適なCPU実装との比較には読み替えません。

| module | CPU first 秒 | CPU warm 3回 秒 | GPU first 秒 | GPU warm 3回 秒 | CPU process peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| audio-1119 | 30.936 | 23.934–26.862 | 8.407 | 5.458–6.789 | 4.135 |
| audio-839 | 30.169 | 25.685–27.353 | 8.298 | 5.232–5.835 | 4.130 |

first / warmは同じ入力で、loadを含みません。表はstage合計にprotocolの確認等も含むwhole-call wallです。
各stage・observer時間・process CPUはJSONへ別に保持しています。
peak RSSはprocess lifetime全体で、モデル専有RAM・最低RAMの証明ではありません。
GPUメモリは前掲のPDH観測で、CPUのRSSと加算する値ではありません。

## 保持する失敗と残作業

旧sourceの変換上限停止、20分のONNX CPU実測停止、5分の単一区間loading停止は保持しています。
今回の最初の1回診断でも、検証orchestrationがsample maskをint32のまま渡して入力guardで停止しました。
原本を維持し、既存入口と同じint64変換を明示した後に、新しい試行として実行しました。
入力guardの停止はencoder数値の不一致とは扱いません。

CPUの独立したload / release計時は今回の記録にはありません。
公開可能な測定入口のTDD・CI統合、画像squareの中間不一致、動作要件の確定、
旧providerコピーの個人情報削除確認、新tag consumer、正式SDK Releaseは残っています。
音声2profileの数値とwall比較の成立を、性能全体・任意音声の品質・正式公開の完了には数えません。
