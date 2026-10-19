# 採用済みONNXのCPU比較経路

既存の[PyTorch CPU診断](reference-cpu-diagnostics.md)では、最終vectorが一致した後も元のencoder中間値に不一致が残りました。その失敗・元参照・許容差を保持し、採用済みの実ONNX演算をCPUで実行する追加の比較経路を用意します。SDKの依存やSentisのGPU推論は変更しません。

## 実行条件

`onnx_cpu_comparison`は既存のdistribution監査を先に実行し、元参照に対する合格と同じ40桁のexecution sourceを必須にします。元入力から接続した5分割・各graphの全SHA256・captureを確認し、ONNX RuntimeのCPUExecutionProviderだけを明示して実行します。graph最適化と実行providerのfallbackは無効です。別ファイルの重みはgraph hashだけでは保証できないため拒否します。

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
