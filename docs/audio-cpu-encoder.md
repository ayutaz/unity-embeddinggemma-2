# 音声の比較用CPU encoder接続

`AuditedAudioCpuEncoder` は、監査済みの5分割ONNXを、比較用の
`CpuReferenceCase` へ接続する所有者です。Unity packageに依存を追加せず、
製品のSentis GPU経路も変更しません。

ロード前に既存の`audit_distribution_report`が元数値条件・配布graph世代を
確認します。指定source、caseとprofile、5分割の順序・実graphの全hashと
静的shapeを検査します。ONNX RuntimeはCPUExecutionProviderのみ、
priority順・最適化なし・4 intra-op / 1 inter-op threadです。既存のCPU
sessionがarena、memory pattern、prepackingとfallbackを無効にします。

`forward`へ渡すのは実際のPCM前処理が返したCPU / FP32の
`(1,1119,128)` tensorです。5 sessionへ実出力を順に渡し、fullでは
`(280,512)`、paddedでは`(210,512)`を返します。参照配列を推論入力へ
代入しません。元参照はadmissionの監査だけに用い、adapter内へ保持しません。

`with`または`close()`でsession所有権を解放します。途中のロード失敗でも
既に生成したsessionを閉じます。ロード中の元report変更、閉じたownerの
再使用、不正な入力tensorを拒否します。解放はnative session所有権の終了で、
OSへの全メモリ返却を保証するものではありません。

## 検証範囲と残件

[TDD記録](results/audio-cpu-encoder-tdd-20261020.json)は24件の未実装失敗、
boolean indexの追加1件失敗 / 25件成功から、関連137件成功・skip0を記録します。
実CPUExecutionProviderを使う小さい5 graphと、合成の前処理・backboneを
既存の4反復測定へつなぎ、出力の受け渡しと解放を確認しました。

このadapterで実EmbeddingGemmaモデルを動かした証拠ではありません。
元PCM・sample/frame mask・元前処理と最終参照のhash照合、固定snapshotの
実モデル読込、各段のcapture・独立数値照合、実測CLIとCIへの接続が残ります。
既存の実encoderだけの120 captureの結果とも分けて扱います。CPU比較全体、
同じhardwareでのCPU/GPU性能、必須端末と正式公開は未完了です。
