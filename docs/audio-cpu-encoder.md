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
固定snapshotの実モデル読込、各段のcapture・独立数値照合、実測CLIとCIへの接続が残ります。
既存の実encoderだけの120 captureの結果とも分けて扱います。CPU比較全体、
同じhardwareでのCPU/GPU性能、必須端末と正式公開は未完了です。

## 元PCMと参照の読込

`load_audio_cpu_reference_cohort`は、元の前処理参照・audit、encoder参照・audit、
Document最終参照の5ファイルを固定SHA256で照合します。全6 caseの宣言と
Document role、revision、snapshotの共通hash、元の許容差を検査します。
元のencoder auditに残る過去のONNX失敗は保持し、native referenceの完了と
全体のsuccessを区別します。

CPU測定の固定条件であるsilence / padded_noiseについて、実PCM・sample mask・
前処理features・frame maskとencoder入力の全bytesを読み、元manifestのsize /
SHA256と相互対応を照合します。返す配列はbytesを所有元とする読み取り専用で、
writeabilityを再設定できません。snapshotの宣言はencoder / backboneの両方から
統合します。元Documentはroleを各caseに明示し、metadataには繰り返していない
ため、全caseの明示roleを検査します。

[loaderのTDDと保持済み元データの実照合](results/audio-cpu-cohort-actual-tdd-20261020.json)
は、未実装19 errorsとschema対応の追加Redを経て関連159件成功・skip0、元の
2条件・snapshot宣言6ファイルの対応を確認しました。モデルgraph・重みの読込、
推論、CPU/GPU性能比較はこのloaderで行っていません。元入力の照合成功を
実モデルの数値合格へ置き換えません。
