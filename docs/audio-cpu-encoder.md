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
固定snapshotの実モデルによる実測CLIの実行と、CIへの接続が残ります。
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

## PCMから最終vectorまでの測定入口

`audio_cpu_reference`の`measure`は、固定sourceの実行ファイル18件、元データ、
元数値gateを通った同じsourceのchain-reportを確認してからsnapshotを取得します。
宣言された全snapshot hashを照合し、実AudioPreprocessorExport、上記の5 session、
実MultimodalEmbeddingExportを読み込みます。モデル読込と4回の推論は分けて記録します。
CLIはTorch 4 intra-op / 2 inter-op thread、ONNX sessionは4 / 1を使用します。

実PCM前処理のfeaturesとframe maskを次のconsumer前にsnapshotし、実encoderの
出力をbackbone前にsnapshotします。コピーの時間は測定区間から除外します。
ONNX wrapperの入力検査・hash検査とTensor / NumPy境界はencoder callbackの
時間に含まれ、検査なしのCPU kernel時間として扱いません。途中失敗では取得済み
captureを保持し、最終vectorだけ一致する場合も中間gateの失敗を維持します。

`audit`はモデルを読み込まず、元データを再照合し、保存した全4回のfeatures・mask・
encoder・最終vectorのhash / shape / 元数値条件を再計算します。source journal、
CPU設定、実行source、前後のcontextを照合します。モデルgraphはaudit成果物に
含めないため、保存captureのauditorがgraph binaryを独立に再hashしたとは記録しません。

```sh
uv run --locked --group cpu-comparison python -m embeddinggemma_tools.audio_cpu_reference measure --preprocessing ../artifacts/original-preprocessing --encoder-reference ../artifacts/original/audio-runtime-reference --document-reference ../artifacts/original-document/audio-embedding-reference.json --chain-report ../artifacts/pinned-audio/frozen-audio-chained-onnx.json --case silence --output ../artifacts/audio-cpu-silence --source "$GITHUB_SHA"
uv run --locked --group cpu-comparison python -m embeddinggemma_tools.audio_cpu_reference audit --preprocessing ../artifacts/original-preprocessing --encoder-reference ../artifacts/original/audio-runtime-reference --document-reference ../artifacts/original-document/audio-embedding-reference.json --output ../artifacts/audio-cpu-silence --source "$GITHUB_SHA"
```

これは配置済みの元データ・同一sourceのgraph/reportを前提とする入口です。
元データの公開取得経路・CI接続と実モデル実行は未完了です。旧artifactが取得不能な
場合に、参照・モデルを再生成して元のものとして置き換えません。

[測定・capture audit・CLIのTDD](results/audio-cpu-reference-tdd-20261020.json)は、
未実装8 errors、auditorの10 failures、CLI / admissionの3 failuresを経て、
新規21件・関連180件成功・skip0です。合成CPU moduleによる検証であり、実モデル
のPCM→最終vector・同じhardwareのCPU/GPU性能・GPU受け入れの証拠ではありません。
