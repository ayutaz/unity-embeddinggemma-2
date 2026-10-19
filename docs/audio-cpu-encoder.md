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

上記の合成moduleによるTDDは、実EmbeddingGemmaモデルの実行証拠と区別します。
初回の実モデルCIの結果と独立照合は末尾に記録し、公式CPU前処理への修正後の実測を残します。
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
宣言された全snapshot hashを照合し、公式CPU前処理、上記の5 session、
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
下記の入力配置と手動CIを実装し、初回の実モデルCIは末尾の成功・失敗を記録しました。
旧artifactが取得不能な場合に、参照・モデルを再生成して元のものとして置き換えません。

[測定・capture audit・CLIのTDD](results/audio-cpu-reference-tdd-20261020.json)は、
未実装8 errors、auditorの10 failures、CLI / admissionの3 failuresを経て、
新規21件・関連180件成功・skip0です。合成CPU moduleによる検証であり、実モデル
のPCM→最終vector・同じhardwareのCPU/GPU性能・GPU受け入れの証拠ではありません。

## 元バイト列と一致する手続き入力と手動CI

`tools/references/audio-cpu-original/`の3 JSONは、保持済みの元前処理参照・auditと
Document最終参照の変更していない309,947 bytesです。完全なhashと全テキストの
privacy検査を行い、Gitの改行変換も対象3ファイルだけ無効にします。大きいencoder
JSONと元のfeaturesは既存の固定公開配布から取得し、モデル・PCMをGitへ追加しません。

`audio_cpu_inputs`は無音と固定seed17のpadded_noiseのPCM・sample / frame maskを
CPUで再現します。これは入力の再構成であり、元artifactの取得と区別します。
元の演算順とpaddingの負のzero bitsを維持し、6 bufferすべての全size / SHA256が
元auditと一致することを必須にします。元参照やモデル、期待値は再生成しません。
参照featuresは固定公開配布の元バイト列を比較用にコピーし、実推論へ代入しません。

```sh
uv run --locked python -m embeddinggemma_tools.audio_cpu_inputs --encoder-reference ../artifacts/original --output ../artifacts/audio-cpu-inputs --fetch-missing-inputs --source "$GITHUB_SHA"
```

`--fetch-missing-inputs`は、元encoder参照のmanifestと固定公開配布の両方に対応する
2つの小さい入力bufferだけを補います。既存の不一致ファイルは上書きせず拒否します。
配置後に既存のcohort loaderも独立に元hash・shape・case・roleを照合します。

**Frozen native CPU context diagnostics**で`audio_cpu_measurement=true`だけを選ぶと、
2条件を別々のUbuntu jobで実行します。手続き入力の完全一致を確認してから、同じ
sourceの実5分割graphを一度だけ生成・元数値条件で検査し、実PCM前処理→encoder→
backboneの4回測定と保存captureの再監査を実行します。他の診断modeと同時選択すると、
依存導入やモデル取得の前に失敗します。各jobの上限は90分です。

privacy guardの成功後に数値capture・安全なJSON・取得receiptだけを保存します。
入力PCM、snapshot、graph、重み、raw logはupload対象にしません。途中失敗は
失敗のまま保持します。[入力配置とworkflowのTDD](results/audio-cpu-inputs-workflow-tdd-20261020.json)
は関連213件成功・skip0とactionlint成功です。実モデルの成功や性能全体の合格とは区別します。

### 診断ログと実入力の追加確認

5つのモデル関連commandの標準出力・標準エラーはrunner内の非公開scratchへ保存し、
公開logにはphaseと成功・失敗だけを出します。実commandの終了codeは保持します。
実際のBashで正常終了 / exit 7の両方を検査し、修正前の10失敗からworkflowの
19件成功・skip0、adapter・cohort・測定・既存画像/音声protocolを含む関連182件
成功・skip0を確認しました。scratch logはartifactに含めません。この確認は実モデル
推論の代わりではありません。

source `4bb9575` では2つの元featuresを匿名の固定公開URLから実取得し、手続き入力
6つと合わせた8 bufferすべてを保持済み原本とbyte比較しました。encoder JSONは
保持済み原本を配置したため、全入力を新規公開取得したconsumerの証拠ではありません。
3 JSONのGit blob・作業file・配置結果も元hashと一致しました。モデル実行はしていません。

[追加確認の記録](results/audio-cpu-private-logs-and-inputs-20261020.json)では、元の
入力・数値条件、公開取得と保持済み原本の区別、実モデル未実行の状態を保持します。

## 初回実モデルの結果と公式CPU参照への修正

source `9bd7a26` の[初回実モデル結果](results/audio-cpu-export-preprocessing-9bd7a26-actual-20261020.json)
は、2条件・各4回・全32 captureと実2 ZIPのdigest / CRC・私的な完全2コピーを
独立照合しました。無音は全段階で合格し、padded_noiseは前処理の元許容差内でも
encoderが各回33値・計132値の超過で不合格です。全8最終vectorは合格しましたが、
中間gateの失敗とrun全体のfailureを保持します。whole-process peak RSSは約10.8〜11.2 GBで、
SDK専有メモリや最低要件の測定ではありません。

初回はGPU向けの変換用`AudioPreprocessorExport`をCPU比較にも使っていました。
同じ元PCMによる小さい確認では、公式`Gemma4AudioFeatureExtractor`は2条件とも
元featuresの全bytesと一致し、変換用前処理はノイズで最大約6.44e-6の差がありました。
CPU参照側を`OfficialAudioCpuPreprocessor`へ変更し、元producerと同じ公式処理を
実PCMから再計算します。既存の入力検査を共有し、FP32 cast・mask乗算と負のzeroを
維持します。参照featuresの代入や、元参照・許容差・SDKのGPU前処理の変更は行いません。

実ローダーの選択を含む20件のRed→20件Green、前処理の種類をreportとsource journalに
明記して改変を拒否する追加3件のRed→関連205件Green・skip0を
[TDD記録](results/audio-cpu-official-preprocessing-tdd-20261020.json)に保存しました。
前処理単体の元hash一致と、修正後の実モデルによる全captureの合格は別の条件です。
修正sourceの実モデルCIと成果物の独立照合で後者を判定します。
