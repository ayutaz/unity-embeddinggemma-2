# M4: native音声参照と標準ONNX Swish候補

2026-10-13。A4の保存graph数値差に対して、native SiLUを保持するPython参照と標準ONNX SwishのSentis向け候補をTDDで追加した。実重みの元6信号、Sentis全encoderのCPU / 実GPU、最終768次元API / sampleは未受け入れ。正式Release / tagは作成しない。

## 経路と固定した条件

`export_audio_pt2`はSiLUだけをCore ATen分解から除外し、native kernelを保持する。これはPython数値gate用であり、Sentisへの直接Torch importを成功扱いしない。既存の全6信号・profile・公式SDPA参照、`rtol=1e-4` / `atol=1e-5`、cosine >= 0.999999を維持する。

ONNXはTorchScript exporterのopset 20で一時的なSiLU markerを作り、version converterで24へ変換し、標準Swishへ置き換える。最終graphは標準domainのみとし、checkerを通す。custom markerはexport中だけで、native runtime pluginは追加しない。export用temporary directoryは出力内に限定し、temporary symbolic registrationはfinallyで解除する。

`prepare-audio-encoder --native-silu`はnative保存出力を公式参照へ照合した後、保存ONNXを独立`ReferenceEvaluator`で元の全6信号へ照合する。export時と同じ入力だけで検証しない。不正shape / dtype / 非有限値、固定elementwise条件またはcosine不合格は失敗JSONにsource / profile / case / graph hashを残し、成功receiptを作らない。数値不合格の公式 / ONNX出力もartifactへ残す。

native graphとONNX graphをprofileで区別し、全caseのONNX cosine / 最大絶対誤差 / 許容差比を記録する。`onnx_structurally_validated`、`onnx_numerically_validated`、`onnx_case_count`、`unity_validated`を分ける。重複native graph weightsを解放してからONNX evaluatorを読み込み、profileごとに解放する。時間 / peak memoryの改善はまだ測定していない。

## TDDと範囲

- native保持と標準ONNX APIの4 assertion red → 4 green。小さい全encoderを公式出力と独立ONNX評価で照合。
- candidate生成 / CLI選択2 red、default export保持1 red → 関連58 green。以前のnative vector比較2 redも、独立native期待値と完全一致する。許容差は変更しない。
- 数値成功flagの欠落と、構造上正常でも出力を反転したONNXを許可する穴の2 redを確認。red checkpoint `46d7e3a1048df3c269a8e82ef8899bfa3ccbd45b` → 数値gate実装commit `ab8e00367f4e95ac815a5ab0ee5c2d2328c5d851`。小さいモデルの関連43 passed / failed 0、feed-forward回帰17 passed。初回回帰コマンドのtestファイル名不備は未実行として扱い、assertion redに数えない。
- mainとのCLI競合を解消し、audio / prompt / probe各入口の3件を確認した。単独Swish probe PR #57のWindows全4条件の過去の実測は別scope。

[小さいTDD receipt](results/m4-audio-native-onnx-tdd-20261013.json)。ローカルでは重みを取得・実行せず、tools/とuvで必要な小さいテストだけを実行した。初回のnative58件と追加後43 + 17件は別checkpointで、件数を合算して一度の実行と扱わない。

## 次の受け入れ

初回の[実モデルCI 38362823692](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38362823692)はONNXのsilenceで不合格。head `05b48e6a2aa363d229f7eb18c163f9d5c19c94b4`とは別のproducer merge ref `cb7f2b00e8135d016c55ef8e00adfe9faafa387b`を保持する。cosine `0.9999998211860657`は合格でも、最大絶対差 `0.00013065338134765625`、許容差比 `3.971386671066284`、不一致292要素で固定elementwise条件は不合格。後続のONNX信号・全6信号のnative完了・Unityは未受け入れ。nativeの最初のprofileを通過してONNX照合へ進んだことは、保存JSONとproducerの処理順から分かる範囲に限る。

2,457,302,434 bytesのartifactを再取得せず、6回 / 1,551 bytesのHTTP rangeで失敗JSONだけを取得した。ZIPのCRC、producer、失敗status、JSON hashを確認。[実不合格receipt](results/m4-audio-native-onnx-real-failed-20261013.json)。次は同じONNX経路のstage traceを公式hook参照へ比較し、trace最終値が元の不合格graphと完全一致する場合だけ局所化を有効にする。trace追加や二次診断の成功で元の不合格を上書きしない。

ONNX stage診断は5 assertion redから実装し、hash helperのNameErrorによる初回3 failed / 6 passedを修正後に9 passed。さらに元graphとtrace最終値が一致してもnative参照がずれた文脈を許可する穴を1 assertion redで確認し、元の公式referenceへの完全一致guardを追加した。最終関連10 passed / failed 0。first stage・選択layer・全stage比較、元の失敗graph一致、native参照一致、二次診断成功 / 失敗時の元不合格保持を確認した。[診断TDD receipt](results/m4-audio-onnx-diagnostic-tdd-20261013.json)。NameErrorは実装中の失敗であり、意図したassertion red件数へ合算しない。

診断は不合格時だけ実行し、coarse graphの重複weightsを解放してから必要なlayer graphをexportする。traceが元の不合格graphと一致し、hook参照の最終値が元の公式referenceと一致する場合だけ`localization_validated`を有効にする。診断で`encoders_validated` / `end_to_end_validated`を成功にしない。実重みのstage診断結果は次のActionsで確認し、現在は未実行。

後続の[実stage CI 38365896076](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38365896076)はsilence不合格を保持したまま、coarse trace最終値が元の失敗ONNXと全値一致し、hook最終値も元の公式referenceと全値一致した。producer `6f1e14447d7ee2abb175d157de9b7567e6c3fc96` / head `75af5e1cba6506033a3348c136b2bbbc088b930b`。最初の非zero差・固定許容差超過はともにsubsample。shape [1,280,1024]、最大差8.96453857421875e-05 / 許容差比2.1150693893432617 / 562要素不一致。subsample内部の演算原因はまだ未確定で、conv / LayerNorm / ReLU / projectionの順に独立参照から分ける。

同runの最終featureは847要素不一致、最大差0.0002942085266113281 / 許容差比14.494515419006348。元のONNX graph hashは初回と同じでも、292要素の以前の結果と数値が異なる。差の変化を修正・性能改善・回帰原因の確定と扱わず、入力 / 参照 / 数学環境の差は未確認として保持する。3.68GB archiveから11,501 bytesだけを範囲取得し、CRC・producerとJSON hashを確認した。[実stage不合格receipt](results/m4-audio-onnx-stages-real-failed-20261013.json)。当時の診断TDD receiptの実重み未実行flagは過去のcheckpointとして残す。

subsample内部のmasked input → conv → channel-last LayerNorm → channel-first ReLUを2層、projection入力 / 出力へ分ける小さいtraceを追加した。公式moduleのhook参照、full / padded maskと変更入力、参照 / traceずれ、既存hook保持と例外時解除の6 assertion red、shape / dtype / 非有限値 / paddingの4 red → 関連16 green。coarseの元subsample値を次段へ渡さない2 assertion redから接続し、最終関連22 passed / failed 0。[subsample TDD receipt](results/m4-audio-subsample-diagnostic-tdd-20261013.json)。各checkpointの件数を一度の実行として合算しない。

coarseの失敗を保持してsubsample単独ONNXを保存し、単独最終値とfull coarseのsubsample値、hook参照とfull native値の完全一致を確認する。重複coarse weightsを解放してから小さいsubsample graphをexportし、次の比較に必要なinput / mask / native subsampleのみをbinaryで残す。単独diagnosticの成功・失敗を全encoder成功へ流用しない。実重みのsubsample演算別結果は次のActionsで確認する。

後続の[演算別CI 38370900217](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38370900217)は元のsilence不合格を保持。producer `714ee37935b357b8f6f0d1e9c5da9e564482ebc8`。単独traceとcoarseのsubsample最終値、native hookとcoarse native値は両方全値一致した。最初の微差はlayer0.norm（最大4.76837158203125e-06）、最初の許容差超過はprojection（562要素 / 比2.0606276988983154）。projection入力にも前段の差があるため、projection演算だけの原因とは確定しない。

同じ保存subsample ONNX（4,357,846 bytes）とinput / mask / native出力だけを合計6,084,157 bytesの範囲取得で既存Editorへ配置した。uloopでcompile成功、2 NUnit passed / failed 0 / skip 0 / CLI exit 0。Sentis FP32のCPU / 実GPUの両条件は元の固定許容差内。CPU最大差5.53131103515625e-05 / 比0.57182560736176347、GPU最大差4.291534423828125e-05 / 比0.64762583899927439、不一致とも0。GPU計算 / 入力fallback・readback flushは0。[実Sentis receipt](results/m4-audio-subsample-real-sentis-windows-20261013.json)。requested / resolved Runtimeは固定Git `8a7d2f0dea30be95a3158830fb0d9c9de895efc4`、Runtime overlayなし。受け入れharnessだけをconsumerのEditor assemblyへ配置した。前処理からの本番連結の成功とは数えない。

NumPy評価の不合格と、実Sentis subsampleのCPU / GPU合格を分ける。数式をNumPyの結果だけへ合わせて変更せず、次は元6信号の完全なnative参照を保持して全encoderの実Sentisを診断する。現在のRequired数値gateを通過したとは扱わず、未実行・失敗を隠さない。全encoder / 最終768次元 / API / sample / Player / 必須実機は引き続き未受け入れ。

### 全encoder診断へ渡すnative参照収集

`--native-silu --collect-onnx-failures`を明示した診断経路だけは、NumPy不合格を記録して元6信号のnative参照・各profileのONNXを最後まで保存する。native保存参照の不一致は即時拒否する。NumPyに不合格があれば最終`success=false` / `onnx_numerically_validated=false`、全caseの失敗ledger、非zero終了を維持する。`native_reference_complete=true`とUnity / encoder受け入れは別。Required model-referenceにはこのflagを付けず、既存の厳格なgateを変更しない。

別の`Audio native reference for runtime diagnostics` workflowはmanual、または同repoのPRへ`audio-runtime-diagnostics` labelを付けた場合だけ実行する。contents readのみ、Secretsなし、pull_request_targetなし。NumPy不合格のstepへcontinue-on-errorを付けずworkflowも失敗のままにするが、always uploadで診断artifactを保持する。小さいTDDはAPI / CLIの4 red、workflow欠落の1 assertion red → 最終関連18 passed。UTF-8 workflowを既定cp932で読む初回testの1失敗は修正し、数値のredに合算しない。[収集TDD receipt](results/m4-audio-native-runtime-reference-tdd-20261013.json)。実重みの全6参照はこの時点では未実行。

その後の[全参照収集run 38375956180](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38375956180)で、silence / impulse / sine / nyquist / noise / padded_noiseの全6信号、2 profile（280 / 210 soft token）のnative保存経路が固定elementwise条件とcosine条件に合格した。minimum cosineは0.9999997615814209。producer `271f5a12ca2aab081eb330f45cf593d404330374`とhead `b3961877234dc6e47d7ffd47093f2efa03468c72`を区別する。NumPyは全6信号で不合格のまま、validation success=false / workflow failure / 非zero終了を維持した。[全native参照receipt](results/m4-audio-all-six-native-reference-real-20261013.json)。これは全encoderのPython native参照完了であり、ONNX / Sentis / 最終embeddingの完了ではない。

実FP32 Sentisの全6信号をCPU / GPUで照合する`AudioEncoderReferenceTests`を準備した。選択したONNX・入力 / mask・referenceのhash、元の6 ID、native完了sourceを監査し、profileごとにimport / 保存 / 解放する。通常の数値不一致は各caseへ残して他の信号も評価し、全6信号の合格が揃うまでconditionを成功にしない。Float16とGPU前処理 / backbone連結は別gate。現在はcompile成功のみで、実モデルの転送完了後に実行する。

2026-10-13 10:00 UTCに実FP32 SentisのCPU / GPU × 元6信号の全12条件を実行した。compile errors / warnings 0、NUnit 0 passed / 2 failed / skip 0 / CLI exit 1。全12条件でcosine >= 0.999999は合格したが、固定`rtol=1e-4` / `atol=1e-5`のelementwise条件は全て不合格。[全encoderの実不合格receipt](results/m4-audio-full-encoder-fp32-sentis-windows-failed-20261013.json)。CPU最大許容差比15.800193893998493（sine）、GPU最大比13.610273396664804（impulse）。GPU全信号の実出力はGPUCompute、CPU計算 / 入力fallback・readback flush 0。backend実行と数値受け入れを区別し、cosineだけで全encoder成功にしない。

Runtime requested / resolved SHAは `8a7d2f0dea30be95a3158830fb0d9c9de895efc4`、Runtime overlayなし、Unity 6000.3.21f1 / Sentis 2.6.1 / RTX 4070 Ti SUPER / D3D12。最初のモデル転送は大きいrangeで進行後にURL期限切れ403となった。SHA / CRC検証済みの最初のONNXを保持し、残りは8 MiB × 最大4並列のbounded rangeへ切り替えて384.3秒で完了した。全archive・native `.pt2`は再取得せず、選択したONNXと入力 / mask / JSONだけを配置した。転送の観測停滞だけでprocessを再起動せず、進捗guardで停止を撤回し、実403終了後にURLを更新した。

次は実Sentisの段階別出力を元の不合格最終値と対応付け、最初の不一致を小さいredへ分離する。NumPy評価と実Sentis全encoderの両方が未達である。subsample単独の合格を流用せず、元6信号・Float16・前処理 / backbone連結・最終API / sample / Player / 必須実機のgateを維持する。

段階別のnative参照producerを6 assertion red → 6 green、CLI / workflowの未接続を2 red → 2 greenで追加した。不合格の元ONNX報告を保持する分岐も1件green。[TDD receipt](results/m4-audio-runtime-stage-reference-tdd-20261013.json)。source / revision / snapshot / 選択artifact hash、6 ID / 2 profile、finite入力とprefix maskを重み読み込み前に監査する。native各段階の最終値は元encoder参照へ完全一致し、traceのNumPy最終値も元ONNXの最終値へ完全一致しなければ完了報告を作らない。段階別native tensorはhash付きbinaryへ保存し、original ONNX数値判定を変更せず`audio_runtime_stage_context_only`として扱う。

opt-in workflowはnative収集が数値不合格でも、完了manifestがある場合だけalways条件でstage参照生成へ進む。収集stepの非zero終了は維持し、continue-on-errorを使わない。必須model-reference workflowは変更していない。次の実重み実行でこの文脈を確認する。新しいproducerのnative最終値を以前の固定参照と照合し、実Sentis trace最終値も元の実Sentis不合格へ対応付けてから局所診断へ進む。小さいcontext greenを実encoder数値合格へ流用しない。

後続の[実重みrun 38384143753](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38384143753)は元6信号 / 2 profileのstage生成stepとartifact uploadが成功した。CLIは`audio_runtime_stage_context_only` / `context_validated=true`、full encoder / Unity / end-to-endはfalse。producer `bd9cd68e978b05270bc4bd4b0c564a695a94f127`、head `ab3c0034766970f60ebcb4cfc59e3a7f0f8d91d0`を区別する。native収集stepはONNX不一致の非zero終了を維持し、workflow全体はfailure。[CLIとstepの実記録](results/m4-audio-runtime-stage-context-real-20261013.json)。この文脈保証は同runの元encoder参照 / ONNXとの一致であり、以前のproducer `271f…`の固定参照への照合、stage binaryのローカルhash監査、実Sentis段階診断は未実行。全7.49GB archiveを取得せず、次は必要なmetadata / binary / traceだけを選択取得して元の実Sentis不合格へ対応付ける。再生成labelは解除し、後続文書更新で重み生成を繰り返さない。

Actionsで元の実重み6信号を実行し、native保存参照と全encoder標準ONNXの数値結果を分けて判定する。PR #40の過去のsilence不合格・残り5信号未実行はそのsourceの記録として保持する。候補が失敗した場合も固定許容差を緩めず、最初の演算差を小さいredへ落とす。

### 以前の固定参照への文脈監査

stage生成runの7.49GB archiveから、4つの参照JSONだけを25,287,892 bytes / 22 requestで範囲取得した。ZIP64 / CRC、206 Content-Rangeと応答長、memberのパス / サイズを監査するtools/のCLIを6 + 5 assertion redから追加した。取得した新旧encoder参照とstage参照のSHAをmanifestへ照合した後、固定参照監査を4 + 2 red、欠落した値同士を一致とする経路の2 redから追加し、最終関連19 passed。[TDD receipt](results/m4-audio-artifact-and-context-tdd-20261013.json)。最初のimport collection errorはassertion redへ数えない。

実監査の非zero gateはCLI exit 1 / context_validated=false。新producer `bd9cd68…`のnative最終値は以前の`271f5a1…`へ全6信号で固定許容差を超過した。model revision・設定 / 重みhash・Torch / Transformers版・2 profileのmask hashは一致した。入力featureは5信号でhashが一致し、sineのみ異なる。元ONNXのhashも2 profileとも異なるが、protobufのhashだけで演算・重みの差の原因を確定しない。数学環境の比較は記録不足で未確認。[実文脈不合格](results/m4-audio-pinned-context-failed-20261013.json)。同run内のstage context成功を以前の固定参照への一致へ流用せず、trace graph / stage binaryの取得と実Sentis段階診断へ進めていない。元のFP32不合格は保持し、許容差や参照を都合よく置き換えない。

参照・計算環境の再現性は次の切り分け対象で、原因は未確定。これと独立したRuntime API・所有権 / profile・最終API / sampleのTDD実装は進められるが、実モデルの合格判定には数値gateと全chainの実測を別途必要とする。Required CIの[run 38385087239](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38385087239)、head `c7697f4…`もaudio ONNX silence不一致でfailure。小さい監査テストを理由にPRをmergeしない。

取得と固定参照gateはtools/からuvで実行する。モデル全体は以下のmetadata取得へ含めない。

```powershell
uv run --locked python -m embeddinggemma_tools.artifact_zip --repo ayutaz/unity-embeddinggemma-2 --artifact-id 11804593748 --output ../artifacts/runtime-stage-selected --member audio-runtime-stages/audio-stage-validation.json --member audio-runtime-stages/audio-stage-reference.json --member audio-runtime-reference/audio-encoder-validation.json --member audio-runtime-reference/audio-encoder-reference.json
uv run --locked python -m embeddinggemma_tools.audio_pinned_context --current ../artifacts/runtime-stage-selected/audio-runtime-reference --pinned ../../k/artifacts/audio-encoder-runtime --output ../artifacts/runtime-stage-selected/pinned-context-cli-failed.json
```

後者の例のpinned pathは、既存consumerを保持したこのworktreeの相対配置であり、新規環境では監査済みの旧固定参照の配置先へ合わせる。metadata gateの成功もgraph / stage binaryのローカル監査とUnity数値合格を意味しない。

Python全gate合格後も、Sentis import / 保存 / 再読込、全6信号のCPU / 実GPUと採用Float16形式、GPU前処理 → 実encoder → 共通本体の連結、最終API / 検索・再生・明示収録 / Player / 必須実機は別gate。小さいONNX greenや構造checkerを実モデル・GPU成功へ流用しない。
