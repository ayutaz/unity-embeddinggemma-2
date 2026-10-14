# 工程1〜3の作業基準と受け入れ

> 文書整合確認: 2026-10-15。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。ユーザー指定の実行計画3・4は継続中、未完了。直近の所有Unity processは終了済み。CPUを含む旧実測・コマンドはGPU専用の現行受け入れ結果とは区別する。

2026-10-15現在: 元280行のSGEMM文脈とblock192候補が入力投影を全bit再現し、実GPUもstatic64の249,203 bit差→block192の全bit一致を確認した。元golden / coefficient byte / 許容差は維持。全encoder`43c252d`は無音295要素・最大許容差比7.637025で不合格、GPUCompute・計算fallback / flush0、NUnit0 pass / 1 fail / skip0。CPU / offline診断では最初のfeed-forwardの両投影もblock192で全bit一致したが、そのGPUは未実行。[固定結果と限界](results/m4-block192-input-projection-gpu-and-whole-failed-20261015.json)。Python全体はde3a3b0の4環境各793 pass / skip0まで確認。[固定回帰](results/python-full-scalar-mean-rewrite-green-20261015.json)。工程3・4と全信号 / profile / 最終API条件は未完了のまま。以下は前回checkpoint。

2026-10-15現在: block16リセットConv2と8-lane scalar平均FMAのGPU局所Red→Greenを確認し、`de3a3b0`の実全encoder内でもlayer0.conv / norm / act、layer1.conv / norm / actが元参照と全bit一致した。次の微差はinput_proj_linear249,203要素・元許容差超過0。全encoderは無音417要素不合格、GPUCompute・計算fallback / flush0、NUnit0 pass / 1 fail / skip0。0件 / exit0の初回filter失敗は未実行として保持し、実行XMLを別監査する。[局所とTDD](results/m4-source-order-conv-and-scalar-mean-20261015.json) / [固定全体失敗・負荷](results/m4-scalar-mean-full-gpu-and-load-analysis-20261015.json)。Python全体はb4b817aの4環境各787 pass / skip0まで確認、後続sourceは別gate。[固定回帰](results/python-full-source-order-conv-and-scalar-green-20261015.json)。工程3・4と元全信号 / profile / 最終APIの全範囲は未完了のまま。以下は前回checkpoint。

2026-10-15最新: 元runのOS / image / OMP / MKLを追跡し、既存のAVX512保存native再生の完全一致を確認。新しいConv2 CPU Actionsはnative条件もVMのAVX512 flagsが公開されずruntimeAVX2・268,142値bit差・元許容差超過0で失敗。元goldenを置換せず、ISA不足をdownload前に拒否するpreflightを追加して関連40 Green。[固定CPU診断](results/m4-saved-conv-cpu-context-unavailable-20261015.json)。前回source `fcaf611` のPython全体回帰は4環境各767 pass / skip0。[固定回帰](results/python-full-compensated-conv-green-20261015.json)。今回GPU実行はなく、元encoder / API全範囲と工程3・4未達を維持。次はpinned oneDNN3.12 AVX512生成コードから演算順の証拠を取得する。以下は直前checkpoint。

2026-10-15最新の候補: `dbfa061` のConv64実GPU全encoderは219要素不合格、最大許容差比6.78027、計算fallback / flush0、NUnit0 pass / 1 fail / skip0 / exit2。旧201要素版より悪化しており不採用。元係数byteを保持するrewriteとFP16拒否は関連68 Green。[固定候補失敗](results/m4-compensated-conv-full-gpu-failed-20261015.json)。現在CPUのmode / oneDNNと42種類のFMA順でも元conv2のbit再現は未達。次は参照host / kernel文脈を調べ、小さいGPU成分の忠実性を先に確認する。前回契約修正の全体Pythonは4環境各744 pass / skip0、今回の新APIの全体回帰とは区別。[前回sourceの全体回帰](results/python-full-regression-contract-green-20261015.json)。工程3・4、画像全profile・音声全信号 / 両profile・最終API全条件は未達のまま。以下は直前checkpoint。

2026-10-15最新観測: compactの元12層はlayer_9から超過。layer_9抽出は観測値をbit再現し、元入力なら超過0。入力側7段階は最初のCNN / norm / actが全bit一致、最初の微差は2番目のCNN。[GPU切り分け](results/m4-compact-gpu-drift-localization-20261015.json)。同一入力・係数の現在CPUconv2も元bitを再現しないため、参照文脈未達とし、新goldenへ置き換えない。Persistent allocation警告1件の発生元とDisposeは未検証。全encoder201要素不合格・全信号 / 両profile・画像FP32全3 profile・最終API全条件を維持する。全体Python回帰は4環境各744件中契約2件不合格、関連52件を修正後Greenにしたが全体の再確認は未達。[全体回帰失敗](results/python-full-regression-contract-failed-20261015.json)。以下は直前checkpoint。

2026-10-15最新追加: `0d7394c` は元native lconv全体をbit再現した独立replayで、非連続280行のouter RMSをGPU局所2 Red→2 Green、normal / mean / inverse全bit一致、計算fallback / flush0。[局所証拠](results/m4-frozen-outer-rms-compact-gpu-20261015.json)。compact mean / outer mean / 明示layout overrideは関連45 Green。最新全encoder `d8714aa` は201要素・最大許容差比3.73996で不合格、NUnit0 pass / 1 fail / skip0 / exit2、GPUCompute・計算fallback / flush0。[固定全encoder失敗](results/m4-compact-outer-rms-full-gpu-failed-20261015.json)。旧213要素版より数値とgraph負荷は改善したが、元gateを満たさない。次は新候補の元12層観測から最初の超過を切り分ける。画像FP32全3 profile・音声元全6信号 / 両profileと、実入力→最終APIの全条件を維持する。以下は前回checkpoint。

2026-10-15単独層追加: 約100.88 MBのlayer_9は、観測したlayer_8入力の出力をbit再現し、元nativeのlayer_8入力なら元layer_9との不一致0。GPUCompute・計算fallback / flush0、NUnit1 pass / skip0 / exit0。[固定の元入力照合](results/m4-layer9-original-input-gpu-isolation-20261015.json)。初回のbatch未確定による推論前拒否も保持。最初の層超過について前段入力の影響を確認できたが、全encoderや全APIの合格ではない。次はdynamic attention積和を切り分ける。

2026-10-15最新追加: 最初のnormのinverse補正は固定6行の実GPU2 Red→2 Green・全bit一致。[局所結果](results/m4-frozen-first-norm-refined-gpu-20261015.json)。既存initializer byteを保つinverse-only rewriteは小さい6 Red→関連12 Green。ただし元silence全encoderはinverseのみ472要素、static64組合せ287要素不合格。[inverseのみ](results/m4-inverse-only-real-gpu-failed-20261015.json) / [static64組合せ](results/m4-refined-static64-real-gpu-failed-20261015.json)。元nativeの全12層観測は通常最終値とbit一致、0〜8層は許容差内、9層1・10層39・11層1,218要素不合格。初回はflat cacheの参照path誤りで比較未実行、修正後NUnit1 passは観測完了の意味に限定する。[固定全層観測](results/m4-refined-static64-full-layer-observation-20261015.json)。次は元layer_8入力でlayer_9を単独GPU照合する。工程3・4の全範囲と元許容差を維持する。

2026-10-15追加: 保存Welford8のsilence GPUComputeは元参照と355要素不合格、計算fallback / flush0。[固定失敗receipt](results/m4-saved-welford8-gpu-failed-20261015.json)。通常最終値を変えない7段階観測では、最初の畳み込みがbit一致し、最初の微差はlayer0.norm。[観測と経路TDD](results/m4-saved-subsample-gpu-first-difference-20261015.json)。最初の後段normの1要素超過は、実activationのFP64投影でも残るため入力側へ戻す。[独立oracle診断](results/m4-first-feed-forward-gpu-propagation-20261015.json)。全encoder補償のActionsはrunner通信断で数値結果未取得、成果物0件。[別分類の失敗](results/m4-compensated-encoder-runner-lost-20261015.json)。画像FP32全profile、音声全信号、実入力から最終APIの全role・境界・寿命・例外後再利用・text回帰は未達。以降の停止記述は当時の履歴。

2026-10-14。ユーザー指定の範囲は、(1)基準・PR依存整理、(2)画像・音声encoder数値不一致修正、(3)実入力→最終768次元APIの受け入れ。sample・正式Releaseは今回の完了条件へ置き換えない。

## 停止時の状況とGPU実行への変更

追加実測: ユーザー指定の新しい実行計画1・2を実施した。GPU専用変更を含む `b796e50` がcompile成功、tinyと実square 1件のGPU計測は2 pass / skip0 / exit0。Actions `38578451611` のSiLU修正後実ONNXは元6信号・2 profileすべて照合し全6件不合格。元入力・mask・公式native特徴量は変わっていない。[今回の結果](gpu-execution-and-silu-validation.md)。以下の停止時点の未計測・未compile・未検証は当時の履歴であり、上記の指定範囲を超える元encoder全条件・最終768次元は引き続き未達。

2026-10-14、ユーザーの停止指示後、Unityの負荷調査とGPU実行への変更を追加で依頼された。実行中だった画像診断のbatchmode Unity（PID 64068）は停止した。参照元 `5c697a6472e0203d9f24c77da7a466f83735ce84` の部分結果は未完了として保存し、未実行条件を合格にも数値不合格にも数えない。

この診断はCPUとGPUの実モデル推論を交互に実行していた。完了済みsquare GPUの2画像は `actual_backend=GPUCompute`、RTX 4070 Ti SUPER / Direct3D12、計算fallback・readback flushとも0だった。一方、squareとlandscapeのCPU条件も実際に実行されていた。数値補償の診断候補は元1823層から25265層へ展開され、GPU推論前にもCPU側のモデル変換・読み込みが発生する。各段階の時間を計測していないため、負荷全体の割合は未確定。ログにはUnity Acceleratorの接続timeoutもあるが、推論の遅延原因とは確定しない。

今回の対象である `CompensatedVisionReferenceTests` と `CompensatedNormAudioEncoderReferenceTests` はGPUCompute専用へ変更する。GPU未対応・CPU backend指定を拒否し、出力backend一致と計算fallback・flush 0の既存assertionを維持する。集計は画像3 profile / 音声1 backendへ変更し、`cpu_validated=false` を記録する。旧CPU/GPU全条件の完了記録として扱わない。小さい合成契約テストのCPU条件と過去の診断fixtureは履歴・回帰用途で残すが、ローカルの実モデル検証に含めない。

工程1の基準整理は済んでいる。工程2の画像FP32・音声encoder数値gateは未達、工程3のUnity実入力から最終768次元APIも未達。画像のPython Query/Document/Raw参照は18件合格したがUnity全chainの合格ではない。ONNX SiLUの演算順序修正は小さい関連37件と準備回帰4件を確認したが、修正後の実モデルは未検証。今回のGPU専用変更は構成回帰テストで確認し、長時間の実モデル再実行は行わない。

## 基準とPR依存

確認mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。CI run 38428339728は8 job成功。mainの既存ローカル変更は維持する。作業branchは `feat/multimodal-numerical-acceptance`。統合はこの作業branch内で行い、mainへmergeしない。

全22件のopen PR・head SHA・base・CIは [固定snapshot](results/steps-1-3-start-checkpoint-20261014.json) を参照する。作業開始時の状態であり、各実行のsourceへ付け替えない。

| 系統 | 依存と役割 | 受け入れ状態 |
| --- | --- | --- |
| M2 | #31→main。#36の導入helperを含む | Windows条件付き実測。最終空consumer・標準環境・他OS未完了 |
| 画像元参照 | #41 / #53→main | Float16 CPU/GPU計6条件合格、FP32計6条件不合格。#53はnative文脈gate未達 |
| 音声Python | #40 / #60→main | native保存参照は合格、ONNXはsilence等の数値不合格 |
| API | #62→main、#64→main、#65→#62（#64を含む） | Audio/Image APIの合成契約合格。実モデルgateは別 |
| CPU画像 | #67→#65、#68→#67 | 前処理60条件合格。実encoderはFloat16合格・FP32不合格。合成最終chainにも1件不合格 |
| GPU定数 | #69→#67、#70 / #71→#69 | 分割転送の実backbone24件合格、音声全chain24件は最終cosineのみ合格。IL2CPPは重みなしprobe |
| 音声観測 | #72→#71、#73 / #74→#72、#76→#74、#78→#76、#79→#78 | encoder単独全12条件不合格。#73のPow候補は不採用。#79の固定入力post norm単独は合格 |
| native診断 | #75→#60、#77→#75、#80→#77 | native layer / 内部参照のbit再現、FP64投影候補合格。全encoder修正ではない |

API・前処理・定数転送は開発統合候補。#41 / #53 / #68 / #70 / #72〜#80は失敗や診断の記録を含む。CIのgreenを実モデルgateの代わりにしない。依存branchをmainへ渡す際は差分・回帰・受け入れsourceを確認し、mergeは別の明示指示後とする。

## 数値修正

音声はnative exact-inputのpost norm単独がCPU/GPUで合格したため、前段のFP32線形積和と後段での差の増幅を切り分ける。FP64候補はoracleでありRuntimeに採用しない。FP32 block積和＋Neumaier補償和を小さい独立fixtureからTDD比較する。モデル・元golden・`atol=1e-5` / `rtol=1e-4`を維持する。

画像は新旧参照の文脈一致を先に満たし、元の保存graph・入力・重みhashを維持してstage差を局所化する。音声候補の成功だけで画像へ同じ修正を採用しない。

従来の工程2の記録は元画像12条件と元音声6信号×CPU/GPUを含む。ユーザーのGPU実行要件により、再開時の主実行は画像全profile / 音声元6信号のGPU固定数値gateとPython保存exportとする。CPU/GPU全条件の過去の失敗は保持し、GPU専用結果を旧matrix全体の完了へ流用しない。小さいgreen・cosineのみの成功・FP64候補・skipは代替にならない。

## 実入力から最終API

工程2合格後、Texture/RenderTextureとPCMから前処理→encoder→backbone→768次元APIを固定Pythonへ照合する。Query/Document/Rawの採用role、画像全profile、音声mask/長さ境界、例外後再利用、借用/所有権・二重Dispose・既存text15件/検索回帰を確認する。

GPUでは中間readback禁止、要求/実backend一致、計算fallback/flush 0を確認し、profiler証拠を別に残す。最終vectorは有限・単位長、FP32 cosine>=0.999、Float16重み>=0.99。固定参照とsource/model/input hash、完了集計・exit codeを対応付ける。

## 実行環境と今回のledger

ActionsのRepository Secrets / self-hosted runnerは開始時0件。ローカル6000.3.21f1は不在、6000.3.16f1は存在する。21f1のlaunch失敗は環境不足でありTDD redではない。元consumerのdowngrade要求を受け入れず、別consumerを16f1で作成した。初回UPMは`path Received undefined`で失敗し、数値未実行。再起動はこのterminal initialization failureの切り分けとして行う。実際のUnity版と補修条件を必ず記録する。

FP32演算helperはAPI不足13 assertion red→13 green。固定context比較/CLI/producerは5 red（比較API不足3、CLI flag不足1、producer keyword不足1）→既存を含む48 green。workflow条件は1 assertion red/対照1 pass→修正後検証。ローカルPythonは小さいfixtureのみ、tools/uv。実重み比較は専用Actionsへ渡す。

### 固定native入力のFP32候補

[Actions 38523505660](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38523505660)は成功。PR head `7f9dceeb486bfa896782c0fa69b5e324f9b94cd8`、実行merge source `50f4247bb3d8ae8d2741baacf11eb4b6e4b06cd0`を区別する。固定native layer finalと3つのFP32 baselineはbit一致。chunk64のFP32候補はlinear1、linear2、linear2→post normの3点とも不一致0、最大許容差比は順に0.06828808 / 0.04067426 / 0.07749542。[小さいreceipt](results/m4-compensated-projection-actions-20261014.json)。取得はreport JSONのみ、archive全34.4MB中9318 bytesのrange read。重み・候補binaryをGitへ含めない。

これは同じnative入力によるPython候補のみ。Sentis候補、Sentisの実入力差を含むfull encoder、全6信号、画像、工程3の全chainを合格にしない。

### Sentis候補の実行準備

新しいAssets fixtureは2回ともNoTestsFound / 実行0件で、意図したredではない。Package内の既存254件は検出された。追加fixtureをUPM Testsへ移し、Git `afe6932d2697ad57ce15a593d15fea8240e7862a`を要求/解決した。Runtime overlayはしない。

ローカルconsumerのuloop dynamic compilerはPackageCacheの既存templateを読めず、preflight probeが失敗した。uloop3.14.0を同じ内容でembeddedにした。Test Framework1.6.0もembedded。追加試行のasmdef explicit TestRunner参照はoptional TestAssembliesとの重複でcompile失敗・数値未実行。元asmdefへ戻した。これらの環境失敗をTDD redやGPU passへ数えない。

固定Git `afe6932…`のnative batchmode実行で、追加5件が全てAPI不足のassertion red。最小実装Git `31abf50…`でCPU / GPU含む5 green / skip0 / process exit0。さらに実featureの3D shape保持を3 assertion red / 既存5 passから実装し、Git `753e8b2…`で8 green / skip0 / process exit0。[小さいshape受け入れ](results/m4-compensated-sentis-shape-green-20261014.json)。RendererはRTX4070 Ti SUPER / Direct3D12。終了logのallocation warningは保持し、寿命全体の合格とはしない。

実重みfixtureはRuntime packageとは独立したopt-in validation packageに分離した。保存モデル・native parts・係数anchorのSHAを固定し、既存TDD済みmap / norm isolationを変更せず再利用する。初回のJSON reference不足と次のNUnit API不一致はcompile失敗・数値未実行。Runtime source / モデル再import / 元許容差は変更しない。

### 実係数と公式clipの受け入れ

初期候補 `336493c…` はlinear1のみCPU/GPUで合格し、linear2は両backendで197848要素、post normは285418要素が不一致。入力clipを省いた候補であり、不採用。元graphを1.85MBのmetadata readで確認し、linear2入力clipは[-11.1875,11.0625]、native activationの最大値は22.75だった。これは候補の経路不足であり、元encoder不合格の原因確定ではない。旧raw CPU/GPU JSONは次runで置き換わったため、残るNUnit XMLとtool readback数値を根拠とし、旧raw JSONの保持を主張しない。

公式clip保持の3 assertion red / 既存8 pass→11 green / skip0。Git `eee9ea6…` の監査済み実保存weight・同じnative入力はCPU/GPUのlinear1、linear2、post normすべて不一致0。linear2の入力clipは289要素へ適用。GPU最大許容差比は0.06762058 / 0.03730003 / 0.14436543、計算fallback0。NUnit2 pass / skip0 / exit0。[固定実係数のreceipt](results/m4-compensated-sentis-frozen-weights-20261014.json)。全encoderの実入力誤差を含む結果ではない。

Python exportable layerは5 red→既存oracleを含む34 green。保存Torch program、標準ONNX演算のみのNumPy照合、公式clipと借用weights不変を小さいfixtureで確認した。関連全55件green。実モデル全encoder Python exportは別gate。

### 保存graph全体の候補

保存graph変換はAPI不足3 red→Git `bbf8368…`でCPU/GPUを含む3 green。downstreamの接続と元モデルの保存byte列を保持する。全encoderへの初回適用 `cbcd70b…` はrank2以外の定数積を誤って変換対象へ含めたため、CPU/GPU両方とも推論前拒否・数値0件。[未実行receipt](results/m4-compensated-full-encoder-nonmatrix-rejected-20261014.json)。

定数のbroadcast積を保持する1 assertion red / 既存3 pass→修正。Git `a084197…`でモデル変換4件と変更せず再利用したSHA helper7件の計11 green / skip0。元profile・6信号の重い受け入れは同Gitで進行中。Windows CNG hash helperを再利用して同じSHA監査を行い、モデル再生成をしない。結果が揃うまで元encoder gateと工程2を合格にしない。

### 全encoderの完了結果（不合格）

Git `a084197eafa029fbb696cd12dd12b28337274477`の元6信号・2 profileをCPU/実GPUの全12条件で実行完了。すべてcosine>=0.999999だが、固定elementwise条件は全件不合格。最大許容差比はCPU13.67548863 / GPU13.08589078。元1657層に対し候補39317層。GPU計算fallback / readback flush0、NUnit0 pass / 2 fail / skip0 / process exit2。[全条件receipt](results/m4-compensated-full-encoder-all-six-failed-20261014.json)。原raw reportは上書き前に別名へ保持した。

補償積和だけで元encoder gateは解消していない。候補を製品Runtimeへ採用せず、次はTDD済みWorkerObservationで同じ候補の元最終出力を保持し、最初に固定許容差を超える層・前段入力差を局所化する。画像FP32・native文脈の未達も保持し、工程3の全chain・role/境界・text回帰を完了にしない。

工程1は本snapshotと依存表で整理済み。工程2・3は未完了。各候補の実測をこのledgerへ追記し、失敗や未実行を残す。

### 補償候補の誤差伝播（silence限定）

Git `5337431264f0f790deaf40d25a9e5604c402b051` の12 norm出力観測はCPU/GPUとも最終出力bitsを保持した。最初の不合格は両backendのlayer_0で、CPU6要素 / GPU11要素。観測テストの成功は数値gateの成功ではない。[観測receipt](results/m4-compensated-layer-observation-20261014.json)。

Git `f38ea33ee52fc91624244048df10489063a5d46a` のsubsampleからfeed_forward1までの6段階も通常最終出力bitsを保持した。CPUはlinear1で1要素、post normで284要素が不合格。GPUはpost normで5要素が不合格。NUnit0 pass / 2 fail / skip0。[段階別receipt](results/m4-compensated-feed-forward-observation-20261014.json)。

Git `1477d88e99c98f1bf8937355437f238599c9951e` では同じ観測に中間FP32 binary、SHA、importerで0.5統合されたnorm係数とepsilonを追加した。通常最終bitsは不変、CPU/GPU両条件とも元assertionが不合格で終了した。binaryはignored artifactsに保持し、Gitには記録とhashだけを置く。

独立FP64 RMSNorm oracleと誤差分解は12 assertion redから実装し、既存補償投影を含む35 green。保存したlinear2出力を独立oracleへ入れると、CPU284 / GPU5のpost norm不一致を再現した。一方、実Sentis post norm出力対同じ入力のoracleは両backendで不一致0、最大許容差比0.00137561 / 0.00124218。native出力対native入力のoracleも不一致0。分解残差は0。**このsilence候補では、正規化演算そのものより前段から伝播した入力誤差が主要因**と判断する。他の信号・後続layerへの一般化や、subsample内の原因確定はしない。[誤差予算](results/m4-audio-norm-propagation-budget-20261014.json)。

次の原因切り分け用に、subsampleのconv / LayerNorm / activation × 2と最終projectionの7箇所を採取するPython producerを7 red→既存関連を含む56 greenで追加した。固定入力のSHA・サイズと、既存native subsample最終値の完全一致を満たす場合だけ出力する。既存外部hookを保持し、成功・失敗のどちらも自身のhookを除去する。workflowの固定メル入力取得・hash監査は1 red→1 green。実重みActionsの結果は別gateで、tiny-model greenを実参照文脈の合格に数えない。

Actions [38540183522](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38540183522)は成功。PR head `57714c79…`、実producerのmerge SHA `e868446e44fb96494605b61b99f86cf3a5f5f973`を区別する。メル入力572928 byteのSHAを維持し、native subsample最終値は既存参照と完全一致。7箇所のbinaryを部分取得し、サイズ・SHAを監査した。[固定native参照receipt](results/m4-audio-subsample-native-reference-20261014.json)。

Git `a89ac05c1e970045420747522512be191ee1a169`の実Sentis観測もCPU/GPU各7箇所すべて元elementwise許容差内、通常最終bitsは不変。最初のconv出力はnativeと差0で、最初の非零差はlayer0.normの最大6.67572021484375e-6。subsample最終投影の最大許容差比はCPU0.73225614 / GPU0.66381648。NUnit2 pass / 0 fail / skip0 / process exit0。終了時allocation warningは保持し、資源寿命の合格にはしない。[Sentis観測receipt](results/m4-compensated-subsample-observation-20261014.json)。前段7箇所の合格と、後段post normや全encoderの不合格は両立する。次は最初のLayerNormから伝播する差と投影の差を固定入力で切り分け、最小の数値修正をTDDで試す。画像と工程3の未達は維持する。

### 中心化分散と補償和のLayerNorm候補

Sentis 2.6.1のCPU GlobalAverageVariancePoolはFP32の逐次和から `E[x²] - E[x]²` を計算する。Editor候補として、中心化した値の二乗と、平均・分散のNeumaier補償和を使う `CompensatedLayerNorm` を追加した。元のモデル・係数・output IDを保持するtransformも追加し、Git `d72c780…`のAPI不足10 assertion red→Git `bee1e0a…`のCPU/GPUを含む25 green / skip0 / exit0。大きい行数でもinitializerはscale/biasと小さいscalarだけで、元モデルの保存bytesは不変。[red](results/m4-compensated-norm-red-20261014.json) / [green](results/m4-compensated-norm-green-20261014.json)。Runtimeへ採用していない。

Git `4492052…`のprojection chunk64 / norm chunk4を元6信号・2 profileへ実行した。CPU/GPU全12 encoder条件は引き続き不合格。最大許容差比はCPU11.85614177 / GPU13.28313560。silenceの最初のfeed-forward post normはCPU284→1要素、GPU5→3要素となったが、合格ではない。subsample観測2 pass、feed-forward2 fail、encoder2 fail、計NUnit2 pass / 4 fail / skip0 / exit2。[全encoder失敗](results/m4-compensated-norm4-real-audio-failed-20261014.json) / [component結果](results/m4-compensated-norm4-component-results-20261014.json)。

Git `c96b534…`のnorm chunk1ではsilenceのpost normがCPU/GPUとも1要素不一致。最大許容差比1.94292705 / 2.22255104、最終bitsは保持、NUnit0 pass / 2 fail / skip0 / exit2。全encoderのnorm1受け入れは未実行。[norm1の失敗](results/m4-compensated-norm1-feed-forward-failed-20261014.json)。候補 `8ef7e84…` はinverse standard deviationを一度計算して乗算するFP32式を試し、25件の小さい回帰が合格。実silence feed-forwardもCPU/GPU各1要素不合格、最大許容差比2.09704983 / 1.32994859、最終bits保持、NUnit0 pass / 2 fail / skip0 / exit2。[reciprocal候補の失敗](results/m4-compensated-norm1-reciprocal-feed-forward-failed-20261014.json)。CPU/GPUで改善の方向が異なり、候補の単独合格を全encoderの合格へ置き換えない。後続の実行結果は別directoryへ保存し、旧norm4の全encoder失敗reportを上書きしない。

固定native内部参照は再利用し、producerが変わっていないC#候補更新で重い参照生成を繰り返さないため、PRの任意 `audio-layer-replay-diagnostics` labelを外した。Required CIと実モデル数値gateは維持し、任意workflowのskipをpassに数えない。

### 工程3のrole別参照生成

画像・音声の最終Python参照に `role="query" / "document" / "raw"` とCLIの `--role` を追加した。既存callerのdefaultはdocument。採用roleの固定promptを検査し、不正roleはモデルloadより前に拒否する。row/metadataへroleを記録し、3 roleの独立したtokenizer照合・公式forwardと保存chainをtinyモデルで確認した。

新しい参照生成コードの `source_commit` と、再利用するencoder入力の `encoder_source_commit` / `vision_source_commit` を分ける。CLIは `--encoder-source-commit` / `--vision-source-commit` を受け付け、省略時だけ旧callerと同じsourceを要求する。入力producer、モデルrevision、全入力fileのhashを監査する。旧encoderの数値失敗を新しいrole参照の成功へ置き換えない。

12 red→既存producerを含む34 green。独立tokenizerによるID/mask照合を追加後、対象12件もgreen。実モデルのQuery/Raw参照、Sentis全chain、GPU中間readback・所有権・境界・text回帰の受け入れは未完了。[role TDD](results/multimodal-role-reference-red-20261014.json) / [green](results/multimodal-role-reference-green-20261014.json)。新しいroleは別destinationへ生成し、旧固定参照を上書きしない。

### 画像候補の大きいinitializerと元profileの受け入れ

補償投影の初期total/correctionは出力shapeと同じゼロ配列だった。実vision相当 `[1,2520,3072]` のinitializer制限テストで、要求24576 byteに対し30965760 byteの1 assertion red / 既存11 passを観測。スカラーの0をbroadcastする最小修正でGit `67ac642…` の既存norm/model回帰を含む26 green / skip0 / exit0。出力tensorそのものの大きさは変えず、不要なゼロinitializerを除去した。[red](results/m3-compensated-scalar-zero-red-20261014.json) / [green](results/m3-compensated-scalar-zero-green-20261014.json)。

Git `8cefbcdae2d2419f950b0d9cc7767090eeafefdc` は旧fixed vision reference / validation JSON・全入力・保存FP32モデルのSHAを監査し、画像encoderの公開APIへ補償投影候補を渡した。Unity 6000.3.16f1 / Sentis2.6.1、3 profile×CPU/GPU、各2画像の計12 numeric caseを完了した。すべて元elementwise許容差で不合格。最大許容差比はCPU4.98814232 / GPU5.18880544。candidate25265層、最大constant7741440 byte、GPU計算fallback / flush0。NUnit0 pass / 6 fail / skip0 / process exit2。[全profileの失敗receipt](results/m3-compensated-vision-all-profiles-failed-20261014.json)。新native参照を期待値へ置き換えていない。旧Unity21の実測と今回のUnity16を同一環境のA/B比較としない。Float16、画像前処理、最終backbone接続はこのrunの対象外。

### role参照用の監査済みbackbone再利用

`--backbone-source` で既存のPython最終参照directoryを指定できる。reference/validationの一致、成功scope、producer、model revision、snapshotの全input hash、全profileのsequence length・画像/音声feature数・幅、graphの相対path・サイズ・SHAをモデルload前に検査する。元graphのbyte列を新しいdestinationへコピーし、再exportせずに別roleの公式forwardと保存encoder→backboneの数値を照合する。コピー先graphのSHAは元と一致する。再利用producerとreference JSONのSHAを新metadataへ記録する。

audit8 red、producer連結2 red、CLI2 redから実装し、role・既存producerを含む44 green。source/destinationが同一または相互の配下の場合は旧receiptを変更する前に拒否する6 red→修正、path/CLI/監査の16 green。Windowsのテストmanifestがbackslashを保存していた2 fixture失敗はPOSIX相対pathへ修正し、監査を緩めていない。[reuse red](results/multimodal-backbone-reuse-red-20261014.json) / [連結red](results/multimodal-backbone-reuse-integration-red-20261014.json) / [destination red](results/multimodal-backbone-reuse-destination-red-20261014.json) / [green](results/multimodal-backbone-reuse-green-20261014.json)。tiny-model greenであり、実モデルのQuery/Raw参照やSentisの工程3を完了にしない。

### 実role参照の失敗と再現診断

音声Actions [38555519280](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38555519280) は、固定native・backbone manifestのSHA監査後にDocumentの公式encoder featureが旧固定参照と一致せず失敗した。PR head `678330657ba77bb21f93b0c204d3fa6938642271`、実producer `da22414ba70b6a91869972f435c4276f7f816526`。Query/Rawは未実行。モデルrevision・入力hashの一致だけで演算文脈の一致を証明できない。元goldenを変更せず、数値差と実行環境の記録を先に追加する。[失敗receipt](results/m4-role-reference-real-context-failed-20261014.json)。

画像・音声の文脈不一致時に、失敗JSON・実FP32出力・最大差/許容差比・Torch CPU capabilityとthread数、関連環境変数を記録するTDDは2 assertion red→2 green、role/既存producer回帰も38 green。成功validationは出さず、旧期待値は変更しない。Git `a98a2f805c7c673d08acf9daf9c70fce473f5f4b` の音声診断run `38558654493` は起動時queue待ちで、重複run `38558655724` はcancelした。任意labelを外し、pushごとの同じ重い参照生成を避ける。[diagnostic red](results/role-reference-feature-drift-red-20261014.json) / [green](results/role-reference-feature-drift-green-20261014.json)。

画像にも3 profileの固定native graphと共有backboneを再利用するopt-in workflowを追加した。workflow契約2 red→range extraction/音声workflow回帰を含む10 green。入力はmain `c868df5…` の別role参照生成用artifactであり、工程2の旧vision期待値 `e2a6dc0…` の代替ではない。実モデルrole照合はworkflow greenとは別。[workflow red](results/image-role-reference-workflow-red-20261014.json) / [green](results/image-role-reference-workflow-green-20261014.json)。工程2・3は引き続き未完了。

音声diagnostic run `38558654493` はDocument/silenceで失敗し、288要素、最大絶対差0.0001811981201171875、最大許容差比8.712747897611987を取得した。実producer `6da5da2fa12861b6e6f5f04c455a75130f6fb81c`、Torch2.14.1+cu130 / AVX2 / intraop2 / interop4。環境OMP/MKLは4で、runtime thread数とは区別する。原因は未確定。元goldenは不変。[診断receipt](results/m4-role-reference-real-context-diagnostic-failed-20261014.json)。次の診断は同環境の保存native encoder対旧期待値も比較し、公式forwardの文脈と保存graphの再現性を分ける。2 assertion red→2 greenだが実モデル結果は別。[saved context red](results/role-reference-saved-context-red-20261014.json) / [green](results/role-reference-saved-context-green-20261014.json)。

画像run `38558995411` は固定manifest SHAの監査後、producerが全manifestを監査する際に取得対象へ含めていなかった `square/0.png` で失敗した。数値0件。PNG6枚とprocessor_config.jsonを取得対象へ加える1 assertion red / 1 pass→関連10 green。監査処理は緩めず、元数値gateを成功にしていない。[manifest red](results/image-role-reference-manifest-red-20261014.json) / [green](results/image-role-reference-manifest-green-20261014.json)。

Git `a20ff165aa7bdc170350b68f20a86aaec660f399` のnorm chunk1・reciprocal・scalar-zero候補を、元6信号×CPU/GPU全12条件で完了。全件不合格、最大許容差比はCPU10.75637790 / GPU15.96514096。候補42851層、GPU fallback/flush0、NUnit0 pass / 2 fail / skip0 / exit2。旧norm4のraw reportは変更せず、新directoryとarchiveを保持した。[全encoder失敗](results/m4-compensated-norm1-scalar-zero-real-audio-failed-20261014.json)。局所normの改善を全encoderの修正とせず、次は現在の候補の最初の不合格stageを調べる。

Git `465d98373aed61240e5c69892b38c997a0fc2f86` のsilenceに対する12 norm出力観測は、通常の最終bitsを両backendで保持した。最初の不合格norm出力はCPU layer_5（最大許容差比1.02823035）、GPU layer_6（1.02760732）。layer_11ではCPU24.44634017 / GPU23.94601719となる。NUnit2 pass / skip0 / exit0は観測の忠実性だけの合格で、encoder数値合格ではない。[観測receipt](results/m4-norm1-scalar-zero-layer-observation-20261014.json)。

Git `02413f7…` の音声run `38560553234` と画像run `38560640093` は、PNG等の取得不足を解消して実数値へ到達した。両runnerはTorch2.14.1+cu130 / AVX512 / intraop2 / interop2。音声silenceは公式145要素（最大許容差比3.11813114）、保存native graph251要素（8.87123701）で旧期待値と不一致。画像square-0は公式・保存graphとも3要素（1.18285830）で不一致。Query/Rawは未実行。前回音声AVX2 runnerとの違いは観測したが、CPU ISA単独が原因とは証明していない。[実文脈診断](results/multimodal-real-saved-context-drift-failed-20261014.json)。

同じ保存graph・入力・旧期待値を1/2/4 intraop threadで比較するfailure-only診断を6 assertion red→6 green、画像/音声の連結2 greenで追加した。各requested/actual threadを記録し、実行失敗は数値不一致と分け、最後にcallerのthread数を復元する。golden・通常参照生成の設定は変更しない。[thread red](results/cpu-reference-context-red-20261014.json) / [green](results/cpu-reference-context-green-20261014.json)。次の実Actionsでthread依存を検証する。工程2・3は未完了。

音声thread診断run `38561998469` はAVX2 runnerで完了した。保存native graphのsilenceは1/2/4 threadすべて288要素不一致、最大許容差比8.71274790。このrunnerの当該fixtureはthread数変更で解消しない。ここで全tensor bit一致や他runner/他信号の同値性は主張しない。元期待値は変更していない。[実thread診断receipt](results/m4-real-reference-thread-contexts-failed-20261014.json)。

次はnative設定と `ATEN_CPU_CAPABILITY=avx2` / `MKL_CBWR=AVX2` / `DNNL_MAX_CPU_ISA=AVX2` を要求する設定を、別processのActions matrixで比較する。CPU hostもlscpu JSONで記録する。Intelの[条件付き数値再現性](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/get-started-with-conditional-num-reproducibility.html)ではnon-Intel CPU等でAVX2設定がAUTOへ戻り得るため、環境変数だけで演算文脈一致と断定しない。oneDNNのISA制限は[公式service API](https://uxlfoundation.github.io/oneDNN/group_dnnl_api_cpp_service.html)を参照した。

専用workflowは監査済み保存native graphと最初のfixtureだけを部分取得し、backboneやHF snapshotを取得しない。旧参照JSON・manifestの固定SHA、producer/revision、選択graph・入力のpath/サイズ/SHAを検査してからloadし、診断のみのJSONを出す。通常の参照生成の全manifest監査を弱めていない。standalone replay4 assertion red、workflow2 red→画像分岐と関連回帰を含む23 green。[replay red](results/saved-reference-replay-red-20261014.json) / [workflow red](results/reference-context-workflow-red-20261014.json) / [green](results/reference-context-workflow-green-20261014.json)。最初のfixtureの診断であり、全encoder・工程2/3の合格ではない。

専用matrix [38563722587](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38563722587) はhead `d78066b2f5d608ab158773b2a64c1750299f0fc3` で4 jobの実handleを確認した。画像nativeは開始済み、残りはqueue待ち。任意labelは同じrunの確認後に外した。別の画像role/thread run `38562190876`（head `9332de5…`、job `115845754166`）は公式実モデル比較stepでliveを確認し、未完了なので再起動しない。結果が揃うまで工程2/3は未完了のまま維持する。


## 最後の確定結果と再開計画

実装・実モデル検証は停止中。現在のコード基準は `a237dee2874cdb0fbcdf4fe333aebae7193ad390`。画像norm4の中断、GPU専用構成2 red→2 green、対象変更後のUnity未検証は[中断receipt](results/m3-vision-norm4-user-interrupted-20261014.json)と[文書更新snapshot](results/documentation-checkpoint-20261014.json)を参照する。

画像の[Query / Document / Raw実Python参照](results/m3-real-image-all-role-reference-20261014.json)はActions `38562190876`、producer `891ce9004deecc262437e56b2fd9a2eaf1054632`で全18件合格した。上の旧runの「Query/Raw未実行」は当時の履歴。元vision FP32 gate、Unity全chainの未達は維持する。音声Query / Rawは未完了。

[CPU参照文脈matrix](results/native-first-fixture-context-matrix-20261014.json)のActions `38563722587` は4 job実行完了。最初のfixtureのみの検証で、native画像3要素不一致、AVX2要求画像は一致、native音声silenceは一致、AVX2要求音声はthread別に不一致だった。hostと設定の交絡があるためCPU ISA単独を原因確定せず、旧goldenを更新しない。

[dynamic16候補](results/m4-norm1-dynamic16-full-encoder-failed-20261014.json)はsource `998fa43d89814a90c9eebb899c71439879b79390`、norm1 / projection64 / dynamic16、候補45623層。元音声12数値条件は全件不合格、NUnit0 pass / 2 fail / skip0 / exit2、GPU fallback / flush0。GPU最大許容差比11.47812797で、候補採用・全encoder修正とは扱わない。

[標準ONNX SiLU修正](results/native-silu-division-order-green-20261014.json)はmarkerを `Neg→Exp→Add→Div` へ置換し、FP32除算順序の小さい反例を3 red→3 green、関連37 greenへした。準備回帰は旧Swish検査の失敗を標準4演算の検査へ更新し4 green。実モデルは未実行で、元silence不合格の修正完了とは言えない。

再開順は[全体計画R0〜R5](multimodal-tdd-plan.md)。最新compileと小さいGPU確認、既存SiLU修正のActions全信号照合、一つのGPU失敗経路の最小修正、全profile / 全信号、実全chain、sample / 必須環境 / 配布準備の順に進む。段階別負荷と5分の外部上限を再開時に設定・確認し、未変更の全suite反復や新しい大規模補償候補の同時展開を避ける。工程2・3は未完了。
