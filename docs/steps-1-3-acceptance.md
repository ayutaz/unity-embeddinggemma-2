# 工程1〜3の作業基準と受け入れ

2026-10-14。ユーザー指定の範囲は、(1)基準・PR依存整理、(2)画像・音声encoder数値不一致修正、(3)実入力→最終768次元APIの受け入れ。sample・正式Releaseは今回の完了条件へ置き換えない。

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

工程2の完了証拠は元画像12条件と、元音声6信号のPython保存export・Sentis CPU/実GPUの固定数値gate。小さいgreen・cosineのみの成功・FP64候補・skipは代替にならない。

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

Git `c96b534…`のnorm chunk1ではsilenceのpost normがCPU/GPUとも1要素不一致。最大許容差比1.94292705 / 2.22255104、最終bitsは保持、NUnit0 pass / 2 fail / skip0 / exit2。全encoderのnorm1受け入れは未実行。[norm1の失敗](results/m4-compensated-norm1-feed-forward-failed-20261014.json)。次の候補 `8ef7e84…` はinverse standard deviationを一度計算して乗算するFP32式を試し、25件の小さい回帰が合格。実feed-forwardの結果は別gateとして収集中。

固定native内部参照は再利用し、producerが変わっていないC#候補更新で重い参照生成を繰り返さないため、PRの任意 `audio-layer-replay-diagnostics` labelを外した。Required CIと実モデル数値gateは維持し、任意workflowのskipをpassに数えない。

### 工程3のrole別参照生成

画像・音声の最終Python参照に `role="query" / "document" / "raw"` とCLIの `--role` を追加した。既存callerのdefaultはdocument。採用roleの固定promptを検査し、不正roleはモデルloadより前に拒否する。row/metadataへroleを記録し、3 roleの独立したtokenizer照合・公式forwardと保存chainをtinyモデルで確認した。

新しい参照生成コードの `source_commit` と、再利用するencoder入力の `encoder_source_commit` / `vision_source_commit` を分ける。CLIは `--encoder-source-commit` / `--vision-source-commit` を受け付け、省略時だけ旧callerと同じsourceを要求する。入力producer、モデルrevision、全入力fileのhashを監査する。旧encoderの数値失敗を新しいrole参照の成功へ置き換えない。

12 red→既存producerを含む34 green。独立tokenizerによるID/mask照合を追加後、対象12件もgreen。実モデルのQuery/Raw参照、Sentis全chain、GPU中間readback・所有権・境界・text回帰の受け入れは未完了。[role TDD](results/multimodal-role-reference-red-20261014.json) / [green](results/multimodal-role-reference-green-20261014.json)。新しいroleは別destinationへ生成し、旧固定参照を上書きしない。
