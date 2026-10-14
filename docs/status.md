# 現在の状態と残タスク

更新日: 2026-10-15。最新の完了済み実GPU全encoder候補sourceは `43c252d3b908ab9fee657ec9e06cb2c1e2843224`、入力投影GPU局所Greenは `16b9e40a656564aae38f6095ac8d9fcf050b5156`、確認mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。この文書が現在の状態の入口で、実行順は[全体計画](multimodal-tdd-plan.md)、固定source別の証拠は[工程1〜3](steps-1-3-acceptance.md)にまとめる。

## 現在の判定

**2026-10-15最新の入力投影修正:** 元subsample.layer1.actと投影係数a59650e6…を固定した。現在CPUの3行へ縮めたmatmulは元値と異なったが、元280行のSGEMMはthread1 / 2 / 4、contiguous RHS / linearの全9条件で保存参照を全bit再現した。192項ずつゼロからFMAし、各blockの結果を順にFP32で加える候補でも元280行・286,720値を再現した。これは固定参照から推定した順序で、すべてのMKL shape / ISAに対する保証ではない。API8 Red→関連30 Green、元initializerを保持する1箇所rewrite6 Red→関連22 Green。初回のONNX exportは記号shape推論で巨大allocationを要求し失敗したため、固定Kと正のpermute軸を明示して修正した。失敗と修正後GreenのXMLを別に保持する。

`16b9e40`の実GPU固定投影は、元のstatic64経路で249,203 bit差→block192で全bit一致。元許容差超過は両方0、GPUCompute・計算fallback / flush0、NUnit各1件のRed→Green。元goldenは変更していない。`43c252d` / graph `63f24d48…`の全encoderは無音295要素不合格・最大許容差比7.637025、68,094層、NUnit0 pass / 1 fail / skip0 / exit2。前候補417要素より件数は減ったが最大比は悪化し、旧201要素の候補も含め全encoder未合格。model load2.337秒、CPU変換18.639秒、CPU schedule18.322秒、GPU完了待ち＋最終download0.818秒で、速度の受け入れにも流用しない。[固定投影修正・全体失敗・TDD](results/m4-block192-input-projection-gpu-and-whole-failed-20261015.json)。所有Unityはterminal、現在停止済み。

次の独立offline診断では、最初のfeed-forwardの2つの投影も、元280行CPU SGEMMとblock192候補が保存参照を全bit再現した。入力clip / 出力clipと係数byteを元graphから固定し、FP64は独立FMA丸めoracleだけに使用した。これら2投影の実GPU・全モデル内bit一致は未検証。次は固定GPUでRed→Greenを確認し、元全encoderへ戻る。工程3・4、音声元6信号 / 両profile、画像FP32全3 profile、実入力から最終768 APIの全条件は維持する。

`de3a3b0`の[Python全体回帰38640507832](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38640507832)は4環境各793 pass / fail0 / skip0でterminal success。[固定receipt](results/python-full-scalar-mean-rewrite-green-20261015.json)。後続のXML guard / block192 / matmul rewriteはこのsourceに含まれず、新しいsourceの全体回帰は別判定。CIのconv-reference-context workflowはSC2086の引用不足2件で失敗し、パスの変数展開をquoteした`52b6f3b`の[workflow lint38641650350](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38641650350)でterminal success、既存workflow契約14 passも確認。取消されたCIを成功に数えず、最新Required CIの全job完了は別gate。以下は前回checkpoint。

**2026-10-15最新の全体結果と負荷:** `de3a3b0` / graph `c050c2ef…` のGPU全encoderは無音417要素不合格・最大許容差比6.736437438で、元golden / 許容差は変更していない。GPUCompute・計算fallback / flush0、64,050層、NUnit0 pass / 1 fail / skip0 / exit2。同じモデルの7段階観測は通常最終値を変えず、layer0.conv / norm / actとlayer1.conv / norm / actの6段階を元値と全bit一致させた。次の微差はinput_proj_linearの249,203要素、元許容差超過0・最大許容差比0.134108。局所修正の成功と全encoder不合格を別に保持する。[固定全体・観測・負荷receipt](results/m4-scalar-mean-full-gpu-and-load-analysis-20261015.json)。所有Unityは両方terminal、現在停止済み。

今回の検証用拡張graphは、model load約2.094秒、CPU側runtime graph変換19.245秒、worker作成0.102秒、CPU schedule10.431秒、GPU完了待ち＋最終output download0.661秒だった。GPU実行中にもCPUの準備・投入処理は重い。scheduleとGPU処理は重なるため0.661秒を総GPU推論時間と扱わず、通常製品graphの性能保証にも流用しない。先行する0件実行ではcold ONNX import16.805秒も確認。Monoのpeak working setは0で欠測、0 bytesの使用と解釈しない。最初のテストfilter指定は0件 / exit0で未実行として保持し、終了確認後に動作済みregexへ修正した。再発を防ぐXML監査は5 Red＋runner連結3 Red→関連11 Green、終了0でも0件・skip・件数不整合・数値失敗を成功にしない。

`b4b817a`の[Python全体回帰38639201752](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38639201752)は4環境各787 pass / fail0 / skip0でterminal success。[固定receipt](results/python-full-source-order-conv-and-scalar-green-20261015.json)。その後のscalar graph rewrite / 実行XML監査は含まれず、最新sourceの全体回帰は別実行。次は固定済みの元layer1.act入力と元投影係数を使い、input_proj_linearを小さい独立参照へ分離する。工程3・4、音声全6信号 / 両profile、画像FP32全3 profile、実PCM / Textureから最終APIの全受け入れは未達。以下は前回checkpoint。

**2026-10-15最新の元演算順修正:** 固定oneDNNの生成コードに沿ってConv2を16入力channelごとにゼロから積和し、blockを保存するときに前の結果を加える順序へ修正した。連続したFMAの候補探索とは異なる。独立offline oracleで元286,720値を全bit再現し、API7 Red→関連30 Green。`8df85b3`の実GPUは修正前224,650 bit差→修正後全bit一致、元許容差超過はいずれも0、GPUCompute・計算fallback / flush0。`7688991`の全encoderは無音417要素不合格・最大許容差比6.73644であり、旧201要素の候補を上回る数値合格にはなっていない。`b065c1f`の通常最終値を変えない7段階観測では、layer0.conv / norm / actとlayer1.convが元値と全bit一致し、次の微差はlayer1.normの850要素（元許容差超過0）。

元[PyTorch moments_utils.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/moments_utils.h)のscalar AddMomentsの平均更新に沿い、8-laneの最後の平均集約だけをFP32 FMAにするopt-inを追加した。API4 Red→関連25 Green、監査rewrite5 Redとaxis拒否不足1 Red→関連18 Green。16-lane変更は改善せず不採用。`b4b817a`の固定6種類 / 元8,960行を保持したGPU局所テストは、修正前40 bit差→修正後全bit一致。計算fallback / flush0。[固定証拠](results/m4-source-order-conv-and-scalar-mean-20261015.json)。全encoderへ組み込んだgraph `c050c2ef…` は別の数値gateで判定する。局所一致を工程3・4の完了へ流用しない。

`5e773cc`の[Python全体回帰38635383175](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38635383175)は4環境各776 pass / fail0 / skip0でterminal success。[固定receipt](results/python-full-native-conv-context-green-20261015.json)。その後のblock16とscalar FMA修正は含まれず、最新sourceの全体回帰・モデルRequired CI・実GPU全chainは別判定。画像FP32全3 profile、音声元全6信号 / 両profile、実Texture / PCM→最終768 APIの全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰は未達。以下は前回checkpoint。

**2026-10-15最新の参照環境調査:** 元参照run `38398829258` はUbuntu24.04.5 / image20261004.327.1、OMP / MKL各4を確認。ただし元runはCPU ISAを記録していない。既存の保存native再生 `5a2496d` はAMD EPYC9V74・AVX512でthread1 / 2 / 4の最終値が元参照と完全一致し、AVX2制限版は不一致だった。この既存証拠はCPU診断で、GPU全encoder合格ではない。

保存native.pt2から元Conv2係数をhashで抽出する診断を6 Red→6 Green、別workflowを1 Red、失敗reportのruntime / 要求hash保持を1 Red、native ISAをdownload前に確認する不足1 Redから修正し、関連40 Green。Actions `38632854677` はnative / avx2とも実際のruntimeがAVX2、元Conv2と268,142値のbit差（元許容差超過0）でterminal failure。native VMは旧成功時と同じEPYC9V74というmodel名でも、lscpuにAVX512 flagsを公開していなかった。nativeというjob名をISAの証拠にせず、期待値は置き換えない。[固定診断とhardware差](results/m4-saved-conv-cpu-context-unavailable-20261015.json)。新workflowはdefault branchにないため直接dispatchの404後、PR診断labelで起動し、run開始後はlabelを外して後続pushの大きいdownload反復を防いだ。

oneDNN3.12.0 / commit `80afa71049cd69a3df32adcccb623b12cd7baa22` の実際のConv2は `jit:avx2`、src / dst `aBcd8b`、weight `ABcd8b8a` と判明。実装・memory formatは[公式verbose仕様](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)に沿って記録する。次は同commitの[AVX512 kernel生成コード](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/cpu/x64/jit_avx512_common_conv_kernel.cpp)を調べ、GPUで元参照の演算順を保つ成分へ戻す。今回はGPU試行を新たに起動せず、Unityは停止済み。工程3・4、画像FP32全3 profile、音声元全6信号 / 両profile、実入力→最終APIの全条件は未達。

畳み込みAPIを含む前回source `fcaf611` のPython全体回帰 `38631235343` は4環境それぞれ767 pass / fail0 / skip0でterminal success。[固定Python成功](results/python-full-compensated-conv-green-20261015.json)。今回の新しいCPU診断 / preflightはこのsourceに含まれず、関連40 Greenと区別する。以下は直前checkpoint。

**2026-10-15最新の畳み込み候補:** FP32 unfold + chunk64 Neumaier積和を15 Red→15 Greenで追加し、元Convの4D係数をそのまま使う1箇所rewriteを7 Red→関連67 Greenで追加。FP16 Castを監査が拒否しない不足1 Redを修正して関連68 Green。runtime / ONNXにFP64を入れず、既定のexportと元goldenを維持する。`dbfa061` の実GPU全encoderは無音219要素不合格・最大許容差比6.78027で、以前の201要素・3.73996より悪化したため採用しない。GPUCompute・計算fallback / flush0、61,003層、約96.65秒、NUnit0 pass / 1 fail / skip0 / exit2。[固定失敗とTDD](results/m4-compensated-conv-full-gpu-failed-20261015.json)。直近の所有Unityは終了済み。

現在CPUのgrad / no_grad / inference mode、oneDNN有無、module呼び出しを比較しても元conv2のbit再現は未達。元の8,960入力patchは6種類の固定値へ集約できることを監査し、channel / kernel / block順と2 / 4 / 8 accumulatorの42種類を独立offline FMA oracleで比較したが全bit一致の順序は見つからなかった。これは候補探索の診断で、GPU成功や参照文脈合格ではない。次はimmutable参照のhost / convolution kernel文脈を復元し、小さいGPU演算の忠実性を確認してから全encoderへ戻る。旧golden・許容差を変えず、画像FP32全3 profile、音声元全6信号 / 両profile、実Texture / PCM→最終768次元APIの全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰 / 中間readbackなしを維持する。工程3・4とも未完了。

前回のworkflow契約修正 `2a6d135` のPython全体回帰 `38628144281` は4環境それぞれ744 pass / fail0 / skip0でterminal success。[固定回帰成功](results/python-full-regression-contract-green-20261015.json)。今回追加した畳み込み・rewriteテストはこのsourceに含まれず、関連68 Greenと区別する。最新sourceの全体回帰はまだ未検証。モデル生成を含むRequired CI、元encoder / 全chainの合格へ流用しない。以下は直前の切り分けcheckpoint。

**2026-10-15最新の切り分け:** compact候補の12層観測 `5a8799e` は通常最終値をbit再現し、元最終gateの201要素不合格も再現。layer_0〜8は許容差内、layer_9は1・layer_10は42・layer_11は1,119要素超過。約101.22 MBのlayer_9抽出 `f24a539` は観測入力で観測出力を全bit再現し、元layer_8入力では元layer_9との超過0、最大許容差比0.26133。入力側7段階の観測 `dbd836b` ではlayer0.conv / norm / actが全bit一致し、最初の差はlayer1.convの224,650値（元許容差超過0）。いずれもGPUCompute・計算fallback / flush0・NUnit1 pass / skip0 / exit0は切り分けの完了であり、全encoder合格ではない。[固定切り分けreceipt](results/m4-compact-gpu-drift-localization-20261015.json)。

同じ元conv2入力・係数で現在CPUを再計算すると元参照と268,142値のbit差があり、元許容差内でも参照文脈のbit再現は未達。現CPU値を新goldenに採用しない。次は保存conv2値に対して、固定CPU畳み込みの集計順 / 文脈と一般的なGPU FP32補償候補を切り分ける。FP64は独立offline oracleのみ。Unity終了時のPersistent allocation 1件の警告を保存しており、発生元・所有権 / Disposeの受け入れは未達。

Python全体回帰 `38625994348` / `139283e` はWindows / Ubuntu・Python3.13 / 3.14の各744件で742 pass / 2 fail / skip0。失敗はimmutable artifact取得の`actions: read`と、複数upload中のnative artifactを旧前提で選ぶ契約テスト2件。読み取り専用権限とfailure後のnative stage保持を維持して判定対象を修正し、関連52 Green。[旧全体回帰と局所修正](results/python-full-regression-contract-failed-20261015.json)。修正後の全体回帰はまだ未検証。画像FP32全3 profile・音声元全6信号 / 両profileと実入力→最終APIの全条件は未達のまま。以下は直前の数値修正checkpoint。

**2026-10-15最新checkpoint: 実行計画3・4は継続中、未完了。直近の所有Unity processは終了済み。** 最新候補 `d8714aa` の無音encoderはGPUCompute・計算fallback / flush0で201要素不合格、最大許容差比3.73996。以前のlayout候補の213要素・6.90199から改善したが、元gateは不合格。Sentis層数78,457→60,546、実行111.94→91.71秒は各1回の比較で、benchmarkではない。[最新全encoder不合格](results/m4-compact-outer-rms-full-gpu-failed-20261015.json)。元golden、入力・mask・係数byte、`atol=1e-5` / `rtol=1e-4`は変更しない。

dynamic16 attention + static64は248要素不合格。投影chunk32は364、chunk16は229で、最大誤差とgraph増大も残るため採用しない。[attention / 投影の失敗](results/m4-refined-attention-projection-gpu-failed-20261015.json)。最初のfeed-forward RMSはGPU局所2 Red→2 Green、全bit一致。[局所RMS](results/m4-frozen-rms-cascade-gpu-20261015.json)。これを109箇所へ一律に適用した全encoderは327要素不合格。畳み込み後の12 normが非連続layoutであると確認し、除外した97箇所版も213要素不合格だった。[両候補の失敗](results/m4-full-rms-candidates-gpu-failed-20261015.json)。

畳み込み後のCPU入力はshape `[1,280,1024]` / stride `[286720,1,280]`。固定8-lane CPUのouter reductionは最初の256行と残り24行で集計順が異なる。元のlconv全体を保存nativeとbit一致させた独立replayから、280行のnorm入力・期待値を固定し、GPU局所2 Red→2 Green、normal / mean / inverse全bit一致、計算fallback / flush0を確認した。[outer RMSの局所証拠](results/m4-frozen-outer-rms-compact-gpu-20261015.json)。この2 Redは元許容差より強いbit目標の失敗で、元許容差内の値を全encoder不合格と誤分類しない。

独立したchunk / accumulator / laneをまとめ、依存する加算順を維持して1024 meanのONNX 842→121 node、1536 meanは1,234→124 nodeへ縮小。compact / outer APIは4 Red、prefix shapeの焼き込みによる1回の回帰失敗を修正し、明示layout overrideの4 Redを含む関連45 Green。12箇所のouter meanと97箇所のinner meanを適用したgraphは31,727 node。legacy Padが残すsymbolic中間shapeは最初に拒否され、元入力固定profile・native replay・12個の同じ長さのcausal convを監査して、そのprofileだけ `[1,280,1024]` に明示束縛した。dynamic profileやpadded候補の成功は主張しない。

次はcompact候補を元12層の保存native値へ観測し、最初の許容差超過へ切り分ける。画像FP32全3 profile、音声元全6信号・両profile、実Texture / PCM→GPU前処理→encoder→backbone→正規化768次元APIのQuery / Document / Raw、mask / 境界、所有権 / Dispose、例外後再利用、text回帰、中間readback / CPU計算fallbackなしは未達。通常CIのモデル生成を含むrunはcancelし、Required CIは不合格として記録する。局所合格を全encoder / 全chain合格へ流用しない。以下は前回checkpointの履歴。

**2026-10-15前回checkpoint（inverse修正）:** 元の全35,840行を保つ6種類の固定入力で、最初のLayerNormのGPU平均値はbit一致、逆標準偏差の2値は不一致だった。平方根と逆数の収縮を避け、FP32 residualで2段階の丸めを補正するopt-inを追加。GPU局所2 Red→2 Green、全normal / mean / inverseが元nativeとbit一致、計算fallback / flush0。[固定局所結果](results/m4-frozen-first-norm-refined-gpu-20261015.json)。FP64は独立scalar oracleのテストだけで、runtime / ONNXに含めない。候補の対象統計域は1e-6〜1e6で、範囲全体のGPU bit保証ではない。既定export / 元goldenは変更しない。

元重みを再exportせず、保存ONNXの2つのinverseだけへ補正templateを挿入する小さいrewriteを6 Red→関連12 Greenで追加。FP64入力 / Castの拒否不足2 Redも修正し、関連14 Green。既存initializer全byteの集約hashを前後で照合し、変換17.57秒で監査済み別graphを生成した。ただし元silenceのGPU encoder gateはinverseだけで472要素不合格、元Welford8の355より悪化。[実モデルの失敗](results/m4-inverse-only-real-gpu-failed-20261015.json)。既存static64投影補償との組合せは287要素不合格、最大許容差比9.27709、40,633層。数は減ったが最大誤差が増えているので全体改善・受け入れとは扱わない。[組合せの失敗](results/m4-refined-static64-real-gpu-failed-20261015.json)。両試行はGPUCompute・計算fallback / flush0、skip0。

static64候補の12層を元producer `4edbe17` の固定native中間値へ照合し、通常最終値を変えないことを確認。layer_0〜8の9層は各固定許容差内、最初の超過はlayer_9の1要素、layer_10は39・layer_11は1,218要素。NUnit1 passは観測完了であり、12数値条件の全合格ではない。[全層の観測](results/m4-refined-static64-full-layer-observation-20261015.json)。

layer_9を約100.88 MBへ抽出して単独GPU照合済み。全モデルで観測したlayer_8入力を使うとlayer_9出力がbit一致し、抽出経路の忠実性を確認。その後、元nativeのlayer_8入力を使うと元layer_9との不一致0、最大許容差比0.25350。NUnit1 pass / skip0 / exit0、計算fallback / flush0。この最初の層超過には前段入力差の伝播が関与すると確認した。[単独層の元入力照合](results/m4-layer9-original-input-gpu-isolation-20261015.json)。初回は未確定batchを残して推論前に拒否され、固定profileのbatch=1へ明示して再実行した。次は前段の誤差を抑えるためdynamic attention積和の補償を切り分け、元encoder gateへ戻す。画像FP32全profile・音声元全6信号・padded候補、実Texture / PCM→最終APIの全role・境界・寿命・例外後再利用・text回帰は未達のまま。以下は前回checkpointの履歴。

**2026-10-15前回checkpoint:** 保存済みWelford8の実音声モデルをGPUComputeで元silenceへ照合し、計算fallback / flush0、355要素不合格。入力側7段階の観測は通常最終出力とbit一致し、最初の畳み込みは元native値とbit一致、最初の微差はlayer0.norm。7段階は各々の許容差内だが、最初のfeed-forward post normでは1要素が超過する。元係数・config・graph epsilonを監査した独立診断では、実activationにFP64投影を適用してもその1要素は残り、前段入力の差が支配的。FP64はoffline oracleのみ。8 lane候補と現在CPUの最初のLayerNormは元の全6種類の入力行にbit一致し、16 laneへの変更は改善根拠がない。[実GPUの不合格](results/m4-saved-welford8-gpu-failed-20261015.json) / [後段への誤差伝播](results/m4-first-feed-forward-gpu-propagation-20261015.json) / [入力側の最初の差](results/m4-saved-subsample-gpu-first-difference-20261015.json)。

全encoderのWelford8 + compensated64 Actions `38600337962` / source `1f52193` はterminal failure。runner通信断のannotationがあり、成果物0件・数値結果なし。OOMは未証明であり、数値不合格や成功とは判定しない。同じ重い処理をそのまま再試行しない。[実行未達の記録](results/m4-compensated-encoder-runner-lost-20261015.json)。次は最初のGPU Welford128を、元の全入力行を保つ6種類の固定行へ切り出し、統計・平方根 / 逆数の微差を分離して小さいRedから修正する。画像FP32全profile・音声全6信号のGPU gateと、実Texture / PCMから最終APIの全role・境界・所有権・例外後再利用・text回帰は引き続き未達。小さい経路bindingは3 Red→実装後のMoveDim不足1 fail→関連6 Greenであり、全encoder合格ではない。以下の「実行中」は前回checkpointの履歴。

**2026-10-15追記: ユーザー指定の実行計画3・4を進行中。いずれも未完了。** GPUの小さいFMA / Welford正規化probeは固定Git `6542bc6`、4 pass / fail0 / skip0 / exit0、全件GPUCompute・計算fallback / flush0。実音声ONNX Welford8候補はActions `38595761111` / source `6bceed7` で元6信号・2 profileを固定済み旧goldenへ直接照合し、全6件不合格。最新runnerの公式native値にも旧goldenとの差があるため、同run内のtrace一致を元の原因特定とは扱わない。[GPU局所結果](results/m4-welford-gpu-probes-windows-20261015.json) / [参照文脈の未達](results/m4-first-onnx-failure-and-original-context-20261015.json) / [元6信号の候補不合格](results/m4-welford8-pinned-onnx-failed-20261015.json)。次は補償付き静的投影の候補を同じ元6信号へ照合する。画像FP32全profile、Sentis音声全信号、実入力→最終APIの全role・境界・所有権・例外後再利用・text回帰は未達のまま。以下の「1・2まで」は前回終了時の履歴。

「ここからの計画」の工程1・2までを実行した。工程1のコンパイル・tiny GPU・実画像1件の段階計測は合格。工程2のSiLU修正後の実ONNX照合は元6信号・2 profileを全件実行し、6件とも数値不合格。数値修正の工程3へは着手していない。[今回の結果と負荷分析](gpu-execution-and-silu-validation.md)。計画の実行番号1・2と、以前の「工程1〜3の受け入れ」の番号は別なので、元encoder数値修正と全chain未完了を維持する。

[Draft PR #81](https://github.com/ayutaz/unity-embeddinggemma-2/pull/81)はOPEN、baseは `test/audio-post-norm-isolation`。GitHubで23件のopen PRを確認し、main未統合のAPI・診断候補とmainの機能を分けた。公開tag / GitHub Releaseは各0件。mainの既存ローカルUnity設定変更は保持している。[文書更新の確認snapshot](results/documentation-checkpoint-20261014.json)。

| 範囲 | 確認済みの到達点 | 未達・未検証 |
| --- | --- | --- |
| 工程1 / 基準整理 | PR依存・参照hash・環境・判定条件を整理済み | 採用candidate変更時のsource・実測更新 |
| M1 / M2 | WindowsテキストのEditorと条件付きPlayer・固定Git導入の過去受け入れあり | 最終candidateの標準空consumer、性能・寿命、macOS / iOS / Android実機 |
| M3 / 画像 | GPU前処理・prompt・共通backboneはmain。Image APIはDraft実装・合成契約合格。Python Query / Document / Rawは6画像×3 roleの18件合格 | 元FP32 encoder gate、Unity実Texture→768次元の全chain・全role、検索sample・Player |
| M4 / 音声 | PCM・GPU前処理はmain。Audio APIはDraft実装・合成契約合格。SiLU修正後の実ONNX照合は6信号・2 profile全件実行 | 実ONNX全6件が不合格。Sentis encoder固定数値gate、音声Query / Raw、実全chainの全条件 |
| GPU専用変更 / 負荷 | 対象2 fixtureのGPU専用変更を含む固定Git consumerがコンパイル成功。tiny / 実square 1件の計測2 pass・skip0・exit0。GPUCompute・fallback / flush0 | 計測は数値合格を主張しない。対象2 fixtureの全profile・全信号は未再実行、全legacy fixtureのGPU専用化ではない |
| 正式公開 | 開発版 `0.1.0-pre.1`、モデルをGit / LFSへ含めない | M2〜M4・全必須環境・最終配布監査が揃ってから公開工程 |

**工程2の数値修正と工程3の実入力→最終768次元APIは未完了。** Pythonのrole参照、局所TDD、最終cosine、観測テストの合格を、元encoderのelementwise gateやUnity全chain合格へ置き換えない。

## 最後に得られた証拠

- [GPU段階計測](results/m2-gpu-phase-profile-windows-20261014.json): source `b796e50`、Unity 6000.3.16f1 / Sentis 2.6.1 / RTX 4070 Ti SUPER / D3D12。2 pass、skip0、exit0。実square 1件のload約3.93秒、補償graph変換約13.13秒、schedule約2.07秒、最終readback約1.30秒。プロセスpeak working set約3.94 GiB。数値合格・全chain合格はfalse。
- [SiLU修正後の実音声ONNX](results/m4-division-silu-real-onnx-failed-20261014.json): Actions `38578451611` / source `a8033bb`。元6信号・2 profileすべて実行、全6件不合格。入力・mask hash、モデル重みとlibrary版は元参照と一致、公式native特徴量も全6件bit一致。旧golden・許容差を変更していない。SiLUの局所修正だけでは全encoderは解決しなかった。
- [画像3 roleのPython参照](results/m3-real-image-all-role-reference-20261014.json): Actions `38562190876`、producer `891ce9004deecc262437e56b2fd9a2eaf1054632`。18件合格。元vision FP32 gateの解決とUnity連結はfalse。
- [音声norm1 / dynamic16候補](results/m4-norm1-dynamic16-full-encoder-failed-20261014.json): source `998fa43d89814a90c9eebb899c71439879b79390`、元6信号×CPU/GPUの12数値条件すべて不合格。最大許容差比CPU12.21667173 / GPU11.47812797、NUnit0 pass / 2 fail / skip0 / exit2。候補45623層。旧norm1候補の結果とは別記録。
- [SiLU演算順序修正](results/native-silu-division-order-green-20261014.json): 標準ONNX `Neg→Exp→Add→Div` にしてnativeの除算順序を保持。関連37 pass、準備回帰4 pass。実モデルのsilence不合格が解消したとはまだ言えない。
- [CPU参照文脈matrix](results/native-first-fixture-context-matrix-20261014.json): Actions `38563722587` の4 jobは実行完了。画像nativeは旧期待値と3要素不一致、画像AVX2要求は最初の画像で一致。音声nativeはsilenceで一致、音声AVX2要求はthread別に不一致。異なるhostも含むためISA単独を原因と断定しない。全画像・全信号の成功ではない。
- [画像norm4の中断記録](results/m3-vision-norm4-user-interrupted-20261014.json): source `5c697a6472e0203d9f24c77da7a466f83735ce84`。square / landscapeのCPU / GPU各条件は完了・不合格、portrait CPUは途中、portrait GPUは未開始。NUnit完了XMLはない。未実行条件を数値失敗にも合格にも数えない。

## GPU実行と負荷について

停止したPID 64068は通常UIではなく、別consumerのbatchmode画像診断だった。CPU/GPUを交互に実行しており、CPUモデル推論も負荷に含まれていた。完了GPU条件ではRTX 4070 Ti SUPER / D3D12、出力GPUCompute、計算fallback / readback flush 0を確認した。

診断候補の層数は1823→25265。モデル変換・shape解析・読み込み・hash監査・GPUへの転送やdispatch準備はCPU側にも仕事がある。GPU使用だけではこの準備負荷は消えない。段階別時間・peak memory・CPU/GPU利用率は未計測で、層数増加を遅延全体の原因として確定しない。Accelerator接続timeoutもログにあるが、推論の遅延との因果は未確認。

上の停止時点の未計測を今回補った。実square 1件ではload + transformが約17.07秒、全7段階約20.53秒の約83%。この2段階のCPU時間は約15.13秒。GPUモデル推論中でも、この候補のCPU準備が大きい。これは単発測定で、旧モデルとの性能比較、全profile、GPU利用率の時系列、Accelerator timeoutの因果は未検証。[段階別の範囲と限界](gpu-execution-and-silu-validation.md)。

`CompensatedVisionReferenceTests` と `CompensatedNormAudioEncoderReferenceTests` はGPUCompute専用へ変更済み。前処理・encoder・backboneを同じGPU backendへ揃え、GPU未対応をCPUで代替しない。Runtime APIのCPU選択肢と小さいCPU契約テストは残るため、他の旧suiteを一括実行するとCPU実モデル検証も含み得る。[自動操作の実行方針](automation.md)を先に確認する。

## 再開時の計画

以下の1・2（GPU確認とSiLU照合）は今回実行済み。次は一つのGPU失敗入力と最初の数値差を切り分ける作業だが、今回の「2まで」の範囲では開始しない。外部5分上限は `tools/embeddinggemma_tools/bounded_process.py` で実装し、所有processのtimeout停止をTDDで確認した。古いignored `run-native.ps1` の無制限待機は残るので、新しい計測実行には使わない。

1. 固定Git consumerのrequested / resolved SHA、Editor・Sentis・GPUを確認し、対象変更のcompileと小さいGPU probeを先に行う。全suiteを起動しない。
2. 実装済みSiLU修正をActionsで元6信号・2 profileの保存ONNXへ照合する。失敗したら一つの入力・最初の差に絞る。旧golden・hash・`atol=1e-5` / `rtol=1e-4`は維持する。
3. Unity側は監査済み元モデルの一つのGPU入力から始め、load / transform / Worker準備 / upload / schedule / readback / disposeを別々に計測する。今回は実square 1件を計測済み。1試行5分の上限は新しいbounded_processで実装・使用し、超過は中断・未完了へ分類する。旧ignored launcherの上限は未実装のまま。
4. 一つの失敗経路の最小修正が合格してから、GPUで画像全3 profile・音声元6信号と採用重み形式を照合する。新しい補償候補を全encoderへ展開する前に、層数・準備時間・memoryの増加を評価する。
5. 工程2のgateが揃った後、実Texture / PCM→前処理→encoder→backbone→768次元APIの全role・profile・mask・長さ境界・所有権・例外後再利用・text回帰を受け入れる。
6. sample・混在検索・長い反復・Player・必須実機・最終空consumer・配布監査へ渡す。main mergeと正式公開は別の明示指示が必要。

今回の実行はGPU確認・段階計測とSiLUの実ONNX照合まで。数値修正・全chain・sample以降は実行していない。GPU専用の新しい結果と、従来のCPU/GPU受け入れ履歴は別scopeで管理する。

## 過去のcheckpointと実測履歴

以下の「次の作業」やPR head / CI / 実行中の記述は当時の記録で、上の現在状態と再開計画が優先する。実測source・失敗・未実行は保持する。


## 2026-10-14: 工程1〜3の作業checkpoint（main未統合）

確認mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。全22件のopen PRのhead・base・CI、API / 前処理 / GPU転送 / 診断の依存を[工程1〜3の受け入れ](steps-1-3-acceptance.md)へ整理した。作業branchは `feat/multimodal-numerical-acceptance`、[Draft PR #81](https://github.com/ayutaz/unity-embeddinggemma-2/pull/81)。mainへのmergeや正式Releaseは行わない。

音声FP32の分割積和＋補償和は小さいfixtureのRedから関連50件Green。固定native入力による[専用Actions 38523505660](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38523505660)も3段階・不一致0で合格した。これはPython候補の検証で、Sentis候補や元encoder全条件の修正ではない。追加Sentis fixtureの0件実行、環境compile失敗は受け入れに数えない。

**工程1は整理済み。工程2の元画像 / 音声数値gate、工程3の実入力→最終768次元APIは未完了。** 未統合PR #69の分割GPU転送、#70の音声全chain最終cosine24件合格、#71の重みなしIL2CPP probe、#72のcurrent encoder全12条件不合格を区別する。以下は過去checkpointである。

追加checkpoint: 補償候補の全encoderは元6信号×CPU/GPUの12条件すべてelementwise不合格。silenceの中間値と独立RMSNorm oracleで、post normのCPU284 / GPU5不一致は前段入力の差から再現し、同じ入力に対するSentis norm自体は許容差内と確認した。これはsilence限定の原因切り分けで、修正完了ではない。subsample内部7箇所の固定Python参照producerを7 red→関連56 green、workflowを1 red→1 greenで追加した。詳細と固定sourceは[工程1〜3 ledger](steps-1-3-acceptance.md#補償候補の誤差伝播silence限定)へ保持する。

LayerNormの中心化分散＋補償和候補は10 red→既存回帰を含む25 green。norm chunk4の全encoderは全12条件不合格、norm chunk1のsilence post normもCPU/GPU各1要素不合格。候補はEditor/testのみ。工程3のQuery/Document/Raw参照生成と入力producerの別SHA監査は12 red→関連34 greenだが、実モデルのrole別参照とSentis全chainは未完了。固定許容差・元の失敗・Required CIを維持して進める。

画像候補の約31 MBゼロinitializerは1 red→スカラーbroadcast修正→既存26 green。保存実モデルの3 profile×CPU/GPU、各2画像は全12 numeric caseで固定elementwise不合格、GPU fallback/flush0、NUnit0 pass / 6 fail / skip0 / exit2。監査済みbackboneを再exportせずrole別参照へ使うAPI/CLIは関連44 green、source保護・path/CLI監査16 green。工程2・3の未達は維持する。

## 履歴checkpoint: 工程1〜3の受け入れ作業

2026-10-14確認。mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`、[main CI 38428339728](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38428339728)は8 job成功。以下の追加実装・検証は[Draft PR #81](https://github.com/ayutaz/unity-embeddinggemma-2/pull/81)の作業branchであり、mainへの統合・正式リリースは行っていない。Audio/Image APIもDraft PR #62 / #64 / #65の実装で、mainの機能と区別する。

**工程1は整理済み、工程2（元encoder数値差）・工程3（実入力から最終API）は未完了。** [作業基準・依存・受け入れledger](steps-1-3-acceptance.md)を最新の詳細とし、以下の過去checkpointは各sourceの実測として保持する。

| 対象 | 今回の確定結果 | 次の作業 |
| --- | --- | --- |
| M1 / M2 | Windowsテキストの既存受け入れを維持。最終candidate導入・他OSは未完了 | マルチモーダル候補の採用後にtext回帰と固定Git consumerを確認 |
| 画像FP32 | 補償投影候補の3 profile×CPU/GPU、各2画像の12 numeric caseは全件不合格。GPU fallback/flush0 | 元期待値と許容差を維持し、参照文脈と最初の数値差を局所化 |
| 音声FP32 | norm chunk1/reciprocal/scalar-zero候補の元6信号×CPU/GPU計12条件は全件不合格。最大許容差比CPU10.75637790 / GPU15.96514096 | 現候補のsilence norm出力はCPU layer_5、GPU layer_6から不合格。後段演算と伝播を切り分ける |
| 参照再現性 | 同じ固定graph・期待値に対し、実AVX512 runnerで音声saved251要素、画像saved3要素が不一致。原因未確定 | 1/2/4 threadの同graph replayを診断する。旧goldenは変更しない |
| 最終API / role | role別参照とbackbone再利用のTDDは合格。実Document参照がencoder contextで止まりQuery/Raw未実行 | 実参照の再現性解消後、Texture/PCM→768次元、role/境界/所有権/GPU中間readback禁止/text回帰を受け入れる |

根拠は[画像全profile失敗](results/m3-compensated-vision-all-profiles-failed-20261014.json)、[音声全6信号失敗](results/m4-compensated-norm1-scalar-zero-real-audio-failed-20261014.json)、[silence層観測](results/m4-norm1-scalar-zero-layer-observation-20261014.json)、[実Python文脈診断](results/multimodal-real-saved-context-drift-failed-20261014.json)。観測NUnit2 passは最終bitsを保持した診断の合格で、encoder数値合格ではない。

実SentisはUnity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12。旧21f1結果との同環境A/B比較はしない。局所TDD・cosineのみの合格・skipを全encoder/全chainの成功と数えない。重い処理はActions優先、Pythonはtools/uv、main直push・無断merge禁止を維持する。

## 過去checkpoint: 全音声encoder検証後

2026-10-13確認。mainは `a88ed9cffd942999bb271e597c86563298425f2c`。全体TDD計画PR #58まで統合済み。今回の更新は文書のみで、モデル生成・Unity実行は追加しない。

**M2〜M4は未完了。正式リリースは全実装・全必須検証・配布準備の完了後。** [全体TDD計画](multimodal-tdd-plan.md)、[M3](m3-plan.md)、[M4](m4-plan.md)のRed → 最小実装 → Green → 実モデル受け入れで進める。工程6まで公開tag / GitHub Releaseを作らない。

| 対象 | 確認済み / 不合格 | 次のTDDと完了条件 |
| --- | --- | --- |
| M2 | WindowsテキストEditor / Playerの条件付き実測。PR #31はdraft | 導入・寿命・cache・性能の原因別red → 修正 → 最終candidateの空Git consumerと正常終了。Unity版更新・他OSも残る |
| 共通 | PR #43 backbone、#55公式prompt / maskは統合済み。合成featureのCPU / GPU × 2精度は合格 | 実encoderの数・順序・mask・所有権・中間readbackのred → GPU連結 → 最終768次元 / text回帰 |
| M3 | #52 GPU前処理と#56最終Python参照は統合済み。#41はFloat16全6条件合格、FP32全6条件不合格。#53診断は参照文脈未達 | 文脈不一致を拒否するred → 原因別修正 → 元12条件。Image API・画像検索sample・Playerは未完成 |
| M4 | #48 PCM入力 / #50 GPU前処理は統合済み。draft #60のnative保存参照は全6信号合格。NumPy ONNX全6信号とSentis FP32全12条件は不合格 | 元の失敗を再現しないtrace / 参照を拒否するred → 演算差修正 → 全条件再照合。最終音声参照・Audio API・検索 / 再生 / 明示収録・Playerは未完成 |
| 全統合 / 配布 | Windows以外の必須実機受け入れは不足 | 各OSで3モダリティ・実CPU / GPU・UI・反復資源 / 性能・終了。全合格後、空Git consumer・version / 文書 / license・配布監査 |

音声の根拠はPR #60の確認head `70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1`に固定する。[native / NumPyの全6信号](https://github.com/ayutaz/unity-embeddinggemma-2/blob/70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1/docs/results/m4-audio-all-six-native-reference-real-20261013.json)、[Sentis CPU / 実GPUの全12条件](https://github.com/ayutaz/unity-embeddinggemma-2/blob/70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1/docs/results/m4-audio-full-encoder-fp32-sentis-windows-failed-20261013.json)。cosineだけの合格やsubsample単独の合格をencoder全体の成功にしない。旧PR #40の残り5信号未実行は当時のsourceの記録で、#60の最新範囲とは別。

重い検証はActionsを優先し、Pythonはtools/ + uv。ローカルは小さいTDDと必要な実GPU / UI / 端末検証、Unity Editorは1つとする。すべて作業ブランチ / PR経由、必要checkと対象の受け入れを揃えてから統合する。以下は過去checkpointで、現在の作業順は上記と全体計画を使う。

## 過去checkpoint: PR #50時点

確認日: 2026-10-13。main / PR / CIをGitHub APIで再確認し、PR #50統合後の全体TDD計画を更新した。今回の計画更新ではUnity実行・モデル取得 / 再export・buildを追加していない。前処理RuntimeのTDD、固定Git package回帰、公式CPU / 実GPU信号照合は[既存の実測](results/m4-audio-preprocessing-runtime-windows-20261013.json)を参照する。Secrets / runner・tag / Release・main保護の件数と[直前の計画snapshot](results/tdd-plan-checkpoint-20261013.json)は過去の記録として保持する。

**M2〜M4は未完了。未実装のAPI・前処理・sampleと数値不合格の修正が残る。正式リリースは全対応・全必須検証が終わった後。** [全体TDD計画](multimodal-tdd-plan.md)を6工程で更新し、[M3画像](m3-plan.md) / [M4音声](m4-plan.md)の各タスクに先に確認するRed、最小実装、Green・実モデル受け入れを定義した。配布準備では公開tag / GitHub Releaseを作成しない。

## mainと作業ブランチ

| 対象 | 今回確認した状態 |
| --- | --- |
| main | `f41546dfd48ab77347d27772612b2ea43c992b70`、PR #50まで統合。[CI 38336919527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38336919527)成功 |
| [PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31) | draft / 未統合。head `a34bf9057694e141e939ac41bea3e2b713b0a15b`。Windows Player終了・sampleの条件付き実測あり。導入helperのPR #36はこのブランチへ統合済みで、mainには未反映。最終空consumer・性能・native入力・他環境が残る |
| [PR #40](https://github.com/ayutaz/unity-embeddinggemma-2/pull/40) | 未統合。head `e13adcd96ed3e10802131dfc94c48d9533f37ee2`。feed-forward診断をTDD追加して関連76件green。[実モデルCI 38331914586](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331914586)とRequired CIは不合格。最初の微差はlayer 0.feed_forward1のactivation、許容差超過はlayer 10。残り5信号未実行、修正未完了 |
| [PR #41](https://github.com/ayutaz/unity-embeddinggemma-2/pull/41) | draft / 未統合。head `b5766154a209622849e9d71d11955b9bd7e0a65c`。[CI 38320212347](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38320212347)成功。画像Sentis Float16全6条件合格 / FP32全6条件elementwise不合格を維持 |
| [PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43) | mainへ統合済み。最終head `4feac7551bd99f2571a667f237ace85ba2d7093f`。[CI 38331086059](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331086059)全8 job成功。小さい配置 / cache契約68件と実1024長backbone4条件は別runで合格。固定Git `21e598c`のRuntime更新consumerも4条件合格。実測sourceを統合headへ付け替えず、実encoder接続・公式prompt / 前処理 / sampleと最終空consumerを別gateとする |
| [PR #48](https://github.com/ayutaz/unity-embeddinggemma-2/pull/48) | mainへ統合済み。最終head `55221e5c53763ef0de9f00fa6459861dce3756d7`、全8 CI成功。PCM入力25 red → 25 green、固定Git Runtimeの型 / 全102ファイル監査とpackage assembly 87 green。[受け入れ範囲](m4-pcm-input-validation.md) |
| [PR #50](https://github.com/ayutaz/unity-embeddinggemma-2/pull/50) | mainへ統合済み。最終head `6dc6651dacce126ad88c37eb89f01c4fda6d5049`、[全8 CI成功](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38336350087)。前処理Runtime 21 red → 21 green、固定Git package 108 green、公式CPU / 実GPU各6信号と次段検証graphへのTensor受け渡し合格。実encoder / 最終APIは別gate |
| [PR #46](https://github.com/ayutaz/unity-embeddinggemma-2/pull/46) | mainへ統合済み。音声前処理のWindows Sentis CPU / 実GPU受け入れを追加。製品PCM入力API・encoder・end-to-endの成功ではない |
| Unity / Sentis | 開発検証6000.3.21f1 / 2.6.1。mainのEditor固定はPR #31未統合のため6000.3.16f1。実測のsource / package / model hashを混同しない |

## 履歴の検証範囲

| 対象 | 確認済みと証拠 | 未達・限界 |
| --- | --- | --- |
| M1テキスト | Windows Editor tokenizer15件、FP32 / Float16重み × CPU / GPU。[完了記録](m1-completion-validation.md) | text128長の受け入れを画像 / 音声・他OSへ流用しない |
| 共通backbone固定Git Runtime | 既存consumerを固定SHAへ更新し、Runtime source overlayなし、requested / resolved一致。実text重み・合成project済みfeature、1024長・最終768次元、CPU / GPU × FP32 / Float16の4 passed / fail 0 / skip 0、CLI exit 0。[receipt](results/multimodal-git-runtime-upgrade-windows-20261013.json) | 公式Test Framework 1.6.0をembeddedにした条件。新規空consumer・実encoder・GPU前処理・公式prompt・sample / Player・他OSの合格ではない。初回import失敗は過去receiptへ保持 |
| 画像encoder | 3 profile × CPU / GPU × FP32 / Float16。[12条件](results/m3-vision-windows-plan-snapshot-20261013.json) | Float16 6条件合格、FP32 6条件はcosine合格でもelementwise不合格。I3以降の実装が残る |
| 音声前処理 | CPU / 実GPU各6信号、保存 / 再読込、mask完全一致。最大差 `0.0038890838623046875`。[実測](m4-audio-preprocessing-sentis-validation.md) | 診断出力のreadbackあり。製品PCM APIとencoder→backboneのGPU連結は別gate |
| PCM入力境界 | PR #48統合済み。25 red → 25 green、固定Git更新consumerで実Runtime型と全102ファイルを監査、package assembly 87 green。[receipt](results/m4-pcm-input-windows-20261013.json) | 新規空consumer・実encoder / 最終APIの受け入れとは分ける |
| 音声encoder診断 | 層内traceと単独feed-forwardが元の失敗経路へ一致。silence保存対eager最大差 `0.00021576881408691406`、許容差比 `9.733773231506348`、不一致368要素。[CI 38331914586](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331914586)。最初の微差はactivation | 診断76件greenはencoder成功ではない。丸め差の修正・元の6信号合格・Sentis実encoderが残る。過去328要素不一致も保持。固定許容差を維持 |
| Windows Player | 21f1精度4条件、sample日英GUI / FP32 GPU・全順位 / cache、独立2起動の実終了0。[条件と結果](m2-unity-21-upgrade-validation.md) | 既存実測SHA / 補修条件を維持。最終candidate・native入力・Float16性能・3モダリティのPlayerは未達 |
| Android / macOS / iOS | Androidは既存実APK buildとpayload監査まで。[Android記録](m2-android-validation.md) | Android実機とMac / iOSの3モダリティ・CPU / 実GPU・sample・終了は未実行 |

## 次の作業と依存

I3画像GPU前処理はPR #52で実装・Windows固定Git受け入れまで進んだ。固定source `71284e3`でpackage118 green、公式24ケースのpixel / position一致、最大8192px入力のdispatch修正も合格。[receipt](results/m3-image-preprocessing-final-git-windows-20261013.json)。次は画像FP32 / 音声encoderの数値差と公式prompt / mask。GPU profiler・反復資源 / 性能、実encoder・最終API / sample、他環境と最終空consumerは別gate。全体のred / green・完了条件は[全体計画](multimodal-tdd-plan.md#直近の実行順と依存)を参照する。M2〜M4、必須環境、配布準備の全完了後に正式リリースを扱う。

1. M2の導入helper / 21f1更新をmainへ渡す条件を整理し、残る性能・安定性を原因別red → 最小修正 → 同条件比較で進める。
2. PCM境界PR #48と共通Runtime PR #43は統合済み。前処理Runtime PR #50はTDD21 green、固定Git package108 green、公式CPU / 実GPU各6信号・mask・次段Tensor受け渡しが合格。[条件と失敗履歴](m4-audio-preprocessing-runtime-validation.md)。実encoder / backbone接続、最終空consumerは別gate。
3. 画像FP32差と音声activation差をそれぞれ小さい独立参照redへ分離し、修正後に元の実モデル条件を確認する。診断追加だけでは合格にしない。
4. I3画像GPU前処理と、A4合格後の実encoderへのGPU接続を境界・所有権・中間readback禁止のredから実装。I4 / A5の最終768次元、I5 / A6の検索sample、I6 / A7のPlayer・明示収録へ渡す。
5. Windows / macOS Editor・iOS / Android実機で、3モダリティ・混在索引・切替・cache・UI・CPU / 実GPU・時間 / memory / 正常終了を確認する。環境確保とライセンス / runner準備は早期から並行する。
6. すべての必須gate合格後、版 / 文書 / license / 配布監査と固定candidateの空Git consumerを確認する。正式Release / 公開tagは全対応後の別工程。

全振る舞いのTDD、tools/ + uv、Actions優先、main直push禁止を維持する。ローカルEditorは1つ、同operationの観測timeoutで重複実行しない。計画変更は文書レビューで確認し、架空のred / greenを追加しない。下記は過去の状態であり、現在の完了判定には上のscopeと固定receiptを使う。

## 履歴: PR #12時点のmainとブランチ

| 対象 | 確認した状態 |
| --- | --- |
| 確認基準main | `4ee0cbbde07fc99ca87d33ed52a44e6100ca0b0c`。PR #1〜#4・#6・#8〜#12統合済み。M1、CI・UPM、検索サンプル・モデル手順・準備最適化と統合記録を反映済み |
| [PR #9](https://github.com/ayutaz/unity-embeddinggemma-2/pull/9) | 計画更新を2026-10-10 02:05:30 JSTにsquash merge。commit `de82a29fc737bca0478c87d06ffcc4d94fb07e64`、統合後CI run 37963795561成功 |
| [PR #10](https://github.com/ayutaz/unity-embeddinggemma-2/pull/10) | 検索基盤を2026-10-10 02:09:08 JSTにsquash merge。commit `be791e00a1b344c5453f70037d7116184169c17f`、統合後CI run 37964228515成功 |
| [PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11) | 最終head `8c80526`のCI run 37974713580全8 job成功後、2026-10-10 03:41:24 JSTにsquash merge。commit `758cb04`、統合後CI run 37975026704全8 job成功。Windows consumer再現・UI・TDD完了 |
| [PR #12](https://github.com/ayutaz/unity-embeddinggemma-2/pull/12) | 統合後の計画・証跡を2026-10-10 03:47:25 JSTにsquash merge。commit `4ee0cbb`、統合後CI run 37975720534全8 job成功。文書のみでUnityを再実行していない |
| [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6) | CI整備・UPM化を2026-10-10 00:07:13 JSTにsquash merge。commit `5ae4e8c29a3a4c8639bd94f7848e3683d063f4d2` |
| [PR #8](https://github.com/ayutaz/unity-embeddinggemma-2/pull/8) | 文書更新を最新mainへ更新し、全8チェック成功後に2026-10-10 00:11:01 JSTにsquash merge。commit `c7d11897e43d374fbb87f3761f65fb3faec60c7c` |
| main保護 | サーバー設定済み。PR必須、strict Required CI（GitHub Actions App 15368）、管理者適用、force push / 削除禁止、linear history・会話解決。人手承認数0 |
| [PR #5](https://github.com/ayutaz/unity-embeddinggemma-2/pull/5) | 文書変更をPR #6へ含め、未マージでクローズ |
| [PR #7](https://github.com/ayutaz/unity-embeddinggemma-2/pull/7) | 文書のみPRのCI確認用。検証後に未マージでクローズ |

全PRを対象とするCI入口、UPMコードと検索APIはmainへ統合済み。サンプルとモデル準備手順はPR #11で追加。mainへの直接pushや保護の迂回は行っていない。[開発規則](../AGENTS.md)に従い、マージは依頼された範囲でCI・差分・競合を確認して行う。

## 履歴: 導入・検索基盤の検証

| 対象 | 証拠・結果 | 限界 |
| --- | --- | --- |
| M1 | [完了検証](m1-completion-validation.md)。tokenizer全15件一致、fp32 / Float16重みのCPU / GPUCompute各15件、保存・再読み込み・時間 / メモリ測定合格 | Windows Editor 6000.3.16f1 / Sentis 2.6.1、固定モデル・batch 1 / length 128 / 768次元 |
| 確認基準mainのCI | [run 37975720534](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37975720534)。main `4ee0cbb`の全8 job成功。Python4環境、実モデルPython M1 / 検索照合、lint・パッケージ監査・Required CI成功 | Python照合・パッケージ静的監査であり、新たなUnity実行ではない |
| PR #6のCI | [run 37812870591](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37812870591)。Ubuntu / Windows × Python 3.13 / 3.14各102件、lint、実モデルPython15件、パッケージ監査、Required CIの全8 job成功 | 実際のcheckoutは `9ccff937ae9fc675c12aee8e1533d08cc0502e9a`。確認したPR headとtree一致。Python照合はSentis実行ではない |
| 文書のみPR | [run 37813857997](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37813857997)。PR #7はMarkdown 1ファイル・6行追加だけで全8 job成功 | baseは `feat/ci-upm-package`、headは `9633a1ca8d5f3a01eff427599187329c2447b68c`。古いmainへの統合済みという意味ではない |
| UPM導入 | [パッケージ検証](m2-package-validation.md)。別の空のUnityプロジェクトでローカルフォルダ依存解決・compile・契約29件成功、URP依存なし | 当時はモデル不要の導入検証。後続consumerの実モデルGPU結果は別行。Git URL導入は未実行 |
| UPM移行後の回帰 | 元プロジェクトで契約29件・測定契約1件、実モデルM1 3件・API 2件成功、failed / skipped / inconclusive = 0 | 保存・量子化・ベンチマーク全体を移行後に再実行した結果ではない。元M1実測を維持 |
| 検索サンプルのCI | [run 37959339466](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/37959339466)。PR #11の実装headで全8 job成功、Python4環境各132件、M1 15入力 + 検索10入力のexport照合合格 | 実checkout `13f194266162d2a728b2b2a0da9e9e58ae314ff6`。Python照合をSentis GPUの成功に数えない |
| Windows検索実モデル | [検索検証記録](m2-search-validation.md) / [全順位と数値](results/m2-search-root-windows-20261010.json)。fp32 / Float16重み × CPU / GPUComputeの4 passed、failed / skipped / inconclusive = 0。固定6文書 / 4queryの全順位一致 | PR側のサンプル、元の検証プロジェクト。他環境や任意入力の品質保証ではない |
| 検索画面操作 | 元プロジェクトGPUとconsumer CPUでGame View入力による準備・日英検索・空入力・解放を確認。元プロジェクトではモデル欠落も確認 | 画面画像と操作記録を保存。任意入力の品質評価ではない |
| 新規consumer | [改善後記録](results/m2-search-consumer-completed-windows-20261010.json)。別checkout / 初期空consumer、導入・監査・変換、sample 18 passed、実モデル4 passed / skip 0、CLI完了応答・全順位・画面操作 | 以前の中断を履歴保持。ローカル依存 / artifactキャッシュは再利用。allocation / fontログの発生元は未確定 |

パッケージは `com.ayutaz.embeddinggemma` / `0.1.0-pre.1`、未リリース。
PR #6のUPM移行時にはRuntimeと契約テストの19ファイルが移行前とGit blob一致し、assembly名・GUIDを維持した。後続の検索API・契約追加は別のTDDと検証記録で扱う。
直接依存はSentis 2.6.1 / Newtonsoft JSON 3.2.2。モデル・ネイティブプラグイン・URP・uloop・Pythonを配布Runtimeへ含めない。

## 履歴: 実装と検証の進展

PR #29統合SHA `fc66af7`を新しい空のGit consumerへ導入し、requested / resolved SHA、配布92 / sample 52ファイル、compile、UPM 38 / sample EditMode 66 / PlayMode 2、token15、実モデル4条件・全順位 / score、日英UI / 空入力 / 欠落モデル / 解放 / file URL初回・warm cacheを確認した。[結果と制約](m2-font-git-consumer-validation.md)。最初のUIでfont Warning 1 / Editor検索DB Error 1を保持した。測定fixtureが破棄済みfontをcacheへ残す経路を実証し、Editor test 2ファイルだけの修正 `bc3ac49`で1 failed → 1 passed、全67 passed、invalid font参照0、未準備UI 3回の停止前後Console 0を確認した。検索DB原因・native allocation・Player sample画面 / 他OS実機・候補 / tag / 正式Releaseは未完了。

Windows Editorのsample fontが停止で破棄されてもIMGUI cacheに参照が残る経路を再現し、Editor内で1つのnative fontを共有する修正をTDDで確認した。red 2 failed → green 2 passed、sample EditMode 66 / PlayMode 2、未準備UI 3回の停止前後Console 0、実Float16 GPU日英検索の全6順位・score照合、空入力・解放・停止が合格。[当初の実測と制約](m2-editor-font-lifetime-validation.md)を参照。修正source `f7ffbfd`の当初検証は作業ソース配置で、後続 `fc66af7`の新Git導入は上記で別に確認した。正常Editor終了時の所有font破棄も確認済み。次は残るEditor検索DB / native allocation診断、Player sample画面、他OS / 実機・正式Release gateを進める。

ユーザーが安定化から正式リリースまでの実行を依頼した。[詳細実行計画](m2-release-plan.md)に4段階の受け渡しと完了条件を記載した。新しい実測はその実行SHA・環境で記録し、以下の既存履歴を書き換えない。

PR #14 / #15は統合済み。main `f564987` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38008701772) は成功した。Git URLは固定commitのPackage Manager解決まで成功し、Editor初回起動は失敗している。Playerの監査済み実モデルbundle準備は成功し、build / 実行は未実行。Windows Editorの完全SHA-256改善は28契約テスト合格、実モデルキャッシュ照合7.38秒の単回測定を確認した。[高速化記録](results/m2-editor-sha256-windows-20261010.json)と詳細計画に現在の証拠を追記し、以下の既存履歴を維持する。

PR #16も統合し、main `0a6e299` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38009534632) 全8 job成功を確認した。続いてPlayer用のhash監査・厳密な参照照合APIをTDDで追加し、合成契約40件が合格した。実行component / build / 実機は未実行。[Player準備](player-validation.md)と詳細計画に受け渡しを記録している。

そのAPIはPR #17で統合し、main `e7b3a54` の [CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38010849703) 全8 job成功を確認した。後続のWindows Playerでは起動・build・測定を接続し、契約65件と実モデル4条件の独立した2回の起動が合格した。M1 15ケース、検索全順位、25入力のtoken一致と終了 / 再起動を確認した。[実測と初回失敗](player-validation.md#windows-player実測-2026-10-10)を参照。Development / Mono2x / stripping Disabled限定で、build警告と終了時のmemory診断は残っている。Git URL consumerと他環境の完了を意味しない。

Windows Playerの変更はPR #18の全8 CI成功後にmain `f44976b` へ統合した。続いて[新しいGit consumerの実動作](m2-git-consumer-runtime-validation.md)を確認し、Git固定SHA解決、compile、UPM 38 / sample 28契約、token15入力と実モデル検索4条件、Game Viewの日英検索 / 空入力 / 解放 / Play停止が合格した。元のconsumer起動失敗は履歴保持。標準layout・project path・SHA等を変えた成功であり、以前の停止原因は確定していない。画面の文字欠けとPersistent allocation Log 1件はリリース前に追跡する。

1. macOS Editor・iOS・Androidのrunner / toolchain / 実機を確保し、精度・backend・速度・メモリを実測する。
2. 測定と並行してhash確認の負荷を改善し、再現しなかったallocationログとEditor font警告を切り分ける。domain reloadの原因は未確定。
3. Git URLのcommit / tag固定導入とサンプル起動、文書・ライセンス・CHANGELOGを確認し、テキスト版をリリースする。ここまででM2完了。
4. M3で画像エンコーダ・GPU画像前処理・テキスト→画像検索を実装・照合する。
5. M4で音声エンコーダ・GPUメル前処理・テキスト→音声検索を実装・照合する。

詳細な依存・受け入れ条件は [M2計画](m2-plan.md)、全体のゴールは [ゴール](goal.md)。

## 未実行と失敗の扱い

- macOS / iOS / Android実機 / WebGPUは未検証。Android build・payload監査は成功したが実機成功ではない。旧16f1 sampleのshutdown失敗は履歴保持。21f1の通常Windows精度Playerとsampleでは独立2起動の実終了0を確認した。検証assemblyのpreserve-allや小さい契約モデルを任意consumer / 実機の成功へ一般化しない。
- クラウドUnityは任意手動Linux CPU検証。Secrets / Variablesは未登録、Editor job未実行。必要なのはこのクラウド経路の利用時で、検索サンプルなどの開発を止める条件ではない。
- PR #6当時のconsumerのuloop launch readinessはタイムアウト。後続run-testsのcompile・29 passedとは分けて履歴を保持する。検索導入では長いパスの失敗後、別の短いconsumerの起動・compile・テストが成功した。
- GPU skipやCPU代替をGPU成功として扱わない。モデルと大きいartifactはGit管理外、LFSを使わずActionsで再生成する。

## 履歴: 各checkpointの結果

PR #26をhead `d888e66`の全8 CI成功後にsquash mergeし、main `d6c8442`の[CI run 38027319740](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38027319740)も全8 job成功。元checkoutのUnity設定変更を保持した。モデルなし空sceneでPackage Manager Errorを観測し、回復時の直接uloop起動ではALLUSERSPROFILEが欠けていたことを確認した。旧PID終了・Editor本体0確認後、既存uvハーネスで環境補完して1回起動し、compile Error / Warning 0。以降の空scene 3回はConsole 0件、未準備sample UI 3回は0 / Warning 1 / Warning 1で、font警告をモデル推論なしに再現した。元scene / leak設定は復元済み。6回の比較ではallocation診断は出なかったが、次のPlay開始直後にFound leak診断15件を観測した。責任箇所は未確定。font原因・解消とdomain reload単体の高速化は未確認。[条件・結果・次の切り分け](m2-stability-environment-validation.md)。

Androidの現在のC#はcompileと12契約が合格した。約37分の継続したMainThreadBlockedを確認した後、旧Editor PID 110956の終了・本体0確認を経て1回だけ回復起動し、新PID 127008で実APKをbuildした。source `751c30a`、ARM64 / IL2CPP / compiler Release / High stripping、504.158秒 / errors 0 / warnings 970。SDK v2 debug署名検証と、実APK 1,720,031,692 bytesの全payload完全hash・ARM64監査が成功した。scene / settingsとsample payload / metaは復元済み。実receiptの固定BuildOptions対応は1 redからAPK19 green。[結果と限界](m2-android-validation.md)、[機械可読結果](results/m2-android-apk-build-20261010.json)を参照。実機・jar runtime・Android CPU / GPU・sample Player画面は未実行。停止原因・警告解消・他OS・安定性・tag / Releaseは未完了。PR #27はhead `6ab5278`の全8 CI成功後にmain `c3281cd`へ統合済み。PR #26もその文書更新を取り込んだ。当時の観測timeoutはEditor操作の終了証拠にはせず、その後の回復と区別する。元checkoutのUnity設定変更は保持した。

Android Player報告の監査CLIを追加した。固定APK provenance / 同梱参照hash、全4条件のactual backend / 15ケース / 全順位・score / 解放、初回jar展開とwarm cache、3回warm timingと14段階memory、異なる非重複runを照合する。39 redと追加5 redから関連73 passed。小さいfixtureによる整合性監査で、ADB実行・端末identity・process終了は別の実測証拠を要求する。[回収手順と範囲](m2-android-validation.md#実機結果の回収と報告監査)。実APKは上記で確認済みだが、実機 / 他OSと正式Release gateは残る。

Android用build helperとAPK payload監査の準備時点では、Windows targetでhelperの9 red、graphics保存リスト復元の2 redから既存を含む110 passed。小さいAPK fixtureは16 redから関連29 passedだった。その後のAPI変更を含むC#は当時再検証待ちでdraftにした。現在は上記のAndroid12契約と実APKの成功を追加したが、以前の110件の合格範囲は拡大しない。[準備履歴と現在の結果](m2-android-validation.md)。

PR #24統合main `01f3d86`を新しい空のGit consumerへ固定導入し、requested / resolved SHA・88ファイル・import済みsampleを照合した。compile、UPM 38 / sample EditMode 64、token15入力、実モデル検索FP32 / Float16 × CPU / GPUの4条件と全順位・scoreが合格。sample PlayMode 2件は受理後のCLI接続失敗を保持し、同じ実行のUnity保存XMLから2 passedを回収した。Game Viewの日英入力・準備・検索・空入力・解放・warm完全hash・Play停止も確認した。[新規Git導入の結果](m2-latest-git-consumer-validation.md)。旧Editorの終了を確認してから新Editor1つを起動し、再起動なし。停止時Persistent allocation Log 1件の原因、実APK / 他OS / 実機 / tag導入と正式Release gateは残る。

配布TextSearch sampleの選択モデル配置と所有cacheを実装し、実行source `2e463a8`でEditMode 64 / PlayMode 2 / uv Python 22件が合格した。Windows Editorの実file URL転送・FP32 / Float16のactual GPU準備・固定日英2queryの全順位とscore・空入力・解放・欠落モデル拒否・実worker使用中のleaseを確認した。warmではmanifestだけを転送しモデル再転送0、完全CNG監査約803ms / 418ms、成功後は選択Float16 1組の約586MBだけを保持した。[実測と制約](m2-sample-cache-validation.md)。stage時間はmodel deserialize / 文書推論を含まない。変更sample SHAの新規Git導入、実APK / 実機と新sample Player画面は未実行、font Warning 1件の由来も未確定。次は新規固定Git導入と実APK / 他環境のgateを進める。

Windows IL2CPP / compiler Release / managed stripping Highのbuildと実モデル2起動を確認した。release build helperの3 red → 99 green、build 257.1秒 / errors 0 / warnings 485、両Player終了code 0。4条件のactual CPU / GPU、25入力token一致、M1・検索全順位・score・解放が合格。file URLの初回6件展開から2回目manifest 1件だけのcache再利用へ移行し、実CNGの完全hash監査1.47秒 / 1.55秒、モデル再転送0、保持1 bundleを確認した。[結果と範囲](m2-player-il2cpp-validation.md)。検証assemblyにpreserve-allを指定した条件で、任意consumerのstripping・他OS・native leak解消を証明した結果ではない。次は配布sampleのAndroid配置、実APK / 他OS / 実機、残る安定性とrelease gateを進める。

PlayerのURL配置は、完全SHA-256を毎回照合する所有cacheへ改善した。cacheのred / greenとWindows CNGのred / green、全96契約が合格。同じ約1.68GBの実bundleでwarm監査3回が合格し、モデル再転送0、監査約1.5〜1.8秒を確認した。[測定と限界](m2-bundle-cache-validation.md)にbaseline約108秒の観測とNUnit timeout失敗を保存する。保持modelは成功後1 bundle、更新中最大2 bundle。後続のWindows Player / IL2CPP結果は上記に追加し、他OS・実機 / sample配置と安定性・正式releaseのgateは残る。

Player用のStreamingAssets展開adapterは新規12件のredから、既存を含む77件のgreenへ移行した。Windows Editorの実UnityWebRequestによる小さいfile URL転送と欠落処理を確認し、jar transportは注入して検証した。[配置経路と残る条件](player-validation.md#streamingassetsからの展開adapter)を参照。展開後も完全SHA-256・token・全backend条件を要求する。後続のWindows実Player / cache確認は上記に追加した。実APK / Android端末 / 配布sample配置は未実行。環境inventoryではGitHub Secretとrunnerは0、ADB接続端末も0だった。他OS / 実機の合格や正式releaseを完了とはしない。

検索UIの文字欠けは、実OS fontのglyph境界を使うTDDで再現し、font・sizeを明示したcached styleへ修正した。EditMode 34件とPlayMode 1件が合格し、Windows Game Viewの日英GPU検索の全順位・scoreが修正前と完全一致、空入力・解放・モデル欠落表示・Play停止も確認した。[表示修正の実測](m2-ui-font-validation.md)を参照。修正したsampleはGit consumerへ作業ソースとして配置したもので、修正SHAの新規Git導入成功とは扱わない。allocationとdomain reloadの原因、他OS / 実機と正式リリースの条件は未達のまま残す。Windows IL2CPP / release strippingの後続結果は上記を参照。

## 文書の読み方

README、ゴール、M2計画、CI、API、automationは現在の利用・作業手順。
M1の各plan / validationは実行した段階の履歴で、当時のcommit・数値・失敗を維持する。
[技術調査](technical-approach.md)は採用済みのテキスト経路と未実装の画像 / 音声設計を区別する。
[新規性調査](embeddinggemma-2-unity-novelty.md)は2026-10-07の調査に2026-10-09の限定検索を追記したもので、先行実装の不存在を保証しない。
package内の文書は導入 / API / 未リリースと、mainへ統合したサンプルの使い方を説明する。AGENTS.mdとApacheライセンスは規則・条文を維持する。

## 文書一覧と現在の読み方

2026-10-14に次の全文を読み込み、現行コード・証拠・停止状態との参照関係を確認した。ライセンスと開発規則は変更せず、既存結果JSONも書き換えない。履歴文書の未実行や次の作業は当時のscopeで、上の現在判定が優先する。

| 文書 | 位置づけ |
| --- | --- |
| [README.md](../README.md) | 現在の仕様・手順 |
| [docs/automation.md](automation.md) | 現在の仕様・手順 |
| [docs/ci.md](ci.md) | 現在の仕様・手順 |
| [docs/embeddinggemma-2-unity-novelty.md](embeddinggemma-2-unity-novelty.md) | 固定sourceの履歴 |
| [docs/goal.md](goal.md) | 現在の仕様・手順 |
| [docs/gpu-constant-il2cpp-validation.md](gpu-constant-il2cpp-validation.md) | 固定sourceの履歴 |
| [docs/gpu-constant-upload-validation.md](gpu-constant-upload-validation.md) | 固定sourceの履歴 |
| [docs/m1-completion-plan.md](m1-completion-plan.md) | 固定sourceの履歴 |
| [docs/m1-completion-validation.md](m1-completion-validation.md) | 固定sourceの履歴 |
| [docs/m1-plan.md](m1-plan.md) | 固定sourceの履歴 |
| [docs/m1-runtime-plan.md](m1-runtime-plan.md) | 固定sourceの履歴 |
| [docs/m1-runtime-validation.md](m1-runtime-validation.md) | 固定sourceの履歴 |
| [docs/m1-validation.md](m1-validation.md) | 固定sourceの履歴 |
| [docs/m2-android-validation.md](m2-android-validation.md) | 固定sourceの履歴 |
| [docs/m2-bundle-cache-validation.md](m2-bundle-cache-validation.md) | 固定sourceの履歴 |
| [docs/m2-editor-font-lifetime-validation.md](m2-editor-font-lifetime-validation.md) | 固定sourceの履歴 |
| [docs/m2-font-git-consumer-validation.md](m2-font-git-consumer-validation.md) | 固定sourceの履歴 |
| [docs/m2-git-consumer-runtime-validation.md](m2-git-consumer-runtime-validation.md) | 固定sourceの履歴 |
| [docs/m2-latest-git-consumer-validation.md](m2-latest-git-consumer-validation.md) | 固定sourceの履歴 |
| [docs/m2-package-validation.md](m2-package-validation.md) | 固定sourceの履歴 |
| [docs/m2-plan.md](m2-plan.md) | 現在の計画と履歴 |
| [docs/m2-player-il2cpp-validation.md](m2-player-il2cpp-validation.md) | 固定sourceの履歴 |
| [docs/m2-release-plan.md](m2-release-plan.md) | 現在の計画と履歴 |
| [docs/m2-sample-cache-validation.md](m2-sample-cache-validation.md) | 固定sourceの履歴 |
| [docs/m2-search-plan.md](m2-search-plan.md) | 現在の計画と履歴 |
| [docs/m2-search-validation.md](m2-search-validation.md) | 固定sourceの履歴 |
| [docs/m2-stability-environment-validation.md](m2-stability-environment-validation.md) | 固定sourceの履歴 |
| [docs/m2-ui-font-validation.md](m2-ui-font-validation.md) | 固定sourceの履歴 |
| [docs/m2-unity-21-upgrade-validation.md](m2-unity-21-upgrade-validation.md) | 固定sourceの履歴 |
| [docs/m3-image-cpu-preprocessing-validation.md](m3-image-cpu-preprocessing-validation.md) | 固定sourceの履歴 |
| [docs/m3-image-embedder-validation.md](m3-image-embedder-validation.md) | 固定sourceの履歴 |
| [docs/m3-image-embedding-reference-validation.md](m3-image-embedding-reference-validation.md) | 固定sourceの履歴 |
| [docs/m3-image-feature-encoder-validation.md](m3-image-feature-encoder-validation.md) | 固定sourceの履歴 |
| [docs/m3-image-preprocessing-validation.md](m3-image-preprocessing-validation.md) | 固定sourceの履歴 |
| [docs/m3-plan.md](m3-plan.md) | 現在の計画と履歴 |
| [docs/m3-vision-export-validation.md](m3-vision-export-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-embedder-validation.md](m4-audio-embedder-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-encoder-runtime-validation.md](m4-audio-encoder-runtime-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-encoder-validation.md](m4-audio-encoder-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-final-reference-validation.md](m4-audio-final-reference-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-native-onnx-validation.md](m4-audio-native-onnx-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-preprocessing-runtime-validation.md](m4-audio-preprocessing-runtime-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-preprocessing-sentis-validation.md](m4-audio-preprocessing-sentis-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-preprocessing-validation.md](m4-audio-preprocessing-validation.md) | 固定sourceの履歴 |
| [docs/m4-audio-real-chain-validation.md](m4-audio-real-chain-validation.md) | 固定sourceの履歴 |
| [docs/m4-current-coefficient-bound-layer-validation.md](m4-current-coefficient-bound-layer-validation.md) | 固定sourceの履歴 |
| [docs/m4-current-encoder-observation-validation.md](m4-current-encoder-observation-validation.md) | 固定sourceの履歴 |
| [docs/m4-feed-forward-stage-observation-validation.md](m4-feed-forward-stage-observation-validation.md) | 固定sourceの履歴 |
| [docs/m4-fp64-projection-oracle-validation.md](m4-fp64-projection-oracle-validation.md) | 固定sourceの履歴 |
| [docs/m4-layer-zero-norm-observation-validation.md](m4-layer-zero-norm-observation-validation.md) | 固定sourceの履歴 |
| [docs/m4-native-feed-forward-reference-validation.md](m4-native-feed-forward-reference-validation.md) | 固定sourceの履歴 |
| [docs/m4-native-layer-replay-validation.md](m4-native-layer-replay-validation.md) | 固定sourceの履歴 |
| [docs/m4-native-silu-swish-probe-validation.md](m4-native-silu-swish-probe-validation.md) | 固定sourceの履歴 |
| [docs/m4-pcm-input-validation.md](m4-pcm-input-validation.md) | 固定sourceの履歴 |
| [docs/m4-plan.md](m4-plan.md) | 現在の計画と履歴 |
| [docs/m4-post-norm-isolation-validation.md](m4-post-norm-isolation-validation.md) | 固定sourceの履歴 |
| [docs/model-license.md](model-license.md) | 現在の仕様・手順 |
| [docs/model-preparation.md](model-preparation.md) | 現在の仕様・手順 |
| [docs/multimodal-backbone-validation.md](multimodal-backbone-validation.md) | 固定sourceの履歴 |
| [docs/multimodal-prompt-validation.md](multimodal-prompt-validation.md) | 固定sourceの履歴 |
| [docs/multimodal-runtime-validation.md](multimodal-runtime-validation.md) | 固定sourceの履歴 |
| [docs/multimodal-tdd-plan.md](multimodal-tdd-plan.md) | 現在の計画と履歴 |
| [docs/player-validation.md](player-validation.md) | 固定sourceの履歴 |
| [docs/runtime-api.md](runtime-api.md) | 現在の仕様・手順 |
| [docs/status.md](status.md) | 現在の仕様・手順 |
| [docs/steps-1-3-acceptance.md](steps-1-3-acceptance.md) | 現在の仕様・手順 |
| [docs/technical-approach.md](technical-approach.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/CHANGELOG.md](../Packages/com.ayutaz.embeddinggemma/CHANGELOG.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/Documentation~/audio-feature-encoder.md](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-feature-encoder.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/Documentation~/index.md](../Packages/com.ayutaz.embeddinggemma/Documentation~/index.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/README.md](../Packages/com.ayutaz.embeddinggemma/README.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/Samples~/README.md](../Packages/com.ayutaz.embeddinggemma/Samples~/README.md) | 現在の仕様・手順 |
| [Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md](../Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md) | 現在の仕様・手順 |
| [validation/com.ayutaz.embeddinggemma.validation/README.md](../validation/com.ayutaz.embeddinggemma.validation/README.md) | 現在の仕様・手順 |
