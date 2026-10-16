# M3: 画像対応のTDD計画

> 文書整合確認: 2026-10-15。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。冒頭の現在方針が、下の旧checkpointの着手順より優先する。

## 現在の画像API検証

2026-10-16: [実Texture全6件 × Query / Document / Raw](results/m3-final-image-api-all-cases-all-roles-gpu-20261016.json)の全18条件が最終GPU APIで合格。原本参照・encoder `atol=1e-5, rtol=1e-4`を維持し、元FP32全6 encoderの全bit一致を前提として、前処理→encoder→backbone→正規化768次元を照合した。最小cosine 0.9999999999976248、追加の最終要素診断も超過0。ケースごとのsourceと同一Runtime treeを記録し、単一head実行へ付け替えない。

[実RenderTextureと拒否 / 解放 / 再利用・text回帰](results/m3-rendertexture-recovery-and-text-gpu-regression-20261016.json)では、不正入力6種類と実GPU stage後キャンセルから全3 roleへ復帰し、PNG最終captureと全bit一致。実text128はFP32 / Float16重みとも10ベクトル・全4順位 / score合格、選択した契約回帰90 Green。非同期APIは8 Red→38 Greenで追加。元の[実行工程4](results/goal4-scope-audit-20261016.json)の画像API側を進めた結果であり、音声PCM側と最終checksの完了は別に判定する。M3全体の検索sample・Player / device / 長期memory等は後続の工程5に保持する。Persistent allocation警告を成功へ置き換えない。

以下は以前のcheckpointと当時の状態。

画像encoder全体は[元FP32全6ケース・3 profileでGPU合格](results/m3-adjacent-reciprocal-all-image-gpu-passed-20261015.json)。固定source `38baf10`で元778,240値全bit一致、6 pass / fail0 / skip0、fallback / 中間readback0。旧候補の未達とは別結果。製品接続adapterは6 Red→関連36 Greenまで実装し、[製品adapterと実Texture](results/m3-production-image-adapter-and-real-texture-gpu-20261015.json)は、元square-1の131,072値がGPU前処理から全bit一致・1 pass / skip0。次は残り実画像ケースとbackbone / 全role最終768 APIへ進む。初回PNGのARGB32形式拒否は2 Red→関連23 Greenで修正し、推論前の失敗履歴も保持する。sample / Player・必須環境・Required CI・長期memoryも残す。Python回帰はsource `cd08e0f`で4環境各1,070 pass / skip0。

最新の優先順: [元CPU参照の再現・第4層への局所化](results/m3-reproduced-cpu-context-and-prefix-localization-20261015.json)と[実GPU逆数丸めの修正](results/m3-gpu-reciprocal-midpoint-and-fourth-layer-fixed-20261015.json)。元参照を再現したCPU診断から、最初の差は第4層token173のQuery RMSと判明。実GPUのseedはCPUと異なり、Newton補正後も正しい逆数より1 ULP下だった。隣接候補を補償FP32残差で比較し、3 Red→関連42 Green、実GPU65,601統計値・7 RMS / GELU・接続4層1,935,360値全bit一致を確認した。次は修正済み元画像全6ケース→製品GPU encoder採用→実Textureから全roleの最終768 API・寿命・例外後再利用・text回帰。元goldenと許容差は維持する。工程3/4全体は未完了。

以下は前回checkpoint。当時実行中だったCPU診断は上記の通り完了している。

最新局所化: [集約前の最終encoder出力](results/m3-last-encoder-boundary-and-ordered-diagnostic-20261015.json)。元square-0の第1層全bit一致に対し、16層後・集約前は1,894,029 bit差 / 13,977要素超過・最大比33.287118の1 fail / skip0。GPUCompute・CPU fallback / 中間flush0・元係数保持。poolingだけでは説明できない。次は明示CPU診断の元patch / 第1層SHA→最終bit一致gateを通った捕捉だけで層を局所化する。CPU softmax単独483,840値bit一致・関連61 Greenは、元encoderの再現やGPU全体合格とは別。Actions 38739806295は実行中。製品GPU経路と元参照・許容差を維持する。

直前checkpoint: [pooling単独の修正と全体未達](results/m3-native-pooling-component-and-whole-failure-20261015.json)。標準Sentisの元constant-left poolingは78,179 bit差・2要素超過、同じ入力と係数のblock192は196,608値全bit一致。固定left演算と分割境界を9 Red→関連23 Greenで追加したが、接続した元square-1は5要素超過・最大比1.409017の1 fail / skip0。batchを停止し、最新pooling候補の他5件は未実行。旧rotary候補の下表4 pass / 2 failと分離する。既存Runtime既定経路を維持し、製品採用・工程3/4の完了とは扱わない。

局所checkpoint: [第1層と演算順序の実GPU照合](results/m3-component-order-and-first-layer-gpu-20261015.json)。元nativeのpatch→第1層1,935,360値がGPUで全bit一致し、slice回帰を含む7 pass / skip0。AVX512同一入力のsoftmax / GELU / scale前RMS、AVX2の4行列形状のblock192、固定65,536統計値の逆平方根も各限定範囲で全bit一致。最初のscale境界誤比較とslice index誤分類はharness失敗、ISA-only環境のAVX512不足はモデル未実行として保持。第2層以降や全体の合格には数えない。その後のpooling結果は上のcheckpointを参照。

全画像GPU checkpoint: [元FP32全6ケースの実測](results/m3-native-rotary-all-image-gates-20261015.json)。**全6ケースを実行し、4 pass / 2 fail / skip0。** 不合格はsquare-1の5要素とlandscape-0の1要素で、元参照・`atol=1e-5, rtol=1e-4`を維持する。各receiptは実際の固定source SHAを保持する。profileやモデル全体の合格へ未実行 / 不合格を混ぜない。

| case | 実GPU判定 | 許容差超過 | 最大許容差比 |
| --- | --- | ---: | ---: |
| square-0 | pass | 0 | 0.593776 |
| square-1 | fail | 5 | 1.406222 |
| landscape-0 | fail | 1 | 1.321967 |
| landscape-1 | pass | 0 | 0.224743 |
| portrait-0 | pass | 0 | 0.203042 |
| portrait-1 | pass | 0 | 0.056726 |

全ケースでGPUCompute、計算/入力CPU fallback0・中間flush0、元学習済み投影係数の保持を確認。派生rotary表のみ、元nativeの入力非依存値へ有界補正したvalidation候補。最大host peakは約4.64GiBで、各caseは別native process / 300秒上限 / owned 12GiB監視付き。実行＋最終取得は約14.7〜25.7秒の単発観測で、正式benchmarkではない。全runner / watcherはterminal、Unity終了済み。

全3 profileのrotary表は[入力非依存の元graphだけから取得済み](results/m3-original-rotary-profile-captures-20261015.json)。Squareは既存の元native byteを再現。profile対応付けは4 intended Red→関連Editor15 Greenで、誤対応 / 曖昧さ / 非有限を拒否する。FP32 252行tail templateも各該当profileで使用した。source `a5423c4...`のPython回帰はWindows / Linux ×3.13 / 3.14で各1,029 pass / skip0。これは最新headのRequired CIやGPUモデル全条件合格とは別である。

次は不合格2ケースの境界値から残る算術差を切り分け、同じ全6 gateへ戻す。製品ImageFeatureEncoderへの採用、実PCM / Texture→GPU前処理→encoder→backbone→正規化768 API全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4 sample / Player / 必須環境 / Required CI / 長期memoryは未完了。既存音声6ケース / 2 profileの元GPU証拠は保持し、画像の未達を縮めず工程4まで進める。

以下は前回checkpoint。

最新GPU checkpoint: [元rotary定数の丸め差・修正と全体GPU結果](results/m3-original-rotary-and-mkl-context-20261015.json)。元保存modelはCos / Sin演算やfull-widthの定数Concatを持たず、`[1,2520,1,32]`の4種類の派生rotary表をQ / K ×16層で32回ずつ持つ。元nativeの対応表へcos各半分21,936 bit差、sin各半分12,576 bit差、最大差5.96e-8を静的auditで確認した。初めの2種類の探索失敗は数値未実行のsetup failureとして分離。元graph / caller配列を変更せず、1e-6以内の位置表補正だけを独立所有で再bindするhelperを4 API Red→元表2 Redを含む関連6 Greenで確認。これはvalidation-onlyの定数補正であり、学習済み投影係数や元goldenは変えていない。

補正4表を128箇所のrotary乗算へ接続した`d345886...`は、**元square-0の全GPU gateに1 pass / skip0、許容差超過0・最大比0.593776で合格**。GPUCompute・計算/入力CPU fallback0・中間flush0・元学習済み係数保持。host peak4,941,717,504 bytes（約4.60GiB）、実行＋最終取得25.539秒 / process86.900秒の単発観測。続く`759f116...`のsquare-1は**5要素超過・最大比1.406222で1 fail / skip0**。host peak4,646,338,560 bytes、実行＋最終取得14.711秒 / process72.828秒。正方形profile全体や他4画像caseの合格へ広げず、製品のImageFeatureEncoder既定経路には未採用。runner / owned監視は全てterminalでUnity終了済み。

CPU参照の追加調査は、Intel Xeon8573C / 実AVX512のnativeでは598〜600超過、MKLのみAVX2制限ではpatch全bit一致でも全体5超過を保持し、いずれも新しいgoldenにしない。次は元保存graphの画像入力から独立したrotary依存部分だけを評価し、全3 profileの位置表をActionsで取得する。無関係な前段やencoderを実行しないFX sliceとdispatchを2 Red→関連52 Greenで追加した。横長 / 縦長の元4 GPUケースとsquare-1の修正は未完了。Python回帰のsource `9cf2292...`はWindows / Linux ×3.13 / 3.14すべて各1,020 pass / skip0で、最新headのRequired CIやモデルGPU合格とは別。元画像全6条件、既存音声6ケース / 2 profile、実PCM / Texture→GPU前処理→encoder→backbone→正規化768 API全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4 sample / Player / 必須環境 / 長期memoryの全範囲を維持する。工程3・4は未完了。

最新参照checkpoint: [実AVX512でも元参照を再現しなかったhost差](results/m3-avx512-host-context-failed-20261015.json)。[Actions 38723926213](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38723926213)は実AVX512事前確認後、Intel Xeon Platinum 8573Cでthread 1 / 2は598要素、4は600要素が元golden超過しfailure。以前のAMD EPYC 9V45では元値を全bit再現した。同じ保存graph / 入力SHA・torch版・AVX512表示でも、今回のpatch `add_1`は最初から異なり、rotaryは一致。全捕捉fileのSHAと通常 / 観測の最終bit一致を確認したが、新しい捕捉を元参照として採用しない。AVX512の有無だけで参照文脈を同一と扱わない。次はATen nativeを保ちMKLだけAVX2に制限する仮説をActionsで確認する。元patch SHAと全bit一致した時だけ全encoder参照再生へ進む前段gateを7 Red→関連50 Greenで追加。prefix以降が実行されないこと、非有限 / FP64の拒否、graph / 入力の保持を小さいFXで確認。MKL仮説の実モデル結果はまだ未取得。画像GPU元全体の残り1要素、全6画像条件と工程3・4 API / sample / Player / 必須環境 / Required CI / 長期memoryの未達を維持する。ローカルUnity終了済み。

追加checkpoint: [252行の末尾RMSとprofile形状の修正](results/m3-tail-profile-shape-gpu-20261015.json)。正しい252行templateを拒否し、252行のsourceへ256行templateを誤接続できる問題を4 intended Redで確認。sourceの実中間shapeを推論してtemplateと照合する修正により形状5 Green、source / templateの非破壊性と異なるprofile / widthの拒否を確認した。元AVX512 square末尾の先頭252行を使う独立fixtureで、Python / 標準FP32 ONNX / 実GPUの193,536値が全bit一致。GPU1件と形状5件の合計6 pass / skip0、fallback / flush0。252行の生成templateを元全画像6ケースのharnessへ接続し、接続後のC# compile / 形状5 pass / skip0も確認。横長 / 縦長の実encoder全体は未実行で合格とは扱わない。代表square-0の元全体gateは1要素超過のまま。元golden / 許容差、実入力API・sample / Playerを含む工程3・4の全範囲を維持する。AVX512 prefix取得と回帰CIは既存Actions runを待ち、同一jobを再投入しない。

最新checkpoint: [GELUの丸め切り分けと実GPU結果](results/m3-native-gelu-rounding-gpu-20261015.json)。保存済みLinux AVX2のGELU先頭49,152要素に対し、旧分離式194 bit差、graph内FMA候補130 bit差、SLEEF tanh接続後82 bit差だった。境界の直接観測で前段に203 bit差、同じGPU前段入力へのtanh自体は全bit一致と判明。FP32の明示FMA / 非FMA境界を単一compute kernelで保持するopt-in経路に修正し、同一入力のGELUを全49,152値で全bit一致、入力所有権 / 再利用 / Dispose / 不正入力と既定tanh回帰を含むGPU13 pass / skip0、package22 Greenを確認した。既定GELU経路は保持する。これはAVX2同一入力の局所証拠であり、元AVX512全encoderの合格ではない。

全体接続の最初の実行`8d67fa8...`は旧32 Worker行のharness assertionが16単一kernel行を拒否し、数値比較前に失敗。検証を修正した`155d552...`の元square-0全体は、**元goldenの許容差超過3→1要素、最大許容差比1.085958で未合格・未採用**。残りはrow25 / column118、実値0.0167587101 / 期待値0.0167713910、絶対差1.26809e-5に対して許容1.16771e-5。GPUCompute・計算/入力CPU fallback0・中間flush0・元係数保持、113 RMS / 178 GPU演算 / 179 segment。1 fail / skip0、他5画像case未実行。host peak4,940,636,160 bytes（約4.60GiB）、実行＋最終取得16.837秒、process77.341秒の単発観測。runner / owned監視はterminal。数値差を解消した候補だけを全画像条件へ広げる。

元graphの最初のsoftmax prefixも同一入力483,840値でLinux AVX2 nativeへ全bit一致し、関連GPU5 pass / skip0。次は元goldenを再現したAVX512環境から同じprefixを取得し、その文脈のRMS / attentionを比較する。Actionsに実AVX512をmodel download前に確認する明示opt-inを2 Red→関連26 Greenで追加し、そのmodeでは既知のAVX2再実行を省く。実AVX512がない場合は前提失敗・モデル未実行として記録する。元golden・学習済み係数・`atol=1e-5, rtol=1e-4`と、元画像全3 profile / 6 GPUケース、既存音声6ケース / 2 profile、実PCM / Texture→GPU前処理→encoder→backbone→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4 sample / Player / 必須環境 / Required CI / 長期memoryの範囲を維持する。工程3・4は未完了。ローカルUnity終了済み。

以下は前回checkpoint。

元square-0参照の再現条件を[AVX512実行で確認](results/m3-original-reference-avx512-reproduced-20261015.json)。元保存graph・元入力を固定し、AMD EPYC 9V45 / AVX512のthread 1 / 2 / 4全条件で全131,072値を元goldenへ全bit一致。同型CPUのAVX2指定jobは3要素超過のまま。patch / rotaryは前AVX2捕捉と同一、最初の層からbit差が現れる。同じ捕捉上で公式 / 保存poolingのfeature・RMS・最終値は全bit一致し、配置差仮説は支持されない。次は元golden再現文脈を参照として最初の層のRMS / attentionを比較し、GPUの残る5要素超過を修正する。他5case未実行、工程3・4全体未完了。以下は前回checkpoint。

末尾RMS / block192投影は[今回の固定CPU入力へ実GPU全bit一致](results/m3-original-saved-tail-rms-gpu-20261015.json)。Python6 Red→関連38 Green、GPU局所1 pass / skip0。新しい末尾RMSの統計量上限2e6は明示opt-inで、既存の既定1e6は保つ。全113 RMSへ接続した全体候補は[元固定golden5要素超過のまま](results/m3-all-rms-original-gpu-gate-still-five-20261015.json)、全5要素のbitも前候補と同じ。GPUCompute・fallback / 中間flush0・元係数を保持、1 fail / skip0、他5case未実行。次は取得する元最終encoder入力上で公式poolingの転置280行 / 保存graphの連続256行をActionsで比較する。原因と断定せず元golden・許容差を保つ。Unity終了済み。

最新は[元graphの直接中間観測](results/m3-original-saved-graph-node-context-20261015.json)と[block192全体GPU](results/m3-original-context-block192-gpu-20261015.json)。観測あり / 通常CPU実行の最終全bit一致を確認し、rotaryは旧診断nativeと全bit一致、差はpatchで既に存在すると分かった。今回CPU patchは保存済みGPU block192と全bit一致する。元の全体候補をblock192へ接続した1ケースはGPUCompute・fallback / 中間flush0・元係数保持のまま、元固定golden超過566→5、最大比1.256657へ減少。まだ1 fail / skip0、他5case未実行。今回CPU出力へも3要素超過が残る。host peak約4.37GiB、実行＋最終取得30.050秒、Unity終了済み。次は取得済み末尾入力で残るRMSと最終投影を単独比較する。参照と許容差を変更せず、API全条件 / sample / Player / 必須環境の範囲を維持。以下は前回checkpoint。

元保存graphのActions再実行は完了し、両AMD EPYC 7763 / AVX2 runnerのthread 1 / 2 / 4全条件で元square-0参照3要素超過・最大比1.182998。AVX512は未実行。推論なしのarchive照合ではmetadata以外の2,321演算 / 218保存dataは元と診断で全て一致した。SHA差だけを演算変更とせず、元保存graphから直接中間値を捕捉し、診断参照との最初の相違を確認する。元goldenは保持する。

最優先は[元の固定参照と診断参照の文脈不一致](results/m3-original-reference-context-misalignment-20261015.json)の解消。入力と元定数contentは揃うが、診断native出力自体が元固定参照にsquare-0で598要素超過する。診断manifestの一致flagは、そのjobで再exportした別SHAのgraphとの一致だった。局所stage / layer合格を元固定golden合格としない。178 custom GPU演算を接続した候補は元固定参照566要素超過、診断nativeへ6要素超過で不採用。他5case未実行。元graph・参照・許容差は保持し、Actionsで元保存graphのnative / AVX2・thread条件を再現してから追加の全体GPU修正へ戻る。学習係数違い、ISA、threadを原因として断定しない。工程3・4の全範囲は未完了のまま。以下は前回checkpoint。

Editor描画frameの進行を明示する検証harnessで、同じ全体候補の旧最終bitを保持したままhost peak低下を確認した。[固定receipt](results/m3-editor-frame-progress-and-buffer-accounting-20261015.json)。GC診断後のrunだけでなく、それを含めない独立run `b887820...`で約3.34GiB（旧9.55GiB）、GPU実行＋最終取得2.762秒、描画frame 4→141。約2秒間隔のdevice VRAM観測最大6,246MiBはbackgroundを含み真のpeakではない。元1層のDispose後、通常のEditor test yieldでは描画frameが止まり、player loopの明示更新で遅延解放が進むことも別診断で確認した。元層 / inline RMS層は2観測pass・通常GPU bit / source保持であり、元native CPU数値合格には数えない。全体は元係数digest / 最終hash `43fa98f5...`、GPU計算と中間readbackなしを維持し、571要素超過・最大比6.883229の各1 fail / skip0として保持する。Runtime算術と参照 / 許容差を変えず、強制GCは診断のみ。package22 Green。必要なEditor検証はframe進行するharnessで行い、残るRMS / scalingと局所合格した投影 / attentionの全profile接続へ戻る。全実入力の最終API全条件・工程3 / 4 sample / Player、Required CI・長期memory等の範囲は未完了のまま。以下は前回checkpoint。

stage fence・内部GELU cache・backend scratch / 完了済みML contextの解放をTDDで確認したが、4つの同一全体workload比較は元参照571要素超過・高いmemory peakのままで未採用。[固定receipt](results/m3-gpu-stage-lifetime-and-cache-peak-unresolved-20261015.json)。最後の全体source `99a6565...`は18 backend scratch0 / context再作成と32 GELU cache0、旧候補hash `43fa98f5...` / 元係数digest保持、host peak約9.55GiB。実行中device VRAMも高止まり。元layer 0の旧候補hash / 元native許容差gateは保持した。36関連Green、恒等GEMMが最適化で消えた前提失敗の後に非恒等GEMMを含む6 Green、package22 Green。継承private fieldの初回lookup失敗も別記し、数値Redや合格へ混ぜない。static auditは推論なしで、区間間保持44.3MiB・未消費内部出力0・分割計算重複なし。次は全体cache比較を反復せず、1層 / 1区間で実GPU buffer容量とnative / driver allocationを照合する。新方式はopt-inのまま、既定constructor / cache動作と算術kernelを維持。全FP32 profile・残るRMS / scaling・実入力からの最終API全条件・工程3 / 4のsample / Playerは未完了。以下は前回checkpoint。

元attentionの先頭16行だけを捕捉し、通常layer 0の旧GPU hash / source bytesを保持した。行prefix取得は3 Red→既存観測を含む10 Green。softmax候補は現CPU式へのbit差を419,773→157,609へ減らしたが全bit一致ではなく、これだけを元層へ追加した候補は1要素超過で不合格として保持。2つの行列積は捕捉入力上の現CPU式へ実GPU全bit一致。初回の3次元head shape前提の失敗は数値未実行として別記録した。動的な両入力と残差を保持する分割は2 Red→関連24 Green。`eca9428...` の元layer 0は2つの行列積を追加して、標準softmax / 候補softmaxとも元native gate超過0・最大比0.739177 / 0.613358。2 pass / skip0、GPUCompute、計算/入力CPU fallback・中間flush0、source bytes保持、package22 Green。[固定receipt](results/m3-attention-softmax-and-matmul-gpu-20261015.json)。元native内部値への全bit一致や画像全体の合格ではない。既存GPU演算と音声経路は不変。全encoderは再実行せず、前候補571要素超過と高いmemory peakを保持する。次は残るattention/scaling / 元RMSとallocation寿命を確認し、ピーク対策後に全profileへ接続する。全実API / role / mask / 境界 / 寿命 / text回帰、工程3・4のsample / Playerは未完了。以下は前回checkpoint。

unused GPU cache解放とコピー済みWorker出力の退役をopt-inで追加。cache 3 Red→3 Green＋linker1 Red→関連14 Green、出力退役 / 保持fieldの2 Red→関連15 Green、package22 Green。入力 / 係数 / 結果寿命 / 例外後再利用を維持し、元layer 0の前候補bit・元native許容差gateを保持した。[固定receipt](results/m3-gpu-cache-release-and-output-retirement-20261015.json)。全体square-0もcache-only / 出力退役の最終bitが完全一致したが、元参照571要素超過で各1 fail / skip0。単発観測の準備4.163秒・実行＋取得7.111秒に対し、host peak約9GiB・VRAM観測15,674MiBでmemory peakは未解決。機械的なcache解放を資源受け入れへ置き換えず、全体反復は止める。次は元attention / softmaxの先頭行だけをGPU捕捉して独立参照へ照合。残る元RMS / 全profile / 実API / sample / Playerの範囲を維持する。Player linker保持metadataは確認済み、実Playerは未実行。以下は前回checkpoint。

GpuGeluの所有権 / GPU専用 / 再利用を5 Redから実装。捕捉gate7,741,440値に対する現CPU式のbit差は既存3,048,817→候補6,716、許容差超過0で全bit一致ではない。元layer 0への接続は関連16 Green・元native許容差超過0 / source bytes保持。全体の112 RMS＋16 GELU＋patch384候補square-0は571要素超過、host peak約9.252GiB・VRAM snapshot15,638MiBで未採用。[固定receipt](results/m3-gelu-gpu-layer-and-full-failure-audit-fix-20261015.json)。全体反復を止め、次はstageごとの未使用GPU cache解放をopt-inで局所TDDし、attention / softmaxを独立照合する。係数監査の46.802秒は、同じ全係数bytes / digestを保持するWindows CNG経路へ置換し、3 Red→関連10 Green、単体0.467秒を確認した。全推論のbenchmarkや画像合格へは流用しない。初回実GELU fixtureのreshape前提失敗と監査testのcompile失敗は別記録。既存GPU演算 / 音声経路は不変。残る元RMS、全3 profile、最終API / sample / Playerは未完了。以下は前回checkpoint。

RMSを全体へ戻す際のWorker増加を抑えるため、除算をFP32残差で補正するopt-in exportを追加。3 Red→関連79 Green、現CPU式の全1,935,360値へ1 Workerの実GPU全bit一致。元layer 0への7 RMS組込みも、1 Red→関連20 Greenで前回のqualified GPU出力bitを全保持し、元native layer gateの超過0、segmentは15→8。[固定receipt](results/m3-inline-rms-layer-equivalence-and-whole-failed-20261015.json)。全16層の112 RMSを組み込んだsquare-0は577要素超過・最大比7.556308で不合格。元RMS1か所は未変更、他5caseは未実行。host peak約4.331GiB、GPU実行＋取得4.142秒だが、準備46.964秒には係数hash監査等も含むので製品の純粋な準備benchmarkではない。この全体候補は未採用・反復停止。次は活性化とattention積和の局所参照を確認してから元全profileへ戻す。最終API / role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

元layer 0の全7 RMSと7投影をGPU接続し、固定元layer参照の許容差超過0・最大比0.766697へ到達。bit差1,518,970は残り、全bit一致ではない。4次元境界 / 元scale借用の5 Red→関連22 Green、全RMS marker / callable headの4 Red→27 Green。768幅4か所と64幅3か所は捕捉入力上の現CPU式へ実GPU全bit一致、元model / scale bytes・参照 / 許容差を維持した。[固定receipt](results/m3-seven-rms-original-layer0-gpu-20261015.json)。追加fixtureのcompile失敗 / XML欠測は別証拠として保持。既存Runtimeは未変更。画像全体は再実行せず、以前のsquare-0の566要素超過が未達のまま。次はGELU / attention積和を局所照合する。小さいGELU CPU probeは分離式にも55 bit差が残り未採用で、GPU成功や元内部参照の再現とは扱わない。全3 profile / 実API / sample / Playerの範囲は維持する。以下は前回checkpoint。

元layer 0の7 RMSを、通常GPU最終bitと旧hashを保持して捕捉（1 pass / skip0）。64幅3か所は新しいscaled / unscaled exportで13 Red→関連30 Green・音声算術 / package込み76 Green、捕捉元入力と元scaleを使う現CPU式へ実GPU全bit一致（3 pass / fail0 / skip0 / fallback0 / flush0）。768幅4か所も既存exportのCPU式全bit一致だが、GPU確認は最初の1か所のみ。[固定receipt](results/m3-all-rms-observation-and-head-gpu-20261015.json)。次は4次元unary境界 / 元scale借用をTDDで実装し、残る768幅3か所をGPU照合してから7 RMSを元layer 0へ戻す。元native内部参照や全encoder合格とは扱わず、下記wholeの566要素不合格と全profile / 最終API / sample / Playerを未達のまま保持する。以下は前回checkpoint。

最初のRMS＋7つのblock384投影を元layer 0へ戻した実GPU gateは、元参照・係数・入力・許容差のまま超過0。境界 / RMS単体を含む15 pass / fail0 / skip0。全bit一致ではなく、画像全体合格へは流用しない。確認済み最初の層だけを元全encoderへ接続したsquare-0は566要素超過・最大比6.791765で1 fail / skip0。10 segment、host peak約4.017GiB、実行＋取得3.660秒だが、定常性能合格ではない。[固定receipt](results/m3-first-layer-rms-reconnection-20261015.json)。次は全体反復を止め、7か所のRMS入力 / 出力 / scaleを通常GPU最終bitを保持して捕捉し、残る64 / 768幅の正規化・活性化・積和を局所照合する。元3 profileの全条件、実Texture→768 API、role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

最初のRMS単体は、元patch / scaleと現CPU式を使い、実GPUの全1,935,360値で全bit一致した。原因はSentisが除算768→epsilon加算をScalarMadへまとめた丸め差で、明示GPU除算の境界を保持して修正。元RMS観測の再現・失敗候補・統計FMA診断・1 pass / fail0 / skip0を[固定receipt](results/m3-first-rms-gpu-explicit-division-20261015.json)へ分離した。元native内部参照は未取得で、画像encoder合格には数えない。次はこのRMSを元layer 0へ戻し、元固定参照で改善を検証する。他のRMS/活性化、全3 profileの全条件、最終API / role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

現在の次工程はRMS正規化の固定GPU照合。元patch入力のlayer 0は通常4要素超過、7投影のblock384候補は1要素超過。全114投影を元square-0 encoderへ戻しても579要素超過・最大比5.757522で未合格だった。実行＋取得30.978秒・host peak10.969GiBのため、この全体候補を反復せず小さい正規化/活性化テストへ戻る。768幅平均は7 Red→CPU/ONNX関連25 Green、元patch64行の平均もbit差0。有界1万点のCPU probeでは補正inverseがnative powerへbit差0だが、GPU成功・元全context合格へは数えない。[固定証拠と残件](results/m3-layer0-static-gpu-and-rms-next-20261015.json)。以下は前回checkpoint。

元画像のpatch prefixで1,256,626 bit差を再現し、係数を保つGPU block384投影で元1,935,360値へ全bit一致した。128/192/256/512/768の不一致も保持する。sliceの入力番号問題を小さいCPU/GPU Redから修正し、GPU投影・slice関連24 Green、package22 Green。[固定receipt](results/m3-patch384-gpu-and-original-encoder-20261015.json)。ただし、この投影を元全encoderへ戻したsquare-0は579要素超過・最大許容差比7.406214で不合格。局所一致を全体合格へ置き換えない。次は正確なpatch出力を起点にlayer 0の残る投影・演算を切り分け、元3 profileの全条件を確認してから実Texture→正規化768 APIへ進む。画像全chain / role / mask / 所有権 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

GPU画像前処理はmain、ImageFeatureEncoder / ImageEmbedderはDraft実装・合成契約合格。画像PythonのQuery / Document / Rawは3 profile・6画像×3 roleで18件合格した。[receipt](results/m3-real-image-all-role-reference-20261014.json)。元FP32 encoderの固定gateとUnity実Texture→768次元の全chainは未達。

補償投影候補の元12数値caseは全件不合格。最後のnorm4実行はユーザー中断で、square / landscape CPU / GPUは不合格、portrait CPUは途中、portrait GPUは未開始。[中断receipt](results/m3-vision-norm4-user-interrupted-20261014.json)。小さいGPU契約やPython role参照でこの未達を置き換えない。

再開は[全体計画R0〜R5](multimodal-tdd-plan.md)に従う。対象fixtureのGPU専用変更をcompileし、一つのGPU入力で参照文脈・数値差と準備負荷を確認する。元3 profile、FP32 / 採用Float16、全role・実Texture / RenderTexture・所有権・text回帰・sample・Player / 必須環境を保持する。層数25265への展開を性能合格の候補と扱わない。

## 以前の画像計画・固定sourceの履歴

更新日: 2026-10-13。**GPU画像前処理はPR #52でmainへ統合・Windows単独受け入れ済み。画像end-to-end API / sampleと実画像から最終768次元の受け入れは未完了**。画像encoder / projectorのPython実モデル照合はPR #37で合格した。Sentis encoderはPR #41のWindows実測でFloat16のCPU / GPU全6条件が合格、FP32全6条件は固定elementwise許容差超過で不合格。共通RuntimeのPR #43はmainへ統合済み。下記の実測sourceは過去の固定headとして維持し、統合headへ付け替えない。[全体計画](multimodal-tdd-plan.md)の工程3として進め、M3完了時には正式リリースせずM4と全必須環境の検証へ渡す。

I1 / I2の固定profile encoder / projector exportと公式processor参照・Actions gateをTDDで実装した。[実装と検証範囲](m3-vision-export-validation.md)。小さい回帰66 passed / skip 0。[実モデルPython CI 38310718250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38310718250)の3 profile / 6ケースでminimum cosine `0.9999997615814209`。出力は512次元soft tokenであり、画像の最終embeddingではない。

### 履歴checkpoint

公式prompt / mask契約PR #55と最終画像参照PR #56はmainへ統合済み。実モデルPythonで3 profile / 6画像の768次元、native feature → 保存backbone、保存encoder → 保存backboneを固定許容差で照合した。producer `35d357263ae29de4db1ad93d1cac3616ea1e61b4`、[CI 38356261690](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38356261690)。最新baseのhead `9814ce9d8bd70fa07cbd191867d93d7f4dd461f5`の[CI 38364660005](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38364660005)全8成功を確認して統合。Python参照の合格と、画像Runtime・Sentis全chainの未実行は別に記録する。

次はzero-audio profileのimport、入力 / feature順・mask / 所有権・途中失敗のred → greenからGPU前処理・実encoder・共通本体を接続する。元のFP32全6条件の不合格を解消し、最終768次元・全順位・画像検索sample・Playerへ進む。下記の過去の実測sourceは維持し、C3の公式prompt構築の未完了記述はこの最新checkpointで更新する。

### 直近のTDD順序

I3の画像GPU前処理は[PR #52](https://github.com/ayutaz/unity-embeddinggemma-2/pull/52)で実装。固定Git Runtime `71284e302d5e3e643e4ed98272c5774640582c1e`で全package 118 green、Texture / RenderTexture × linear / sRGBの公式24ケースは最大pixel差0。8192pxのdispatch不具合も1 redから修正し、最大入力・paddingを確認した。[実測と未達](m3-image-preprocessing-validation.md)。次はI2の数値差修正と、統合済みC3公式prompt / maskへの実encoder接続。I4の実モデル連結の合格判定はI2 / I3 / C3が揃ってから行う。GPU profiler、長い反復 / memory、Playerと他OSは別gate。最終API・sample・PlayerもTDDで実装し、encoder単独の合格でM3を完了にしない。

1. I3のGPU前処理実装と単独Windows照合は完了。次の実encoder接続では、Tensor寿命・途中失敗・中間readbackを新しい契約のredから確認し、GPU profiler / 長い反復memoryの証拠を追加する。
2. I2のFP32差をstage / 演算別に特定し、独立参照との不一致を小さいテストでredにしてから修正する。現在の `rtol=1e-4` / `atol=1e-5`は緩めず、3 profile / 全12条件の実モデル再照合と保存 / 再読込を確認する。
3. C3の[PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43)、確認head `21e598c5b53d5ad73557ee6d606b12770254d73b`では配置・cache修正の68件greenに加え、実1024長backboneのCPU / GPUCompute × FP32 / Float16全4条件が合格した。[固定headの結果](https://github.com/ayutaz/unity-embeddinggemma-2/blob/21e598c5b53d5ad73557ee6d606b12770254d73b/docs/results/multimodal-backbone-sentis-windows-20261013.json)。最初の4条件はconsumer Editor assembly内source、後続の固定Git Runtime更新consumerも4条件合格した。[Git更新のreceipt](results/multimodal-git-runtime-upgrade-windows-20261013.json)。いずれも合成project済みfeatureの受け入れであり、実画像feature接続・統合済み公式prompt / maskとの連結・最終candidateの空consumerは別gateとする。placeholder / feature数・順序の不一致をred → greenにし、実画像の最終vectorを照合する。
4. I4〜I6のAPI → 横断検索sample → Playerを下表の契約順でTDD実装する。最終768次元と全順位、実操作・解放・正常終了の必要条件を満たすまでM3は未完了とする。

PR #53の実Sentis診断は2026-10-13 04:56 UTCにCPU / GPUの2テストが独立参照の文脈確認で不合格。case_0のtrace最終値は元保存Sentisと一致したが、新native参照と元の固定参照の一致gateが未達で、case_1は未実行。原因を確定したとは扱わない。元 / 新参照の入力・環境を照合し、途中の成功flagを拒否する報告監査のredを追加する。診断文脈の合格後に演算別red → 最小修正 → 全12条件へ進む。[今回のcheckpoint](results/tdd-acceptance-plan-checkpoint-20261013.json)。

[12条件の実測](results/m3-vision-windows-plan-snapshot-20261013.json)はencoderのみ。全GPU条件で実出力はGPUCompute、CPU計算 / 入力要求・readback flushは0件だが、GPU画像前処理と共通本体へのfeature受け渡しは別に検証する。CLI観測timeoutと実operationの保存報告を区別し、未変更の重い検証を再起動しない。

## 範囲と参照固定

`Texture` / `RenderTexture` → GPU前処理 → 画像encoderのsoft token → 共通本体 → 768次元を実装する。encoder出力を直接L2正規化して最終embeddingとする実装は採用しない。入力は最初に単一画像 / batch 1と参照で確定した解像度profileを扱い、縦横比・許容寸法・上限・複数画像の対応範囲をAPIで明示する。

最初に固定revisionの公式processorからresize計算・補間・色空間 / RGB・alpha扱い・値域・patch順・位置ID・pooling・padding / mask・soft token数・promptを確認する。既存調査のpatch 16px / 標準280 tokenは調査値であり、Unity profileの採用値はこの確認後に固定する。共通本体C3の長さ拡張なしに128長へ切り詰めない。

参照fixtureは小さい単色・勾配・checkerboard・縦横非対称画像、縦長 / 横長、奇数寸法、RGBA・RenderTextureを用意する。これらは前処理の検出用であり、意味検索の評価は別の権利確認済み固定画像 / 日英queryを使う。参照生成はActions / tools / uv、重みと大きい中間テンソルはartifactへ保存する。

## 実装順とRed / Green

| ID / 依存 | 先に観測するRed | 最小実装 | Greenと実モデルの受け入れ |
| --- | --- | --- | --- |
| I1 参照・C2 | processor / profile / revisionの違い、patch位置・mask・期待feature数不整合をfixture監査が見逃す | processor参照CLI、fixture manifestと段階別参照・encoder export | 小さいmanifest検査green、Actions実モデルの前処理・encoder・最終vector参照とhashを固定。数値許容差を根拠付きで先に決める |
| I2 export・I1 | 実重みを使うexportの出力数 / 並び / paddingが固定Python参照と不一致。互換性の小さいgraphテストも先に作る | encoderを対応演算へ展開し`.pt2` export、import / `.sentis`保存・採用Float16重み形式 | Actionsで元Python対export照合、Unity CPU / GPUで実encoder feature照合。import成功だけで合格にしない |
| I3 GPU前処理・I1 | 向き反転、sRGB / linear混同、resize丸め、channel / patch順、padding値で参照と不一致。借用Texture破棄の契約もred | GPU上のTexture変換・resize・patch / position処理。固定profileで始める | 合成画像の中間tensor比較と実GPU実行、前処理のCPU readbackなし。Texture / RenderTexture両経路で確認 |
| I4 共通本体・I2 / I3 / C3 | image placeholderとfeature数不一致、feature順違い、prompt欠落、mask / projection / pooling違いで最終vector不一致 | featureをGPU tensorのまま共通本体へ渡す。仮API名 `ImageEmbedder`、名称・所有権は契約PRで確定 | 実モデルfp32 cosine >= 0.999 / 採用Float16重み >= 0.99、有限768次元・単位長。CPU / actual GPUを別記。既存text128回帰 |
| I5 検索・sample・I4 / C5 | text→imageの全順位 / score、同点ID順が不一致。未準備、対応外画像、欠落モデル、停止後再操作の契約がred | 画像事前embeddingとtext query、結果画像 / score表示、入力拒否・解放 / 復帰 | 固定日英queryの全順位一致、実UIで準備→検索→表示→解放。画像向き・表示・エラー後再操作も確認 |
| I6 Player・I5 / C4 / C6 | encoder欠落 / hash破損 / profile違い・GPU fallback・途中結果をpassにできる。繰返し切替で寿命契約違反がred | bundle / cache監査、Player harness、画像sampleの配置・測定 / 終了監査を接続 | Windows IL2CPP / Release / High strippingで実モデルCPU / GPU、独立2起動の実終了0、cache・切替 / memory。他OSは全体公開条件へ |

I2のSentis非互換は演算子とshapeを保存して調査する。標準演算への分解を先に小さいPython / Unity比較で検証する。標準ONNXが有利なら別PRで同じ参照へ照合し、native runtimeプラグインを追加しない。

入力のnull、破棄済み / 対応外Texture、範囲外寸法、feature数・profile不一致は推論前に明示的に拒否する。自動crop / 切捨て / CPU fallbackで期待値を合わせない。画像upload・最終vector取得と、中間GPU tensorを区別してprofilerで確認する。

## 測定・完了条件

- I1〜I6の意図したred / green、source / model / processor / graph hash、Unity / Sentis、requested / actual backendを記録する。
- 前処理、encoder、共通本体、load / 初回 / warm反復、切替 / Disposeの時間・メモリと測定APIを残す。画像入力のCPU readbackと中間転送がないことを実GPUで確認する。
- 実画像の参照embeddingとtext→image全順位、既存テキスト検索が合格し、空Git consumerのsampleを実操作できる。小さい画像テストのgreenで実encoder互換性を完了にしない。
- Float16重みは実測した採用形式だけを宣言する。OOM / GPU非対応 / 他OS未実行を別記し、全体の正式公開条件へ引き継ぐ。

PRはI1 / I2の参照・export、I3のGPU前処理、I4 / I5のAPI・sample、I6のPlayer接続に分ける。各変更のgreen後に影響する実モデル照合を行い、変わっていない大きいモデル生成・Unity起動を繰り返さない。
