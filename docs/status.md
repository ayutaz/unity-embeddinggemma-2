# 現在の状態

更新日: 2026-10-15。**Editorの描画frame進行がGPU資源の遅延解放に関係していた。検証harnessでframe更新を明示すると、同じ画像候補の出力bitを保持したまま、独立processのhost peakが約9.55GiB→3.34GiBへ低下した。** [固定receipt](results/m3-editor-frame-progress-and-buffer-accounting-20261015.json)。画像の元FP32 gateは571要素超過のまま、工程3・4は未完了。

元layer 0と7 RMSをinlineした層の2つの容量診断は2 pass / skip0、観測と通常GPU実行の全bit一致・source bytes保持。元層は旧GPU hash `bc417e7f...`を保持したが、元native CPU参照の合格には数えない。追跡可能なGPU buffer容量の最大は約706 / 739MiBで、その大半はstorageのfree pool約644 / 658MiB。per-layer `ScheduleIterable`はタイミングが通常と異なり、graphics-driver counterはEditor全体を含むため、これを通常実行benchmarkへ流用しない。最初のprofile processは134.91秒、NUnit内の実行は7.48秒で、固定package更新等の起動待ちを推論時間へ混ぜない。

元1層のDispose直後はgraphics-driver counterが約919MiBのまま、描画frame 4→6で約303MiBへ低下し、その後の`yield null`ではframeが進まなかった。GC / UnloadUnusedAssetsでは減らない。`EditorApplication.QueuePlayerLoopUpdate`を診断harnessで呼ぶと描画frame 6→14、約178MiBまで低下した。これを受け、全体候補のawait中にもEditor player loopを進めた。mixed診断後の1 runでhost peak3,679,625,216 byte、さらに**GC診断を含めない独立run** `b887820...`で3,584,634,880 byte（約3.34GiB）、GPU実行＋最終取得2.762秒を観測した。描画frameは4→141。実行中device VRAMの観測最大6,246MiBはbackgroundを含む約2秒間隔のsampleで、真のpeakは保証しない。単発観測であり、他環境 / 長期memory受入や定常benchmarkではない。

両全体runは同じ18 segment・元係数digest / 最終hash `43fa98f5...`を保持し、GPUCompute・計算/入力CPU fallback・中間flush0。元参照571要素超過・最大比6.883229で数値gateは各1 fail / skip0のまま。mixed runのdisposal診断1 passを全体数値合格へ置き換えない。今回の変更は診断とEditor検証harnessであり、Runtime算術の変更や強制GCを製品へ追加していない。package22 Green。次はこのframe進行するharnessで必要な検証を行い、残るRMS / scalingと、局所合格した投影・attention等の元全profileへの接続へ戻る。全画像FP32条件、実PCM / Texture→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4 sample / Player、Required CI・Persistent allocation警告・長期memoryは未完了。Unity終了済み。

以下は前回checkpoint。

更新日: 2026-10-15。**GPU stage単位の完了待ち、GELU内部cache、backend scratchと完了済みML演算子contextの解放を確認したが、全体の高いmemory peakは解決していない。** [固定receipt](results/m3-gpu-stage-lifetime-and-cache-peak-unresolved-20261015.json)。GPU計算・元係数・固定参照・許容差は維持し、画像全体571要素超過、工程3・4未完了を保持する。

`ExecuteAsync`はUnity main threadで各stageのgraphics fence完了を待ち、途中tensorのCPU取得を追加しない。重複呼び出しを拒否し、cancel / Disposeは投入済みstageのGPU完了後に所有tensorを解放する。4 intended Red→既存26 Green。GELUの内部2 Worker cacheは既存保持1 pass＋新API1 Red→関連32 Green。継承されたprivate fieldのlookupを誤った最初のscratch診断の2前提失敗を別記し、修正後はscratch解放・完了済みcontext解放・linker保持の3 intended Red（既存保持1 pass）→元layer 0のbit保持を含む36 Green。初回GEMM診断は恒等行列の最適化で前提失敗と記録し、非恒等GEMM / async / linkerの6 Greenへ修正。package22 Green。元layer 0は`fe3b9fe2...`全bit保持・元native許容差超過0。既存constructor / cacheの既定動作は維持し、算術kernelは変更していない。Player linker metadataの保持を実Player合格には数えない。

**4つの全体比較を性能改善として採用しない。** 同じ18 segment / square-0候補で、fenceのみ、GELU内部cacheのみ、その併用、さらにbackend scratchとGPU完了済みML contextの解放を比較した。すべて最終hash `43fa98f5...`と元係数digestを保持し、571要素超過 / 最大比6.883229で各数値gate不合格。最後のsource `99a6565...`では18 backendのscratch0 / context再作成と32 GELU cache0を確認したが、host peak10,257,428,480 byte（約9.55GiB）。実行中のdevice VRAMも高止まりした。各runは単発観測で、定常benchmark / 真のVRAM peak / 他GPUへ一般化しない。最初のfence-only resource監視は準備中に終了したため、実行中VRAM peak欠測として別記。

推論を行わないstatic auditでは、分割layer総数19,518で再計算の重複なし、区間を跨いで保持するboundary tensorは最大46,448,640 byte（約44.3MiB）、未消費内部出力0だった。これらの確認やcacheの機械的解放を全体memory改善へ置き換えない。GPUComputeの実装はこの環境ではUnity 6の`GfxDeviceBackend`で、CPU fallbackではない。次は同じ全体cache実験を反復せず、元の1層または1区間で実buffer容量とnative / driver allocationを分けて調べる。その後、数値合格済みattention等を全profileへ接続する。全画像FP32 / 残る元RMS・scaling、実PCM / Texture→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4 sample / Playerは未完了。Required CI、Persistent allocation警告・長期memoryも未達。Unity終了済み。

以下は前回checkpoint。

更新日: 2026-10-15。**画像attentionの2つの行列積は、元GPU入力の先頭16行に対する独立した現CPU式へ実GPUで全bit一致した。最初の層への再接続も元native参照の許容差超過0。ただし画像全encoderと高いmemory peakは未解決。** [固定receipt](results/m3-attention-softmax-and-matmul-gpu-20261015.json)。工程3・4は未完了のまま。

診断用の行prefix取得は3 intended Red→既存観測を含む10 Green。元の`[1,12,2520,2520]`を全量downloadせず、各headの先頭16行だけを取得した。観測後の元layer 0出力は旧GPU hash `bc417e7f...` と全bit一致し、source model serialized bytesも保持。これは診断readbackであり、製品の中間readbackなしを証明する試験ではない。

softmaxは同じ483,840入力値に対する現CPU式で、標準GPUの419,773 bit差→既存候補157,609 bit差、両者の許容差超過0。候補は全bit一致ではない。softmaxだけを前のRMS / GELU / 投影候補へ追加した `143312e...` の層は1要素超過・最大比1.182683で不合格として保持する。行列積の初回診断はheadをまとめた3次元shapeを4次元と誤認して前提確認で停止し、数値比較未実行として別記録。shape監査修正後は元層の入力3行列を全量、巨大出力とsoftmax入力は先頭16行だけ捕捉し、元最終GPU bit保持を確認した。

`d41c4b3...` の2つのGPU行列積は、捕捉入力に対する独立した現CPU式へそれぞれ483,840値 / 12,288値の全bit一致。元native内部値の取得ではない。両GPU入力と残差の保持は2 intended Red→既存分割 / pipeline / 行prefixを含む24 Green。`eca9428...` の元layer 0は7 RMS / 7投影 / GELU / 2行列積を接続し、標準softmaxで元native gate超過0・最大比0.739177、候補softmaxも接続すると超過0・最大比0.613358。最終hashはそれぞれ`747e55d6...` / `40cf083c...`、全bit一致ではない。2 pass / skip0、GPUCompute・計算/入力CPU fallback・中間flush0、source bytes保持。既存GPU演算カーネルと音声経路は変更していない。package22 Green。

全encoderはこのcheckpointで再実行していない。前候補square-0の571要素超過と高いmemory peakを未解決として保持する。他5画像caseも未実行。次は残るattention/scaling・元RMSと全体のallocation寿命を確認し、ピーク対策後に合格した演算を元全profileへ接続する。実PCM / Texture→GPU前処理→encoder→backbone→正規化768 API、全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。Required CI、Persistent allocation警告・長期memoryは別gate。Unity終了済み。

以下は前回checkpoint。

更新日: 2026-10-15。**未使用GPU cache解放とコピー済みWorker出力の退役をopt-inで追加し、元layer 0と全体候補の出力bit保持を確認した。ただし画像全体の571要素超過と高いmemory peakは未解決。** 既存constructorの既定動作は維持。3 intended Red→cache契約3 Green＋linker保持1 Red→関連14 Green、コピー済み出力 / 保持メンバーの2 Red→関連15 Green、package22 Green。[固定receipt](results/m3-gpu-cache-release-and-output-retirement-20261015.json)。

解放はWorkerが所有する未使用poolだけに適用し、全境界のCopyOutput完了後は既にコピー済みのWorker出力を退役する。入力・係数・caller所有の結果・借用operationを保持し、反復・例外後再利用・二重Disposeを確認。最初の層は前候補hash `fe3b9fe2...` と全bit一致し、元native layer gate超過0 / source model bytes保持。Player用link.xmlには実際に参照するprivate field / generic pool constructor・Disposeを明示追加し、広いpreserve指定は使わない。保持metadataの検証であり、Player実行合格ではない。既存GELU / tanh / softmax / batched matmul / 音声算術は不変。

全体square-0のcache-only `5a20d5095df29ad82d58b285718e309aaf7cba19` と出力退役 `1ee8f2c00408e2466e7df3c8ca55608e24f442cc` は最終出力hash `43fa98f5...` が完全一致した。両者とも元参照571要素超過・最大比6.883229、各1 fail / skip0。CPU computation/input fallback・中間flush0。前者は準備4.309秒 / GPU実行＋最終取得8.068秒 / host peak9,630,670,848 byte、後者は4.163秒 / 7.111秒 / 9,652,461,568 byte。係数監査は前後0.4〜0.5秒。同じ仕事量の単発観測で、定常benchmarkや他環境の性能合格ではない。退役runは約2秒間隔でresourceを記録し、実行中のdevice VRAM観測最大15,674 / 16,376MiB、OS host peak観測最大9,653,788,672 byte。device値は他processを含み、真のVRAM peakを保証しない。未使用poolが空になることとコピー済み出力の退役は確認したが、peak低減の受け入れは未達。

この全体候補は未採用・反復停止、他5case未実行。次は元attention / softmax入力の先頭行だけをGPUで捕捉し、小さい独立参照で照合する。残る元RMS1か所、全3 profile、実PCM/Texture→GPU前処理→encoder→backbone→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。ローカルUnity終了済み。Required CI、Persistent allocation警告・長期memoryは別gate。

以下は前回checkpoint。

更新日: 2026-10-15。**GPU GELUと元layer 0の局所gateは合格、画像全体は571要素超過で未合格。** 新しい `GpuGelu` は固定shapeの2 Worker＋既存GPU tanhを所有し、入力を借用、結果を独立所有で返す。API 5 Red、初回は関連9 Green＋実データ境界確認1 fail（投影後reshapeの見落としで数値比較未実行）、境界marker 2 Redを経て、関連16 Green / skip0。元の投影入出力・shapeを監査してから照合した。[固定receipt](results/m3-gelu-gpu-layer-and-full-failure-audit-fix-20261015.json)。

捕捉gate入力7,741,440値に対する独立した現CPU tanh-GELU式では、既存処理3,048,817 bit差→候補6,716 bit差、双方の許容差超過0。候補最大比0.015367。元native内部値の全bit再現ではない。`61a4dcea5d86bd6f5ef465d280918b6991896d1a` の元layer 0は7 RMS / 7投影 / GELUをGPU接続して許容差超過0、最大比0.842903、source model bytes保持。GPUCompute・計算/入力CPU fallback・中間flush0。既存tanh / softmax / batched matmul / 音声経路は未変更で、Runtime追加はGpuGeluのみ。package22 Green。

全体候補 `ca6883efdd696db4d0ef0355b5ede2afe666f48c` のsquare-0は112 RMS＋16 GELU＋patch384、18 segmentで571要素超過・最大比6.883229、1 fail / skip0。未採用・反復停止、他5case未実行。host peak観測9,934,274,560 byte（約9.252GiB）、実行中の別snapshotでVRAM15,638 / 16,376MiB。GPU実行＋最終取得25.903秒、process全体170.406秒。各WorkerのTensorPool保持がメモリ増加へ寄与する可能性はソースからの推定で、制御したcache解放比較は未実施。元係数descriptor / bytes / 参照・許容差は保持。

**遅い準備のうち係数監査を修正した。** instrumented runはfile SHA0.517秒、model load1.006秒、係数監査前46.802秒 / 後46.135秒、graph変換1.551秒、GPU pipeline準備1.402秒だった。同じ順序・全係数bytesを借用するread-only streamから既存Windows CNG hashを使うよう変更し、未実装3 Red→既存hash回帰を含む10 Green。`1db7f961f989b3ae4ab74098dd36466472424999` の実係数単体監査は0.467秒、旧digest `3e14a1b4...` と完全一致。失敗後のstream巻き戻し / fallback / ArraySegment offset / 借用bytes保持も確認。追加testのcompile失敗は意図したRedと別記録。これは同じ係数の単発監査比較で、全推論の速度改善や数値合格ではない。監査修正後に重い全encoderは再実行していない。

次は未使用GPU cacheだけを任意に解放する方針を小さいTDDで検証し、live tensor / 入力 / 結果bit / 例外後再利用を保持する。その後attention / softmaxを局所照合してから元全profileへ戻す。残る元RMS1か所、実PCM/Texture→GPU前処理→encoder→backbone→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。ローカルUnity終了済み。Required CI、Persistent allocation警告・長期memoryは別gate。

以下は前回checkpoint。

更新日: 2026-10-15。**除算のFP32残差補正を一つのモデル内へ置き、最初のRMSは1 Workerで現CPU式の全1,935,360値へ実GPU全bit一致した。** Python 3 Red→音声算術 / packageを含む79 Green、`8c4ea1c898ff91384aa6720b3be4b37efe9a3b74` のGPU 1 pass / skip0。元input / scale / 比較値を保持し、source model bytesも不変。[固定receipt](results/m3-inline-rms-layer-equivalence-and-whole-failed-20261015.json)。元native内部値の合格や全encoder合格ではない。

7 RMSのgraph組込みは1 intended Red→関連20 Green。`1c8ed7fd06cbf2d36fec3ae61a8af3a9c8ae6d56` の元layer 0は、前回7 RMS方式の出力hash `8312ee...` を全bit保持し、元native layer参照の許容差超過0。segment 15→8、1,220層。元係数・model bytesは不変。単発の実行＋取得は0.689秒だが、定常benchmarkや他環境の性能合格ではない。

**全16層候補は未採用。** `341934265036f90527efaa19c9ea1201058a1d9b` のsquare-0で112 RMSを組込み（元RMS1か所は残る）、19,535層の候補を実GPU実行した。577要素超過・最大比7.556308、1 fail / skip0 / exit2。元coefficient descriptor / 全coefficient bytesのSHAを前後で保持。準備46.964秒は読込・係数hash監査・graph変換・Worker準備を含み、GPU実行＋最終取得は4.142秒、host peak4,650,835,968 byte（約4.331GiB）。後段の係数監査・起動終了を含むprocess全体138.771秒を推論時間へ混ぜない。全件GPUCompute・計算/入力CPU fallback・中間flush0。以前の別候補566要素超過も保持し、改善や採用へ置換しない。他5画像caseはこの候補では未実行。

次は活性化 / attention積和の局所照合へ戻り、合格した候補だけを全encoderへ接続する。元全3 profile・残るRMS・実PCM/Texture→GPU前処理→encoder→backbone→正規化768 APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。既存Runtime package filesは前checkpointから未変更。ローカルUnity終了済み。Required CI、Persistent allocation警告と長期memoryは別gate。

以下は前回checkpoint。

更新日: 2026-10-15。**元layer 0の全7 RMS＋7投影をGPUで接続し、元の固定layer参照の数値gateに合格した。** `c189347998e2a6950a80f75a9ab1e5f1bf11a8f5`、関連27 pass / fail0 / skip0 / exit0。layer 0の許容差超過0、最大比0.766697、bit差1,518,970。元Model serialized bytes / 6つのscale / 7投影係数・入力・参照・許容差を保持。GPUCompute、計算/入力CPU fallback・中間flush0、15 segment。[固定receipt](results/m3-seven-rms-original-layer0-gpu-20261015.json)。全bit一致や画像全encoder合格には数えない。

4次元unary境界 / 元scale借用は5 intended fail＋旧2次元1 passから関連22 Green。追加fixtureの括弧誤りによるcompile失敗 / XML欠測も別記録し、TDD Redや数値結果へ混ぜない。768幅4か所は元scaleを借用して現CPU式へ実GPU全bit一致、template / coefficient source bytes保持。全7境界 / callable head操作は4 intended failから27 Green。64幅q/k/vもwrapper経由で現CPU式へ全bit一致した。これらは捕捉入力上の局所式の証拠で、元native内部値は未取得。

**画像全体は未合格のまま。** このcheckpointでは全encoderを再実行せず、前source `dd38756...` square-0の566要素超過を保持する。次のGELU切り分けとして、捕捉gate入力の先頭65,536値を現CPUのtanh版GELUへ照合すると、分離演算55 / FMA候補273 / 因数分解5,058 bit差、局所許容差超過は各0だった。候補は採用していない。[固定PyTorch source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/Gelu.h)も確認した。次は活性化とattention積和の局所参照を確認してから全profileへ戻す。実PCM/Texture→GPU前処理→encoder→backbone→正規化768 API、全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。既存Runtime package filesは前checkpointから未変更、全Unity process終了済み。Required CI、Persistent allocation警告・長期memoryは別gate。

以下は前回checkpoint。

更新日: 2026-10-15。**64幅のhead RMS 3か所も、捕捉GPU入力と元scaleを使う独立した現CPU式へ実GPUで全bit一致した。** `e07c878249f06dbdd7dab3ce7c600f38e5e69655`、scaled q/k・unscaled vの3 pass / fail0 / skip0 / exit0、GPUCompute・計算/入力CPU fallback・中間flush0。13 Red→CPU/ONNX関連30 Green、音声算術・packageを含む76 Green。[7 RMS観測とhead GPUの固定receipt](results/m3-all-rms-observation-and-head-gpu-20261015.json)。

先に元layer 0の7つのRMS入力 / 出力 / scale / 単体saved modelを捕捉し、通常GPU最終値が旧固定hash `bc417e7f...` と全bit不変であることを確認（1 pass / skip0）。768幅4か所は同じ捕捉入力上で既存exportが現CPU式へ全bit一致したが、実GPU確認は最初の1か所だけ。残る768幅3か所のGPU照合、4次元unary境界と元scale借用のTDD、層への再接続が次の作業。元native内部値は未取得であり、画像全体は下記square-0の566要素超過を未解消のまま保持する。全3 profile・最終API・sample・Player・Required CI・Persistent allocation警告と長期memoryは未完了。ローカルUnityは検証後に終了済み。

以下は前回checkpoint。

更新日: 2026-10-15。**最初のRMSと7つのblock384投影を元layer 0へ戻し、元の固定layer参照のGPU数値gateに合格した。** `ac2dc3355b5d957064a6df463bc95c4e422a3402`、境界・RMS単体・実layerを含め15 pass / fail0 / skip0 / exit0。layer 0の許容差超過0、最大比0.842903、bit差1,539,872であり、全bit一致ではない。元Model serialized bytes / scale / 入力・参照・許容差を保持。[固定receipt](results/m3-first-layer-rms-reconnection-20261015.json)。

**画像全体は未合格。** `dd3875616488e9fb0d8f6c1dd34de52887e4f312` のsquare-0へ、確認済みpatch・最初のRMS・layer 0の7投影だけを接続（8投影＋RMS、10 segment）すると566要素超過、最大比6.791765、1 fail / skip0 / exit2だった。GPUCompute・計算/入力CPU fallback・中間flush0。準備1.949秒、実行＋最終取得3.660秒、host peak4,312,829,952 byte（約4.017GiB）。単発観測であり、115 segmentの旧候補との同仕事量benchmarkではない。他5 caseはこの候補で未実行。全体の反復を止め、残るRMS / 活性化 / 積和を局所照合するため7つのRMS境界観測へ進む。工程3・4全体、最終API / sample / Playerは未完了。

以下は前回checkpoint。

更新日: 2026-10-15。**最初の画像RMS正規化は、元patch入力と元scaleを使い、現CPU式の全1,935,360値へ実GPUで全bit一致した。** `2fe78b5acda3522bdb2c6a16171c036b48e10e8a`、GPUCompute・計算/入力CPU fallback・中間flush0、1 pass / fail0 / skip0 / exit0。元Modelのserialized bytesも保持した。[正規化単体・除算境界修正の固定receipt](results/m3-first-rms-gpu-explicit-division-20261015.json)。元Linuxの未取得native内部値へ一致した証拠ではなく、画像全encoder・全profile・最終APIの合格ではない。

切り分け順: まず元RMSだけの7層を切出し、旧layer0観測のprojection入力をGPUで全bit再現（1 pass / skip0）、scale hash / epsilonを固定した。新しい768幅RMS exportは10 Red→CPU/ONNX関連35 Green、全元patch入力でも現CPU式へbit差0。最初のGPU候補は325,961 bit差・局所許容差超過0で厳密gateに失敗した。成功扱いへ置き換えず保存している。

小さい統計tensorの観測で通常GPU最終bitを保持（1 pass / skip0）した上で、除算768とepsilon加算が単一ScalarMadへ変換されたことを確認。2,520統計値のうち1,406値が現CPUの除算→加算と異なり、GPU値は「FP32逆数乗算＋epsilonを一度丸めるFMA」のoffline oracleに全bit一致した。FP64はこの単一FMA丸め診断だけで、RuntimeはFP32。既存 `GpuScalarDivide(768)` を二つのGPU segmentの間へ置き、epsilon加算を保持すると差が0になった。新しい重み・参照値・許容差変更はない。

次はこの正規化を元layer 0へ戻し、元の固定layer参照で確認する。他のRMS / 活性化、画像全3 profileの全条件、実PCM/Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。全体モデルの再反復はまだ行っていない。package22 Green、既存Runtimeは前checkpointから未変更。Unityは終了済み、最新Required CIと長期memoryは別gate。Draft PR #81、main未統合。

以下は前回checkpoint。

更新日: 2026-10-15。画像layer 0を元の正確なpatch入力から切り出すと、通常GPUは4要素超過、7静的投影をblock384へ変えた候補は1要素超過だった。元の入力・係数・参照・`atol=1e-5` / `rtol=1e-4`は維持。通常と診断downloadの最終bit不変、全projection入出力・係数のhash/shapeを記録した。[層0・全体候補・次のRMS切り分けの固定receipt](results/m3-layer0-static-gpu-and-rms-next-20261015.json)。1要素超過も不合格で、元全encoderの合格には数えない。

saved-model分割は、複数live出力・将来の残差・係数bytes・固定形状を保持し、切出した投影の再計算を拒否する。正しい残差fixtureの6 Redから、既存sliceを含む12 Greenへ進めた。同processの実layer 0だけは1 fail / skip0。初回の不正GUIDによるcompile失敗/XML欠測と、fixture前提の誤りを含む初期runは別に記録し、意図したTDD Redへ置き換えない。

**全114静的投影候補も未採用:** `a9aac194b88dadb4da79b5d4d61560c2463964df` のsquare-0は115 segmentをGPU接続し、579要素超過・最大許容差比5.757522で1 fail / skip0。計算/入力CPU fallback・中間flush0。準備8.780秒、実行＋最終取得30.978秒、host peak 11,777,482,752 byte（10.969GiB、VRAMではない）。数値gateが未達で負荷も増えるため、この候補の全体反復を止めた。他5画像caseはこの候補では未実行。次は正規化と活性化の小さい固定テストへ戻る。

768幅のRMS平均候補を独立CPU/ONNXへ照合し、7 Red→関連25 Green、package22 Green。元patchの先頭64行でも平均のbit差0。Windows Torch 2.14.1+cpuの有界1万点probeでは、`pow(x,-0.5)`と単純な`1/sqrt(x)`に1,321 bit差があり、既存の補正inverse式は同probeのnative powerへbit差0だった。これは小さいCPU算術の証拠で、元Linux全contextの再生・GPU RMS合格・画像全体合格ではない。次はこの768幅平均と補正inverseを元の固定norm入力で実GPU照合してから、全体候補を再評価する。

このcheckpointでは既存Runtimeファイルは前回から未変更。音声6ケース/2 profileと直近impulseの固定合格は下のsourceに保持する。画像全3 profile、実PCM/Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未完了。Draft PR #81、main未統合。ローカルUnityは終了済み。最新Required CIとPersistent allocation警告/長期memoryは別gate。

以下は前回checkpoint。

更新日: 2026-10-15。画像FP32の最初の差を5層のpatch prefixまで絞った。元の係数・入力・参照を固定し、通常GPUの1,256,626 bit差（局所許容差超過0）を再現。入力・出力・係数をhash付きで捕捉し、診断downloadが通常prefix最終値を変えないことを確認した。これは画像全encoderの合格ではない。[画像前段・全体・音声回帰の固定receipt](results/m3-patch384-gpu-and-original-encoder-20261015.json)。

FP32投影の加算区切り128 / 192 / 256 / 384 / 512 / 768を同じ固定入力で比較すると、**384だけが元patch出力の全1,935,360値に全bit一致**した。他5候補の不一致も保存している。これは特定入力の値を置換する補正ではなく、係数を保持する逐次FP32 FMAとblock保存加算の順序。既存3引数constructorは192を維持し、明示4引数constructorだけ1〜4096の区切りを選べる。元Block192 shader entry本体の不変も確認した。新API 9 Red→slice / 既存GPUを含む24 Green、package関連22 Green。

saved-model切出しは元のlayer descriptorをコピーし、係数backing bytesを借用する。基本2 Red、外部境界2 RedからCPU/GPUを確認。実prefix接続でSentisの入力番号不整合を検出したため、入力list位置へ全tensor indexを衝突なく再配置した。小さい2 Red→関連6 Greenを保存。番号不整合で止まったrunは数値候補の成功/失敗へ混ぜない。

**画像全体候補は未採用:** `47226d3dc2d7365e5c4bcb11345ca4bdea2c5168` のsquare-0で、正しいpatch384投影を元encoderへ戻すと579要素が元FP32許容差を超過し、最大許容差比7.406214、1 fail / skip0。GPUCompute・計算/入力CPU fallback / 中間flush0。準備1.353秒・実行と最終取得3.357秒はこの単独runのphase観測で、定常benchmarkではない。次は正確なpatch出力を起点にlayer 0内の残る投影・演算差を絞る。画像全3 profileの全条件、実Texture/PCMから正規化768次元API、role / mask / 境界 / 所有権 / Dispose / 例外後再利用 / text回帰、工程3・4のsample / Playerは未達を維持する。

同sourceの元音声impulse全encoder回帰は1 pass / fail0 / skip0、許容差超過0・最大絶対誤差1.907349e-6、GPUCompute・計算/入力CPU fallback / flush0。過去に合格した音声6ケース・2 profileの固定証拠は下に保持し、全件を最新headで再実行したとは扱わない。Unityは検証後に終了。最新Required CIと長期memory / Persistent allocation警告は別gate。Draft PR #81、main未統合。

以下は前回checkpoint。

更新日: 2026-10-15。**元音声encoderの全6ケース・2 profileが実GPU数値gateに合格。工程3・4全体は未完了。** 通常5ケースは `a26317f`、`padded_noise` は `d955554335bed9ea8006d6287b47a3a1908932c2` の固定証拠。両source間で既存Runtimeの数値処理は未変更。RTX 4070 Ti SUPER / Direct3D12 / GPUCompute、各ケースの許容差超過・計算/入力CPU fallback・中間flushは0、fail0 / skip0。元参照値・学習済み係数・`atol=1e-5` / `rtol=1e-4`は維持した。[通常5ケース](results/m4-scalar-moment-fma-full-profile-gpu-20261015.json) / [パディング・エンコーダーAPI接続](results/m4-padded-audio-and-gpu-encoder-adapter-20261015.json)。全6件を最新headで再実行したという主張ではない。

`padded_noise` は正確な `frames-1119-valid-839`、最終210×512出力を使う。元full/padded graphのSHAを固定して比較すると、差分はbinary mask initializer 3個・最終crop Constant 1個・出力形状で、その他583 initializer / 5,638 nodeは一致した。CPUモデル推論は行わず、bounded local構造比較は6〜9秒で完了。Actions 38680976728はrunner待ちのため所有jobをcancelし、Actions passには数えない。

差分はIdentity aliasをたどって安定したconsumerへ照合し、全partitionを監査してから14パーツ・15 tensor箇所へ適用。maskが変わるため「全initializer不変」はfalse、「学習済み係数保持」はtrueと分ける。継承されたplan metadataを訂正した履歴もreceiptへ保存した。Python比較9 Red、workflow1 Red、consumer/alias各1 Red、適用を含む10 Redから関連27 Green。元profile不一致を拒否するUnity 1 Redを保持し、正しいpadded modelは1 pass / fail0 / skip0 / exit0、最大絶対誤差1.430512e-6、最大許容差比0.042265。

新しい `PartitionedAudioFeatureEncoder` は `IAudioFeatureEncoder` を実装し、Workerと返却結果を所有、入力・model・GPU operationを借用する。3 Red→既存encoderを含む26 Greenで、Dispose後の結果寿命、CPU入力の拒否、prefix不一致、例外後再利用を確認。`cf3ed61` の元impulse実モデルもこのadapter経由で1 pass / fail0 / skip0、280×512 GPU出力・許容差超過0。これは保存feature→encoderの証拠で、実PCM前処理→正規化768次元APIの合格ではない。

次は画像FP32全3 profile、その後に実PCM / Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 長さ境界 / 所有権 / Dispose / 例外後再利用 / text回帰。[全体計画](multimodal-tdd-plan.md)の工程3・4にあるsample / Player等も未達として維持する。最新Required CI、Persistent allocation警告と長期memory安定性は別gate。Draft PR #81、main未統合。Unityは各検証後に終了し、実モデル1件/process・300秒上限・owned PID監視を維持する。

以下は前回checkpoint。

更新日: 2026-10-15。**通常音声profileの元5信号が、同一候補の実GPU全encoder gateに合格。工程3・4全体は未完了。** `a26317fbfbcfbf896dff4a6577124109c6cf977c`、RTX 4070 Ti SUPER / Direct3D12 / GPUCompute。元参照・モデル係数・`atol=1e-5` / `rtol=1e-4`は変更していない。Draft PR #81、main未統合。

前段2正規化のscalar moment更新で内側の乗算・加算をFP32 FMAにし、外側のmoment加算順を保持した。8レーンの順番・係数・幅32/128を監査してから適用し、既存initializer bytesを保持する。exportの意図した2 Red→6 Green、ONNX rewriteの8 Red→幅32/128を含む関連21 Green。元impulseの前段出力は1,290 bit差→候補の全bit一致となり、実GPU 1 pass / fail0 / skip0。診断captureのreadbackは製品経路の証拠と分ける。

同じ候補を元全encoderへ接続し、通常profile `frames-1119-valid-1119` の各信号を別Unity processで検証した。全5件が各1 pass / fail0 / skip0 / exit0、計算・入力CPU fallback / 中間flush0。

| 元信号 | 許容差超過数 | 最大絶対誤差 | 最大許容差比 |
| --- | ---: | ---: | ---: |
| silence | 0 | 1.4305115e-06 | 0.037593716 |
| impulse | 0 | 1.9073486e-06 | 0.046572928 |
| sine | 0 | 1.4305115e-06 | 0.041204525 |
| nyquist | 0 | 1.4305115e-06 | 0.043833878 |
| noise | 0 | 1.9073486e-06 | 0.042264722 |

[実行XML・source SHA・前段capture hash・係数保持・Python TDDを結び付けたreceipt](results/m4-scalar-moment-fma-full-profile-gpu-20261015.json)。これは検証候補の音声encoder合格で、公開AudioEmbedder経由の実PCM→768次元・全profile・画像の合格ではない。従来のmean-only候補の521 bit差 / 全encoder360要素不合格は、下の失敗receiptに保持する。

次は `padded_noise` の正確な `frames-1119-valid-839` 用モデルを準備し、元maskと出力210行を保持してGPU検証する。その後、画像FP32全3 profile、実PCM / Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 長さ境界 / 所有権 / Dispose / 例外後再利用 / text回帰へ進む。[全体計画](multimodal-tdd-plan.md)の工程3・4の範囲を維持し、sample / Player等の未検証項目も完了へ数えない。

到達不能Acceleratorへの待ちは検証consumerのproject cache無効化で対処済み。root project / global設定は保持した。1 processに実モデル1ケース・300秒上限・owned PIDのmemory監視を続ける。Persistent allocation警告・長期memory安定性、現sourceのRequired CI全合格は別gate。モデル・重み・大きなcaptureはGit/LFSへ入れない。

以下は前回checkpoint。

更新日: 2026-10-15。工程3・4は未完了。元無音の全encoder GPU gateは合格済みだが、元音声全6信号・両profile、画像FP32全3 profile、実PCM / Texture→正規化768次元APIの全条件は未達。元参照・モデル係数・`atol=1e-5` / `rtol=1e-4`を保持する。Draft PR #81、main未統合。

**Unityの待ちへの対処:** 現在の検証consumerがglobal Accelerator設定を継承し、到達できないサーバーへの接続timeoutを繰り返していた。`create_consumer(automation=True)`の既存方針へ戻し、consumer内だけでCacheServerMode.Disabled / download0 / upload0を適用した。所有Unityが停止していることを確認し、設定の変更前backupと前後SHAを保存。root projectのEditorSettingsとglobal preferencesは変更していない。現在のローカルbounded runnerも起動前にこの方針を再適用する。

`39c2559afa61c33aa029dc1328f781dc12442261`のEditor API確認は変更前1 Red。変更後は同sourceでcache設定・CPU/GPUのshape捕捉・tiny GPU profilerを含む4 pass / fail0 / skip0、全process29.157秒。到達不能サーバーと接続timeoutのログは消え、tiny GPUはGPUCompute・計算/入力CPU fallback / flush0。変更前は40.164秒だったが、test数が違う各1回の観測なのでbenchmarkや一律の速度改善率とは呼ばない。[設定・ログ・GPU固定receipt](results/local-consumer-cache-and-gpu-check-20261015.json)。[Unity公式のproject単位Disabled設定](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/CacheServerMode.Disabled.html)。Persistent allocation警告の原因・長期安定性は別問題として未解決。

**音声前段の切り分け:** 全encoderの繰返しを減らすため、元のsegment-0と最初のGPU投影だけを使う小さい固定テストを追加。実際のTensorShapeも捕捉し、通常の前段出力を診断downloadが変えないことを確認する。shape捕捉はCPU/GPUの2 Redから2 Green。`0420495`では元impulse subsample出力との1,290 bit差を再現し、局所許容差超過は0。6内部境界・投影前後・最初のConv係数をhash付きで保存した。診断用readbackは製品経路のreadbackなしという主張へ含めない。

最初のConvについて独立offline FP32 FMA oracleを比較すると、GPUの4,587,520値がkernel height/width順の単一FMAと全bit一致し、乗算・加算を別々に丸める版とは2,473,076値が異なる。これはGPUの計算順の確認で、取得できていないnative中間値の合格ではない。最初の正規化のGPU値も、scalar mean更新をunfusedで計算するoffline式へ全bit一致した。

**候補は未採用:** 最初の正規化のscalar mean更新8か所をFP32 FMAへ変えたderived segmentを作り、既存initializer bytesの一致を確認した。`f268d1c`の前段は元impulseとのbit差が1,290→521、局所超過0・最大許容差比0.0765705。しかし厳密bit gateは1 fail / skip0。元全encoderも360要素超過・最大許容差比20.973450で1 fail / skip0（以前の396要素・13.077703より件数は減るが最大比は悪化）。GPUCompute・計算/入力CPU fallback / flush0。候補の局所改善を全体合格や採用へ置き換えない。[参照不一致・捕捉・候補失敗receipt](results/m4-impulse-subsample-context-and-first-norm-fma-20261015.json)。

**CPU参照の現状:** signalを選ぶ`subsample`診断scopeを追加し、既定のfull gateは保持した。元feature hash・stage input/finalを固定した[38676791482](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38676791482)は、元subsample finalを再現できずfailure。失敗診断を追加した[38677340121](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38677340121)もfailure。後者はCPU実行条件24通り（threads1/2/4、oneDNN有無、係数/feature配置）を全て実行したがbit一致は0件。各条件の局所許容差超過0を元context合格と扱わず、新しいnative partsは保存していない。診断後は元のCPU設定・入力・重み・hookを保持する。workflow1 Red→関連41 Green、診断3 Red→関連51 Green。これらはCPU診断・Python単体の証拠で、GPU全体合格ではない。

動的feature選択へ対応する既存workflowテストも、default/fullのsilence入力・元hash・subsample gateを検証する形に更新した。旧文字列形式による1 Redを確認し、packageを含む関連Python回帰は74 pass / fail0 / error0 / skip0。全環境のRequired CI合格へは流用しない。

次は保存した小さいGPU前段で、正規化の残るmoment/varianceの演算順と元CPU参照の実行文脈を調べる。全体の順序と範囲は[全体計画](multimodal-tdd-plan.md)を維持し、音声全6信号・2 profile、画像全3 profile、最終APIの全role / mask / 長さ境界 / 所有権 / Dispose / 例外後再利用 / text回帰を未達のまま追う。生成model・重み・大きなcaptureはGit/LFSへ含めない。

以下は前回checkpoint。


更新日: 2026-10-15。工程3・4は未完了。**元音声encoderの無音ケースはGPU数値gateに合格したが、他信号は未合格。** `c1f90de187f2b17f30461f61cbaf22d522bacd34`の元silenceは許容差超過0、最大絶対差1.430512e-6・最大許容差比0.037594、NUnit 1 pass / fail0 / skip0。RTX 4070 Ti SUPER / Direct3D12 / GPUCompute、計算・入力CPU fallback / 中間flushは0。元参照・モデル係数・`atol=1e-5` / `rtol=1e-4`は変更していない。Draft PR #81、main未統合。[今回の固定receipt](results/m4-native-table-tanh-and-original-audio-signals-20261015.json)。

通常profile `frames-1119-valid-1119` の実GPU結果（各1ケース／Unity process）は次のとおり。無音の合格を全信号・全profile・最終APIの合格へ流用しない。

| 元信号 | 許容差超過数 | 数値gate |
| --- | ---: | --- |
| silence | 0 | pass |
| impulse | 396 | fail |
| sine | 423 | fail |
| nyquist | 783 | fail |
| noise | 1 | fail |

前3ケースは `c1f90de`、後2ケースは `5d4f5ba`。後者の変更は観測テストだけで、Runtimeは同一。全5件skip0、requested / actual GPUCompute、計算・入力CPU fallback / flush0。`padded_noise`の `frames-1119-valid-839` は対応する分割候補をまだ準備しておらず、この候補では未実行。

**今回の修正:** Intel IGCのMITライセンス付き一般27区間・6次多項式係数をFP32 high / residualへ分け、GPU上のFP32だけで評価する。特定のモデル入力値を置き換える補正ではない。元layer_3のtanh固定テストは4 bit差の1 Redから0 bit差へ修正し、layer_0・1・3の計165,888値が元nativeへ全bit一致。入力借用・Disposeを含む4 GPUテストはpass4 / fail0 / skip0。小表版oracleの不一致も保存し、選んだ係数表のsource SHA・MIT / BSL noticeをpackageに保持する。Windows CPUの小さい同一入力診断も元値へ全bit一致したが、それはGPU合格の代わりには数えない。

`5d4f5ba`の元impulse・12層観測は通常最終bitを変えず1 pass / fail0 / skip0。layer_0から13,864値のbit差があり、最初の局所許容差超過はlayer_8の1要素。`4b2dbb7b2cd652df59e4bb1cc6759cd20d557256`でsubsampleを加えた13境界観測も1 pass / fail0 / skip0、通常最終bit不変。**subsample出力ですでに1,290値のbit差（局所超過0、最大許容差比0.114856）がある。** 次はこの前段の最初の差を切り分ける。観測成功は元impulse全encoderの396要素不合格を取り消さない。

元の信号を明示してnative層参照を生成する機能を追加。producer / CLIの3 Red、workflowの1 Redから関連40 Greenへ進めた。既定のfull scopeは元silenceのsubsample gateを保持し、attention scopeだけで元full-profile信号と層を選択できる。`91284239131a4fdc64a8eaddadd816c5de6f7677`の[impulse layer_0参照38676003460](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38676003460)は、元の入力・最終値を固定してnative finalへ全bit一致。45 partsの生成とreport SHAを確認した。binary partsはローカル未取得で、subsample内部のnative参照はまだ生成していない。

package関連Pythonテストは22 pass / fail0 / error0 / skip0、現packageの配布監査もsuccess。これはGPU全chainやRequired CI全合格ではない。`5d4f5ba`のCI38675402233はcancelledを確認し、成功へ数えない。既存の独立Python915件×4環境の固定結果は前sourceの証拠として保持し、今回の信号選択追加後の全回帰へ置き換えない。

次の順序: (1) impulseのsubsample差と残る音声数値誤差を修正、(2) 元音声全6信号・両profileと画像FP32全3 profileの実GPU gate、(3) 実PCM / Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 長さ境界 / 所有権 / Dispose / 例外後再利用 / text回帰、(4) 証拠とPRを整備する。[全体計画](multimodal-tdd-plan.md)の全範囲を維持する。最終API・全環境・正式公開は未達。Persistent allocation警告の原因と長期メモリ安定性も未解決。

以下は前回checkpoint。


更新日: 2026-10-15。工程3・4は継続中、未完了。最新の実モデル全encoder gateは `a2ad709067422f0040e0533f665b0ae2521828ac`、元silenceで**179要素不合格・最大許容差比4.642802**。GPUCompute / RTX 4070 Ti SUPER / D3D12で、計算・入力CPU fallback / 中間readback flushは0。参照値・モデル係数・`atol=1e-5` / `rtol=1e-4`は変更していない。main未統合、Draft PR #81。実行順と全範囲は[全体計画](multimodal-tdd-plan.md)を維持する。

モデル準備はCPUで行う処理を含むが、モデルの計算はGPU。`a2ad709`の一実行では読込・graph変換1.435秒、native operationの係数確認・構築12.626秒、Worker作成0.073秒、入力作成・pin0.002秒、推論投入1.748秒、最終download・GPU待機0.523秒。投入から最終取得は2.271秒で、純GPU kernel時間や定常benchmarkではない。Windows native counterでhost peak working setは4,160,483,328 byte（約3.87 GiB）。これはVRAMではなく、変更前とのメモリ削減量も未計測。[負荷と寿命の固定receipt](results/gpu-intermediate-lifetime-and-phase-profile-20261015.json)。

`GpuProjectionPipeline`は中間Tensorを最後の消費stage後に解放するよう修正。残差の将来利用と最終出力を保持し、呼出側入力・演算objectを借用する。修正前 `56c9f5e` の解放タイミングテストは意図した1 fail、修正後 `c8b8901` は関連5 pass / fail0 / skip0。元silence全encoderの超過179要素は変更後も同じで、数値合格へは数えない。Persistent allocation警告1件は小さい失敗テストにも存在し、今回の修正で警告や長期安定性が解消したとは主張しない。

`eb4dedb`のtanh係数残差修正はlayer_0・layer_1の元固定入力合計110,592値へ全bit一致し、APIを含む3 pass / fail0 / skip0。元全encoderの無音は532から179要素不合格へ変化した。12層観測は通常最終bitを変えず1 passで、**layer_0〜2が全bit一致、最初の層出力差はlayer_3の5,216値**。最初の局所許容差超過はlayer_10の8要素、layer_11は627要素。[tanh局所成功と全体未合格](results/m4-native-tanh-layer1-and-all-layer-observation-20261015.json)。

元layer_3の入力・最終出力を固定したCPU参照[38673432736](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38673432736)は、元最終値へ全bit一致。`91f2620`の35境界GPU観測は1 pass / fail0 / skip0、通常最終bit不変・選択層入力も元bitと一致。FF1、Q/K/V、relative K、matrix_ac / matrix_bd、tanh入力まで全bit一致し、**tanh出力で初めて4値のbit差**がある。`9980926`の独立した固定tanhテストも4 bit差で1 fail / skip0。局所許容差超過は0だが、全体gateへ影響する候補として未修正の失敗を固定した。[layer_3の発生箇所と失敗receipt](results/m4-layer3-first-tanh-rounding-difference-20261015.json)。

独立offline oracleは現GPUを全bit再現する。FP64 tanhをFP32へ丸めても元nativeとの同じ4値差が残り、精度を上げるだけでは元値を再現できない。固定PyTorch sourceにはMKL有効時のVML Tanh経路があるが、元ジョブの実経路確認は未完了。次は実行文脈・MKL VMLの演算順を確認する。特定入力値だけを置き換える補正や参照値の更新は行わない。

独立Python回帰[38671700527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38671700527) / `3ee8279`は、取得した4 XML各915 pass / fail0 / error0 / skip0。選択層参照生成とworkflow変更を含む。後続C# / shaderを含む最新sourceの全Required CI合格とは扱わない。[Python固定receipt](results/python-full-selected-layer-replay-green-20261015.json)。

次の順序: (1) 元音声encoderの残る数値誤差修正、(2) 音声全6信号・2 profileと画像FP32全3 profileの実GPU gate、(3) 実PCM / Texture→GPU前処理→encoder→backbone→正規化768次元APIの全role / mask / 長さ境界 / 所有権 / Dispose / 例外後再利用 / text回帰、(4) 証拠を整理してPRをレビュー可能にする。全chain・全環境・正式公開は未達。モデル生成・CPU参照はActionsを優先し、ローカルUnityは実モデル1ケース／process・300秒上限、所有PIDのメモリ監視で区切る。

以下は以前のcheckpoint。


**最新のnative除算・softmax接続:** `GpuScalarDivide`は残差FMA補正を使い、固定native softcap除算55,296値を全bit一致、2 Red→実GPU2 Green。plannerは9 Red→関連65 Green。`GpuSoftmax`は元指数関数と8 laneのFP32加算順を用い、固定native55,296値を全bit一致、2 Red→実GPU2 Green。plannerは6 Red→関連71 Green。ともにGPU入力借用・独立出力・CPU入力拒否後の再利用・Disposeを検証した。[固定GPU成功](results/m4-native-division-and-softmax-frozen-gpu-green-20261015.json)。他GPU・全native入力・全profileのbit一致は未証明。

元initializer digestを保つ255 operation / 256 segmentを生成。除算までの`961f617`は元silence全encoder269要素不合格。softmaxまでの`b6c96cf`は**125要素不合格・最大許容差比3.300957**、NUnit0 pass / 1 fail / skip0 / exit2。準備19.774秒、実行＋最終診断download6.740秒、Unity全process87.526秒。これらは計測した一実行の値で、製品benchmarkや純GPU時間ではない。計算fallback / flush0を確認した。

同sourceの最初のAttention内部13境界（matrix_ac / matrix_bd / weighted_valuesの左右入力・出力と、tanh / softmaxの入力・出力）は**すべて元参照へ全bit一致**。観測は通常最終bitを変えず、独立したUnity processで1 pass / fail0 / skip0、66.102秒。全体125要素不合格を再現し、観測成功を数値合格へ数えない。次は後段のpost投影とlconv1d境界を比較する。[全体gateと13境界観測](results/m4-native-softmax-whole-and-first-attention-20261015.json)。Persistent allocation警告1件は残り、リーク解消は未証明。

独立Python回帰38666006647 / `2a0fea6`は4 XML各879 pass / fail0 / error0 / skip0を確認。[Python固定証拠](results/python-full-native-tanh-planner-green-20261015.json)。除算・softmax planner追加後の38667280645 / `b6c96cf`は別runで、未確定の間は成功へ数えない。音声全6信号・2 profile、画像FP32全3 profile、実PCM / Textureから最終768次元GPU APIまでのgateは引き続き未達。以下は以前のcheckpoint。

**最新のnative tanh切り分け:** `223f47d`の`GpuTanh`はSLEEF固定commitの演算をFP32の残差ペアでGPU実装し、元55,296値へ全bit一致。実GPU2 Red→2 Green、符号付きゼロ・CPU入力拒否後の再利用・借用入力と独立出力・二重Disposeも確認した。最初のCopyOutput呼出しのcompile失敗は未実行として別記録。plannerは5 Red→関連56 Green、元の重みdigestを保持する231 operation / 232 segmentを生成した。[単体GPUとsoftmax oracle](results/m4-native-tanh-frozen-gpu-and-softmax-oracle-20261015.json)。

`2a0fea6`の接続観測は元silenceで320要素の許容差超過を記録し、直前166要素より悪化した。最初のmatrix_ac / matrix_bdの6境界とV入力は全bit一致するが、tanh入力21,923値・出力17,473値、softmax後の重み25,153値にbit差が残る（局所許容差超過0）。単一の実モデル観測を新しいUnity processで実行し、通常最終bit不変・11境界観測1 pass / fail0 / skip0、54.387秒で完了。これは観測成功で、全encoder数値合格ではない。

固定native BMM出力から元rel_shiftとFP32加算を再現すると、`sum / float32(50)`はnative tanh入力へ全bit一致し、`sum * (float32(1)/float32(50))`は現在のGPU tanh入力へ全bit一致した。次は元除算のFP32丸めをGPU上で保持する。softmaxも元SLEEF expと元vector加算順のoffline oracleで55,296値へ全bit一致したが、GPU softmaxは未実装・未検証。[接続観測と除算原因](results/m4-native-tanh-connected-and-division-diagnosis-20261015.json)。

全体数値gateと観測を同じprocessで連続した最初のrunは13.76GiBに達し、所有PIDを停止、XML欠測。既存の未完了出力をarchiveする前に開始した一度の再試行は出力重複guardで失敗し、推論未実行として記録。その後の単独観測のみが上記1 pass。以後の実モデル全体ケースは1 processに1ケースとする。独立Python回帰38664601644 / `8975030`は4 XML各874 pass / fail0 / error0 / skip0。[Python固定証拠](results/python-full-native-query-scale-planner-green-20261015.json)。tanh planner追加後の38666006647 / `2a0fea6`は別runで、未確定の間は成功へ数えない。以下は以前のcheckpoint。

**最新のQスケーリング修正:** 元の`(Q × scalar) × vector`は固定native出力へ全bit一致する一方、`Q × (scalar × vector)`は修正前GPU出力へ全bit一致し、93,060値の差の原因を再現した。元の2回のFP32丸めを保持する`GpuSequentialScale`は2 Red→実GPU2 Green、286,720値を全bit一致。plannerは厳密な2 Mul chain、係数・shape・内部値の専用利用を確認し、8 Red→関連51 Green。元initializer digestを保つ219 operation / 220 segmentを生成した。[固定GPUと接続観測](results/m4-native-query-scale-frozen-and-connected-gpu-20261015.json)。

`b9cf8d6`の観測は使用済みWorker・中間GPU tensorを各段で解放し、小型単入力 / 二入力観測を含む3 pass / fail0 / skip0、68.110秒で完了。通常最終bitを変えず、最初のmatrix_ac / matrix_bdの左右入力と出力の6箇所がすべて元参照へ全bit一致。weighted_valuesの左入力に25,744値、出力に189,856値のbit差が残る（局所許容差超過0）、右入力Vは全bit一致した。

`a0ec884`の元silence全encoderは**166要素不合格・最大許容差比4.229706**。準備12.692秒、実行＋最終診断download2.094秒。固定native入力の標準GPU tanhは31,467値、softmaxは14,877値のbit差（ともに局所許容差超過0）。3つの厳密gateは0 pass / 3 fail / skip0 / exit2。次はこれらの丸めを固定参照で切り分ける。[全体と非線形演算の失敗証拠](results/m4-native-query-scale-whole-and-nonlinearity-failures-20261015.json)。

独立Python全体回帰 [38663414106](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38663414106) / `746fbc0`はterminal success、4 XMLで各866 pass / fail0 / error0 / skip0を確認した。[固定回帰証拠](results/python-full-native-bmm-planner-green-20261015.json)。Qスケーリング追加後の`8975030`を対象とする38664601644は別runで、未確定の間は成功に数えない。以下は以前のcheckpoint。

**最新のGPU行列積接続:** 固定native入力の3種類のAttention行列積はGPU上で全bit一致し、API検証を合わせて5 pass / fail0 / skip0。二入力のGPU pipeline接続は4 pass / fail0 / skip0。[固定入力での成功](results/m4-native-bmm-frozen-gpu-green-20261015.json)。固定shapeを再計算する後続subgraphが切出し済み演算へ戻る問題は、小型再現1 Red→関連43 Greenで修正。形状を参照する元GPU tensorを保持し、元の演算・重みを維持した208 segmentの生成に成功した。

ただし、接続した元silence全encoderは**415要素不合格・最大許容差比9.637922**。元許容差・参照値は変更していない。準備12.692秒、実行＋最終診断download2.266秒、NUnit0 pass / 1 fail / skip0 / exit2。前候補の203要素より件数が増えたため、接続先の入力誤差を観測する必要がある。[全体の失敗証拠](results/m4-native-bmm-whole-gpu-failure-20261015.json)。固定入力での成功を全encoder・全profile・最終APIの成功へ流用しない。

接続入力9箇所の初回診断 `62b1501` は通常実行と観測のモデル重複保持で13.14GiBに達し、所有Unity PIDをメモリ上限で停止した。XML欠測・exit4294967295で不合格、観測完了ではない。`9404b92`で通常Workerを先に解放し、既に取得済みの通常最終値を比較に再利用するよう修正。再検証は小型の単入力・二入力観測を合わせて**3 pass / fail0 / skip0、66.965秒で完了**。通常最終bitを変えず、scaled Qは93,060値にbit差（局所超過0）、K・relative-K・V入力は全bit一致。weighted_values入力には26,294値、出力には192,816値のbit差が残る（いずれも局所超過0）。次はQスケーリングの定数・FP32演算順を固定nativeと比較する。[接続演算の観測](results/m4-connected-native-attention-operands-20261015.json)。Persistent allocation警告1件は残り、リーク解消や製品性能の合格は未証明。以下は過去の切り分け記録。

**2026-10-15最新のattention切り分け:** 直接GPU operation出力の観測を1 Red→1 GPU Greenで追加。`1008d6a`の12境界観測は通常最終bitを変えず、最初のFFまでの5境界に加え、Q / K / V / relative-K投影を元のClip境界で全bit一致させた。最初の差はself_attnの201,927値（局所許容差超過0・最大比0.255179）、attention残差68,209値、lconv1d201,063値へ伝播する。観測1 pass / skip0は全encoder成功ではなく、全体203要素不合格を再現した。

native投影の記録は5 Red→関連20 Green。full replay Actions `38659893615` / source `7ba22ac`はCPU subsample終端不一致で失敗し、失敗receiptを保持した。独立したattention scopeを1 Red→関連21 Greenで追加し、既存full / subsample gateをdefaultとして保持。Actions `38660240389` / source `c030222`のattention scopeは成功し、固定layer_0終端と厳密一致、既存12 partのhashも元`99529dc0…`の保存値と全件一致した。この基準でGPUのQ / K / V / relative-Kが全bit一致することを確認した。

次の内部演算captureは4 Red→関連24 Green。native operatorを置き換えずに3つのbmmの左右入力 / 出力、tanh / softmaxの入力 / 出力を記録し、元layer終端のbit一致とhook復元を維持する。独立attention Actions [38661214188](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38661214188) / source `5a51cfff0efa955b6862abc481b2230f312b87ac`はterminal success。取得した30 partのうち既存17 partはすべてhash一致。固定native入力のoffline FP32-FMA丸めoracleでは、K=128 / 128 / 24の3 bmmすべてが連続積和（192項以内なので途中resetなし）で全bit一致し、16項resetの候補には39,299 / 22,407 / 65,721値のbit差が残った。FP64はこのoracleのみで、GPU成功には数えない。[固定native演算とoracle](results/m4-native-attention-operation-reference-and-fma-oracle-20261015.json)。次はGPU上のscaled Q/Kと入力を照合し、native順のbatched matmulを実GPUで検証する。[固定GPU観測・native投影・full失敗・TDD](results/m4-first-attention-projections-bit-exact-20261015.json)。製品計算はGPUのままで、raw GPU snapshotのClipと元残差和は診断比較だけに使う。工程3・4の全条件は未達のまま。以降は前回checkpoint。

**2026-10-15最新のSiLU全接続:** GPU投影とSiLUを借用する`IGpuTensorOperation`を追加し、既存の投影専用constructorを保持した。混在接続は実GPU1 Red→関連3 Green。plannerは元SiLUのNeg / Exp / Add / Div chainを厳密に確認し、内部値の外部参照・係数 / shape不一致を拒否してGPU境界へ分割する。7 Red→関連35 Green。元exportの内部3 nodeが無名であるケースは1 Red→関連36 Greenで、無関係な無名nodeも保持するよう修正した。無名nodeの識別失敗とshape未確定の準備拒否はGPU未実行として別記録。

Sentisのmetadata監査で元の483境界を確定（NUnit1 pass、推論なし）。元initializer digest `1154ac70…`を保つ135投影＋36 SiLU / 171 GPU operation / 172 segment / 41,830 ONNX nodeを生成した。`3ae76bd`の元silence全encoderは**203要素不合格・最大許容差比6.967976**、GPUCompute・計算fallback / flush0。前の全135投影539要素から件数は減ったが、元許容差を満たさず、以前の2投影175要素より優れるとも判定しない。準備13.295秒、実行＋最終診断download2.497秒。数値gateと観測を合わせたNUnitは1 pass / 1 fail / skip0 / exit2。

初期5境界（入力投影・最初のpre norm後の入力・活性化後の入力・最初のFF出力・norm_pre_attn）は元参照へすべて全bit一致し、診断readbackを加えても通常最終bitは変わらなかった。元Clip境界を合わせる期待値変換と、参照ファイル本体の不変を分ける。**全encoderへの接続はできたが、数値受け入れは未完了。** 次はbit一致したnorm_pre_attnの直後のattentionブロックを元native参照で観測する。[固定接続・TDD・監査・全体不合格と初期一致](results/m4-native-silu-whole-gpu-and-early-bit-exact-20261015.json)。Persistent allocation警告は再発し、発生元は未検証。所有Unityはterminal。

Python全体の新しい独立Actions [38657424472](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38657424472) / 固定source `3ae76bd182fb80dd74c9b7851a99ce719e4da67d`はterminal success。全4 XMLで各851 pass / fail0 / skip0を確認した。[固定回帰](results/python-full-native-silu-partition-green-20261015.json)。Python成功を実GPU全encoder成功へ流用しない。工程3・4の音声元6信号 / 両profile、画像FP32全3 profile、最終768 APIの全条件は未達のまま。以降は前回checkpoint。

**2026-10-15最新のSiLU修正:** `f774707`の5境界観測は通常最終bitを変えず、input_projectionと最初のpre_layer_norm後の入力を元参照へ全bit一致させた。最初の差はactivation_inputの826,899値（元の局所許容差超過0・最大比0.001836）。最初のFF出力89,170値、norm_pre_attn81,727値へbit差が伝播する。観測値をhash付きのGit対象外binaryへ保存した。元のquantized-linear入力Clipは期待値だけに適用し、参照ファイル / 製品GPU計算は変更しない。

固定PyTorch commit `5c488690…`が参照するSLEEF commit `5a1d179…`のexpf手順を確認した。独立したFP32-FMA丸めoracleは最初のnative SiLU 1,146,880値を全bit再現。FP64はoffline oracleだけ。GPU API / 固定活性化の2 Redを確認後、FP32 range reduction・polynomial・native MAD・除算の残差補正を行うopt-in `GpuSilu`を追加した。BSL-1.0の著作権表記とlicenseを同梱。`e2121e9`の実RTX 4070 Ti SUPER / D3D12は2 pass / fail0 / skip0 / exit0、元SiLU 1,146,880値を全bit一致。CPU入力拒否後の再利用、signed zero、独立GPU出力、op解放後の借用入力 / 出力の生存も確認。[固定観測・源コード・TDD・GPU成功](results/m4-native-silu-frozen-gpu-green-20261015.json)。shaderのbounded range外はGPU数学経路で、bit一致は未検証。他GPU / OSの保証にも流用しない。

**全encoderへのSiLU接続は未完了。** 既存135投影候補は539要素不合格のまま。次はSiLUを投影と同じGPU tensorのstageとして接続し、最初のFFと元全encoderへ戻す。音声元6信号 / 両profile、画像FP32全3 profile、最終768 APIの全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰 / 中間readback・CPU計算fallback禁止は未達。既存Persistent allocation警告の発生元も未検証。所有Unityはterminal。

独立Python全体回帰 [38654418805](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38654418805) / 固定source `e25217575b591e37f68e37943d49e47986f19682`はterminal success、4環境各843 pass / fail0 / skip0。全4 XMLで確認した。以降のPython変更はなく、追加GPU SiLU / 新しいUnity観測 / Required CIの成功には流用しない。[固定回帰](results/python-full-projection-shape-audit-green-20261015.json)。以降は前回checkpoint。

**2026-10-15最新の誤差伝播診断:** 観測helperは小さいGPU1 Red→1 Greenで、通常と観測の最終bit一致と、借りた入力 / kernelの生存を確認した。`b29c5ad`の135投影候補は通常最終値539要素不合格を再現し、診断readbackを加えても最終値のbitは変わらなかった。元の12層参照との照合ではlayer_0〜9は許容差内（bit差は残る）、最初の超過はlayer_10の42要素・最大許容差比8.265064、layer_11は2,021要素・19.720919。NUnit1 passは観測完了であり、全encoder数値合格ではない。

layer_10を元graphから切り出した11投影 / 12 segment / 2,941 ONNX nodeへ、元native layer_9出力を入力した。`e252175`の単独GPUは許容差超過0・最大許容差比0.243686、GPUCompute・計算fallback / flush0、NUnit1 pass / fail0 / skip0 / exit0。準備1.186秒、実行＋最終診断download0.771秒。全体のlayer_10不合格には前段入力の誤差伝播が関与するが、layer_10の全bit一致や全encoder合格は証明していない。小さい観測と単独層の終了logにはPersistent allocation 1件の警告があり、発生元 / leak freedomは未検証。所有Unityはterminal。[固定観測・TDD・単独GPUと限界](results/m4-all-static-original-layer-propagation-20261015.json)。

次はlayer_10を繰り返すのでなく、元の最初のfeed-forward固定参照を使って、早い投影の入力と出力の最初の微差を調べる。Python全体は独立Actions [38654418805](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38654418805) / 固定source `e25217575b591e37f68e37943d49e47986f19682`を開始し、結果待ち。工程3・4の全条件は未達のまま。以降は前回checkpoint。

**2026-10-15最新の全135投影候補:** runtime shape監査は123個を確認し、135個の期待数に届かず不合格だった。元ONNXと係数hashで照合した結果、残り12個はすべて定数入力のrelative-key投影で、Sentisの定数畳み込み対象と分かった。畳み込みへの変換という仮説は否定した。元定数のshapeだけを読み、実行時123個と合わせた135個のmetadata監査を追加。元graphの演算・係数を保持して分割境界を出力へ公開する別graphで、Sentisの固定shape411個を確認した。NUnit1 pass / fail0 / skip0 / exit0、推論は未実行でGPU数値合格ではない。

shape補完5 Red→関連19 Green、定数投影の切り出し1 Red→関連20 Green、境界overlay7 Red→関連27 Green、元integer / scalar metadata保持1 Red→関連28 Green。135投影 / 136 segment / 41,938 ONNX nodeを生成し、すべてのsegment inputの固定FP32 shapeと元initializer digest `1154ac70…`を確認した。`a2fca44`の元silence全encoderはGPUCompute・計算fallback / flush0、NUnit0 pass / 1 fail / skip0 / exit2。**539要素不合格・最大許容差比6.730431**で、前の175要素・4.043479から悪化したためこの候補を採用しない。準備12.463秒、実行＋最終診断download2.499秒は1入力の候補計測であり、最終APIの性能保証ではない。[固定監査・TDD・準備・実GPU不合格](results/m4-all-static-partitioned-gpu-failed-20261015.json)。所有Unityはterminal、終了後のlive queryでUnityなしを確認した。

次は135投影候補と前の2投影候補を元の層別参照へ照合し、最初の誤差増加を起こす投影を特定する。全投影の積和順を同一と推測して採用しない。工程3・4、音声元6信号 / 両profile、画像FP32全3 profile、実PCM / Textureから最終768 APIの全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰 / 中間readbackとCPU計算fallback禁止は引き続き未達。以降は前回checkpoint。

**2026-10-15最新の全encoder接続:** 直接GPUの最初の2投影を、元graphの投影前 / 間 / 後段へGPU tensorのまま接続した。partition plannerは6 Red→6 Green、入力残差のbyte / 係数 / 演算を保持する小さい独立照合を確認。`GpuProjectionPipeline`は2 Red→2 GPU Greenで残差・連続実行・独立出力・Workerの解放 / 借りた投影器の保持を確認。初回の実モデルは中間segmentの入力をそのまま出力するONNXをSentisがbroken linksとして拒否し、推論未実行。未使用input拒否不足1 Redと明示Identity不足1 Redから、未使用inputを除き、値を変えないIdentityとoutput aliasで残差を受け渡すように修正し、関連19 Green。元参照は置き換えない。

`e9e5c92`の実GPU全encoderは無音175要素不合格・最大許容差比4.043479。前候補295要素・7.637025より改善したが、元`atol1e-5 / rtol1e-4`を満たさない。GPUCompute・計算fallback / flush0、NUnit0 pass / 1 fail / skip0 / exit2。3 segmentの実layer数9,120 / 7 / 58,004、準備23.371秒、実行＋最終診断download10.948秒。直接GPUは2投影だけで、他のstatic / dynamic投影は従来候補。所有Unityはterminal、現在停止済み。[分割TDD・import失敗・固定全体結果](results/m4-partitioned-first-ff-whole-gpu-failed-20261015.json)。

残りの固定行列を同じkernelへ切り替える準備では、元graphに135個のstatic MatMulを確認したが、ONNX境界の一部に未確定shapeがあり、plannerが変更前に拒否した。既存value_infoを除いてshape inferenceを再実行しても231境界に不明dimensionが残った。これは準備拒否 / GPU未実行で、数値失敗でもGPU成功でもない。次は元の固定modelをSentisでshape監査し、hashに結び付いた確定値だけで次の分割へ進む。工程3・4、音声元6信号 / 両profile、画像FP32全3 profile、実入力から最終768 API全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰 / 中間readbackとCPU計算fallback禁止は未達のまま。以下は前回checkpoint。

**2026-10-15最新の直接GPU投影:** 最初のfeed-forwardの2投影を元280行で検証した。static64は781,856 / 233,045値のbit差（元許容差超過は両方0）、NUnit2 fail / skip0。標準ONNXへFMAを展開したblock192候補は、最初の投影中にprocess working setが36,685,799,424 bytesへ増え、300秒timeout後のkill待機30秒もtimeoutした。数値出力 / NUnit XMLは未取得で、GPU合格ではない。その後のlive queryでUnity processなしを確認した。最初のCS0136 compile失敗は変数名衝突を修正し、推論未実行として別記録。[資源失敗と終了確認](results/m4-first-ff-expanded-gpu-resource-failure-20261015.json)。この大きい展開graphは再試行しない。

`GpuBlockedMatmul`を入力GPU tensor・setup時の係数upload・owned GPU outputだけで動く直接compute kernelとして追加した。API3 Red、追加契約を含む8 Red→8 Green。積和の丸め・192項ごとの集計 / tail・CPU入力拒否・係数検証・不正入力後の再利用・独立した出力 / 二重Disposeを実GPUで確認した。`e9e72e8`の実固定投影2件は、元clipもGPUで実行し、1,146,880 / 286,720要素すべてが元native値と全bit一致。GPUCompute・Clip worker計算fallback / flush0、NUnit2 pass / fail0 / skip0。実行＋最終診断downloadは212.7533 / 27.6472msで、モデル全体のlatencyには数えない。[API TDDと固定GPU成功](results/m4-direct-gpu-first-ff-projections-green-20261015.json)。現在Unityは停止済み。

HLSLの[公式mad仕様](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/mad)は、native命令をfused / nonfusedのどちらにも実装可能としている。この候補の丸めはRTX 4070 Ti SUPER / D3D12の実測で確認したもので、全GPU / OS保証ではない。CPUへの計算fallbackは入れない。次は元encoderを投影境界で分け、GPU tensorのまま直接kernelへ渡して全encoderを再照合する。既存の全encoderは無音295要素不合格のまま、音声元6信号 / 両profile・画像FP32全3 profile・実PCM / Texture→最終768 API全role / mask / 境界 / 寿命 / 例外後再利用 / text回帰は未達。工程3・4の範囲を維持する。以下は前回checkpoint。

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
