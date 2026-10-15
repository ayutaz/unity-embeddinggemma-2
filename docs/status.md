# 現在の状況

更新: 2026-10-16。対象はDraft [PR #81](https://github.com/ayutaz/unity-embeddinggemma-2/pull/81)の`feat/multimodal-numerical-acceptance`。main統合・リリースは未実施です。ユーザー指定「4まで」の完了条件は[元計画の監査](results/goal4-scope-audit-20261016.json)を維持します。

## 手順3・4の検証

| 条件 | 確認済み結果 | 固定証拠 |
| --- | --- | --- |
| 元画像encoder、6画像・3 profile | 全778,240値が元参照とbit一致 | [画像encoder](results/m3-adjacent-reciprocal-all-image-gpu-passed-20261015.json) |
| 元音声encoder、6信号・2 profile | 全824,320値で元atol=1e-5 / rtol=1e-4の超過0 | [分割音声encoder](results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json) |
| 実PNG Textureから最終768次元、全role | 18条件合格 | [画像API](results/m3-final-image-api-all-cases-all-roles-gpu-20261016.json) |
| RenderTexture、実不正入力・GPUキャンセル後の復帰 | 3 role合格、PNGとbit一致、同じAPIで復帰 | [復帰・RenderTexture](results/m3-rendertexture-recovery-and-text-gpu-regression-20261016.json) |
| 実PCMから最終768次元、全6信号・全role | 18条件合格、最小cosine 0.9999997097692651 | [音声API](results/m4-real-pcm-all-roles-staged-gpu-20261016.json) |
| 音声例外後再利用・借用stage・中間Tensor解放 | 実GPU encoder後のbackbone例外から復帰、API破棄後も借用stage再利用 | [音声所有権](results/m4-real-pcm-all-roles-staged-gpu-20261016.json) |
| テキスト回帰・API / 境界 / 非同期契約 | FP32・Float16、6文書・4queryの全順位 / score合格、関連189 pass / skip0 | [最新テキスト回帰](results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json) |
| 必須repository checks | **未達**。最新0618de7のCIはPython4環境・package・workflow lint成功、音声ONNX比較1309要素超過・Required CI失敗 | [CI 38821377276](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38821377276) |

追加確認: source `b378070`の[CI 38774804527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38774804527)では、Windows / Linux × Python3.13 / 3.14の全4環境で各1,111 pass・fail0・skip0、package / workflow lintも合格しました。実モデルの音声ONNX比較は失敗しており、Required CIの完了条件は満たしていません。4環境のJUnit XMLを取得・監査済みです。

GPU結果はWindows Editor / Unity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定sourceで得たものです。計算・非空feature入力のCPU fallbackと、最終API内の中間readback / flushは0。全結果を最新headで再実行したという主張ではありません。source、capture hash、実行条件は各receiptに保持しています。

音声最終vectorは元cosine基準で合格していますが、追加の要素別診断には1213差分が残ります。encoderの厳密比較とは別の診断で、bit一致とは表現しません。Query / Raw参照は元encoder境界を監査して公式backboneへ供給する[明示的な参照生成](results/m4-qualified-frozen-audio-role-reference-20261016.json)で、元Documentも照合しました。この参照生成自体はencoder実行の合格を証明しません。

## Unityの重さとGPU要件

同期音声APIの連続実行は、GPUキャッシュ保持によりホストメモリ13,570,727,936 bytesへ達し、12 GiB guardで停止しました。数値合格・正常終了とは数えません。

`PartitionedAudioFeatureEncoder.EncodeAsync`と`AudioEmbedder.EmbedAsync`がGPU分割境界を待ち、完了済みキャッシュを解放する実装へ変更しました。同じ上限の実PCM検証は6プロセスとも完了し、最大ホストピーク7,660,285,952 bytes（約7.13 GiB）。単発API観測1.28〜3.44秒であり、定常速度・VRAM使用量・長期安定性のbenchmarkではありません。前処理・backboneはメインスレッド同期処理のままです。

終了時のPersistent allocation警告は、Sentis2.6.1がEditModeで登録し損ねる共有64-byte bufferのcleanupへ局所化し、Editor終了 / assembly reload hookで対応しました。[固定診断](results/m4-sentis-editor-shared-gpu-cleanup-20261016.json)。以後の対象音声6プロセスと最新テキスト回帰の正常終了で警告0を確認しましたが、長期的に全allocationが解消済みとは主張しません。

ローカルの実モデルrunnerは終了済み。必要なUnity検証は所有プロセス・最大300秒・ホスト12 GiB guardに限定し、重いCPU変換と診断はGitHub Actionsを使います。

## 残る音声ONNX問題

必須CIの残件は、実音声encoderのONNX出力とPython native出力の数値差です。source `0618de7`の[最新確定CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38821377276)は、Python4環境・package・workflow lintが成功し、音声ONNX1309要素超過とRequired CIが失敗しました。先行f0593c7のLinux /Python3.13 Tanh2件失敗は保存し、最新hostのgreenだけで原因解消とは扱いません。既定export、元参照、重み、許容差と製品Runtimeは変更していません。

[後半3層の独立した185演算比較](results/m4-isolated-audio-primitives-all185-within-tolerance-20261016.json)は、各ONNX演算へ同じnative入力を渡すと全件が元許容差内でした。比較値は後続へ注入せず、元のnative区間出力は元最終参照と全bit一致しています。最大の単体差は`mm_109`の許容差比0.67で、接続したONNX区間の失敗と合わせると、小さな丸め差の蓄積が疑われます。これは単独の原因演算の確定や全encoder合格ではありません。

元のblock192 FMA /store-add順序を標準FP32演算へ直接書き出す方式を追加しました。[実行列積の全1,146,880値](results/m4-real-matmul192-all1146880-bit-exact-20261016.json)が、元最終出力を再現したnative実行から保存した演算出力と全bit一致しました。旧native行列積を通すだけのCPU forwardは、固定順序の実行証拠には数えません。実際のONNX出力を別に取得・SHA照合しています。別のMatMul2によるFMA表現は小区間で遅くなったため不採用とし、コードとexport flagを元へ戻しました。[固定比較と不採用記録](results/m4-fixed-matmul192-order-and-rejected-fma2-20261016.json)。

[元stage12→13のバイアス付き投影](results/m4-original-output-projection-first-block-bias-order-20261016.json)は、元の入力・出力・係数・biasのSHAを照合しました。native全430,080値が元出力とbit一致し、96列の独立FMA比較と実ONNX比較では、最初の192項の部分和へbiasを足してから残りを加える順序が元26,880値とbit一致しました。

[後半区間の実ONNX比較](results/m4-ordered-static-linear-tail-original-tolerance-passed-20261016.json)は、同じホストの標準ONNXが8要素超過、`ordered_static_linear`が元最終143,360値に対して許容差超過0でした。最大許容差比0.8648、bit差133,830で、bit一致とは表現しません。監査した280行の固定係数mm /addmm29か所だけを変更し、13行の相対位置投影と動的係数の積は維持しました。[TDDと範囲制約](results/m4-ordered-static-linear-segment-and-bias-tdd-20261016.json)。関連82 pass /skip0、workflow lint合格。[全12層比較](results/m4-ordered-static-linear-all12-original-tolerance-failed-20261016.json)は完了し、同じAMD EPYC7763ホストでnative区間が元最終出力と全bit一致、標準ONNX414要素超過、固定行列積候補550要素超過でした。後半3層の成功は全層へ一般化できず、この候補の全encoder採用は見送ります。元subsample出力を入力に固定した診断で、元特徴量からの全encoder合格ではありません。

[元特徴量から先頭subsampleまでの実ONNX比較](results/m4-ordered-original-input-stem-tolerance-passed-20261016.json)は、同じホストの標準ONNX505要素超過、既存refined_mean_moment正規化2か所と280行投影1か所の候補が286,720値で許容差超過0、最大許容差比0.1957でした。bit差279,359が残ります。native全encoder自体は元出力から288要素超過のため、独立監査済みの元stage参照による明示的な区間診断として記録し、native文脈合格とは数えません。[先頭経路TDD](results/m4-ordered-original-input-prefix-tdd-20261016.json)。

次の独立した候補として、[元第0・第1層の注意機構BMM6条件](results/m4-six-original-attention-bmm-actual-onnx-bit-exact-20261016.json)を実ONNXで評価し、元native全出力が固定参照と一致した文脈から保存された全760,320値でbit一致を確認しました。対象は監査済み3形状だけで、各層の4番目のBMM投影は未適用です。`ordered_static_linear_attention`は行列積のみの候補と標準ONNXも同じホストで実行し、接続した1層で効果を比較しました。RMS /SiLU /Convは同時に変更しません。既定export、元参照・重み・許容差と製品Runtimeは維持します。

[接続した第0層の比較](results/m4-first-layer-attention-bmm-connected-tolerance-passed-20261016.json)は標準・固定行列積・注意BMM追加の3通りすべてが元286,720値で許容差超過0、最大許容差比0.2108 /0.1498 /0.1352でした。元nativeはbit一致ですが、候補はbit一致ではありません。第0層の改善だけで全encoderの修正とは数えません。

[第1層の元クリッピング付きpointwise投影](results/m4-original-clipped-pointwise-bmm-all286720-bit-exact-20261016.json)は全286,720値が実ONNXで元出力とbit一致しました。元の入力・出力のclip係数、重みと期待値のSHAを照合し、同じclip順序を維持しています。これはclip後のmodule境界の一致で、clip前の生BMM全値のbit一致とは主張しません。`ordered_static_linear_all_bmm`で、元のparameterからview /expand /transposeだけで構成される4番目の投影を追加しました。[全12層比較](results/m4-all-bmm-full12-numerical-failure-and-upload-timeout-20261016.json)は、同じホストでnativeが元出力と全bit一致、標準ONNX414要素超過、候補321要素超過で不合格でした。計算は完了しましたが、モデルのupload中に20分上限でjobがcancelledとなっています。公開済みartifactから全数値captureを回収してSHAと許容差を独立照合し、job成功にも数値合格にも数えません。RMS /SiLU /Convは引き続き維持します。[TDDと範囲制約](results/m4-ordered-parameter-bmm-tdd-20261016.json)は新規red2件→green5件、関連96 pass /skip0、workflow lint合格です。

[元の単体Softmax /GLU /Depthwise比較](results/m4-original-softmax-glu-depthwise-actual-onnx-bit-exact-20261016.json)は、保存済みの元native入力を使った実ONNXで全628,736値がbit一致しました。新規Softmax9件、GLU /Depthwise /Sigmoid13件をTDDで追加し、関連127 pass /skip0です。[元のRMS /SiLU入力](results/m4-original-rms-and-silu-worst-input-onnx-qualified-20261016.json)でも平均280値と逆平方根280値はbit一致、SiLU286,720値は許容差超過0で極小のbit差4を残します。いずれも単体の証拠で、全encoderの合格や既定変換への採用とは数えません。下記の接続診断でこれらを追加しましたが、全体の数値合格には至りませんでした。

[ONNX区間接続のTDD](results/m4-connected-onnx-partitions-no-reference-reset-tdd-20261016.json)で、元subsampleから全12層を3層ずつ変換する明示的な診断モードを準備しました。各区間の実ONNX出力をそのまま次へ渡し、途中の正解Tensorへ戻しません。別のnative制御はcutの物理layoutを保ち、未分割のnativeと全bit一致することを要求します。最初のONNX出力を意図的に誤らせるテストでは最後まで誤差が伝わり、元最終参照の比較で検出されます。区間ごとのcapture /graph SHA、進行段階と経過時間を保存します。関連134 pass /skip0、workflow lint合格。実モデルの区間接続は下記の全12層診断で実行済みですが、数値は不合格です。速度改善や全encoder合格とは数えません。

[次の明示的な接続候補](results/m4-connected-original-operator-candidate-and-upload-tdd-20261016.json)は、検証済みの固定行列積 /全BMMへRMS /SiLU /Softmax /Sigmoid /Depthwiseの元順序を加える`ordered_original_primitives`です。小さな接続モデルの実ONNX比較と関連139 pass /skip0、workflow lintに合格しました。[実モデルの全12層接続](results/m4-connected-original-primitives-all12-numerical-failure-20261016.json)は完走しましたが、標準ONNX414要素超過、候補561要素超過で不合格でした。同じホストのnative全区間とnative区間接続は元参照にbit一致しています。候補の第2・5・8層cutはいずれも元許容差内でしたがbit一致ではなく、最終段で超過しています。診断収集の成功を数値合格とせず、この候補は採用しません。全captureのSHAと区間間のcarryを独立照合済みです。数値JSON /binを任意の大きなONNX graph archiveより先にuploadし、chained modeではgraph archiveを省きます。20分上限は維持し、元参照・重み・許容差・既定exportと製品GPU Runtimeは変更しません。

[元stage13から最後の正規化・投影](results/m4-original-final-rms-and-projection-all143360-bit-exact-20261016.json)は、元の入力・重みを固定した実ONNXで全143,360値がbit一致しました。単体の最終投影は再現可能ですが、全encoderの接続で入力誤差が増幅しない証拠ではありません。[非注入の候補演算診断](results/m4-noninjecting-original-candidate-operators-tdd-20261016.json)は新規8件のredから実装し、新旧18 pass、関連158 pass /skip0、workflow lint合格です。次は後半3層の同じnative入力で候補演算を独立比較し、比較結果を後続へ注入せず、形状・strideごとに最大差の入力と出力を保存します。

[後半3層の候補演算149比較](results/m4-original-candidate-tail149-local-tolerance-and-native-context-failure-20261016.json)は収集完了、115候補演算と未変更演算を同じnative入力で比較し、全件が許容差内でした。ただしIntel8573Cのnative区間自体が元最終参照から10要素超過し、元文脈の合格ではありません。正規化 /Softmax /Sigmoid /Depthwiseと注意BMM3形状はbit一致、SiLUは極小の5bit差、主な差は行列積です。[保存した6行列形状](results/m4-six-saved-matrix-contracts-local-avx2-actual-onnx-bit-exact-20261016.json)では、同じ入力の実ONNX出力がローカルAVX2 nativeに全bit一致しました。これはIntel sourceの元文脈失敗を解消する証拠ではありません。次は[監査済みの元行列1件だけのCPU経路比較](results/m4-small-native-matrix-context-diagnosis-tdd-20261016.json)で環境差を切り分けます。

未変更だった13行の位置情報投影も、保存入力で固定順序の実ONNXとローカルAVX2 nativeが一致しました。元のqualified replayと入力SHAは同じですが、元の演算出力を直接保存した証拠ではないため元文脈の合格には数えません。[明示的な`ordered_original_positions`候補](results/m4-original13row-position-candidate-tdd-and-isolated-smoke-20261016.json)を新規3件のredから実装し、関連81 pass /skip0、workflow lint合格です。既存候補は維持し、各3層区間へ3件だけこの投影を追加します。[実全12層の比較](results/m4-connected-original-positions-all12-numerical-failure-20261016.json)は完了し、AMD EPYC9V74でnative全区間 /native接続が元参照とbit一致、標準ONNX414要素超過、位置投影追加候補307要素超過でした。先行候補561要素超過から減りましたが、この候補も不合格です。全captureと隣接carryのSHAを独立照合済みです。

[小さな元行列のCPU経路比較](results/m4-native-original-matrix-amd9v45-actual-onnx-bit-exact-20261016.json)はAMD EPYC9V45 /AVX512で完了し、native既定 /MKL AVX2指定 /実portable ONNXの全1,146,880値が元のqualified参照とbit一致しました。今回のhostはAMDなので、Intel8573Cの差をAVX2指定で解消できるかは未検証です。既定設定への採用は行っていません。

[元6ケースの入力・参照・profile選択](results/m4-original-all-six-frozen-case-and-profile-selection-tdd-20261016.json)をTDDで追加しました。入力13件・stage15件・CLI3件のredから実装し、関連104 pass /skip0、workflow lint合格です。既定はsilenceのまま、明示的なallは元の5 fullケースと1 paddedケースを独立ジョブへ渡し、ケース別artifactを保存します。[元paddedグラフの保存schema](results/m4-original-padded-serialized-boundaries-audit-20261016.json)から全14境界と最終210行cropを照合しました。1.229GBのモデル全体はローカルへ読み込まず、schema約5.5MBだけを取得しています。[実際の元stage参照全6ケース](results/m4-original-all-six-stage-reference-integrity-20261016.json)も選択し、最終824,320値が元encoder参照と一致しました。これらは参照データの整合性確認で、encoder推論 /GPU /全ケースの数値合格ではありません。

元入力からの先頭区間へ向け、[元の第1・第2畳み込みの固定16座標ずつ](results/m4-static-stem-conv-original-selected2560-bit-exact-20261016.json)を抽出し、端・角・中央の全2,560値が実ONNXで元出力とbit一致しました。全畳み込みや全先頭区間の合格ではありません。[明示的な`ordered_original_stem`候補](results/m4-ordered-original-stem-and-static-conv-tdd-20261016.json)は2畳み込み・2正規化・1投影だけを対象とし、元subsample境界以外を拒否します。大きな実Tensor上のFMA列をexport前に実行せず、native forwardは元Conv2dを通すだけ、実ONNX出力を別に検証します。小さな接続モデルの実ONNXと関連166 pass /skip0で確認済み、[元入力silenceから全先頭区間の実ONNX](results/m4-original-entire-stem-all286720-actual-onnx-bit-exact-20261016.json)は全286,720値が元subsample出力とbit一致しました。native wholeは288要素超過のままで、独立監査済みstage参照による先頭区間の合格です。全encoder /GPU /全6ケースの合格へ拡張しません。既定exportと製品GPU Runtimeは維持します。

未変更だったTanhの[元第3層の保存入力](results/m4-original-tanh-all55296-actual-onnx-bit-exact-20261016.json)では、標準ONNXに9,907bit差があり、単体の許容差超過は0でした。既存native GPU shaderのIGC係数とFP32 pair順序を標準演算へ移し、実ONNX全55,296値が元のqualified出力とbit一致しました。[明示的な`ordered_original_tanh`候補](results/m4-connected-original-tanh-and-operator-counts-tdd-20261016.json)は位置投影候補へ各層のTanh1件だけを追加します。新規redから実装し、関連140 pass /skip0、workflow lint合格。失敗したSLEEF案は不採用で、期待値・許容差は維持しました。[実全12層の接続比較](results/m4-connected-original-tanh-all12-143360-bit-exact-20261016.json)は完了し、silenceの元stem出力から最終143,360値が元参照と全bit一致しました。同じhostのnative全区間・native接続も元参照にbit一致し、標準ONNXは414要素超過でした。全区間の入出力captureと隣接carryのSHA、最終metricsを独立照合しています。元特徴量からの全encoder・他5ケースの合格はまだ証明しません。

[元入力から全先頭区間・全6ケース](results/m4-original-entire-stem-all-six-1720320-actual-onnx-bit-exact-20261016.json)もsource bf3b6cfのActionsで実行し、全1,720,320値が元stage00参照とbit一致しました。各captureのSHAと元参照に対するmetricsを独立再計算しています。これは5 full /1 paddedケースの先頭区間合格で、全12層を含むencoder合格ではありません。native raw wholeの環境差も別に保持します。

[元特徴量から先頭区間→全12層の実ONNX接続](results/m4-original-raw-input-connected-chain-tdd-20261016.json)を明示的なraw_inputモードとしてTDDで準備しました。途中stage正解ファイルはdownload /loadせず、前の実ONNX出力をそのまま次へ渡します。誤った先頭出力が最後まで伝わって失敗すること、未実行 /先頭まで実行 /全区間完了を区別することを小さいモデルで確認しました。関連101 pass /skip0に加え、2Conv先頭候補とTanh body候補を実ONNXでつないだ追加1件も合格、workflow lint合格。実モデルの[raw全6ケース試行](results/m4-original-raw-all-six-attempt-four-pass-sine-two-fail-padded-timeout-20261016.json)は終了し、4件合格・sine2要素不合格・padded_noise最終未取得でした。候補は不採用です。

先行する全silence候補は475要素、RMS追加候補は583要素超過で不採用です。RMS /SiLUの後半候補も[同一ホスト比較](results/m4-fixed-tail-joint-and-same-host-baseline-failed-20261016.json)で不合格でした。同じONNX /入力 /期待値でもAMDとIntelでは結果が変わり、Intelではnative自体の元参照との差も残っています。CPU環境差、変換差、失敗・未実行を分けて保持します。過去のsource付きreceiptは`docs/results/`に残し、部分診断やActions収集の完了をGPU /全encoder合格へ読み替えません。

最新CI bf3b6cfの[失敗時official silence出力](results/m4-fresh-ci-official-vs-original-silence-drift-20261016.json)も取得し、元固定参照から145要素超過することを確認しました。標準ONNX対そのfresh officialは443要素超過です。元固定参照への候補一致だけでfresh nativeのCI合格へ置き換えず、両方の比較を維持します。小さな元行列だけの[Windows host選択](results/m4-small-native-matrix-windows-host-selection-tdd-20261016.json)をTDDで準備し、Intel/MKL経路の仮説を確認できるようにしました。2 red→関連15 pass /skip0、workflow lint合格。新しい実host結果はまだありません。既定referenceやencoder jobのCPU設定は変更していません。

[独立したraw数値監査](results/m4-independent-raw-chain-numeric-auditor-tdd-20261016.json)を追加し、元JSON /auditの固定SHA、元入力、全区間captureと隣接carry、元最終値・新旧native /candidate metrics・cosineを再計算します。途中値の取り違え、偽の差分0、ケース欠落 /重複 /profile変更 /source混在を拒否し、正直な失敗は失敗として保存します。関連27 pass /skip0。実run38818905732の先行5ケースではsilence /impulse /noise /nyquistの4件が元許容差内、sineは2要素超過で不合格でした。padded_noiseは20分上限でcandidate part3評価中にcancelされ、先頭＋第5層まで3partの実出力 /carry SHAは回収・照合できましたが、最終値がないため未達です。[sineの元先頭区間](results/m4-original-raw-sine-cut-localization-20261016.json)はbit一致、第2 /5 /8層cutは許容差内で小さいbit差が増え、最終で超過します。この候補の採用は行いません。

[Linux /Python3.13のTanh2件の失敗](results/m4-linux-python313-tanh-native-context-failure-20261016.json)は同hostのnative出力へのbit比較で、元qualified55,296値の参照差とは区別します。期待値・テスト・許容差は変更していません。[CPU数値環境の保存](results/m4-audio-cpu-context-recording-tdd-20261016.json)をTDDで追加し、今後の成功・失敗時に重みload前のCPU /Torch /許可した数値設定だけを残します。2 red＋stale context1 red→関連24 pass、最終context3 pass。既定のnative /ONNX計算や参照生成は維持します。

[Windowsの小さな元行列診断](results/m4-native-original-matrix-windows-amd7763-bit-exact-20261016.json)はAMD EPYC7763 /AVX2で完了しました。実ONNX、native既定、MKL AVX2指定の全1,146,880値が元qualified値とbit一致し、保存capture /log SHAも照合済みです。Intel hostではないためIntel経路の仮説は未解決です。sineの先頭3層へ向けた[非注入演算比較](results/m4-sine-original-tanh-position-div-noninjecting-tdd-20261016.json)を6 red→関連49 pass /skip0・workflow lint合格で追加し、同じ元入力・前後参照を選択します。接続候補と同じ位置投影 /Tanh関数に加え、未変更のscalar-buffer Divも観測します。比較値を後続へ渡さず、元native区間の参照一致を別判定します。実モデルのsine演算比較はまだ未実行です。

[sine先頭3層の非注入147比較](results/m4-sine-first3-noninjecting-bmm-one-bit-localized-20261016.json)は、native区間 /native interpreterが元stage03とbit一致した文脈で完了しました。117候補演算を含み、104 captureファイルのSHAを独立照合しました。QK行列積bmm_8の1値だけに1bit差、SiLUは極小の22bit差、その他のTanh /Div /正規化 /Softmax /Sigmoid /Convはbit一致です。[元bmm_8の出力1値](results/m4-sine-original-bmm8-exact-fma-midpoint-cause-20261016.json)を正確な有理数FMAで追跡すると、積和k=43で既存FP32エミュレーションがmidpointの低残差を失うことを確認しました。

[独立したmidpoint FMA候補](results/m4-isolated-fp32-fma-midpoint-residual-tdd-20261016.json)は旧実装の実数値red2件から追加し、30境界例と独立有理数oracleの2000正常 /相殺例、実標準FP32 ONNXで確認しました。[元bmm_8の実ONNX](results/m4-sine-original-bmm8-midpoint-all55296-actual-onnx-bit-exact-20261016.json)は全55,296値が元出力とbit一致しました。これは行列積単体の修正で、sine全encoderの合格はまだ証明しません。新しい[ordered_original_midpoint_bmm](results/m4-ordered-original-midpoint-qk-bmm-candidate-tdd-20261016.json)は元Tanh候補を維持し、各層の監査済みQK BMM1件だけをこの丸めへ変更します。3 red＋契約guard1 red→関連32 pass、追加の接続実ONNX1 pass、監査variant2 red→14 pass、workflow lint合格です。

[最新CIのCPU情報](results/m4-latest-ci0618-four-python-pass-audio1309-fail-cpu-context-20261016.json)はAMD EPYC7763 /AVX2、Torch2.14.1+cu130でした。通常のnative /ONNX計算や期待値を変えずに失敗情報を回収しています。[小さなMKL context診断](results/m4-small-native-mkl-tanh-context-tdd-20261016.json)にも固定dense32771 /random514のnative Tanhと元順序の出力を追加し、fresh default /MKL AVX2子プロセスが同じ入力SHAを使うことを要求します。3 red→関連24 pass、拡張関連37 pass。これはnative比較で、Tanhの新たな実ONNX /全encoder合格ではありません。

## 後続工程

手順5はモデル準備の再利用・配布、検索sample、連続推論のメモリ / 速度、Windows Player、他OS / 実機。手順6は依存PR、導入、ライセンス、配布の整理です。merge・公開は別途明示された範囲で進めます。

過去checkpointの長い時系列は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4cdb2330645fa3705b8d46f18fb64e66897a301f/docs/status.md)と`docs/results/`のsource付きreceiptに保持しています。過去の失敗・未実行結果を現状の合格で上書きしません。
