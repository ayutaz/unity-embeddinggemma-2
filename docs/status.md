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
| 必須repository checks | **未達**。確定済み9d40b11eのCIはPython4環境・package・workflow lint成功、音声ONNX比較とRequired CI失敗 | [CI 38798000979](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38798000979) |

追加確認: source `b378070`の[CI 38774804527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38774804527)では、Windows / Linux × Python3.13 / 3.14の全4環境で各1,111 pass・fail0・skip0、package / workflow lintも合格しました。実モデルの音声ONNX比較は失敗しており、Required CIの完了条件は満たしていません。4環境のJUnit XMLを取得・監査済みです。

GPU結果はWindows Editor / Unity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定sourceで得たものです。計算・非空feature入力のCPU fallbackと、最終API内の中間readback / flushは0。全結果を最新headで再実行したという主張ではありません。source、capture hash、実行条件は各receiptに保持しています。

音声最終vectorは元cosine基準で合格していますが、追加の要素別診断には1213差分が残ります。encoderの厳密比較とは別の診断で、bit一致とは表現しません。Query / Raw参照は元encoder境界を監査して公式backboneへ供給する[明示的な参照生成](results/m4-qualified-frozen-audio-role-reference-20261016.json)で、元Documentも照合しました。この参照生成自体はencoder実行の合格を証明しません。

## Unityの重さとGPU要件

同期音声APIの連続実行は、GPUキャッシュ保持によりホストメモリ13,570,727,936 bytesへ達し、12 GiB guardで停止しました。数値合格・正常終了とは数えません。

`PartitionedAudioFeatureEncoder.EncodeAsync`と`AudioEmbedder.EmbedAsync`がGPU分割境界を待ち、完了済みキャッシュを解放する実装へ変更しました。同じ上限の実PCM検証は6プロセスとも完了し、最大ホストピーク7,660,285,952 bytes（約7.13 GiB）。単発API観測1.28〜3.44秒であり、定常速度・VRAM使用量・長期安定性のbenchmarkではありません。前処理・backboneはメインスレッド同期処理のままです。

終了時のPersistent allocation警告は、Sentis2.6.1がEditModeで登録し損ねる共有64-byte bufferのcleanupへ局所化し、Editor終了 / assembly reload hookで対応しました。[固定診断](results/m4-sentis-editor-shared-gpu-cleanup-20261016.json)。以後の対象音声6プロセスと最新テキスト回帰の正常終了で警告0を確認しましたが、長期的に全allocationが解消済みとは主張しません。

ローカルの実モデルrunnerは終了済み。必要なUnity検証は所有プロセス・最大300秒・ホスト12 GiB guardに限定し、重いCPU変換と診断はGitHub Actionsを使います。

## 残る音声ONNX問題

必須CIの残件は、実音声encoderのONNX出力とPython native出力の数値差です。source `9d40b11e`の[最新確定CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38798000979)では、Windows / Linux × Python3.13 /3.14、package、workflow lintは成功しましたが、音声ONNXとRequired CIは失敗しました。既定export、元参照、重み、許容差と製品Runtimeは変更していません。

[後半3層の独立した185演算比較](results/m4-isolated-audio-primitives-all185-within-tolerance-20261016.json)は、各ONNX演算へ同じnative入力を渡すと全件が元許容差内でした。比較値は後続へ注入せず、元のnative区間出力は元最終参照と全bit一致しています。最大の単体差は`mm_109`の許容差比0.67で、接続したONNX区間の失敗と合わせると、小さな丸め差の蓄積が疑われます。これは単独の原因演算の確定や全encoder合格ではありません。

元のblock192 FMA /store-add順序を標準FP32演算へ直接書き出す方式を追加しました。[実行列積の全1,146,880値](results/m4-real-matmul192-all1146880-bit-exact-20261016.json)が、元最終出力を再現したnative実行から保存した演算出力と全bit一致しました。旧native行列積を通すだけのCPU forwardは、固定順序の実行証拠には数えません。実際のONNX出力を別に取得・SHA照合しています。別のMatMul2によるFMA表現は小区間で遅くなったため不採用とし、コードとexport flagを元へ戻しました。[固定比較と不採用記録](results/m4-fixed-matmul192-order-and-rejected-fma2-20261016.json)。

[元stage12→13のバイアス付き投影](results/m4-original-output-projection-first-block-bias-order-20261016.json)は、元の入力・出力・係数・biasのSHAを照合しました。native全430,080値が元出力とbit一致し、96列の独立FMA比較と実ONNX比較では、最初の192項の部分和へbiasを足してから残りを加える順序が元26,880値とbit一致しました。

[後半区間の実ONNX比較](results/m4-ordered-static-linear-tail-original-tolerance-passed-20261016.json)は、同じホストの標準ONNXが8要素超過、`ordered_static_linear`が元最終143,360値に対して許容差超過0でした。最大許容差比0.8648、bit差133,830で、bit一致とは表現しません。監査した280行の固定係数mm /addmm29か所だけを変更し、13行の相対位置投影と動的係数の積は維持しました。[TDDと範囲制約](results/m4-ordered-static-linear-segment-and-bias-tdd-20261016.json)。関連82 pass /skip0、workflow lint合格。[全12層比較](results/m4-ordered-static-linear-all12-original-tolerance-failed-20261016.json)は完了し、同じAMD EPYC7763ホストでnative区間が元最終出力と全bit一致、標準ONNX414要素超過、固定行列積候補550要素超過でした。後半3層の成功は全層へ一般化できず、この候補の全encoder採用は見送ります。元subsample出力を入力に固定した診断で、元特徴量からの全encoder合格ではありません。

[元特徴量から先頭subsampleまでの実ONNX比較](results/m4-ordered-original-input-stem-tolerance-passed-20261016.json)は、同じホストの標準ONNX505要素超過、既存refined_mean_moment正規化2か所と280行投影1か所の候補が286,720値で許容差超過0、最大許容差比0.1957でした。bit差279,359が残ります。native全encoder自体は元出力から288要素超過のため、独立監査済みの元stage参照による明示的な区間診断として記録し、native文脈合格とは数えません。[先頭経路TDD](results/m4-ordered-original-input-prefix-tdd-20261016.json)。

次の独立した候補として、[元第0・第1層の注意機構BMM6条件](results/m4-six-original-attention-bmm-actual-onnx-bit-exact-20261016.json)を実ONNXで評価し、元native全出力が固定参照と一致した文脈から保存された全760,320値でbit一致を確認しました。対象は監査済み3形状だけで、各層の4番目のBMM投影は未適用です。`ordered_static_linear_attention`は行列積のみの候補と標準ONNXも同じホストで実行し、接続した1層で効果を比較しました。RMS /SiLU /Convは同時に変更しません。既定export、元参照・重み・許容差と製品Runtimeは維持します。

[接続した第0層の比較](results/m4-first-layer-attention-bmm-connected-tolerance-passed-20261016.json)は標準・固定行列積・注意BMM追加の3通りすべてが元286,720値で許容差超過0、最大許容差比0.2108 /0.1498 /0.1352でした。元nativeはbit一致ですが、候補はbit一致ではありません。第0層の改善だけで全encoderの修正とは数えません。

[第1層の元クリッピング付きpointwise投影](results/m4-original-clipped-pointwise-bmm-all286720-bit-exact-20261016.json)は全286,720値が実ONNXで元出力とbit一致しました。元の入力・出力のclip係数、重みと期待値のSHAを照合し、同じclip順序を維持しています。これはclip後のmodule境界の一致で、clip前の生BMM全値のbit一致とは主張しません。`ordered_static_linear_all_bmm`で、元のparameterからview /expand /transposeだけで構成される4番目の投影を追加し、元subsampleから全12層への接続を比較します。RMS /SiLU /Convは引き続き維持します。[TDDと範囲制約](results/m4-ordered-parameter-bmm-tdd-20261016.json)は新規red2件→green5件、関連96 pass /skip0、workflow lint合格です。

先行する全silence候補は475要素、RMS追加候補は583要素超過で不採用です。RMS /SiLUの後半候補も[同一ホスト比較](results/m4-fixed-tail-joint-and-same-host-baseline-failed-20261016.json)で不合格でした。同じONNX /入力 /期待値でもAMDとIntelでは結果が変わり、Intelではnative自体の元参照との差も残っています。CPU環境差、変換差、失敗・未実行を分けて保持します。過去のsource付きreceiptは`docs/results/`に残し、部分診断やActions収集の完了をGPU /全encoder合格へ読み替えません。

## 後続工程

手順5はモデル準備の再利用・配布、検索sample、連続推論のメモリ / 速度、Windows Player、他OS / 実機。手順6は依存PR、導入、ライセンス、配布の整理です。merge・公開は別途明示された範囲で進めます。

過去checkpointの長い時系列は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4cdb2330645fa3705b8d46f18fb64e66897a301f/docs/status.md)と`docs/results/`のsource付きreceiptに保持しています。過去の失敗・未実行結果を現状の合格で上書きしません。
