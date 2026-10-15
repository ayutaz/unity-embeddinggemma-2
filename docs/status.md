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
| 必須repository checks | **未達**。確定済み4cdb233のCIはPython4環境・package・workflow lint成功、音声ONNX比較とRequired CI失敗 | [CI 38771922063](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38771922063) |

追加確認: source `b378070`の[CI 38774804527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38774804527)では、Windows / Linux × Python3.13 / 3.14の全4環境で各1,111 pass・fail0・skip0、package / workflow lintも合格しました。実モデルの音声ONNX比較は失敗しており、Required CIの完了条件は満たしていません。4環境のJUnit XMLを取得・監査済みです。

GPU結果はWindows Editor / Unity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定sourceで得たものです。計算・非空feature入力のCPU fallbackと、最終API内の中間readback / flushは0。全結果を最新headで再実行したという主張ではありません。source、capture hash、実行条件は各receiptに保持しています。

音声最終vectorは元cosine基準で合格していますが、追加の要素別診断には1213差分が残ります。encoderの厳密比較とは別の診断で、bit一致とは表現しません。Query / Raw参照は元encoder境界を監査して公式backboneへ供給する[明示的な参照生成](results/m4-qualified-frozen-audio-role-reference-20261016.json)で、元Documentも照合しました。この参照生成自体はencoder実行の合格を証明しません。

## Unityの重さとGPU要件

同期音声APIの連続実行は、GPUキャッシュ保持によりホストメモリ13,570,727,936 bytesへ達し、12 GiB guardで停止しました。数値合格・正常終了とは数えません。

`PartitionedAudioFeatureEncoder.EncodeAsync`と`AudioEmbedder.EmbedAsync`がGPU分割境界を待ち、完了済みキャッシュを解放する実装へ変更しました。同じ上限の実PCM検証は6プロセスとも完了し、最大ホストピーク7,660,285,952 bytes（約7.13 GiB）。単発API観測1.28〜3.44秒であり、定常速度・VRAM使用量・長期安定性のbenchmarkではありません。前処理・backboneはメインスレッド同期処理のままです。

終了時のPersistent allocation警告は、Sentis2.6.1がEditModeで登録し損ねる共有64-byte bufferのcleanupへ局所化し、Editor終了 / assembly reload hookで対応しました。[固定診断](results/m4-sentis-editor-shared-gpu-cleanup-20261016.json)。以後の対象音声6プロセスと最新テキスト回帰の正常終了で警告0を確認しましたが、長期的に全allocationが解消済みとは主張しません。

ローカルの実モデルrunnerは終了済み。必要なUnity検証は所有プロセス・最大300秒・ホスト12 GiB guardに限定し、重いCPU変換と診断はGitHub Actionsを使います。

## 残る音声ONNX問題

固定6 MBのprefix診断では、元係数を維持したrefined Welford候補が合格しました。[完了したprefix実験](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38771944210)。ただし同候補の[全音声ONNX実験](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38771944636)は6信号すべて不合格です。部分成功を全encoderの成功へ拡張せず、必須CIの既定方式へ採用していません。

現在のPython native出力も元の固定参照と異なるため、元保存graphの2か所のLayerNormだけに介入する診断を追加しました。[新しい固定証拠](results/m4-complete-prefix-rejected-onnx-and-reproduced-native-20261016.json)では、AMD EPYC 9V74 / AVX512で元silenceの最終143,360値を全bit再現しました。native controlとInterpreterもbit一致。scalar mean / moment両方のFMA候補は同じ最終bitを再現し、他2候補は許容差内ですが464 bit差を残しました。これらは1信号のCPU診断で、全6信号・ONNX・GPUの新たな合格ではありません。元参照・係数・許容差・製品GPU経路は変更していません。

次はこの診断結果で最初の差を局所化し、原因を再現する小さいテストから修正し、全6信号と必須CIを再確認します。必須CIが失敗している間、手順4全体は未完了です。

追加の固定参照比較: [最初の投影](results/m4-original-context-subsample-onnx-three-variants-20261016.json)は標準ONNX563要素超過に対し、明示LayerNormの2候補は超過0。[第1層終端](results/m4-fixed-original-layer0-three-onnx-variants-20261016.json)も元286,720値に対して標準284要素超過、2候補とも超過0でした。現在ホストのnative全encoderは元参照と251要素が異なり、その失敗を保持したまま、監査済みの元stage出力を明示的な比較対象としています。これはsilenceの部分区間の結果で、全encoder・全6信号の合格ではありません。

保存グラフのmask型変換とSoftmaxの内部演算名を、同じCPU FP32計算の公開APIで表す診断用bridgeをTDDで追加しました。元グラフ・製品GPU処理は変更していません。[最終encoder単一候補](results/m4-saved-full-silence-onnx-candidate-failed-20261016.json)は元最終143,360値に対して475要素が許容差を超過して不合格でした。[第6・第9・第12層終端](results/m4-late-stage-localization-and-rms-candidate-tdd-20261016.json)では、前2区間は超過0、最終層終端は1827超過でした。ただし別CPUホスト間の比較なので、最初に差が生じる演算を確定したとは扱いません。

追加した[明示RMS候補](results/m4-saved-full-silence-ordered-rms-onnx-failed-20261016.json)も109か所すべてを置き換えて583要素超過となり、不採用です。元参照・重み・許容差は維持しています。これらのActions成功表示は診断収集の完了であり、数値合格ではありません。

[固定した小区間の比較](results/m4-fixed-rms-and-final-projection-components-20261016.json)では、第1層末尾RMSは8-lane加算・sqrt後の逆数で元286,720値とbit一致、最終投影も元stage13を入力にすると元143,360値とbit一致しました。NumPyで最終投影を計算した場合も許容差超過0です。全体の誤差を切り分けるため、元stageを入力にして後続の保存graphだけを実行する明示的な診断経路を追加しています。この経路は前半を実行しないため、全encoder・GPU・最終APIの合格証拠には数えません。

source `9741d1e`の[最新確定CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38785455573)も、Python4環境・package・workflow lintは成功、model-referenceとRequired CIは失敗です。製品Runtimeは実GPU検証source `0857edc8`から変更されていません。

[元の第9層終端を入力にした後半3層の比較](results/m4-original-layer8-to-final-onnx-14-failures-20261016.json)は、同じ入力に対するnative計算が元最終出力と全bit一致、標準ONNXは14要素超過でした。後半だけでも変換後の差が発生する証拠であり、最初の不一致演算を確定した結果ではありません。[固定したSiLU小区間](results/m4-fixed-original-silu-fp32-candidate-20261016.json)では、標準NumPy式は許容差内ながら375,259値にbit差があり、既存GPU shaderと同じ順序を標準FP32演算で表す候補は元1,146,880値と全bit一致しました。この候補だけを後半区間へ明示適用して検証します。診断時のメモリ保持を減らすため、ONNX ReferenceEvaluatorの同じkernel・同じ実行順序を維持し、最終利用後に中間値を解放する実行をTDDで追加しました。既定の全体参照ゲートは変更していません。

## 後続工程

手順5はモデル準備の再利用・配布、検索sample、連続推論のメモリ / 速度、Windows Player、他OS / 実機。手順6は依存PR、導入、ライセンス、配布の整理です。merge・公開は別途明示された範囲で進めます。

過去checkpointの長い時系列は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4cdb2330645fa3705b8d46f18fb64e66897a301f/docs/status.md)と`docs/results/`のsource付きreceiptに保持しています。過去の失敗・未実行結果を現状の合格で上書きしません。
