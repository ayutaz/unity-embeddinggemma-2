# 現在の状況

更新: 2026-10-16。最新の結果整理は補助ブランチ`diag/audio-sine-primitive-20261016`に保存し、既知の失敗CIを再実行するpushは保留しています。PR headはe99c442です。対象はDraft [PR #81](https://github.com/ayutaz/unity-embeddinggemma-2/pull/81)の`feat/multimodal-numerical-acceptance`。main統合・リリースは未実施です。ユーザー指定「4まで」の完了条件は[元計画の監査](results/goal4-scope-audit-20261016.json)を維持します。

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
| 必須repository checks | **未達**。最新e99c442のCIはPython3環境・package・workflow lint成功、Linux/Python3.14のTanh2件・音声ONNX比較・Required CI失敗 | [CI 38825766738](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38825766738) |

追加確認: source `b378070`の[CI 38774804527](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38774804527)では、Windows / Linux × Python3.13 / 3.14の全4環境で各1,111 pass・fail0・skip0、package / workflow lintも合格しました。実モデルの音声ONNX比較は失敗しており、Required CIの完了条件は満たしていません。4環境のJUnit XMLを取得・監査済みです。

GPU結果はWindows Editor / Unity6000.3.16f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12の固定sourceで得たものです。計算・非空feature入力のCPU fallbackと、最終API内の中間readback / flushは0。全結果を最新headで再実行したという主張ではありません。source、capture hash、実行条件は各receiptに保持しています。

音声最終vectorは元cosine基準で合格していますが、追加の要素別診断には1213差分が残ります。encoderの厳密比較とは別の診断で、bit一致とは表現しません。Query / Raw参照は元encoder境界を監査して公式backboneへ供給する[明示的な参照生成](results/m4-qualified-frozen-audio-role-reference-20261016.json)で、元Documentも照合しました。この参照生成自体はencoder実行の合格を証明しません。

## Unityの重さとGPU要件

同期音声APIの連続実行は、GPUキャッシュ保持によりホストメモリ13,570,727,936 bytesへ達し、12 GiB guardで停止しました。数値合格・正常終了とは数えません。

`PartitionedAudioFeatureEncoder.EncodeAsync`と`AudioEmbedder.EmbedAsync`がGPU分割境界を待ち、完了済みキャッシュを解放する実装へ変更しました。同じ上限の実PCM検証は6プロセスとも完了し、最大ホストピーク7,660,285,952 bytes（約7.13 GiB）。単発API観測1.28〜3.44秒であり、定常速度・VRAM使用量・長期安定性のbenchmarkではありません。前処理・backboneはメインスレッド同期処理のままです。

終了時のPersistent allocation警告は、Sentis2.6.1がEditModeで登録し損ねる共有64-byte bufferのcleanupへ局所化し、Editor終了 / assembly reload hookで対応しました。[固定診断](results/m4-sentis-editor-shared-gpu-cleanup-20261016.json)。以後の対象音声6プロセスと最新テキスト回帰の正常終了で警告0を確認しましたが、長期的に全allocationが解消済みとは主張しません。

ローカルの実モデルrunnerは終了済み。必要なUnity検証は所有プロセス・最大300秒・ホスト12 GiB guardに限定し、重いCPU変換と診断はGitHub Actionsを使います。

## 残る音声ONNX問題

手順4は未完了です。[最新CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38825766738)は終了し、音声ONNX対fresh officialでsilenceの1,309値が元許容差を超過しました。最大許容差比24.0435です。Linux/Python3.14のnative Tanh bit比較2件も失敗し、固定random514入力の実ONNX対nativeは8値が1 ULP異なります。他のPython3環境、package、workflow lintは成功、Required CIは失敗です。[固定した失敗情報](results/m4-ci-e99-audio1309-and-linux314-tanh-two-failures-20261016.json)。PR headはe99c442、CIのPR merge実行sourceは9e18b1cで、両者を区別します。元参照、重み、許容差、既定export、製品GPU Runtimeは維持しています。

[Tanhテストの対象整理](results/m4-tanh-pinned-captures-and-native-rounding-contract-tdd-20261016.json)では、PyTorchのoneMKL高精度モード呼出しとIntelのCPU依存丸め仕様を確認しました。毎回のCPU出力への無条件bit一致を、保存済み元contextの33,285入力・出力への厳密bit比較と、現在のnativeへの元encoder許容差比較へ分けました。実ONNXの固定参照bit比較も514値から33,285値へ拡張し、参照から1 ULP変わればnativeと一致しても失敗します。過去に保存・監査済みの重みなしfixture266,280bytesを使用し、候補から期待値を生成していません。記録済みCI5値のred1件とfixture不足red4件から、関連29 pass /skip0になりました。これは単体テスト仕様の修正で、実モデルの計算や既定CIの音声比較を変えたものではありません。新しい実モデルCIは実行していません。

実CIのCPUはAMD EPYC7763 /AVX2 /Torch2.14.1+cu130、NumPy2.5.3のBLASはscipy-openblas0.3.34.106.0 /Haswellでした。これは保存した実行条件で、数値差の原因確定ではありません。

元のQK行列積にあった1bit差は、有理数FMA追跡でmidpointの低残差欠落へ局所化しました。[元bmm_8の実ONNX](results/m4-sine-original-bmm8-midpoint-all55296-actual-onnx-bit-exact-20261016.json)は修正候補で全55,296値がbit一致しました。明示的な`ordered_original_midpoint_bmm`は各層の監査済みQK BMM1件だけを修正し、他のBMMと元の係数・mask・clipを維持します。[先行するraw sine全12層](results/m4-original-raw-sine-full12-midpoint-qk-tolerance-passed-20261016.json)は143,360値で元許容差超過0、最大許容差比0.482376、bit差34,679でした。bit一致とは表現しません。

[現在のraw全6ケース検証](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38825780838)はsource e99c442で終了しました。5件が数値合格、noiseは20分上限でcancelledです。元特徴量→2Conv先頭区間→全12層の実ONNX出力を次へ渡し、途中stage正解ファイルをdownload /loadしません。silence、impulse、sine、nyquist、padded_noiseの計680,960値は独立監査済みで許容差超過0です。[全6件試行の終了記録](results/m4-raw-midpoint-all-six-five-pass-noise-timeout-20261016.json)。noiseは先頭＋第5層まで3partの実出力とcarry SHAを照合しましたが、第6〜8層の評価中に停止し、最終値はありません。全6件合格とは数えません。前回padded_noiseの20分timeoutは失敗のまま保存し、今回は実測に基づく35分上限で最後まで取得できました。

Windows2022の[小さな行列/Tanh診断](results/m4-native-windows2022-amd9v74-avx512-matrix-tanh-bit-exact-20261016.json)はAMD9V74 /ATenAVX512で全bit一致でしたが、後続の[無変更native全6ケース](results/m4-native-full-six-windows2022-amd7763-original-context-failed-20261016.json)は同じrunner指定でもAMD7763 /AVX2となり、全6件で元許容差を超過しました。全824,320値、capture SHAとmetricsを独立照合しています。収集jobのsuccessは数値合格ではありません。runner名だけでCPU命令セットを保証できず、ランダムなhost再試行は行いません。新しい診断の関連26件と追加19件のテスト、workflow lintは合格しましたが、fresh official /ONNX /GPU合格への置き換えには使いません。

元runの結果回収と取得済み出力の独立照合は完了しました。残件はnoiseの最終値取得と、必須CIの音声ONNX /native Tanh比較です。[noise再開と監査のTDD](results/m4-pinned-noise-prediction-resume-and-independent-auditor-tdd-20261016.json)を追加し、保存済みの実出力から残り2区間だけを評価します。先頭3区間は元係数からグラフを再生成して旧SHAへ一致させ、推論は繰り返しません。元の計算コード・依存lock計61ファイルをe99c442へ固定し、異なる新しいjob sourceを明示します。関連39 pass /skip0、追加集計8件のred→green、最終再開＋集計23 pass /skip0、workflow lint合格です。実保存記録のcapture /carry /layoutと残区間の演算数もモデルをloadせず照合しました。

[再開run38829612885](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38829612885)、source c994563はrunner待ちです。元6信号・2 profile・824,320値の基準を維持し、途中stage正解を使いません。固定参照に一致する候補と現在のnativeへの一致も別条件として維持し、既定の実モデルCIの不合格は未解決です。Unityは停止中です。

過去の候補、失敗とTDD経緯は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/e99c44214ab7d7f526b0fede5b561862460f1d66/docs/status.md)とsource付き`docs/results/`に保持しています。

## 後続工程

手順5はモデル準備の再利用・配布、検索sample、連続推論のメモリ / 速度、Windows Player、他OS / 実機。手順6は依存PR、導入、ライセンス、配布の整理です。merge・公開は別途明示された範囲で進めます。

過去checkpointの長い時系列は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4cdb2330645fa3705b8d46f18fb64e66897a301f/docs/status.md)と`docs/results/`のsource付きreceiptに保持しています。過去の失敗・未実行結果を現状の合格で上書きしません。
