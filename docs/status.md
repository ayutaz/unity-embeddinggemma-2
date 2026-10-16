# 現在の状況

更新: 2026-10-17。[PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31)はmain `58a6aa2`へマージ済みで、[main CI38921023089](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38921023089)も成功しました。今回の「3まで進める」の画像・音声検索サンプルは実装・実GPU・PlayMode表示・505件の全体回帰を検証済みです。[delivery PR #83](https://github.com/ayutaz/unity-embeddinggemma-2/pull/83)で最終CIと統合状態を確認できます。[現行サンプルの実装・固定検証結果](multimodal-search-sample.md)を参照してください。以下は元計画の画像・音声API検証と、その時点の履歴です。

## 元計画のGPU / API検証記録

更新: 2026-10-16。元計画の手順3・4は、固定sourceのGPU/API検証と[必須CI38850001042](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38850001042)の合格で完了しました。[要件別の完了監査](results/goal4-current-completion-audit-20261016.json)。PR #81は依存ブランチへマージ済みで、[PR #79](https://github.com/ayutaz/unity-embeddinggemma-2/pull/79)でmain `d925824`へ統合済みです。統合候補 `2ec11ab` の[CI38883782797](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38883782797)は必須15job・Python4環境各1,433件が合格し、mainのtreeとの一致も確認しました。

固定Python参照を必須基準にするユーザー指定の方針に従い、同一実行source `1058dbe57b0376589e4fe24910cff4c9de9269f1`で元音声6ケース・2 profile・全12層の実ONNXを評価しました。824,320値で元許容差超過0、bit差162,697です。独立したcapture/carry監査をローカルでも再計算し、CIの監査JSONと一致しました。Windows / UbuntuのPython3.13 / 3.14は各1,433 pass / skip0、必須15jobが合格しています。fresh CPU対ONNXの6件・5,390値の不合格は診断として `success=false` のまま保持し、GPU合格には数えません。

[PR整理の監査](results/main-pr-integration-audit-20261016.json)に、履歴への包含、同一ファイルと変更済みファイル、旧診断の採否、その時点で残したPlayer PR #31を記録しました。重複・旧診断20件をクローズ済みです。PR #31はその後mainへ追従し、新しいPlayer検証とCIを経てマージ済みです。[固定sourceのPlayer受け入れ](results/m2-current-main-player-acceptance-20261017.json)。統合作業では元参照・学習済み係数・許容差を変更していません。

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

## 過去の音声ONNX失敗と解消までの記録

以下は各実行時点の履歴です。現行の合否は冒頭と完了監査を参照してください。

手順4は未完了です。[最新CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38825766738)は終了し、音声ONNX対fresh officialでsilenceの1,309値が元許容差を超過しました。最大許容差比24.0435です。Linux/Python3.14のnative Tanh bit比較2件も失敗し、固定random514入力の実ONNX対nativeは8値が1 ULP異なります。他のPython3環境、package、workflow lintは成功、Required CIは失敗です。[固定した失敗情報](results/m4-ci-e99-audio1309-and-linux314-tanh-two-failures-20261016.json)。PR headはe99c442、CIのPR merge実行sourceは9e18b1cで、両者を区別します。元参照、重み、許容差、既定export、製品GPU Runtimeは維持しています。

[Tanhテストの対象整理](results/m4-tanh-pinned-captures-and-native-rounding-contract-tdd-20261016.json)では、PyTorchのoneMKL高精度モード呼出しとIntelのCPU依存丸め仕様を確認しました。毎回のCPU出力への無条件bit一致を、保存済み元contextの33,285入力・出力への厳密bit比較と、現在のnativeへの元encoder許容差比較へ分けました。実ONNXの固定参照bit比較も514値から33,285値へ拡張し、参照から1 ULP変わればnativeと一致しても失敗します。過去に保存・監査済みの重みなしfixture266,280bytesを使用し、候補から期待値を生成していません。記録済みCI5値のred1件とfixture不足red4件から、関連29 pass /skip0になりました。これは単体テスト仕様の修正で、実モデルの計算や既定CIの音声比較を変えたものではありません。新しい実モデルCIは実行していません。

[同一入力の参照区間監査](results/m4-same-input-official-original-disjoint74-acceptance-intervals-20261016.json)では、失敗CIのofficial出力・ONNX出力・入力だけ約1.72MBをrange取得し、入力SHAが元参照と同一、ONNX対officialの1,309値失敗を独立再計算しました。official自体は元参照から288値が許容差超過し、74値は両参照の許容区間が重なりません。`abs(a-b) > 2*atol + rtol*(abs(a)+abs(b))`を各値の正確な有理数でも確認しました。同じ出力へ両方の基準を要求することは、この記録済みcontextでは不可能です。固定参照を必須基準としてCPU差を診断に残すか、現在CPUの基準も別変換経路／固定contextで必須維持するかは方針確認が必要です。CI gate・参照値・許容差は変更していません。

実CIのCPUはAMD EPYC7763 /AVX2 /Torch2.14.1+cu130、NumPy2.5.3のBLASはscipy-openblas0.3.34.106.0 /Haswellでした。これは保存した実行条件で、数値差の原因確定ではありません。

元のQK行列積にあった1bit差は、有理数FMA追跡でmidpointの低残差欠落へ局所化しました。[元bmm_8の実ONNX](results/m4-sine-original-bmm8-midpoint-all55296-actual-onnx-bit-exact-20261016.json)は修正候補で全55,296値がbit一致しました。明示的な`ordered_original_midpoint_bmm`は各層の監査済みQK BMM1件だけを修正し、他のBMMと元の係数・mask・clipを維持します。[先行するraw sine全12層](results/m4-original-raw-sine-full12-midpoint-qk-tolerance-passed-20261016.json)は143,360値で元許容差超過0、最大許容差比0.482376、bit差34,679でした。bit一致とは表現しません。

[先行するraw全6ケース試行](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38825780838)はsource e99c442で終了しました。5件が数値合格、noiseは20分上限でcancelledです。元特徴量→2Conv先頭区間→全12層の実ONNX出力を次へ渡し、途中stage正解ファイルをdownload /loadしません。silence、impulse、sine、nyquist、padded_noiseの計680,960値は独立監査済みで許容差超過0です。[全6件試行の終了記録](results/m4-raw-midpoint-all-six-five-pass-noise-timeout-20261016.json)。noiseは先頭＋第5層まで3partの実出力とcarry SHAを照合しましたが、第6〜8層の評価中に停止し、最終値はありません。全6件合格とは数えません。前回padded_noiseの20分timeoutは失敗のまま保存し、今回は実測に基づく35分上限で最後まで取得できました。

Windows2022の[小さな行列/Tanh診断](results/m4-native-windows2022-amd9v74-avx512-matrix-tanh-bit-exact-20261016.json)はAMD9V74 /ATenAVX512で全bit一致でしたが、後続の[無変更native全6ケース](results/m4-native-full-six-windows2022-amd7763-original-context-failed-20261016.json)は同じrunner指定でもAMD7763 /AVX2となり、全6件で元許容差を超過しました。全824,320値、capture SHAとmetricsを独立照合しています。収集jobのsuccessは数値合格ではありません。runner名だけでCPU命令セットを保証できず、ランダムなhost再試行は行いません。新しい診断の関連26件と追加19件のテスト、workflow lintは合格しましたが、fresh official /ONNX /GPU合格への置き換えには使いません。

元runの結果回収と取得済み出力の独立照合を終え、noiseの最終値を取得するために以下の再開処理を追加しました。[noise再開と監査のTDD](results/m4-pinned-noise-prediction-resume-and-independent-auditor-tdd-20261016.json)を追加し、保存済みの実出力から残り2区間だけを評価します。先頭3区間は元係数からグラフを再生成して旧SHAへ一致させ、推論は繰り返しません。元の計算コード・依存lock計61ファイルをe99c442へ固定し、異なる新しいjob sourceを明示します。関連39 pass /skip0、追加集計8件のred→green、最終再開＋集計23 pass /skip0、workflow lint合格です。実保存記録のcapture /carry /layoutと残区間の演算数もモデルをloadせず照合しました。

[再開run38829612885](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38829612885)、source c994563は正常終了し、先頭3グラフの再生成SHAが元記録と一致しました。実際の前回予測から残り2区間を実行したnoiseは143,360値で元許容差超過0、最大許容差比0.251034、bit差45,930でした。全capture /carry SHAと最終metricsを独立再計算しています。[全6ケースの固定結果](results/m4-original-raw-all-six-midpoint-candidate-resumed-noise-tolerance-passed-20261016.json)は計824,320値、元許容差超過0、bit差162,697です。全6件を同じ新job sourceで実行したという意味ではなく、e99c442の元5件・noise先頭3区間とc994563の続行を、同一の元計算コード61ファイルとグラフSHAで監査しています。途中stage正解は使っていません。

この段落までの候補・続行結果は当時の記録です。その後、冒頭に記した単一sourceの全6件と必須CIが合格しました。fresh CPUの不合格は保持し、新たなGPU実行やPlayer・他deviceの合格とは扱いません。

過去の候補、失敗とTDD経緯は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/e99c44214ab7d7f526b0fede5b561862460f1d66/docs/status.md)とsource付き`docs/results/`に保持しています。

## 後続工程

手順5はモデル準備の再利用・配布、検索sample、連続推論のメモリ / 速度、Windows Player、他OS / 実機。mainと依存PRの整理を先行し、手順6の導入、ライセンス、配布の整理は引き続き残します。merge・公開は別途明示された範囲で進めます。

過去checkpointの長い時系列は[更新前の固定文書](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4cdb2330645fa3705b8d46f18fb64e66897a301f/docs/status.md)と`docs/results/`のsource付きreceiptに保持しています。過去の失敗・未実行結果を現状の合格で上書きしません。
