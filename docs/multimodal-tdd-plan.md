# M2・M3・M4から正式リリースまでのTDD計画

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。冒頭の現在方針が、下の旧checkpointの着手順より優先する。

## 2026-10-14の実行方針と再開順

2026-10-15: 実行計画の3（数値修正）・4（最終API）までが新しく許可された。R2の音声数値差を進行中で、R2 / R3の完了は未証明。小さいGPU演算probeの4 passを全encoderへ流用しない。実Welford8 ONNXは旧golden全6件で不合格。最新native参照の再計算にも環境差があるため、新しいnative値を期待値に採用せず、保存した実ONNX出力を旧goldenへ直接比較する。[現在の判定](status.md)が以下の前回checkpointより優先する。画像FP32全profileと実APIの全role・境界・資源寿命・例外後再利用・text回帰の範囲は維持する。

ユーザー指定の実行計画1・2までを実施した。最新Unity実測は `b796e504367fb9165a4992ff036a205fb2e7d429`、mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。Draft PR #81のGPU専用変更を含むcompile、tiny / 実square 1件のGPU計測は合格。SiLU修正後のActions実ONNXは元6信号・2 profile全件実行、全6件不合格。[今回の結果](gpu-execution-and-silu-validation.md) / [最新状態](status.md)。R2の数値修正以降は未着手。

| 順序 | 着手条件と作業 | 渡す証拠 / 完了条件 |
| --- | --- | --- |
| R0 GPUと負荷 | 固定Git SHA / Unity / Sentisを監査し、compileと小さいGPU probe。CPUへ代替しない | requested / actual backend、実GPU / API、fallback / flush 0。load / transform / Worker / upload / schedule / final readback / disposeの時間とmemory |
| R1 実装済み修正 | 標準ONNX SiLU除算順序修正をActionsの元6信号・2 profileで照合 | 保存graph hash・独立評価・固定許容差。小さい37+4 passは実モデル成功の代替にしない |
| R2 数値差 | 一つのGPU入力で参照文脈と最初の差を確認し、独立した小さいredから最小修正 | 元golden・入力 / model hash・`atol=1e-5` / `rtol=1e-4`不変。局所合格後に画像全profile・音声全信号のGPU gate |
| R3 最終API | R2のencoder gate後、実Texture / PCM→encoder→backbone→768次元 | Query / Document / Raw、全profile・mask・境界、所有権 / Dispose・例外後再利用、GPU中間readbackなし、text回帰 |
| R4 全体受け入れ | M2〜M4の残りsample・混在検索・反復 / 性能・Player・必須実機 | Windows / macOS / iOS / Androidのsource対応結果、最終candidateの空Git consumer、正常終了 |
| R5 配布準備 | 全必須gateと依存差分を確認 | version / CHANGELOG / license / 配布監査。公開tag / Releaseは準備後の別工程 |

ローカル実モデルは1試行・1失敗経路に絞り、新しいbounded_processで5分の外部上限を適用する。上限超過時は所有processだけを止めてPIDと部分結果を保存し、未完了と記録する。旧ignored launcherへのtimeoutは実装していない。重い全条件・モデル取得 / exportはActions優先。GPU runner / ライセンス不在は未実行としCPU結果で補わない。

今回の計測では新しい `embeddinggemma_tools.bounded_process` の外部300秒上限を実装・使用した。timeoutで所有processだけを停止する3件の小さいテストと、実Unityの正常終了を確認済み。以前のignored launcherの上限は依然未実装。R0はWindows単発測定の範囲で完了、R1は検証実行完了・数値不合格としてR2へ渡す。

CPUを含む下記の以前の検証matrixは互換性の履歴・契約として保持する。今後の主実行経路はGPUで、従来のCPU/GPU全条件の完了flagをGPU専用結果へ流用しない。RuntimeのCPU対応を削除したとは主張しない。

更新日: 2026-10-14。**正式リリースはM2〜M4の全対応と全必須検証の完了後**とする。工程6までは開発・検証・配布準備であり、公開tag / GitHub Releaseを作成しない。振る舞いを追加・変更するすべての実装と自動化を、独立した期待値によるred → green → refactorで進める。

## 以前の基準と当時の優先順

2026-10-14のmainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。追加実装はDraft PR #81の依存branchで、mainと分ける。最新の実測・未達は[status](status.md)、sourceごとの証拠は[工程1〜3のledger](steps-1-3-acceptance.md)を参照する。以下の以前の基準とcheckpointは各sourceの履歴として保持する。

1. 参照・PR依存・環境・採用候補を固定する。工程1の整理は済んでいるが、新しい候補ごとにsourceと実結果を追記する。
2. 旧golden・モデル・許容差を維持し、Python参照の演算文脈を再現して画像FP32/音声ONNX・Sentisの元不合格を修正する。現在は保存native graphも旧期待値に一致しないrunnerがあり、thread条件を制御した診断で切り分ける。音声候補のsilence norm出力はCPU layer_5 / GPU layer_6から不合格。小さいgreenを全encoder成功へ置き換えない。
3. 工程2合格後、実Texture/PCM→前処理→encoder→backbone→最終768次元APIを受け入れる。Query/Document/Raw、全profile・mask・境界・所有権・例外後再利用、GPU中間readback/fallback禁止、text回帰を含む。独立した契約TDDは先に進めてよいが、実全chain合格とは別に記録する。
4. 画像検索・音声検索/再生/明示収録sample、反復資源・性能、Player・必須実機、最終空Git consumer・配布監査へ渡す。実機・Unity CI環境の確保は並行する。公開tag/Releaseは全必須gate後の別工程。

工程2・3は未完了。以下の工程1〜6、各C/I/A契約と全必須環境の範囲を維持する。

## 以前の基準

今回確認したローカルmainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`（計画更新PR #66）。基盤・公式prompt・最終画像Python参照は統合済み。画像 / 音声API、CPU画像前処理、GPU分割転送は各Draft PRで実装・検証を進めているがmain未統合。画像 / 音声sampleは未実装。実音声chainは最終cosineを全24条件で満たしたが、encoder単独の数値gateは未達でM4全体は未完了。残作業は検証だけではない。以下の最新checkpointと、過去の固定sourceの実測を区別する。[以前の計画snapshot](results/tdd-plan-checkpoint-20261013.json)、[音声前処理Runtime受け入れ](results/m4-audio-preprocessing-runtime-windows-20261013.json)、[診断未達](results/tdd-acceptance-plan-checkpoint-20261013.json)は保持する。

### 2026-10-14: GPU分割転送・実音声chain・IL2CPPの追加checkpoint

| 対象 | 固定sourceの証拠 | 残るgate / 次のTDD |
| --- | --- | --- |
| I3 CPU画像前処理 | Draft PR #67で実装、公式60条件のpixel / position一致。実画像encoder APIはDraft PR #68でFloat16 CPU / GPUの6条件合格 | FP32全6条件は元elementwise条件未達。CPU前処理を含む合成chainにも1件の固定許容差超過を保持。実Textureから最終768次元までの全chain、sample |
| C7 GPU定数転送 | [Draft PR #69](https://github.com/ayutaz/unity-embeddinggemma-2/pull/69)。Runtime `66c72fb…`で関連16 green / package253 green。8 MiB以下へ分割。native特徴量 → 実backboneはCPU / GPU × FP32 / Float16の全24条件が元cosine・診断elementwise条件に合格 | 旧Editor終了の原因確定、長い反復 / memory profiler、Byte専用case、実モデルPlayer、最終空consumer。TextEmbedder / encoderのWorkerまで変更したことにしない |
| A5 実PCM全chain | [Draft PR #70](https://github.com/ayutaz/unity-embeddinggemma-2/pull/70)。同Runtime / producer `4edbe17…`で全24出力が元の最終cosine条件に合格。NUnit4 pass / failed0 / skip0 / CLI exit0。最小cosine0.999999800599847 | encoder Python ONNX gateは不合格、current producerのSentis encoder単独gateは未実行。診断elementwise不一致1890を保持し、overall success=false。Raw / Query・入力境界、sample、反復 / Player / 必須環境 |
| C7 パッケージlinker / IL2CPP | [Draft PR #71](https://github.com/ayutaz/unity-embeddinggemma-2/pull/71)。`IUnityLinkerProcessor`を1 assertion red → 1 green、固定Git `f246eea…`でpackage254 green。GUID修正済みGit `adccab2…`でcompile / linker1 green、実IL2CPP Release / High / D3D12 probeの独立2起動でCPU / GPU全2097153要素がbit一致、両process exit0。[検証とreceipt](gpu-constant-il2cpp-validation.md) | 重みなしprobeであり、512 MiB定数・実画像 / 音声chainのPlayer合格にはしない。初回・増分ともbuild warning185を保持。実モデルPlayer・他OS・長い反復 / memoryは別gate |

PR #69 / #70のCIは全8項目成功。PR #71の初回package / Python失敗は33桁GUIDによるもので、修正コードhead `adccab2…`のCIはmodel-reference / Required CIを含め全8項目成功。後続の文書commitのCIとは区別する。GitHub Unity secretsは0件で、ActionsのUnity実行は未実行。小さいローカルPlayerの実測をActionsでのUnity成功や必須全環境の合格にしない。

次は元の画像 / 音声encoder数値gateを原因別TDDで修正し、実画像 / 音声の全chain・各role / 入力境界を固定参照へ照合する。独立したsample契約のTDDも進め、実モデルPlayer、安定性・時間 / memory、macOS / iOS / Android実機、最終空consumer、配布準備まで工程1〜6を維持する。公開tag / GitHub Releaseは作成しない。

### 2026-10-14: GPU分割転送前の画像APIと音声実chainのcheckpoint（過去source）

| 対象 | 固定sourceの証拠 | 残るgate / 次のTDD |
| --- | --- | --- |
| I4 画像encoder API | [Draft PR #64](https://github.com/ayutaz/unity-embeddinggemma-2/pull/64)、head `46cea116f505e6836f9222e53a14ac536c6b176b`。16 assertion red → Git `3c9bf7d616e0fccb4ec808280e1a59185da8e2b9`でCPU / 実GPUの16 green、package158 green / skip 0。headの[CI 38423962516](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38423962516)全8成功。寸法・位置・feature数・backendと結果所有権の合成検証 | 監査済み実vision modelでのAPI受け入れ、元FP32数値gate。constructorでgraph内部の焼き込み位置を証明したことにしない |
| I4 最終画像API | [Draft PR #65](https://github.com/ayutaz/unity-embeddinggemma-2/pull/65)、head `f6794ef4fa8cd85a0e69dcc7d144978d0ab46b59`。24 assertion red → Git `c2c5516e83339450bf7ff4c65432ef9ddae06861`でCPU / 実GPUの24 green、共通package225 green / skip 0。Runtime overlayなし。音声PR #62の共通stage契約をbaseにし、PR #64の画像encoderを含む | CPU Texture前処理は未実装（CPU試験は独立fixture）。実Texture → encoder → 画像専用backboneの全chain、全profile / role・sample / Player / 実機・最終空consumer。依存PR統合後にmain基準へ整理 |
| A5 音声backbone単独 | producer `4edbe17…`、Runtime `9c6f794…`、harness `eeec732…`。native特徴量 → 実backboneのCPU / GPU × FP32 / Float16重みは全24 caseでcosineと診断elementwise合格 | encoder・PCM前処理・最終APIを通した全chainとは別gate。CLI観測はunresponsiveでexit 1、同operationの保存結果は完了・合格。NUnit完了集計を回収したとはしない |
| A5 音声実chain | 同producer / Runtime、harness `da5e8f0…`。実PCMからAPIまでFP32 CPU / GPU、Float16 CPUは各6信号の最終cosine合格。Float16 GPUは2件返却後sineの正規化で失敗、後続3件未実行。NUnit3 pass / 1 fail / skip 0 / CLI exit 1 | 元encoderのelementwise不合格を保持。current producerのSentis encoder単独gate、Raw / Query・長さ境界・sample・必須環境は未達 / 未実行 |
| C7 GPU終了・cache | tiny GPU probeはclean restart後CPU / GPU合格。その後の実chainはFP32 CPU6件完了、GPUは5件保存してEditor終了・EOF、Float16未実行。D3D12 logは16 MiB bufferに512 MiB要求を記録。cacheのsource / graph / 版 / quantization / file size / SHA監査は5 red → 関連24 green | 原因未確定。小さいGPU回復を全chain修正にしない。saved model再利用の重い実測・性能倍率は未測定。転送・資源寿命を原因別のredへ落とし、元条件を再照合 |
| A4 Required CI | Draft PR #60のhead `d1c0d8c`の[CI 38422572898](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38422572898)は`audio ONNX export mismatch: silence`で失敗。Python4環境 / package / actionlintは合格 | 同じ数値gateの再現。network障害として再実行しない。native-referenceのskipはpassに数えず、許容差を維持して修正する |

受け渡しは[画像encoder receipt](https://github.com/ayutaz/unity-embeddinggemma-2/blob/46cea116f505e6836f9222e53a14ac536c6b176b/docs/results/m3-image-feature-encoder-tdd-windows-20261014.json)、[最終画像API receipt](https://github.com/ayutaz/unity-embeddinggemma-2/blob/f6794ef4fa8cd85a0e69dcc7d144978d0ab46b59/docs/results/m3-image-embedder-composition-windows-20261014.json)、[音声の失敗・未完了記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/a9cef60/docs/m4-audio-real-chain-validation.md)で固定する。実測sourceを最新headの再実行に付け替えない。

次は(1)GPU終了とencoder数値差の原因別再現・修正、(2)実画像 / 音声chainの固定参照照合とCPU画像前処理、(3)画像検索 / 音声再生・明示収録sample、(4)安定性・Player / 必須実機、(5)配布準備へ進む。実モデルgateが未達の間も独立したsample・CPU画像前処理のTDDは進められる。以下の工程1〜6・全必須環境は維持し、公開tag / GitHub Releaseは作成しない。

### 2026-10-13のcheckpoint（過去sourceを保持）

| 対象 | 現在の証拠 | 次のTDDと受け入れ |
| --- | --- | --- |
| C3 公式prompt | PR #55はmainへ統合済み。prompt / placeholder / maskの契約実装を保持 | 実encoderのfeature数・順序・所有権・途中失敗を新しいredから接続し、実画像 / 音声の最終vectorとtext回帰を確認 |
| I1 / I4 最終画像参照 | PR #56はmainへ統合済み。実モデルPythonは3 profile / 6画像で768次元と保存encoder → 保存backboneを照合。producer `35d357263ae29de4db1ad93d1cac3616ea1e61b4`、[CI 38356261690](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38356261690)。最新baseのhead `9814ce9d8bd70fa07cbd191867d93d7f4dd461f5`の[CI 38364660005](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38364660005)全8成功を確認して統合 | zero-audio profileのSentis import・CPU / 実GPUの全chain照合、I2の元FP32条件、API / sampleは別gate |
| A4 標準Swish単独 | PR #57は統合済み。[小さいWindows probe](results/m4-native-silu-swish-probe-windows-20261013.json)はCPU / GPUの全4条件合格。GPUは固定許容差内であり、完全一致ではない | 単独operatorの合格を全encoderへ流用しない。全encoder ONNX、保存 / 再読込と元6信号を確認 |
| A4 native音声ONNX候補 | Draft PR #60、確認head `70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1`。native保存参照は元6信号 / 2 profileで固定許容差合格、NumPy ONNXは全6信号不合格。[producer 271f…の記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1/docs/results/m4-audio-all-six-native-reference-real-20261013.json)。Sentis FP32も全6信号 × CPU / GPUの12条件でelementwise不合格、cosineのみ全条件合格。[実測](https://github.com/ayutaz/unity-embeddinggemma-2/blob/70cb9eefae1caa8d6a5c78958f3fefe0121aa1f1/docs/results/m4-audio-full-encoder-fp32-sentis-windows-failed-20261013.json) | 元の不合格を再現するstage参照をTDD追加。native最終値と元参照、trace最終値と元graphの一致を先に確認して原因修正へ。元6信号のNumPy / Sentis CPU / 実GPUを再照合し、採用Float16も別gate。小さいgreen・cosineだけで実モデル成功にしない |

このcheckpointのPR / branch状態は2026-10-13の確認時点。各後続PRでsource SHA・入力 / model hash・red / green・未実行を更新する。工程1〜6の範囲と各C / I / A契約は下記のまま維持し、正式公開は全完了後に別途扱う。

### 音声API・最終参照の追加checkpoint

| 対象 | 確認した状態 | 次のTDD / 受け入れ |
| --- | --- | --- |
| A5 最終Python参照 | Draft PR #60の現在headは `85b84a012f00a0a9d5239af56ab4a71572ec8f86`。producer `4edbe17a8105233c5aaa1514faa2862b08a4f7e0`のDocument-role参照は元6信号 / 2 profile / 768次元で保存native encoder → 保存backboneを照合済み。[固定receipt](https://github.com/ayutaz/unity-embeddinggemma-2/blob/85b84a012f00a0a9d5239af56ab4a71572ec8f86/docs/results/m4-audio-final-reference-real-20261013.json) | zero-image音声専用backboneの実Sentis CPU / GPU × FP32 / Float16を照合する。参照source / hash / 欠落case・数値不一致の拒否を先にredから保証。これはbackbone単独であり、未達のencoder条件を解消したことにしない |
| A4 / A5 Runtime API | Draft PR #62の現在headは `280250e4ebba9d1006a74c377d39181fcbd93a55`。AudioFeatureEncoder / AudioEmbedderのTDD実装あり。最終APIは20 assertion red → 20 green、固定Git `9c6f7943ca6319849dd3db54611acc477e7d0981`でpackage 185 green / skip 0。[固定receipt](https://github.com/ayutaz/unity-embeddinggemma-2/blob/280250e4ebba9d1006a74c377d39181fcbd93a55/docs/results/m4-audio-embedder-composition-windows-20261013.json) | 合成graphの所有権・GPU受け渡し合格と、実モデル全chainを区別。元6信号の実前処理 → 実encoder → backbone → APIを固定Pythonへ照合し、Raw / Query role・長さ境界・text / image回帰も必要なredから追加 |

現headはGitHubのopen PR一覧で確認した。上記の実測は各receiptのsourceであり、現headの再実行結果ではない。元ONNX全6信号・Sentis FP32全12条件の不合格は保持する。新しい最終Python参照の合格やRuntimeの合成greenでM4を完了にしない。

直近は、(1)参照文脈と数値差の再現・修正、(2)画像 / 音声専用backboneの独立受け入れ、(3)最終APIの契約実装と実chain照合、(4)検索・再生・明示収録sample、(5)Player / 必須環境、(6)配布準備の順に進める。数値差の修正中も独立したAPI・sample状態契約のTDDは進められるが、M3 / M4全体の合格判定は元のencoder gateと全chain結果が揃ってから行う。

| 範囲 | 確認済み | 次に満たす条件 |
| --- | --- | --- |
| M2 / C1・C7 | Windows Editor / Playerのテキスト実測あり。導入helperのPR #36はPR #31の作業ブランチへ統合済み | mainへの導入手順反映、安定性・性能の原因別修正、最終candidateの空Git consumer、native入力・他OS。既存consumerの更新を新規導入と数えない |
| C3 共通backbone | [PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43)はmainへ統合済み。最終head `4feac7551bd99f2571a667f237ace85ba2d7093f`、[CI 38331086059](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331086059)成功。配置・cacheの小さい68件greenに加え、実1024長graphのCPU / GPUCompute × FP32 / Float16全4条件合格 | 実測sourceは固定Git `21e598c5b53d5ad73557ee6d606b12770254d73b`。統合headへ実測を付け替えない。公式promptはPR #55で統合済み。実encoder feature・GPU前処理との接続と最終空consumerは別gate |
| M3 / I1・I2 | Python画像encoder / projectorは3 profileで合格。Windows Sentis Float16 CPU / 実GPUの6条件合格 | FP32の6条件はcosine合格でも固定elementwise許容差を超過。不一致演算を小さいredにし、全12条件を満たす。I3の実装 / Windows検証は次行。実encoder接続・最終768次元API / sampleは未完成 |
| M3 / I3 | PR #52でGPU画像前処理を実装。Python14 red → 14 green、Unity13 red → green、最大寸法dispatchの1 redを修正。固定Git `71284e3`でpackage118 green、公式24ケースのpixel / position一致・次段GPU受け渡し。[receipt](results/m3-image-preprocessing-final-git-windows-20261013.json) | 実encoder接続・最終API / sampleは未完了。GPU profiler、長い反復 / memory、Player・他OS・最終空consumerは別gate |
| M4 / A1〜A3 | 前処理単独PR #38 / #46、PCM入力PR #48、前処理Runtime PR #50はmainへ統合。21 red → 21 green、固定Git package全108 green、CPU / 実GPU各6信号の全feature・mask合格、次段検証graphへのGPU Tensor受け渡しを確認。[実測と失敗履歴](m4-audio-preprocessing-runtime-validation.md) | 実音声encoder・backboneへの連結、最終768次元API / sample・最終空consumerは別gate。初回の測定用ReduceMean不一致を成功で消さない |
| M4 / A4 | [PR #40](https://github.com/ayutaz/unity-embeddinggemma-2/pull/40)、確認head `e13adcd96ed3e10802131dfc94c48d9533f37ee2`。feed-forward単独診断をTDD追加し関連76件green | [実モデルCI 38331914586](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331914586)はsilence不合格。layer 0.feed_forward1の最初の微差はactivation、最初の許容差超過はlayer 10。単独診断とfull traceは完全一致。差の発生箇所が分かった段階であり、原因修正と元の6信号の合格は未完了 |
| 必須環境 | Windowsの条件付き証拠とAndroid APK buildあり | macOS / iOS / Android実機の3モダリティCPU / 実GPU、sample、時間 / memory・終了の合格証拠が必要 |

共通backboneの[固定headの実測](https://github.com/ayutaz/unity-embeddinggemma-2/blob/21e598c5b53d5ad73557ee6d606b12770254d73b/docs/results/multimodal-backbone-sentis-windows-20261013.json)と音声診断の[固定headの不合格](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4ae52890da50ffc570cba4902d96e95b182cc9c6/docs/results/m4-audio-layer-real-failed-20261013.json)を参照する。初回backbone import失敗も保持する。PRのCI成功とUnity・実機の受け入れを分け、後続headで過去の実行sourceを置き換えない。

開発検証は承認済みUnity 6000.3.21f1 / Sentis 2.6.1。mainのEditor固定はPR #31未統合のため6000.3.16f1であり、版更新はPRで反映する。固定Git Runtime更新後の実backbone受け入れは同じoperationの完了報告を追跡し、2026-10-13 01:02:13 UTCに4 passed / failed 0 / skip 0 / CLI exit 0を確認した。[更新consumerのreceipt](results/multimodal-git-runtime-upgrade-windows-20261013.json)。Runtime source overlayなし、requested / resolved SHA一致。これは既存consumerの更新で、最終candidateの空consumer再現ではない。公式Test Framework 1.6.0のembedded条件も保持する。今回のRuntime変更に必要なTDD / 公式信号は同Editorで追加実行した。計画更新だけを理由に未変更の重いgraphを再実行しない。

## 着手順と依存

| 工程 | 対象と先に作る失敗テスト | 最小実装と完了条件 |
| --- | --- | --- |
| 1 M2基盤 | C1 / C7。不足依存・導入SHA不一致、resource解放漏れ・待機漏れ、cache破損の誤受け入れを原因別にredへ | 再現可能な導入と安定性修正。固定Gitでcompile・テキスト回帰・検索・正常終了。import / reload / 推論を別測定し、同条件で性能比較 |
| 2 共通基盤 | C2〜C4。revision / hash / profile違い、placeholder数・feature順 / mask違い、借用Tensor破棄・中間CPU readbackをredへ | 固定profile / 公式prompt・特徴量配置、GPU tensorの所有権と受け渡し、Runtime assemblyの実モデル受け入れ。実encoder接続時にも最終768次元を独立参照へ照合。既存text128長回帰 |
| 3 M3画像 | I1〜I6。FP32差の小さい再現、向き・色空間・resize / patch順、画像入力境界、最終vector・全順位・sample状態の不一致をredへ | GPU画像前処理 → encoder → 共通本体 → ImageEmbedder相当API → 画像検索sample → Player。Texture / RenderTexture、実CPU / GPU、有限768次元・単位長、全順位・表示・解放・終了 |
| 4 M4音声 | A1〜A7。encoder差の小さい再現、PCM長 / channel / sample rate・NaN、frame / mask、最終vector・検索・収録状態の不一致をredへ | PCM入力 → GPU前処理 → encoder → 共通本体 → AudioEmbedder相当API → 検索 / 再生sample → 明示収録 / Player。固定PCM精度とマイク入力動作を別判定、停止 / 権限失敗・解放・終了 |
| 5 全統合・必須環境 | C4〜C6。混在索引の次元 / 順位、切替の解放漏れ、actual CPU・skip・途中結果・非zero終了をpassとする監査をredへ | Windows Editor / Player、macOS Editor、iOS / Android実機で3モダリティの実CPU / GPU、sample実操作、cache・切替、時間 / memory・正常終了。未実行・OOMを合格にしない |
| 6 最終配布準備 | 版 / manifest不整合、重み混入、native依存、結果欠落を見逃す配布監査をredへ | 1〜5の全合格後、version / CHANGELOG / API文書・全sample / licenseを整理。固定candidateの空Git consumer再現と配布監査。公開tag / 正式Releaseは後の別工程 |

詳細なRed / 最小実装 / Greenは下記C表と[M3画像計画](m3-plan.md)のI表、[M4音声計画](m4-plan.md)のA表を用いる。機能完了は小さい単体テストだけでは判定せず、実モデル・実backend・sample / Playerの必要な受け入れまで追う。

Mac・端末・署名・Unity CIライセンス / runnerの確保は工程1から並行する。M2の全OS検証完了をM3の着手条件にしない。画像前処理とPCM入力境界の独立した契約は、encoder数値差の解消を待たず準備できる。最終公開に必要な条件は省略しない。日程はモデル数値差と実機確保の見通しを得てから見積もる。

### 直近の実行順と依存

I3の画像GPU前処理RuntimeはPR #52でmainへ統合し、Windows固定Git受け入れまで進んだ。C3の公式prompt / mask契約もPR #55で統合済み。次はI2の画像FP32差とA4の音声encoder差をTDDで解消し、実encoder・公式prompt・共通本体を連結する。I3のGPU profiler / memory / 他環境は別gateとして保持する。工程番号は受け渡しの順序であり、M2の他OS検証完了まで独立したM3 / M4の開発を待たせない。

画像診断PR #53はhead `d51262632b0d0369fe62719dff743e6ecd990a46`の[CI 38346643250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38346643250)が全8チェック成功だが、実Sentis診断は別判定。初回はBurstのimport失敗で推論未実行。回復後の2026-10-13 04:56 UTCの報告はcompile成功・CPU / GPUComputeの2テスト不合格・skip 0で、`New native final must fit the original pinned reference`に失敗した。case_0のtrace最終値は元の保存Sentis最終値と一致したが、独立参照の文脈確認が未達で、case_1へ到達していない。途中の`localization_validated=true`を全体成功や原因確定へ使わない。まず元 / 新参照の入力・生成環境・最終値を照合し、報告監査がこの途中成功を拒否するredを追加する。その後、文脈が揃ったstage / 演算のredから最小修正へ進み、元の全12条件を再照合する。

音声PR #40のsilence不合格は過去の固定sourceの記録として保持する。後継候補PR #60では元6信号のnative保存参照を受け入れたが、NumPy ONNX全6信号とSentis FP32のCPU / GPU全12条件は不合格。subsample単独のSentis CPU / GPU合格はfull encoderへ流用しない。次は元の不合格出力に一致するstage診断の文脈をTDDで保証し、差が出る演算の小さいred → 最小修正 → 関連green → 元6信号の全encoder再照合へ進む。

| 順序 | PRの対象 | 先に観測する失敗と、次へ進む条件 |
| --- | --- | --- |
| 済 / 残る受け入れ | I3 画像GPU前処理 | 境界 / 丸め / 所有権と最大寸法dispatchのred → green、公式24ケース・package 118 greenは完了。以後は実encoder接続・GPU profiler・反復memory・Player / 他OSの証拠を追加し、既存の同じ重い検証を計画更新だけで再実行しない |
| 2 | I2 / A4 数値差の修正、C3 公式prompt / mask契約 | 差の出る演算を小さい独立参照でredへ分離。最小修正・関連回帰後、画像12条件と音声の元6信号を照合。prompt / placeholder数・順序・mask違いは別のredから実装 |
| 3 | I4 / A5 実encoderから最終API | 各encoderが受け入れ済みとなってから実モデル連結を合格判定。入力拒否・途中失敗・再利用・二重Disposeのred → green、最終768次元 / 正規化 / 固定参照一致、既存text回帰 |
| 4 | I5 / A6 / A7 検索・表示・再生・明示収録 | 全順位 / score・未準備・エラー後復帰・停止・権限拒否のred → green。実UI操作と固定PCMの精度を個別に確認し、Playerへ渡す |
| 5 | C1 / C4〜C7 全統合・各必須環境 | 導入・切替・cache・報告監査・正常終了の不正経路をred → green。各OSの実backend、sample、時間 / memoryを最終candidateで照合 |
| 6 | 配布準備 | version / manifest / license・モデル混入 / native依存・証拠欠落を見逃す監査のred → green。全必須結果と固定candidateの空Git consumerを対応付け、公開操作へ渡せる状態にする |

M2の標準導入・安定性修正と、Mac / iOS / Androidの環境確保は上記と並行する。工程6の開始条件は工程1〜5の全必須結果が揃うこと。実行環境が未確保ならその行は未実行として残し、配布準備や正式公開の完了条件を下げない。正式公開は全対応後の別工程とし、今回の開発・検証・準備では公開tag / GitHub Releaseを作成しない。

### 機能単位の受け入れチェックリスト

- [ ] M2: 標準導入手順がmainへ反映され、最終candidateのテキスト導入・sample・安定性 / 性能・正常終了が合格。
- [ ] M3: I1〜I6。画像FP32の既知不合格を解消し、GPU前処理、実画像の最終768次元API、検索sample、Playerを受け入れ。
- [ ] M4: A1〜A7。音声encoderの既知不合格を解消し、前処理から実encoder / 共通本体のGPU連結、最終768次元API、検索 / 再生 / 明示収録、Playerを受け入れ。
- [ ] 全統合: Windows Editor / Player、macOS Editor、iOS / Android実機で3モダリティCPU / 実GPU・sample・切替 / cache・時間 / memory・正常終了の証拠が揃う。
- [ ] 配布準備: 全必須結果を固定candidateへ対応付け、空Git consumer、API文書 / sample、version / CHANGELOG / license、配布監査が合格。

チェックは予定ではなく受け入れ証拠が揃ってから付ける。正式Releaseはこの一覧の全完了後に扱う。

### 次のPR単位の詳細

1. **公式promptと実encoder接続**: PCM入力境界PR #48、共通Runtime PR #43、前処理Runtime PR #50、公式prompt / mask契約PR #55は統合済み。次に実encoderの所有権・受け渡しを、不正入力 / profile / 二重Dispose / 途中失敗のredから実装する。実encoder接続の合格判定はI2 / A4の数値gate合格後、工程6の空consumerは別gate。
2. **音声encoder差の修正**: PR #60の全6信号 / 12条件不合格を保持する。参照source / hash違い、native最終値の不一致、traceが元の失敗を再現しないケースを先にredへ落とす。診断の文脈確認がgreenになってから段階別に差を分離し、演算の独立参照red → 最小修正 → 関連green → 元6信号のONNX / Sentis CPU / GPU再照合へ進む。原因をactivationへ限定せず、単独probe・構造checker・診断追加だけではA4を合格にしない。
3. **画像FP32差の分離**: 独立参照と最初に差が出るstage / 演算の小さいredから修正し、3 profile × CPU / GPU × FP32 / Float16の12条件を確認する。cosine合格だけでelementwise不合格を上書きしない。
4. **GPU前処理から実encoderへの接続**: I3の境界・所有権の実装とWindows単独受け入れは完了。次は実encoderへの受け渡し・途中失敗・中間readback禁止の契約をredから確認する。A3の前処理Runtime単独も受け入れ済みだが、A4合格後の実encoder・共通本体への連結は別契約。I4 / A5の最終768次元とI5 / A6の全順位・sampleへ進む。

最新の音声Sentis実測は元6信号すべてを実行済み。CPUの最大許容差比は15.800193893998493、GPUは13.610273396664804で、全12条件が不合格。NUnitは2 failed / skip 0 / CLI exit 1、compile error / warning 0、GPU fallback / readback flush 0。参照producer `271f5a12ca2aab081eb330f45cf593d404330374`、harness commit `98f33c67d6fcd9181bb2baa52c4908340cbbe90e`を区別する。過去のsilenceだけの失敗を上書きせず、数値の変化を解決や高速化と扱わない。固定した `rtol=1e-4` / `atol=1e-5`は緩めない。

### 性能とローカル負荷

Pythonはtools/とuv、重み取得・変換・参照生成・広い回帰・buildはGitHub Actionsを優先する。ローカルは小さいred / greenと、代替できない実GPU・UI・端末検証に限定する。監査済みartifact / 保存model / Libraryを再利用し、同じ大きい未変更graphを再生成しない。

Unity Editorは1つを再利用する。CLI timeout / 接続切断だけを再起動の理由にせず、同operationの完了報告・PID・保存結果を追う。最適化はgreen後に同じsource / model hash・仕事量・backendで比較し、import時間、初回 / warm推論、前処理 / encoder / backbone、memory・解放を別に記録する。

## 全作業に共通するTDD手順

1. 固定仕様・独立した期待値から振る舞いのテストを書く。数値期待値は実装出力から生成せず、固定Python参照を使う。
2. 未実装・誤処理に起因する**意図したassertion失敗**を確認し、test名、source SHA、失敗理由を保存する。環境不足、download失敗、compile失敗、skipはアルゴリズムのredとは区別する。
3. その失敗を満たす最小実装を行い、対象テストと必要な既存回帰をgreenにする。
4. greenの後に整理・最適化し、同じ入力 / hash / backendで精度と性能を再確認する。速度改善と保存サイズ削減を別々に報告する。
5. 小さい契約テスト、実モデルPython、Unity実モデルCPU / GPU、Player / 実機を別段階で判定する。小さいモデルやmockのgreenで実モデル互換性を完了にしない。

調査・仕様固定は期待値を決める前段、実測は実装の受け入れである。文書変更に架空のred / greenは付けない。実行可能な再現がないタスクは「未実行」とし、公開条件から除外しない。

### 各PRで残す記録

- 対象のC / I / AタスクID、入力契約、独立した参照fixtureとhash、先に固定した数値許容差。
- red時のsource SHA・test名・意図した失敗理由・run URLまたは保存結果。既存コードの改修では、改修前に追加テストが失敗することを確認する。
- green時のsource SHA、対象テスト・影響範囲の回帰件数、failed / skipped / 未実行、実モデルと実backendの結果。
- refactor / 最適化後の回帰結果、必要な同条件の性能比較、残る依存。実装結果から期待値を上書きしない。

環境設定・手動操作・文書作成には架空の単体テストを付けず、実際の読み戻しや受け入れ証拠を記録する。設定を生成・検査するプログラムを追加する場合は、その振る舞いもTDD対象とする。

## 共通基盤のTDD分割

| ID | 先に確認するRed | 最小実装 | Greenと次段の検証 |
| --- | --- | --- | --- |
| C1 標準導入 | 固定版の空consumerで未補修compile失敗を再現。環境対応コードを追加する場合は依存検出・不足時の失敗契約がred | 原因を分離し、再現可能な導入 / テスト構成を確定。vendor cacheの手修正だけを配布手順にしない | source overlayなし固定Git解決、compile、テキスト契約 / 実モデル / sample・独立終了。既存補修条件を明記 |
| C2 参照と成果物 | revision違い、processor設定違い、hash違い、途中生成、欠落encoderを監査が拒否できずred | tools/にマルチモーダル参照 / export / manifest監査を追加 | 小さいfixture green後、Actionsで固定実モデルを生成・照合。重みと大きいテンソルはartifactへ |
| C3 特徴量と長さ | placeholder数とfeature数不一致、異なる差し込み順、padding混入、128長超過、注意窓境界で参照不一致がred | 共通本体のfeature入力とprofile別の長さ / mask処理。固定profileから開始し、対応外は明示エラー | 小さいモデルの局所 / 全体注意照合、実モデルの各採用profile、テキスト15件・既存検索回帰。M1の128長合格を新profileへ流用しない |
| C4 GPU資源と寿命 | featureのCPU readback、借用Texture破棄、二重Dispose、途中失敗のworker残留、切替後の旧bundle喪失を検出するテストがred | GPU tensor受け渡し、所有権、停止 / 再ロード・選択モジュールの解放。cache manifestを拡張 | fakeは寿命契約のみ。実GPU profilerと反復切替 / 正常終了で転送・メモリを実測 |
| C5 共通検索とsample | modality混在で次元 / 正規化不一致、不正ID、同点不定順、未準備・入力不正・エラー後再操作がred | 共通768次元の全件cosine検索と各sampleの状態管理 | 固定Python全順位・score、日英表示と実操作、失敗後復帰・解放。任意入力の品質保証とは分ける |
| C6 実行報告とCI | requested GPU / actual CPU、skip、途中結果、非zero終了、source / model hash不一致をpassにできてred | modality / profile / stage / actual backendを含む結果監査、必要workflow・Player harness拡張 | 小さい偽報告を拒否した後、実runの結果を監査。失敗・未実行を残しRequired CIと公開判定を別にする |
| C7 性能・安定性 | 資源の解放漏れ / 待機欠落など原因を絞った再現テストをredにする。測定分離や報告監査を追加する場合は誤集計を拒否するred | Float16/GPU単独対連続実行、Editor import / reload / 検索DB / allocationを別測定し、根拠のある最小修正 | 同条件の前後比較、精度・全順位・終了回帰。実測から端末別時間 / memory予算を決め、wall-clockだけの不安定な単体テストは作らない |

C2ではM1固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`を起点に、同revisionの公式processor・encoder重み・使用するライブラリcommit / uv lockを確認する。必要な資産がない場合は不足を記録し、revision変更を専用PRで参照一式とともにレビューする。古い調査の画像 / 音声定数を未確認のまま確定値にしない。

C3はprompt、特殊token、feature配置、512→768 projection、pooling対象、L2正規化を参照に合わせる。実画像profileのsoft token数はsquare 256、landscape / portrait 252であり、合成本体検証の280個を流用しない。公式processorの画像開始 / 終了token・実feature数、音声maskの2段downsampleと開始 / 終了token、query / document prefixを固定参照で確認する。誤ったplaceholder数・順序・mask・切捨てを先にredへ落とし、長さ / padding / 上限・注意窓境界を別fixtureにする。可変batch、動画、汎用ストリーミング推論は必須範囲に増やさない。

## 実行場所と証拠

| 検証 | 実行場所 | 必須記録 |
| --- | --- | --- |
| Python契約・processor・export・参照生成 | GitHub Actions、tools/で `uv sync --locked` / `uv run --locked`。依存追加は `uv add` | red / green、uv lock、source / model revision、fixture / graph hash |
| Unity EditMode / PlayMode・import・build | Actionsを優先。適切なUnityライセンスと対応runnerが必要 | Editor / Sentis版、compile、実行数、失敗 / skip、build backend・stripping |
| 実GPU・UI・実機 | 実GPU / 接続端末を持つrunner。代替できない範囲だけuloop / ローカル | requested / actual backend、GPU・graphics API、数値、画面操作、独立2起動と終了code |
| 性能・memory | 同一candidate・hash・入力で対象ごとに測定 | load / 前処理 / encoder / 共通本体 / 初回 / warm反復 / 解放、測定APIと観測範囲 |

ローカルUnityはEditor本体1つ、必要な対象テストだけを実行し、監査済みmodel / Library / build cacheを再利用する。timeoutでは同operation / PID / 保存結果を追跡し、重複起動しない。Secretsを外部forkコードへ渡さず、trusted workflowで実行する。利用者の入力収集は追加しない。

最終出力は有限768次元・単位長、fp32 cosine >= 0.999、採用するFloat16重み形式は >= 0.99、固定横断検索の全順位一致を要求する。中間テンソルの絶対 / 相対誤差、scoreの許容差、単位長の許容差はC2 / M3 / M4の参照測定で根拠を決め、実装前のfixtureに固定する。外れた実装に合わせて許容差を緩めない。

GPUでは画像 / 音声前処理とencoderから共通本体までの中間CPU readbackを禁止する。音声の収録 / decodeと入力upload、テキストtokenize、最終768次元のCPU取得は境界として記録する。Float16重みの保存形式は自動的な半精度演算・高速化の保証ではない。

## 全対応後の受け入れと公開

| 対象 | 正式公開前に必要な証拠 |
| --- | --- |
| Windows Editor / Player | 固定Git導入、text / image / audioの実モデルCPU / GPU、sample操作、IL2CPP / Release / High stripping、独立2起動・正常終了 |
| macOS Editor | 実MacのCPU / Metal GPUで3モダリティの参照照合、sample操作・解放・測定 |
| iOS Player | Mac / Xcode / module / 署名 / 実機、IL2CPP・stripping、CPU / Metal GPUの3モダリティ、配置 / cache・sample・終了 |
| Android Player | 固定版module / SDK / NDK / JDK、実ABI / graphics APIと実機、IL2CPP・stripping、CPU / GPUの3モダリティ、実jar配置 / cache・sample・終了 |

各環境でtext→text / image / audioの全順位を照合し、混在索引とモジュール切替も確認する。全encoder同時常駐を前提にせず、必要なモジュールだけを保持して測る。モバイルOOMや非対応演算は失敗として残し、GPU非対応をCPU fallbackで合格にしない。Web / WebGPU、動画は今回の正式公開の必須条件ではない。

版更新、配布manifest、CHANGELOG、ライセンス監査に自動化を追加する場合も、不一致 / 重み混入 / native依存 / 不正版を拒否するredから実装する。コードApache-2.0とモデル / fixtureの利用条件を分ける。モデルをGit / LFS / UPM / Releaseへ混入させない。

すべての必須結果を最終candidateのsource / model hashへ対応付けてから、version更新PR → 全回帰 → 固定candidateの空consumer再現・配布監査までを準備する。正式公開はその後の段階とし、同commitのtag固定空consumer再現 → 正式GitHub Release → tag / version / Releaseの読み戻しを行う。準備中は公開tag / Releaseを作らない。途中の開発PR統合は可能だが、M2のtext-only Releaseを先に作らない。手動公開はTDDのテスト結果ではなく、監査済み成果物の公開操作として記録する。

## PRの分割

1. M2の標準環境再現と必要な安定性修正（原因別、PR #31の未達条件を解消してから統合）。
2. PR #34のC2 / C3を起点に、Sentis新profile・Unity入力監査・GPU特徴量受け渡しと参照 / artifact監査の残り。大きければ別PR。
3. M3の参照・encoder export → GPU前処理 → end-to-end API / sample。
4. M4の参照・encoder export → GPU音声前処理 → end-to-end API / sample。
5. C4〜C6の残りと全モダリティ・各OSの修正 / 検証。変更ごとに必要な回帰を実行。
6. 全必須条件合格後の最終配布PR。正式公開は準備完了後の別段階。

各実装PRへ対象契約、red / green証拠、実モデルの合格 / 失敗 / 未実行、run URL、残る依存を記載する。main直pushは禁止。モデル形式は検証済み`.pt2` → `.sentis`を起点とし、標準ONNX再exportは非互換または実測の利点がある場合だけ同じTDD・数値検証で追加する。
