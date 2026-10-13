# M2・M3・M4から正式リリースまでのTDD計画

更新日: 2026-10-13。**正式リリースはM2〜M4の全対応と全必須検証の完了後**とする。工程6までは開発・検証・配布準備であり、公開tag / GitHub Releaseを作成しない。振る舞いを追加・変更するすべての実装と自動化を、独立した期待値によるred → green → refactorで進める。

## 現在の基準

今回確認したmainは `5139412ae77f99d7becf2e8dd49aa6bb8b035ab3`。PR #52までの基盤に加え、全体計画PR #54、公式prompt契約PR #55、native SiLU / 標準ONNX Swish単独probe PR #57、環境ごとの診断値照合PR #59が統合済み。画像 / 音声のend-to-end API / sampleは未実装で、残作業は検証だけではない。以下の最新checkpointと、各行の過去の固定sourceの実測を区別する。[以前の計画snapshot](results/tdd-plan-checkpoint-20261013.json)、[音声前処理Runtime受け入れ](results/m4-audio-preprocessing-runtime-windows-20261013.json)、[診断未達](results/tdd-acceptance-plan-checkpoint-20261013.json)は保持する。今回の文書更新では新たなUnity実行・モデル生成は行わない。

### 最新checkpointと直近の受け渡し

| 対象 | 現在の証拠 | 次のTDDと受け入れ |
| --- | --- | --- |
| C3 公式prompt | PR #55はmainへ統合済み。prompt / placeholder / maskの契約実装を保持 | 実encoderのfeature数・順序・所有権・途中失敗を新しいredから接続し、実画像 / 音声の最終vectorとtext回帰を確認 |
| I1 / I4 最終画像参照 | PR #56の実モデルPythonは3 profile / 6画像で768次元と保存encoder → 保存backboneを照合済み。producer `35d357263ae29de4db1ad93d1cac3616ea1e61b4`、[CI 38356261690](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38356261690)。競合解消head `2c6892993b2a198eea671857fd6814efd58af09b`も全8チェック成功。main保護の最新base要件でhead `9814ce9d8bd70fa07cbd191867d93d7f4dd461f5`へ更新し[CI 38364660005](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38364660005)を再検証中 | 現headのCI成功と最新baseを確認して統合。zero-audio profileのSentis import・CPU / 実GPUの全chain照合、I2の元FP32条件、API / sampleは別gate |
| A4 標準Swish単独 | PR #57は統合済み。[小さいWindows probe](results/m4-native-silu-swish-probe-windows-20261013.json)はCPU / GPUの全4条件合格。GPUは固定許容差内であり、完全一致ではない | 単独operatorの合格を全encoderへ流用しない。全encoder ONNX、保存 / 再読込と元6信号を確認 |
| A4 native音声ONNX候補 | Draft PR #60へ固定。4 + 2 + 1 red → 関連58 green、ONNX数値gateの2 red → 関連43 / feed-forward17 green。[実モデルCI 38362823692](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38362823692)はONNX silence不合格。cosine合格でも許容差比3.971386671066284 / 292要素不一致。producer `cb7f2b00e8135d016c55ef8e00adfe9faafa387b`、head `05b48e6a2aa363d229f7eb18c163f9d5c19c94b4`は別 | 元の不合格を保持してONNX stage診断をTDD追加し、trace最終値と元graphの一致を確認して原因修正へ。元6信号とSentis CPU / 実GPUを別判定。小さいgreen・cosineだけで実モデル成功にしない |

このcheckpointのPR / branch状態は2026-10-13の確認時点。各後続PRでsource SHA・入力 / model hash・red / green・未実行を更新する。工程1〜6の範囲と各C / I / A契約は下記のまま維持し、正式公開は全完了後に別途扱う。

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

音声PR #40の実モデルgateはsilence不合格のまま。以前のvector比較2 redはnative演算保持の候補でgreenになり、標準Swish単独もPR #57で受け入れ済みだが、full encoderの実重み修正は未証明。小さいモデルの関連58 greenの候補を固定PRへ渡し、元6信号の実モデルPython → 全encoder標準ONNX → Sentis CPU / 実GPUの順とする。残り5信号の未実行を合格にしない。

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
2. **音声activation差の修正**: 診断と元の実モデル不合格を保持し、native SiLU保持・標準ONNX Swishへの候補を小さいred / greenから別PRへ渡す。小さい関連58 green → 元の実重み6信号green → 全encoder ONNX数値照合 → Sentis CPU / 実GPU受け入れの順で進む。単独probe・構造checker・診断追加だけではA4を合格にしない。
3. **画像FP32差の分離**: 独立参照と最初に差が出るstage / 演算の小さいredから修正し、3 profile × CPU / GPU × FP32 / Float16の12条件を確認する。cosine合格だけでelementwise不合格を上書きしない。
4. **GPU前処理から実encoderへの接続**: I3の境界・所有権の実装とWindows単独受け入れは完了。次は実encoderへの受け渡し・途中失敗・中間readback禁止の契約をredから確認する。A3の前処理Runtime単独も受け入れ済みだが、A4合格後の実encoder・共通本体への連結は別契約。I4 / A5の最終768次元とI5 / A6の全順位・sampleへ進む。

最新の音声実測はsilenceのみで、公式対eager差0、保存対eager最大差 `0.00021576881408691406`、許容差比 `9.733773231506348`、不一致368要素。producer sourceはPR headとは異なる `e81255dcbdf9844752bfef0fa789b54635388a47`。単独feed-forwardではclamp / RMSNorm / 最初のlinearが完全一致し、activationに最大差 `5.960464477539063e-8`が初めて出るが、全9 stageは個別許容差内。残り5信号は未実行。過去の328要素不一致も保持し、数値の変化を解決や高速化と扱わない。固定した `rtol=1e-4` / `atol=1e-5`は緩めない。

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
