# M2・M3・M4から正式リリースまでのTDD計画

更新日: 2026-10-13。**正式リリースはM2〜M4の全対応と全必須検証の完了後**とする。工程6までは開発・検証・配布準備であり、公開tag / GitHub Releaseを作成しない。振る舞いを追加・変更するすべての実装と自動化を、独立した期待値によるred → green → refactorで進める。

## 現在の基準

今回確認したmainは `506b5672c68e22ac8f66076d4462e6ea143f9501`（共通Runtime PR #43、計画PR #49、PCM境界PR #48まで統合）。[main CI 38335082007](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38335082007)成功。M1のWindows受け入れを維持し、画像 / 音声のend-to-end API / sampleは未実装。残作業は検証だけではない。[直前の計画snapshot](results/tdd-plan-checkpoint-20261013.json)と[最新の前処理Runtime受け入れ](results/m4-audio-preprocessing-runtime-windows-20261013.json)を別checkpointとして保持する。

| 範囲 | 確認済み | 次に満たす条件 |
| --- | --- | --- |
| M2 / C1・C7 | Windows Editor / Playerのテキスト実測あり。導入helperのPR #36はPR #31の作業ブランチへ統合済み | mainへの導入手順反映、安定性・性能の原因別修正、最終candidateの空Git consumer、native入力・他OS。既存consumerの更新を新規導入と数えない |
| C3 共通backbone | [PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43)はmainへ統合済み。最終head `4feac7551bd99f2571a667f237ace85ba2d7093f`、[CI 38331086059](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38331086059)成功。配置・cacheの小さい68件greenに加え、実1024長graphのCPU / GPUCompute × FP32 / Float16全4条件合格 | 実測sourceは固定Git `21e598c5b53d5ad73557ee6d606b12770254d73b`。統合headへ実測を付け替えない。実encoder feature、公式prompt構築、GPU前処理との接続と最終空consumerは別gate |
| M3 / I1・I2 | Python画像encoder / projectorは3 profileで合格。Windows Sentis Float16 CPU / 実GPUの6条件合格 | FP32の6条件はcosine合格でも固定elementwise許容差を超過。不一致演算を小さいredにし、全12条件を満たす。GPU画像前処理・最終768次元API / sampleは未完成 |
| M4 / A1〜A3 | 前処理単独PR #38 / #46、PCM入力PR #48はmainへ統合。前処理Runtime PR #50は21 red → 21 green、固定Git package全108 green、CPU / 実GPU各6信号の全feature・mask合格。[実測と失敗履歴](m4-audio-preprocessing-runtime-validation.md) | GPU特徴量 / 整数maskを次段graphへ直接渡す受け入れ。実音声encoder・backboneへの連結、最終768次元API / sample・最終空consumerは別gate。初回の測定用ReduceMean不一致を成功で消さない |
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

### 次のPR単位

1. **公式promptと実encoder接続**: PCM入力境界PR #48と共通Runtime PR #43は統合済み。前処理Runtime PR #50は固定Git / 公式信号の受け入れを記録。次に公式prompt / mask構築と実encoderの所有権・受け渡しを、不正入力 / profile / 二重Dispose / 途中失敗のredから実装する。工程6の空consumerは別gate。
2. **音声activation差の修正**: 診断76件を維持し、単独診断とfull traceの完全一致を前提に、native SiLUと保存graphの丸め差を小さい独立参照の再現redへ落とす。最小修正 → 対象 / 関連回帰green → 元の実重み6信号green → Sentis CPU / 実GPU受け入れの順で進む。診断追加だけではA4を合格にしない。
3. **画像FP32差の分離**: 独立参照と最初に差が出るstage / 演算の小さいredから修正し、3 profile × CPU / GPU × FP32 / Float16の12条件を確認する。cosine合格だけでelementwise不合格を上書きしない。
4. **画像GPU前処理・実encoder接続**: I3の境界・所有権・中間readback禁止をredから実装する。A3の前処理Runtime単独は受け入れ済みだが、A4合格後の実encoder・共通本体への連結は別契約。I4 / A5の最終768次元とI5 / A6の全順位・sampleへ進む。

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

C3はprompt、特殊token、feature配置、512→768 projection、pooling対象、L2正規化を参照に合わせる。既存調査の標準画像280 soft tokenはテキスト128長へ収まらないため、画像 / 音声token数とpromptを含めた採用長を公式processorから決める。複数profileのpadding、上限・注意窓境界を別fixtureにする。可変batch、動画、汎用ストリーミング推論は必須範囲に増やさない。

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
