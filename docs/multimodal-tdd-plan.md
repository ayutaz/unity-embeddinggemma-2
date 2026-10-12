# M2・M3・M4から正式リリースまでのTDD計画

更新日: 2026-10-13。ユーザー指定により、**正式リリースはM2〜M4の全対応と全必須検証の完了後**とする。開発中のPR統合と正式リリースを分け、今回の計画更新では公開操作を行わない。すべての振る舞いの実装・修正・自動化を、以下のred → green → refactorで進める。

## 現在の基準と作業順

- 今回確認したmain: `e35470212f616b020c0b150a4c515a007f0b937f`（PR #42まで統合）。[main CI 38320110239](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38320110239)は成功。テキストUPM / 検索サンプルとWindows M1の証拠は維持する。
- C2 / C3の一部として、project済み特徴量入力・1024長backboneのexport / 参照 / Actions接続は実装済み。[PR #34 CI 38306145598](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38306145598)で実text重みと合成特徴量の2ケース、minimum cosine 1.0を確認した。実encoder featureを使う共通本体とUnity新profileは未検証。[証拠と範囲](multimodal-backbone-validation.md)。
- [PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43)のhead `2e89a61925ff18868fb1e7351ad4b962856b5133`に共通本体Runtime APIを追加した。初回21件と破棄featureの追加1件の意図したredを確認し、最終は新規24件と既存38件の計62 passed / skip 0。小さい実GPUでfeature backend維持・所有権・CPU計算fallbackなしを確認し、[CI 38322389996](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38322389996)も成功。draft / 未統合であり、開発sourceをconsumerのEditor test assemblyへ配置した結果。実backboneのSentis照合、新Runtime assemblyの固定Git導入、実encoder接続の合格ではない。[記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/2e89a61925ff18868fb1e7351ad4b962856b5133/docs/multimodal-runtime-validation.md)。
- PR #31はdraft。6000.3.21f1でのWindows実測成功は検証consumerの補修条件付きであり、未補修compile、新candidateの空Git consumer導入、Float16/GPU性能の確認が残る。詳細は[M2基盤計画](m2-release-plan.md)と[実測記録](m2-unity-21-upgrade-validation.md)。
- 開発対象は承認済みUnity 6000.3.21f1 / Sentis 2.6.1。mainのUnity版変更はPR #31未統合のため未反映。版を黙って変更しない。
- M3の画像encoder / projector exportはPR #37でmainへ統合済み。[実モデルPython CI 38310718250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38310718250)の3 profile / 6ケースでminimum cosine `0.9999997615814209`。`vision_encoder_projector_only`であり、Unity・GPU前処理・最終768次元の合格ではない。
- M4の音声前処理exportは[PR #38](https://github.com/ayutaz/unity-embeddinggemma-2/pull/38)のhead `0424734291dc4cb7606b8bae53086f4911252e94`に実装し、mainへ統合済み。[CI 38313045325](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38313045325)全8 job成功、公式前処理対exportの6ケースで最大絶対誤差 `0.0038890838623046875`。`audio_preprocessing_only`であり、音声encoder / Unityは未検証。
- M3画像 / M4音声のend-to-end API / sampleはmainに未実装。macOS / iOS / Android実機の合格証拠は未取得。Unity CI用Secrets / 専用runnerと端末の確保を並行して進め、CI成功と実機GPU成功を分ける。

### 今回の検証を反映した優先課題

- I2: [PR #41](https://github.com/ayutaz/unity-embeddinggemma-2/pull/41)の画像Sentis fixtureをWindowsで実行した。3 profile × CPU / GPUCompute × FP32 / Float16の12条件は保存報告まで完了し、Float16の6条件は合格。FP32の6条件はcosine条件を満たすが、先に固定した要素単位の許容差を超えて不合格。原因を分離した小さいredから修正する。[全条件と実行範囲](results/m3-vision-windows-plan-snapshot-20261013.json)。GPU encoderの成功をGPU画像前処理・共通本体・最終768次元の成功へ流用しない。
- A4: [PR #40](https://github.com/ayutaz/unity-embeddinggemma-2/pull/40)の最新head `2fb5c83504f46680fd5be40265715ea5eb992eac`は未統合。[CI 38321025453](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38321025453)はSoftplus係数固定・SiLU演算順序のTDD修正後も失敗。silenceでeager wrapper対公式参照の最大差は0、保存graph対eagerは `0.00021576881408691406`、許容差比 `9.733773231506348`、不一致368要素。CIの実行sourceはPR merge ref `5b2f387519de6d2029f52d72525fadbe84152c2a`。次は段階別診断自体をTDDで追加し、最初に差が発生するstageを特定して原因別の小さいredを作る。許容差を変更せず修正・実重み再照合する。小さい契約greenやcosine合格を全条件のgreenとして扱わない。
- C4 / C7: 重いUnity測定は同一operationの完了報告を追跡し、CLI観測timeoutを再実行の理由にしない。import / 保存の再利用、worker / tensor / modelの所有権と解放を先に検証し、改善は同じ仕事量で測る。異なる条件数のrun時間だけで高速化を宣言しない。

この2件の数値不合格解消とM2基盤の残る再現性確認を最初のPR単位にし、共通本体・I3以降・A2 / A3以降へ渡す。以下の6工程とC / I / Aの完了条件は維持する。正式Releaseと公開tagは全対応後の別工程とする。

### 最初の実装単位と停止条件

| 単位 | 先に確認する失敗 | 合格してから渡すもの |
| --- | --- | --- |
| 音声保存graphの段階別診断 | stage欠落・順序違い・診断失敗による元の数値失敗の消失を拒否する小さいテスト | 公式のsubsample / 各encoder層 / projectorと保存graphを比較し、最初の差を特定。原因の再現red → 最小修正 → 元の実モデル6ケースのgreen。診断追加だけではA4合格にしない |
| 共通本体Runtimeの実モデル受け入れ | 新しい契約変更は不正profile / 入力 / 所有権のredから始める。既存APIの受け入れは固定参照との比較で行い、架空のredを作らない | PR #43の62件を維持し、監査済み1024長artifactを保存 / 再読込、CPU / GPUCompute × FP32 / Float16で最終768次元を照合。合成project済みfeatureと実encoder featureの結果を分け、最後にRuntime assemblyの固定Git導入も確認 |
| 画像FP32数値差 | 独立参照に対し最初に差が出る演算 / stageを小さい入力でredにする | 修正対象を絞って再照合し、3 profile × CPU / GPU × FP32 / Float16の12条件を満たす。未変更の重い処理は再生成せず、修正の影響範囲に必要な実測だけを行う |

M2の再現性確認を継続しながら、互いに依存しない参照診断・画像前処理の契約・音声入力境界のテストは準備できる。実encoderの不合格を最終API / sampleの完了で埋め合わせない。OOM、演算非互換、未取得の実機は失敗 / 未実行として残し、その条件に依存しないタスクを進める。

| 順序 | 範囲 | 次へ渡す条件 |
| --- | --- | --- |
| 1 | M2基盤の再現・安定化 | 標準consumerのcompile、固定Git導入、テキスト回帰・終了・resource管理。既知の性能問題の原因と採用条件を記録 |
| 2 | 共通マルチモーダル基盤の残り | 実装済みbackboneをSentis import / CPU / GPUへ接続。固定processor契約、Unity入力profile監査、GPU特徴量受け渡し、artifact監査。テキスト128長の既存APIを回帰確認 |
| 3 | [M3画像対応](m3-plan.md) | GPU前処理 → 画像encoder → 共通本体 → 768次元。実モデル精度、テキスト→画像順位、sample操作・解放 |
| 4 | [M4音声対応](m4-plan.md) | GPU音声前処理 → 音声encoder → 共通本体 → 768次元。実モデル精度、テキスト→音声順位、sample操作・解放 |
| 5 | 全モダリティの統合・必須環境 | Windows / macOS Editor、iOS / Android PlayerでCPU / 実GPU、配布candidate・UI・時間・メモリ・正常終了 |
| 6 | 最終配布準備 | 1〜5の全合格後にversion / 文書 / license / candidate固定consumerを確認。正式リリースは全対応後の別工程 |

端末・Mac・署名・CIライセンスの確保は1から並行する。M2の全OS検証や正式公開をM3・M4の着手条件にしない。共通基盤に必要な再現性を確保できたらM3へ進み、他OSの不足は最終公開の必須条件として残す。M4のprocessor調査・小さい参照fixture準備もM3中に進められる。日程は実測・環境確保後に見積もり、現時点で完了日を約束しない。

### 次に着手する単位

| 順序 | タスクと依存 | 最初に固定するテスト | 次へ渡す証拠 |
| --- | --- | --- | --- |
| 1 | C1 / C7: M2の残る導入・安定性 | 原因別の導入契約、解放 / 待機漏れ、誤集計を拒否するテスト | 未補修と補修条件を分けた固定Git consumer結果、Float16/GPU同条件比較。単なるtimeoutを再起動の条件にしない |
| 2 | C3 / I2: 共通本体と画像exportのSentis受け入れ | 独立Pythonのshape・全feature・数値、実backendとfallbackの検査 | 固定artifact / hashでCPU・GPUCompute × FP32・Float16。既にPythonで合格したexportへ架空のredは付けず、非互換があれば原因の小さい再現テストから修正 |
| 3 | I3 → I4 → I5 / I6: 画像のGPU前処理・API・sample / Player | resize / patch順 / 色空間の不一致、placeholder / mask不整合、検索順位 / 寿命の不正 | Texture / RenderTextureから最終768次元までの実GPU、text→image全順位、UI・解放・独立終了 |
| 4 | A1〜A4 → A5 → A6 / A7: 音声の残るencoder・GPU前処理・API・sample / Player | PR #38の固定前処理を起点に、入力境界・encoder mask / token数・最終vector・検索 / 収録状態の不正 | 音声encoderの独立Python照合後、Sentis実CPU / GPU、text→audio全順位、再生 / 明示収録 / 停止。固定PCMとマイクを別判定 |
| 5 | C4〜C6: 全モダリティ・全必須環境 | 混在次元 / 順位、cache破損、GPU fallback、途中結果 / 非zero終了の誤判定 | Windows / macOS Editor、iOS / Android実機の3モダリティ精度、UI、時間 / メモリ・解放・終了 |
| 6 | 最終配布準備 | 配布監査の版不整合・モデル混入・native依存・不足結果の見逃し | 固定candidateの空consumer再現、API文書・全sample・CHANGELOG・license・結果一覧。公開tag / GitHub Releaseはこの工程では作らない |

未実装の振る舞いは下記C / I / A表のredから開始する。実測で初めて見つかった不具合は、原因を分離した失敗テストを追加してから修正する。各工程は実装済み / 小さいテスト合格 / 実モデルPython合格 / Unity・実GPU合格 / 実機合格を別に更新する。

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
