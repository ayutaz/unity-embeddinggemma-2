# M2・M3・M4から正式リリースまでのTDD計画

更新日: 2026-10-13。ユーザー指定により、**テキストだけの先行リリースは行わず、画像・音声対応と全必須検証を終えてから正式リリースする**。従来文書の「M2リリース後にM3・M4」はこの順序へ変更する。これは計画の更新であり、M3・M4の実装・red / green・実モデル検証はまだ実行していない。

## 現在の基準と作業順

- 計画開始時main: `08c13671acd82dc2324b818eefe8e9aff3a5a54f`（PR #32統合）。テキストUPM / 検索サンプルは実装済み、Windows M1の証拠は維持する。
- PR #31はdraft。6000.3.21f1でのWindows実測成功は検証consumerの補修条件付きであり、未補修compile、新candidateの空Git consumer導入、Float16/GPU性能の確認が残る。詳細は[M2基盤計画](m2-release-plan.md)と[実測記録](m2-unity-21-upgrade-validation.md)。
- 開発対象は承認済みUnity 6000.3.21f1 / Sentis 2.6.1。mainのUnity版変更はPR #31未統合のため未反映。版を黙って変更しない。
- M3画像 / M4音声は未実装。macOS / iOS / Android実機の合格証拠、Unity CI用Secrets / 専用runner、公開tag / Releaseも未取得。CI成功と実機GPU成功は分ける。

| 順序 | 範囲 | 次へ渡す条件 |
| --- | --- | --- |
| 1 | M2基盤の再現・安定化 | 標準consumerのcompile、固定Git導入、テキスト回帰・終了・resource管理。既知の性能問題の原因と採用条件を記録 |
| 2 | 共通マルチモーダル基盤 | 固定processor契約、特徴量差し込み、長さ別export、参照生成・artifact監査。テキスト128長の既存APIを回帰確認 |
| 3 | [M3画像対応](m3-plan.md) | GPU前処理 → 画像encoder → 共通本体 → 768次元。実モデル精度、テキスト→画像順位、sample操作・解放 |
| 4 | [M4音声対応](m4-plan.md) | GPU音声前処理 → 音声encoder → 共通本体 → 768次元。実モデル精度、テキスト→音声順位、sample操作・解放 |
| 5 | 全モダリティの統合・必須環境 | Windows / macOS Editor、iOS / Android PlayerでCPU / 実GPU、配布candidate・UI・時間・メモリ・正常終了 |
| 6 | 正式リリース | 1〜5の合格、version / 文書 / license、固定tag consumer再現、GitHub Releaseと公開内容の読み戻し |

端末・Mac・署名・CIライセンスの確保は1から並行する。M2の全OS検証や正式公開をM3・M4の着手条件にしない。共通基盤に必要な再現性を確保できたらM3へ進み、他OSの不足は最終公開の必須条件として残す。M4のprocessor調査・小さい参照fixture準備もM3中に進められる。日程は実測・環境確保後に見積もり、現時点で完了日を約束しない。

## 全作業に共通するTDD手順

1. 固定仕様・独立した期待値から振る舞いのテストを書く。数値期待値は実装出力から生成せず、固定Python参照を使う。
2. 未実装・誤処理に起因する**意図したassertion失敗**を確認し、test名、source SHA、失敗理由を保存する。環境不足、download失敗、compile失敗、skipはアルゴリズムのredとは区別する。
3. その失敗を満たす最小実装を行い、対象テストと必要な既存回帰をgreenにする。
4. greenの後に整理・最適化し、同じ入力 / hash / backendで精度と性能を再確認する。速度改善と保存サイズ削減を別々に報告する。
5. 小さい契約テスト、実モデルPython、Unity実モデルCPU / GPU、Player / 実機を別段階で判定する。小さいモデルやmockのgreenで実モデル互換性を完了にしない。

調査・仕様固定は期待値を決める前段、実測は実装の受け入れである。文書変更に架空のred / greenは付けない。実行可能な再現がないタスクは「未実行」とし、公開条件から除外しない。

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

すべての必須結果を最終candidateのsource / model hashへ対応付けてから、version更新PR → 全回帰 → 同commitのtag固定空consumer再現 → 正式GitHub Release → tag / version / Releaseの読み戻しへ進む。途中の開発PR統合は可能だが、M2のtext-only Releaseを先に作らない。手動公開はTDDのテスト結果ではなく、監査済み成果物の公開操作として記録する。

## PRの分割

1. M2の標準環境再現と必要な安定性修正（原因別、PR #31の未達条件を解消してから統合）。
2. C2の参照 / artifact監査と、C3の共通本体 / profile拡張。大きければ別PR。
3. M3の参照・encoder export → GPU前処理 → end-to-end API / sample。
4. M4の参照・encoder export → GPU音声前処理 → end-to-end API / sample。
5. C4〜C6の残りと全モダリティ・各OSの修正 / 検証。変更ごとに必要な回帰を実行。
6. 全必須条件合格後の最終配布PRと正式公開。

各実装PRへ対象契約、red / green証拠、実モデルの合格 / 失敗 / 未実行、run URL、残る依存を記載する。main直pushは禁止。モデル形式は検証済み`.pt2` → `.sentis`を起点とし、標準ONNX再exportは非互換または実測の利点がある場合だけ同じTDD・数値検証で追加する。
