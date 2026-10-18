# ロードマップ

更新: **2026-10-18**。公開履歴整理後のmain `4b5bf72` を起点とする現在の作業計画です。確認済みの事実は[検証状況](status.md)、変更しない目標は[プロジェクトのゴール](goal.md)を参照してください。

テキスト・画像・音声のGPU API、Windows text Player、画像・音声検索サンプルとcacheは統合済みです。現在はprivacy対応と固定参照の復旧が先です。その後、他の利用者が準備・操作を再現できる状態を作り、そのcandidateで性能と対象環境を検証します。正式リリースは全必須条件の後に行います。

## 直近の作業順

### 0. CI・個人情報対応・固定参照の復旧

Support待ちと独立して進められる作業を先行します。privacy関連の変更は既存の#89へまとめます。

1. **hookとCIを修正する。** 日本語の正当な入力を拒否するWindows文字コードの問題を、個人情報拒否も含む再現テストで直します。成果物検査を追加したことで不一致となった12件のworkflowテストは、元の失敗時保持条件とprivacy guard成功条件を両方検証する形に修正し、Windows/Linux × Python 3.13/3.14で全体検証します。検査の迂回やskipの成功扱いはしません。
2. **固定参照を復旧する。** Supportへ消失したActions成果物の回収可否を問い合わせ済みです。元の参照JSON・監査JSONはSHA256付きで私的に退避しました。元の生成処理と同一のPython tree・依存lockを残す公開SHAを確認したため、既存の音声参照workflowに既定OFFの手動復旧モードを追加します。実行SHAと旧source metadataの再現を区別し、再生成したグラフの元SHA256一致を検証します。モデルだけの一致と参照JSONを含む原本全体の一致は別に記録し、不一致のモデルを原本として公開しません。取得経路を復旧した後、必須の元音声参照CIを通します。回収不能なら新しい参照世代の作成・差分・再受け入れを別計画として示し、原本の同一性を偽りません。
3. **残存する個人情報を除去・再監査する。** 旧PR refs、古いSHA表示、キャッシュ、ActionsメタデータをSupportの対応結果と照合します。モデル・参照の完全バックアップが確認できるまで大量削除は再開しません。テキスト・画像検査と数値モデル内部の検証も別に記録します。
4. **PRを順に整理する。** #89はprivacy回帰・package・4環境Python・固定参照を含む最新Required CIと差分を確認してから統合を判定します。続いて#86（モデル取得）、#87（性能・操作）を履歴整理後のsourceで再評価します。保護設定を迂回せず、mainへ直接pushしません。
5. **Unityで実GPUの最終受け入れを行う。** 新しいconsumerでtext・image・audioの固定入力、GPUCompute経路、検索、キャンセル・復帰、切替・解放、cold/warmのメモリと速度、Windows Playerを検証します。CPU fallback・未実行・過去sourceの結果を新しいGPU合格に読み替えません。

**完了条件**：上記5項目の証拠がそろい、必須CIが再現可能になっていること。全体の完了時刻はSupportの回収可否・返答時期に依存するため、現時点では約束しません。重い取得・変換・全体検証はCIを優先し、ローカルUnityの重複起動や同じ巨大モデルの再生成は行いません。

以下の製品ロードマップは、この復旧段階と各受け入れ条件に従って進めます。[従来の単独候補の監査](results/runtime-ready-integration-2a29c68-20261018.json)は保存しますが、最新headでの全条件再実行には読み替えません。

### 1. 新規利用者のモデル準備

固定参照・現行bundle形式を使い、空checkoutからtext / image / audioの参照・qualified graph取得、変換、bundle生成を再現可能にします。期限切れartifact、欠落・破損・revision違いからの回復も文書化します。

**完了条件**：既存cacheを所有しない第三者が全6 moduleを準備できること。producer / graph / model SHA、全ファイルhash、成功と失敗を記録します。モデルはGit / LFSへ入れません。

### 2. 利用者操作とサンプルUX

1の手順で用意したモデルと固定Git版を空consumerへ導入し、手動で登録→検索→キャンセル→再登録→切替→解放を確認します。既存bundleでの操作確認は先行できます。結果画像プレビュー・音声再生、マイクの明示開始 / 停止は機能ごとに別PRで実装します。

**完了条件**：日英入力・対応外入力・モデル欠落・権限拒否・deviceなし・中断・復帰を確認すること。固定PCMの数値検証と実マイクの動作検証を分け、無断の保存・送信・収集はしません。

### 3. 性能・安定性の測定と改善

candidate・入力・model hashを固定し、最初に測定します。遅い区間・資源残留の原因を絞って修正し、Float16 / GPU単独と連続実行も区別します。測定設計は2の完了を待たず進められます。

**完了条件**：load・初回・warm・前処理・encoder・backbone・解放、CPU / GPU時間、ホストメモリ / VRAMを別記すること。Editor import・検索DB・allocation・domain reloadも分離し、前後で精度・順位・正常終了を維持します。

### 4. WindowsマルチモーダルPlayer

1〜3で安定したcandidateを使い、画像・音声sampleの配置・cache・切替・失敗回復をPlayerへ接続します。

**完了条件**：IL2CPP / Release / High stripping、実モデルと全モダリティの固定順位・score、実GPU、独立cold / warmの終了code0を記録すること。旧text Playerの証拠を流用しません。

### 5. 必須他環境の受け入れ

macOS Editor、iOS / Android実機を環境ごとのPRで進めます。環境確保は今から並行し、最終受け入れは安定candidateで行います。

**完了条件**：次表の端末・backend・モデル・source・精度・操作・メモリ・終了を記録すること。未実行やCPU fallbackをGPU合格にしません。

### 6. 最終candidateの導入・配布監査

1〜5の必須条件が合格したら、版更新、CHANGELOG、利用手順・ライセンスを整備し、空consumerで再現します。

**完了条件**：全結果とcandidate source / model hashの対応、固定Git導入、packageとモデル配布の分離、文書コマンド・sample再現、required CI成功を確認すること。

### 7. 正式公開

6が完了し、公開の明示依頼を受けてから、承認されたcandidateをtag固定で導入確認し、GitHub Releaseを作成します。

**完了条件**：tag / version / Releaseの一致と読み戻し。画像・音声を含む必須検証を省いてtext-onlyの正式版を先行公開しません。

各行を巨大な1本のPRにはしません。準備、画像表示、音声再生、収録、測定、原因別修正、環境別受け入れをレビュー可能な単位に分割します。所要時間は実機・runnerと測定で決め、未実測の性能目標を達成済みにしません。

## 正式公開まで維持する条件

| 対象 | 必須の証拠 | 現時点で不足するもの |
| --- | --- | --- |
| Windows Editor / Player | 固定Git導入、3モダリティの実モデル・固定順位、sample実操作、CPU / 実GPUを別判定、PlayerのIL2CPP / stripping・独立2起動・正常終了 | 最終candidate全体、画像・音声Player、反復測定・一部UX |
| macOS Editor | 実Mac、CPU / Metal GPUの3モダリティ、sample操作・解放・測定 | 環境確保と実測 |
| iOS Player | Mac / Xcode / build module / 署名・実機、IL2CPP / stripping、CPU / Metal GPU、配置・cache・権限・suspend / 復帰 | 環境確保と実測 |
| Android Player | 対象UnityのSDK / NDK / JDK、実ABI・graphics API・実機、CPU / GPU、jar配置・cache・権限・終了 | 最新toolchain・実機受け入れ。過去APKのbuild監査だけでは不足 |

FP32最終vectorは有限768次元・単位長・参照cosine >= 0.999、採用するFloat16重み形式は >= 0.99、固定横断検索は全順位一致を要求します。encoder中間値・scoreには元の固定許容差を使い、失敗に合わせて緩めません。各profile・role・混在索引とmodule切替を対応する参照で確認します。

CPUでの明示比較はGPU受け入れと分けます。利用者向け画像・音声推論の主経路はGPUであり、非対応演算・OOMをCPUへ切り替えて成功扱いにしません。CPUの数値文脈差、失敗・skip・未実行はそれぞれ記録します。

Web / WebGPU・動画・学習・本格的なベクトルDBは必須公開範囲に追加しません。モバイルのメモリ不足が判明した場合も記録と計画を更新し、未達を黙って除外しません。

## 開発・実行のルール

- 振る舞いの変更は、独立した期待値で意図したRed→最小実装→Green→整理。文書変更に架空のRed / Greenは付けません。
- Pythonは`tools/`とuv、重い取得・変換・回帰・buildはActionsを優先。実GPU・UI・端末の代替できない確認だけをローカルで行います。
- 監査済みmodel・Libraryを再利用し、変更のない巨大graphを再生成しません。Unityは重複起動せず、timeoutでは同じPID・operation・保存結果を確認します。
- 各PRにsource・model / fixture hash、Red / Green、実モデル・実backend、失敗 / skip / 未実行、CIと残件を記録します。main直pushは禁止、マージは依頼された範囲で行います。
- 実GPUの実行条件と数値結果、Python CIの成功、UI操作、公開操作を別々に判定します。最新headで未実行の条件を過去の成功で埋めません。

実装時のC1〜C7・I1〜I6・A1〜A7の詳細は[全体TDD設計](multimodal-tdd-plan.md)・[画像設計](m3-plan.md)・[音声設計](m4-plan.md)に保持します。そこにある過去checkpointの「次の作業」より、本ページの現在の順序を優先してください。
