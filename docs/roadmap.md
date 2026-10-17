# ロードマップ

更新: **2026-10-18**。main `35d18f8`を統合済み基準とする現在の作業計画です。未統合PRの追加検証はSDK `e28a5fd`に対応します。確認済みの事実は[検証状況](status.md)、変更しない目標は[プロジェクトのゴール](goal.md)を参照してください。

テキスト・画像・音声のGPU API、Windows text Player、画像・音声検索サンプルとcacheは統合済みです。次は、他の利用者が準備・操作を再現できる状態を作り、そのcandidateで性能と対象環境を検証します。正式リリースは全必須条件の後に行います。

## 直近の作業順

### 1. 新規利用者のモデル準備

固定参照・現行bundle形式を使い、空checkoutからtext / image / audioの参照・qualified graph取得、変換、bundle生成を再現可能にします。期限切れartifact、欠落・破損・revision違いからの回復も文書化します。

**完了条件**：既存cacheを所有しない第三者が全6 moduleを準備できること。producer / graph / model SHA、全ファイルhash、成功と失敗を記録します。モデルはGit / LFSへ入れません。

### 2. 利用者操作とサンプルUX

1の手順で用意したモデルと固定Git版を空consumerへ導入し、手動で登録→検索→キャンセル→再登録→切替→解放を確認します。既存bundleでの操作確認は先行できます。結果画像プレビュー・音声再生、マイクの明示開始 / 停止は機能ごとに別PRで実装します。

**完了条件**：日英入力・対応外入力・モデル欠落・権限拒否・deviceなし・中断・復帰を確認すること。固定PCMの数値検証と実マイクの動作検証を分け、無断の保存・送信・収集はしません。

### 3. 性能・安定性の測定と改善

candidate・入力・model hashを固定し、最初に測定します。遅い区間・資源残留の原因を絞って修正し、Float16 / GPU単独と連続実行も区別します。測定設計は2の完了を待たず進められます。

**完了条件**：load・初回・warm・前処理・encoder・backbone・解放、CPU / GPU時間、ホストメモリ / VRAMを別記すること。Editor import・検索DB・allocation・domain reloadも分離し、前後で精度・順位・正常終了を維持します。

2026-10-18時点で、Float16の全6 module・固定16ケースの単独GPU参照比較は完了しました。FP32連続試験は12GiB guard中断、16GiB別試験の900秒fixture timeoutを保持し、連続実行は未合格です。次は実GPU時間を取得する同一入力の初回・warm・反復測定を行い、load・frame待機・推論・Editor処理の寄与を分けます。同条件のFP32 / Float16、連続順位・解放まで確認してから、必要な改善をTDDで加えます。単独試験の所要時間や保存容量の減少から高速化を推定しません。

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
