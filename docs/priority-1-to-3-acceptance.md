# 優先1〜3の残タスクと完了条件

更新: 2026-10-18。依頼「3まで進めてください」の範囲は、[ロードマップ](roadmap.md)のモデル準備・利用者操作・性能改善です。測定項目だけを終えて全体を完了にはしません。以下は固定された検証記録の監査であり、進行中のプロセスの成功を先取りしません。

## 残タスク

| 要件 | 現在の証拠 | 残る作業・完了条件 |
| --- | --- | --- |
| 1a: 全6 moduleの公開入力からの新規準備 | PR #86に固定descriptor・object検証を伴う取得ツール。配布候補は既存ローカルbundle | 公開先と固定commitを確定し、空の入力・出力cacheから全6 moduleを取得。producer、revision、descriptorと全objectのhashを照合し、実行receiptを保存する |
| 1b: 中断・破損・期限切れからの回復 | 取得の小さなfixture契約テストあり | 実配布データで再利用・中断後再試行・同サイズ破損修復を確認。欠落・期限切れ・revision違いを成功にせず、文書のコマンドを再現する |
| 2a: 固定Git consumerの操作 | [画像・音声登録、日英検索、キャンセル・再試行・切替・解放](results/fresh-git-native-ui-e28a5fd-20261018.json)と正常終了 | 公開手順で取得したモデルと最終candidateで再確認。非対応入力・モデル欠落・復帰も含め、最新sample修正を過去sourceの操作成功だけで合格にしない |
| 2b: プレビュー・再生・停止 | 上記の画像プレビューと再生、別のwarm起動での[音声途中停止・解放](results/native-audio-early-stop-e28a5fd-20261018.json) | 最終candidateでの操作と統合。再生位置の進行を聴感品質の評価へ拡張しない |
| 2c: 実マイク | 明示操作・失敗処理の実装と契約テスト。実録音は未実行 | 録音の明示許可後、開始・停止・中断復帰を実機で確認。権限拒否・deviceなしの扱いも記録。固定PCMやfake deviceを実マイク成功にしない |
| 3a: 初回・warm・反復とstage別CPU/GPU時間 | 新SDK `d7b2233` の画像3・音声2 profile×両形式・40回で同一入力CPU/wall・独立参照・解放・exit0、encoder 17,176と前処理/backbone 80のGPU区間を確認。[media集約](media-gpu-submission-observations.md)と[同SDKのtext両形式4反復・停止CPU履歴](text-current-sdk-observations.md)を照合 | load/releaseのGPU時間の有無と測定範囲、非同期allocationの実モデル適用を確認する。GPU Recorderと履歴の差を保持し、CPU wallとGPU queue区間を加減してkernel時間にしない |
| 3a: ホスト・VRAM・Editor負荷 | media 9プロセスとtext両形式でtimestamp付きGPU memoryを確認し、欠測を明記。[Editor検索割り当て・空/再利用Library](editor-search-allocation-observations.md)は各10 pass / skip0 / exit0。[frame GC control](async-frame-allocation-observations.md)も10 pass / skip0 / exit0だが実モデル未適用 | 明示的なindex再構築・背景index・非同期allocationの実モデル適用と測定不能の根拠を確認し、受け入れ不足を保持。プロセス全体・stage・寿命ピークと、重なるEditor時間を加算しない |
| 3b: FP32/Float16と原因別改善 | [出力確保の新SDK候補](gpu-write-only-output-allocation.md)はTDD・GPU演算等120件、両形式の固定画像各4回byte一致と計測付きwarm反復の改善、[全6 module×両形式の単独参照32ケース](results/gpu-output-d7b2233-all-module-isolated-reference-20261018.json)、両形式の混在全順位・再試行・解放・exit0を確認 | 新SDKの40回の同一入力・CPU/wall/GPU・メモリ観測を基準に残る性能項目を受け入れる。元の精度・全順位・GPU要件・正常終了を維持し、旧sourceの成功や保存容量の減少を新SDK全体の合格・高速化にしない |
| delivery: PRと最終監査 | PR #85 UX、#86取得、#87性能・Float16が未統合。#87は#85に依存 | 最新headのRequired CI、差分・依存を確認し、依頼範囲内で統合。最終sourceとmodel hashへ全記録を対応させ、上記全行の証拠を揃えてから優先1〜3を完了とする |

## そのまま維持する失敗と境界

- 最新sampleを含む固定Git `680af07` の新規consumerでは、実マウス・キーボードで画像と音声を登録し、日英2 queryの順位・scoreと768次元vectorを独立参照へ照合した。ただし所有Editorは900秒の上限でexit1となったため、[部分実行の監査](results/fresh-git-native-ui-680af07-partial-timeout-20261018.json)は操作全体を不合格とする。cache準備中のキャンセルと再試行、モデル欠落からの復帰、画像表示は観測済み。別の[同consumerのwarm追試](results/native-audio-warm-completion-680af07-20261018.json)では、音声のGPU登録・日本語検索を再照合し、空query拒否後の再実行、再生位置の進行、停止ボタン、登録・preview・session解放、終了code0を確認した。2回の停止は11.2秒音声の終端に近く、早い途中停止の証拠にはしない。終了後のpackage 315ファイル・sample 86ファイル・受動observerとroot user 4ファイルを照合した。Unity CLI Loopの欠落assembly診断・observerのtimeSamples警告・JobTempAlloc等は監査に保持し、無警告の成功としない。さらに[非対応入力の追試](results/native-input-recovery-680af07-early-stop-unproven-20261018.json)で、画像modeへの音声入力を拒否し、audio-1119への切替・再登録後の768次元参照一致と解放・exit0を確認した。同追試の十分早い停止gateは監査scriptのAssertionErrorとして保持する。公開取得・十分早い音声途中停止・実マイクは未完了である。
- FP32の12GiB guard中断、16GiB EditModeでの900秒fixture timeoutを保持する。別のPlayMode成功で過去の失敗を消さない。
- GPU RecorderとProfiler履歴の値の差、既知配列を検出できなかった割り当てAPI、短い呼び出しのVRAM欠測を保持する。未知を0や合格へ置き換えない。
- SDK `e28a5fd`、UX修正 `99cc826`、この監査前の性能branch `10a3ed5` の `Packages/com.ayutaz.embeddinggemma/Runtime` Git treeは全て `4e59c35842ef153f8ab65e46fa848c380d3c5844`。これはruntime sourceの同一性だけを示し、sample・計測コード・consumer・実行条件の同一性や最新版の操作合格は示さない。
- モデル公開と実マイクの許可待ちの間も、独立して進められる測定・文書・回帰を進める。公開先へのuploadや録音を許可済みと推定しない。

## 作業順と後続範囲

実GPUの残る計測を1本ずつ進め、Editor負荷と分離して原因を特定する。並行して公開手順と最終candidateの受け入れ準備を整え、許可された公開取得・マイク確認を実施する。最後に最新CI・source対応と全行の監査を閉じる。

Windows画像・音声Player、macOS/iOS/Androidの最終受け入れ、版更新・正式Releaseはロードマップの4〜7であり、今回の1〜3の完了条件に置き換えたり、黙って完了扱いしたりしない。
