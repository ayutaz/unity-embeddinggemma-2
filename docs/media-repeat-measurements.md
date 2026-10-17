# 同一入力の画像・音声反復測定

2026-10-18。SDK `e28a5fd`、検証コード `0911348` を固定した別consumerで測定する。既存の数値検証で入力を変えて呼び出した結果は、同一入力のwarm反復として数えない。

各module / precisionを別のEditorプロセスで起動し、同じTextureまたはPCMを使ってfirst / warm / repeat-1 / repeat-2を実行する。load、入力準備、前処理、encoder、backbone、releaseを分け、各出力768次元を固定Python参照と独立に比較する。参照一致・資源解放・skipなし・正常終了が必要で、失敗と未実行は別記する。firstは新しいプロセス・sessionの初回であり、OS、shader、Library cacheを消去したcoldではない。

CPU値はEditorプロセス全体のCPU時間、wall値は待機を含む経過時間である。両者の差をGPU時間と解釈しない。GPUメモリはWindowsの対象PIDのDedicated / Shared Usageを実際のcounter timestampで対応付ける。観測最大値は保証されたpeakや必要VRAMではなく、短い区間にサンプルがなければnullを保持する。Profilerや別プロセスのshader compilerの時間は、このCPU値から独立している。

## 確認済み

- image-square / FP32: 同一入力4回の参照一致、解放、1 pass / fail0 / skip0 / exit0。[CPU・wall監査](results/media-repeat-image-square-fp32-0911348-first-audit-20261018.json)、[GPUメモリ観測](results/media-repeat-image-square-fp32-0911348-first-memory-audit-20261018.json)。counter timestampの修正後に観測を開始したため、loadと初回前処理のメモリは欠測。
- image-square / Float16重み: 同じ条件の4回、参照一致、解放、1 pass / fail0 / skip0 / exit0。[CPU・wall監査](results/media-repeat-image-square-float16-weights-0911348-first-audit-20261018.json)、[GPUメモリ観測](results/media-repeat-image-square-float16-weights-0911348-first-memory-audit-20261018.json)。
- counter区間集計: timezone、対象PID、半開区間、実0と欠測、異常status、古いrequest-start timestampの除外、重複検出を11件Red→既存込み23件Green / skip0で確認。[TDD記録](results/gpu-memory-intervals-tdd-20261018.json)。

正方形画像のwarm反復ではencoder区間が支配的だった。FP32とFloat16の実行順やcache条件は一致しないため、両者の差を形式による高速化と主張しない。CPU fallbackの証拠とも扱わない。

## 次の測定と完了条件

音声audio-1119 / audio-839、画像landscape / portraitを両precisionで同じ手順で測定する。対象Unityを1つずつ実行し、各回のsource / model / input hashと実行後のsource一致を監査する。GPU stage時間の校正、非同期処理の割り当て、Editor import / search DB / domain reloadは別の残件である。優先度1の公開empty-cache取得、優先度2の実マイク・最終candidate操作も、これらの測定では完了にならない。
