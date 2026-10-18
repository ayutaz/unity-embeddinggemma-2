# 同一入力の画像・音声反復測定

2026-10-18。SDK `e28a5fd`、検証コード `0911348` を固定した別consumerで測定する。既存の数値検証で入力を変えて呼び出した結果は、同一入力のwarm反復として数えない。

各module / precisionを別のEditorプロセスで起動し、同じTextureまたはPCMを使ってfirst / warm / repeat-1 / repeat-2を実行する。load、入力準備、前処理、encoder、backbone、releaseを分け、各出力768次元を固定Python参照と独立に比較する。参照一致・資源解放・skipなし・正常終了が必要で、失敗と未実行は別記する。firstは新しいプロセス・sessionの初回であり、OS、shader、Library cacheを消去したcoldではない。

CPU値はEditorプロセス全体のCPU時間、wall値は待機を含む経過時間である。両者の差をGPU時間と解釈しない。GPUメモリはWindowsの対象PIDのDedicated / Shared Usageを実際のcounter timestampで対応付ける。観測最大値は保証されたpeakや必要VRAMではなく、短い区間にサンプルがなければnullを保持する。Profilerや別プロセスのshader compilerの時間は、このCPU値から独立している。

## 確認済み

- image-square / FP32: 同一入力4回の参照一致、解放、1 pass / fail0 / skip0 / exit0。[CPU・wall監査](results/media-repeat-image-square-fp32-0911348-first-audit-20261018.json)、[GPUメモリ観測](results/media-repeat-image-square-fp32-0911348-first-memory-audit-20261018.json)。counter timestampの修正後に観測を開始したため、loadと初回前処理のメモリは欠測。
- image-square / Float16重み: 同じ条件の4回、参照一致、解放、1 pass / fail0 / skip0 / exit0。[CPU・wall監査](results/media-repeat-image-square-float16-weights-0911348-first-audit-20261018.json)、[GPUメモリ観測](results/media-repeat-image-square-float16-weights-0911348-first-memory-audit-20261018.json)。
- counter区間集計: timezone、対象PID、半開区間、実0と欠測、異常status、古いrequest-start timestampの除外、重複検出を11件Red→既存込み23件Green / skip0で確認。[TDD記録](results/gpu-memory-intervals-tdd-20261018.json)。

正方形画像のwarm反復ではencoder区間が支配的だった。FP32とFloat16の実行順やcache条件は一致しないため、両者の差を形式による高速化と主張しない。CPU fallbackの証拠とも扱わない。

## 全5 profile・両形式の結果

全10プロセス・計40回で、同じ入力の4反復、独立Python参照との一致、解放、各1 pass / fail0 / skip0 / exit0を確認した。実行後の固定source 538ファイルも再照合した。[集約監査](results/all-media-identical-input-cpu-wall-memory-0911348-20261018.json)から個別のCPU・wall・メモリ監査へたどれる。環境はWindows / Unity6000.3.21f1 / RTX4070 Ti SUPER / D3D12。

下表は初回後3回の観測範囲（秒）。平均性能や形式間の高速化ではなく、ロード済みsessionの同一入力に限る経過時間である。

| profile | FP32 warm・反復 | Float16重み warm・反復 |
| --- | ---: | ---: |
| image-square | 19.035–31.843 | 14.513–19.755 |
| image-landscape | 14.321–14.601 | 13.840–14.167 |
| image-portrait | 13.067–13.215 | 14.209–14.901 |
| audio-1119 / silence | 5.206–5.343 | 5.226–5.856 |
| audio-839 / padded_noise | 5.307–5.397 | 5.646–5.751 |

ロードは画像32.35–40.44秒、音声60.04–79.55秒だった。GPUの使用と低遅延は別の条件であり、速度改善は未確認である。CPU・wallの差から純粋なGPU稼働時間を逆算しない。GPU memory counterの観測最大値はprecision間で単調には減っておらず、Float16の保存容量減少をVRAM削減の保証へ拡張しない。

同じログからEditor startup telemetryとreload / refreshを別記した。startup / reload / refreshは重なり、これらを合計して起動時間や推論時間にしない。startup集計は10件Red→関連26件Green / skip0。[TDD記録](results/editor-startup-observation-tdd-20261018.json)。既存Libraryを使った反復起動のimport観測と、空LibraryやSearch DBの再構築時間は区別する。

## Editor Searchの別測定

モデルなし・nographicsの新規measurement projectで、既定indexの初期化・準備完了待ちと、Packages / WantsMoreを指定したasset provider検索を分けた。[監査](results/editor-search-a62b1a1-observation-20261018.json)。固定SDKファイルを4回とも取得し、1 pass / fail0 / skip0 / exit0、実行後540 sourceファイル一致を確認した。初期列挙0.208070秒、追加の準備待ち0.016279秒、準備後の検索0.230232 / 0.035131 / 0.009126 / 0.007996秒だった。

検索にはファイルパス候補と既定indexの経路が含まれる。これはindex単独の検索時間、明示的なindex再構築時間、モデル推論、GPU、GC allocationの受け入れではない。先行の[未準備probe](results/editor-search-first-not-ready-20261018.json)と、[準備完了後もパッケージ候補0だったprobe](results/editor-search-ready-no-package-match-da21f2c-20261018.json)は各1 fail / skip0 / exit2のまま保持する。既知ファイルの取得条件は緩めていない。

## 次の測定と完了条件

GPU stage時間の校正、非同期処理の割り当て、必要なEditor index再構築・cold import / allocationの受け入れが残る。明示的なCreateIndex / rebuildを呼ばなくても、Search APIが既定のLibrary indexを初期化する場合があるため、純粋な読み取りと主張しない。優先度1の公開empty-cache取得、優先度2の実マイク・最終candidate操作も、これらの測定では完了にならない。
