# Editor再構築のフレーム全体GCとnativeメモリ

2026-10-18。SDK `d7b2233` / validation `e035022` / Unity6000.3.21f1 / Windowsのモデルなし専用consumer。batchmode / nographicsのEditModeで、両sample 86ファイルを導入済みのLibraryを再利用した。GPU推論や実モデルのメモリ要件の合格ではない。

[実行・校正・失敗の監査](results/editor-rebuild-frame-allocation-e035022-20261018.json)は7 pass / fail0 / skip0 / exit0。5件の再構築呼び出し契約、1件の実Editor counter校正、1件の実index再構築テストである。終了後580 source・86 sample・公式Test Framework1.6.0の1168ファイルをbyte照合し、所有fixtureの削除も確認した。既存の[再構築・直接検索](editor-index-rebuild-observations.md)と、PlayModeの[別GC校正](async-frame-allocation-observations.md)はそれぞれの実行条件を保持する。

[統合sourceとの対応](results/editor-frame-allocation-integration-source-20261018.json)では、実行したvalidation全体のGit treeとSDK RuntimeのGit treeが統合先と一致することを確認した。最新sampleの実操作やモデル取得を再実行した証拠ではない。

## EditModeでの既知payload

[`ProfilerRecorder`](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Unity.Profiling.ProfilerRecorder.html)で `GC Allocated In Frame` を直接取得し、全体のProfiler / driver記録は無効のままにした。単位Bytes / Int64、wrapしない上限65536 samples、完了後最初の新counter sampleまでという既存の集計契約を使う。

| 既知payload | counter合計 | 含めたsamples |
| --- | ---: | ---: |
| 0 | 9,646 bytes | 2 |
| 8MiB・4MiBを2回 | 8,392,345 bytes | 2 |

2個の大きなpayloadは別々のcounter更新を待って確保し、どちらも対応sampleに現れた。差分は8,382,699 bytes。各windowのpayload〜payload+1MiB、差分7〜9MiBという事前の条件を維持した。noopにも割り当てがあり、indexだけのbytesを測る方法ではない。

## 実際のindex再構築

通常の `Build()` とready / updating / needsUpdateの完了待ちは維持し、その外側にopt-in captureを追加した。完了を検出した元のwall / CPU計測を止めた後、GC counterの最後のsampleを待つ。fixtureと両sample sceneはfile-path fallbackを使わず、直接index検索で確認する。

| 条件 | Build完了wall | process CPU差分 | 含まれるframeの累積managed bytes | counter samples |
| --- | ---: | ---: | ---: | ---: |
| 最初の明示再構築 | 8.053724秒 | 3.234375秒 | 24,371,666 | 56,518 |
| 同じEditorでの反復 | 0.070739秒 | raw 0秒 | 274,417 | 124 |

両方とも40文書、fixtureと両sceneを保持した。CPUのraw 0は処理なしを示さない。56,518はEditor counter samplesであり描画frame数ではない。容量超過・wrap・欠測は失敗とし、長い再構築へこの固定bufferでの成功を一般化しない。

managed Processが0を返すWindows環境では、既存のnative `GetProcessMemoryInfo` helperを使った。最初のbegin / end working setは1,341,722,624 / 1,347,969,024 bytes、process lifetime peakは1,348,427,776 bytes。反復時のlifetime peakは1,354,088,448 bytesだった。これはOSのプロセス全体の値で、GC累積割り当てやindex専有bytesではなく、両peakを足さない。

counterにはpoll、observer、含まれるframeの他のEditor処理が入る。再構築・検出・counter完了待ちの区間と、wall / CPUの区間も異なる。背景index threadの専有allocationはnullのままである。warm Libraryや既存index artifactの状態も過去試行と異なり、以前の9.146613秒との速度改善率を作らない。

## 先行失敗を保持する

- `642dd24`: 校正記録と再構築中の記録を要求し、5 pass / 2 fail / exit2の意図したRed。
- `01534eb`: 校正は成功したが、`SearchImporter.IndexingRequest` のNon-matching Profiler.EndSampleで6 pass / 1 fail / exit2。全体の成功ではない。
- `12f2e87`: CPU領域設定用の変数名が既存のCPU時間変数と衝突し、CS0136・exit1。テスト未実行。
- `fe62878`: 最初は前の失敗が残した所有fixtureの上書き防止guardで6 pass / 1 fail / exit2。内容・GUID・配置を確認し、退避のhashを照合して既知の3ファイルだけを片付けた別試行も、CPU領域を無効にした全体記録で同じProfilerエラーとなり6 pass / 1 fail / exit2。
- `e035022`: 全体記録を始めずRecorderだけで校正と再構築を実行し7 pass / skip0 / exit0。ログ抑制・LogAssert.Expectの追加や、GC上限・元の直接検索・完了条件の緩和は行っていない。

この差だけでUnity内部エラーの根本原因を断定しない。今回のcounterに不要だった全体記録を外した条件の成功であり、以前の失敗を消すものではない。captureは `EG_EDITOR_REBUILD_FRAME_ALLOCATION=1`、sample要求は `EG_EDITOR_INDEX_EXPECT_SAMPLES=1` のopt-in。通常のindex観測を実モデル・公開取得・実マイク・最終UIの受け入れへ拡張しない。
