# Editor Searchの明示的な再構築

2026-10-18。SDK `d7b2233`、validation `99bd89c`、Windows / Unity6000.3.21f1の専用consumer。モデルを読み込まず、batchmode / nographicsのEditModeで実行した。GPU推論・利用者project全体の性能の合格には数えない。

[統合sourceとの対応](results/editor-index-rebuild-integration-source-20261018.json)では、実行した3つのprobe / testとmeta、assembly定義が統合ブランチと同じGit blobで、SDK Runtime treeも同じことを確認した。全validation sourceの一致や最新head全体の再実行を意味しない。

## 固定fixtureでの再構築

[固定記録](results/editor-explicit-index-rebuild-99bd89c-20261018.json)は6 pass / fail0 / skip0 / exit0。5件は呼び出し・例外の契約テストで、1件の実Editorテストが既定Assets databaseの `Build()` を2回実行した。どちらも直後の `ready=false / Loading`、その後の `ready=true / Complete` と更新終了、直接インデックス検索でのfixture保持を確認した。SearchServiceのasset providerやfile-path候補へのfallbackは使わない。

| 条件 | 送信までのwall | 完了までのwall | 準備完了までのpoll |
| --- | ---: | ---: | ---: |
| 最初の明示再構築 | 0.570 ms | 3.615 ms | 7 |
| 同じEditorでの反復 | 0.379 ms | 2.683 ms | 10 |

測定前のfixture準備待ちは8.086秒で、上表には含まない。この条件のAssets indexは1文書であり、大きなprojectの再構築時間、index artifactの新規生成や平均性能へ広げない。再構築は既存のasset / index artifactを再利用し得る。Unityの[参照実装](https://github.com/Unity-Technologies/UnityCsReference/blob/6000.3/Modules/QuickSearch/Editor/Indexing/SearchDatabase.cs)はAPI調査に使い、実行した固定Editorのassembly hashは監査に保存した。

## 修正前の失敗

最初のvalidation `03b63ab` は5 pass / 1 fail / skip0 / exit2。databaseがreadyでも、fixtureの照合に失敗したため再構築へ進まなかった。この失敗を保持している。

検索結果のIDをパスやGUIDと直接比較していた。実行結果のIDは `GlobalObjectId` だったため、`SearchResult.index` から `SearchIndexer.GetDocument` を取得し、文書のsource / IDで固定assetを確認するよう修正した。独立した呼び出し契約は5件Red→5件Green、固定Editor DLLを参照したコンパイルも警告・エラー0だった。修正前後の576 sourceファイルと元のuser変更4ファイルを再照合した。

## メトリクスの境界

短い2区間のProcess CPU差分はともに0を返した。これはCPU仕事量が0という証拠ではなく、背景index処理に専有したCPU時間でもない。managed `Process` のworking set / lifetime peakも0を返したため、監査では利用不能と分類し、0 bytesの使用量にはしない。背景threadの専有allocationは未測定で、nullを保持する。再構築・import・domain reloadなど重なる時間を加算しない。

この確認は明示的な再構築の開始・完了と直接検索の保持を証明する。公開モデルの取得、実マイク、最新sampleの十分早い音声停止、ロード・解放のGPU時間、非同期allocationと背景index負荷を含む優先1〜3全体の受け入れは別に残る。

## 両サンプルを導入した既存Library

同じSDKのText Search 52ファイルとImage and Audio Search 34ファイルを、停止した専用consumerへすべて導入した。[代表条件の監査](results/editor-representative-index-rebuild-9b6f28e-20261018.json)はvalidation `9b6f28e`、6 pass / fail0 / skip0 / exit0。Libraryは再利用し、検証コードだけ新しい固定snapshotへ向けた。SDKの変更、Libraryの複製、モデルのimportは行っていない。

開始時はreadyでも0文書で、新しいfixtureも索引に存在しなかった。最初の明示再構築で40文書となり、fixtureだけでなく、両サンプルsceneもそれぞれの名前を使った直接インデックス検索で見つかった。同じEditorでの2回目も40文書と3 assetの保持を確認した。

| 条件 | 完了までのwall | プロセスCPU差分 | 準備完了までのpoll |
| --- | ---: | ---: | ---: |
| 両サンプル導入後の明示再構築 | 9.146613秒 | 2.34375秒 | 55,492 |
| 同じEditorでの反復 | 0.0232275秒 | raw 0秒 | 115 |

CPU差分は背景・Editor・計測のpoll処理を含み、短い反復のraw 0はCPU処理なしを意味しない。既存artifactの再利用状態も異なり、上の1文書の条件と比較して速度改善率を作らない。import / 起動時間と上表の再構築時間は別の観測である。

一つ前の同条件probeは5 pass / 1 fail / exit2。再構築前からfixtureが索引にあることを要求してしまい、readyでも空のcacheで停止した。次のprobeは初期内容を記録し、実際の再構築後にfixtureと両sceneが必要という条件にした。準備完了だけで新しいassetが索引に含まれるとは判断しない。失敗は監査に保持している。前のlauncherには実行後のfixture変数を上書きするKeyErrorもあり、後の独立監査で86ファイルのsource / destination hashを再照合した。API調査中の `GetDocuments` コンパイル失敗も保持し、固定Editorに公開されている `Search` / `GetDocument` へ修正して警告・エラー0を確認した。

終了後は新旧の576 sourceファイル、導入した86ファイル、manifestの切替前後、元のuser変更4ファイルを照合した。managed Processのmemory値はこの条件でもraw 0だったため利用不能とし、背景index専有allocationはnullのままである。

別の[EditMode counter校正・frame全体GCとnativeメモリ観測](editor-frame-allocation-observations.md)では、全体Profiler記録を無効にしたRecorder単独の条件で7 pass / skip0 / exit0を確認した。実再構築2回の40文書・両scene保持、counter全体の24,371,666 / 274,417 bytes、native processメモリを記録した。背景threadの専有allocationや、以前の試行との速度改善を示すものではなく、先行のProfiler不整合・compile未実行・所有fixture guardの失敗も保持する。
