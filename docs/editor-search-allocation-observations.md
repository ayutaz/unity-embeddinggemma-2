# Editor検索の割り当てと初回・再起動の観測

2026-10-18。SDK `d7b2233`、validation `93faef7`、Unity6000.3.21f1 / Windowsの検証専用consumer。SDK・validation・embedded Test Framework 1.6.0を導入し、モデルを読み込まず、batchmode / nographicsのEditModeで実行した。通常の利用者projectの起動時間やモデル/GPUの受け入れには一般化しない。

[空Libraryからの起動](results/editor-search-calibrated-allocation-93faef7-20261018.json)と[同じLibraryを再利用する起動](results/editor-search-calibrated-allocation-93faef7-warm-20261018.json)は、それぞれ10 pass / fail0 / skip0 / exit0。両実行後の558 sourceファイルと、起動前の公式Test Framework 1168ファイルを照合した。共有OS・registry・compiler cacheは消去していない。

## メモリ割り当て

停止したCPU profilerの正確なmain-thread markerについて、`HierarchyFrameDataView.columnGcMemory` をdoubleで読む。両実行の同じcapture内でnoop / 64KiB / 128KiB配列は0 / 65,568 / 131,104 bytesだった。欠測を0にせず、重複marker・別thread・非整数・非有限・不正frameを拒否する。契約テストは9件Red→9件Greenで、Unityの各10件にもこの9件を含む。

Search indexの準備完了を待った後、既定databaseの列挙と、Packages / WantsMoreを指定したasset providerの同期検索を4回測った。検索では毎回、固定SDKのTextEmbedder.csを取得できた。CreateContext・GetItems・ToArray・context解放が検索markerの内側で、参照先の確認とJSON出力は外側である。

| 起動条件 | 最初の検索の時間 | 最初の検索の割り当て | 続く3回の割り当て | 列挙の割り当て 初回 / 以降 |
| --- | ---: | ---: | ---: | ---: |
| 空Library | 0.210503秒 | 1,145,332 bytes | 34,766–38,638 bytes | 392 / 216 bytes |
| 再利用Library | 0.230176秒 | 1,091,372 bytes | 36,038–36,406 bytes | 392 / 216 bytes |

値はmarker内のinclusive managed bytesで、他thread・background index job・native/GPUメモリ・非同期完了全体を含まない。検索にはfile-path候補も含まれ、index単独の性能や明示的な再構築の測定ではない。Profilerの影響も含むため、少数の観測を平均性能・通常APIの速度改善へ拡張しない。

## 起動とimportを分ける

正確なPIDのProjectInfo telemetryでは、空LibraryのprojectLoadは55.121954秒、assetDatabaseRefreshは47.343149秒、実importは2658件だった。再利用LibraryはprojectLoad 15.267854秒、assetDatabaseRefresh 7.081417秒、実import 0件だった。これらの起動・refresh・domain reloadは重なり、合計して起動時間や推論時間にしない。reloadの個別stageと失敗・不完全な記録の有無は、それぞれの監査に保持する。

## 残る受け入れ

この観測で、同期Editor検索のmain-thread割り当て、既知サイズによる校正、空Libraryと再利用Libraryのimport/startupを記録できた。明示的なindex再構築は[追加の観測](editor-index-rebuild-observations.md)で、両sample導入後の40文書・固定fixtureと両sceneの直接検索保持を確認した。背景index専有allocation・非同期推論の割り当ては別途監査が必要である。公開empty-cache取得、実マイク、最終candidateの手動操作・最新head CIを含む優先1〜3全体は未完了である。
