# Windowsのプロセス別GPU engineを観測する

2026-10-18。Windows / Unity6000.3.21f1 / Sentis2.6.1 / RTX4070 Ti SUPER / D3D12。ロード・解放のGPU時間をCPU wallから推定せず、OSのカウンターを別に観測した。結論は、今回のサンプル間隔では実モデルのload/release全体のGPU時間を確定できない、というもの。既存のGPU時間nullを0へ変更しない。

## 単位・時計・集計

`GPU Engine / Utilization Percentage` のraw numeratorを使う。取得した型は `Timer100Ns` で、Microsoftの[型の説明](https://learn.microsoft.com/en-us/windows/win32/wmisdk/timer-algorithm-counter-types)と[計算式](https://learn.microsoft.com/en-us/windows/win32/perfctrs/calculating-counter-values)に従い、同一engineの有効な2観測の差を100ns単位の活動時間として扱う。型が `NumberOfItems64` で単位説明が空だった `Running Time` の名前だけから単位を決めない。

このホストのCounter DateTimeはLocalだった。raw `Timestamp100NSec` をUTCのFILETIMEとして解釈すると9時間ずれたため、DateTimeをUTCへ変換した時刻でstageの境界を照合する。raw provider時刻も診断用に残す。既存のGPU memory収集はUTC変換済みであり、今回の誤った初期prototypeと混同しない。

PIDだけでなくprocess作成時刻、LUID・physical adapter・engine全体のidentityを保持する。stage内に完全に入る観測点同士だけで差分を求め、境界へ補間しない。欠測、engine消失、無効status、時刻重複、counter reset、活動量が経過時間を超える値を拒否する。複数engineは重なり得るため足さず、SDK専有時間とも扱わない。

[契約・時計の監査](results/windows-gpu-engine-contracts-and-clock-70be411-20261018.json)では、集計17件のscaffold Red→17 Green、収集protocolの3 fail / 1 pass→修正後関連21 passを確認した。中間の19 pass / 2 failはdot-sourceしたmockの変数衝突で、mock変数を分けて修正した。無関係なprocessのproviderエラーは診断として残し、対象PIDのrowのstatusを個別に検査する。収集のprotocolテストはmockであり、GPU実行の合格ではない。

## 元のSDK text APIとの対応

SDK `d7b2233` / validation `10962fa` の通常text APIをFP32で2回、新しいprocess/sessionから実行した。固定日本語queryを各4回、計8 vectorの独立参照・単位長、解放、各1 pass / fail0 / skip0 / exit0、終了後554 sourceファイル一致を確認した。モデル・Libraryのコピーや再変換、SDKの演算・await・fence変更は行っていない。source・model・入力・capture hashは下記の数値監査に記録する。

[統合sourceとの対応](results/windows-gpu-engine-integration-source-20261018.json)は、SDK RuntimeのGit tree、測定したtext fixtureのGit blob、収集・集計と関連テスト4ファイルのGit blob一致を確認する。sourceの一致を、新しい実UIや公開取得の合格へ拡張しない。

- 最初の[数値・GPU API監査](results/text-current-fp32-d7b2233-gpu-engine-first-20261018.json)は合格。[OS観測の監査](results/text-current-fp32-d7b2233-gpu-engine-first-engine-observation-20261018.json)は9観測中2回の全体providerエラーにより欠測を持ち、loadのengine差分は未利用可能とした。元traceを保持する。
- 収集修正後の[数値・GPU API監査](results/text-current-fp32-d7b2233-gpu-engine-provider-fixed-20261018.json)も合格。[OS観測の監査](results/text-current-fp32-d7b2233-gpu-engine-provider-fixed-engine-observation-20261018.json)は9回を取得した。process終了後の最終queryはidentity検査で無効とし、データへ採用しない。

修正後のload wallは16.308541秒。load内には3観測点があり、14 engineすべてで、その最初と最後の間の10.620599秒の差分は0だった。これは観測点間だけの有効な0であり、load前後の未観測区間やGPU uploadの活動量を表さない。load全体のGPU時間0という結論にはならない。

releaseは10.1547msで、区間内の観測点は0個だった。4回の個別query内にも観測点はない。GPU正数を取得したSDK queryのRecorderと、この粗いOSサンプルの欠測は矛盾しない。process全体はGPU context生成前にengineがない観測を含むため、engineの欠落をまたいで全processの差分を計算しない。どちらの試行も、pure kernel時間、driverの退役完了、SDK load/release全体、通常API平均性能・最低VRAMの合格ではない。

## 収集コード

`tools/observe_owned_gpu_engine.ps1` は `-Receipt` に既存の所有process receipt、`-Output` に新しいJSONL名を取り、実processのidentityを確認して観測する。対象Unityの起動・終了やモデル取得は行わない。Python集計は `embeddinggemma_tools.windows_gpu_engine.summarize_engine_intervals`、単位・境界・失敗条件のテストは `tools/tests/test_windows_gpu_engine.py`、収集protocolは `tools/tests/test_gpu_engine_collector.py`。

この方法で分かった範囲と欠測を[ロード・解放の範囲監査](load-release-measurement-scope.md)に加える。解放の約10msを数秒間隔のcounterから再構成したり、推論のGPU値・CPU wallを代入したりしない。優先1〜3全体の受け入れは未完了である。
